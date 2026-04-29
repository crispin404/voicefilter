import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.enhancement_dataset import EnhancementDataset, enhancement_collate_fn
from model.embedding_adapter import EmbeddingAdapter
from model.model import SnoreFilter
from utils.audio import Audio
from utils.dataset_index import get_data_noise_count, load_jsonl, normalize_noise_count, resolve_manifest_path
from utils.dvector import use_d_vector
from utils.enhancement_eval import evaluate_item, summarize_rows
from utils.embedder_checkpoint import resolve_embedder_path
from utils.hparams import HParam


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def get_train_value(hp, name, default):
    return hp.train.get(name, default)


def get_train_bool(hp, name, default):
    value = get_train_value(hp, name, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('1', 'true', 'yes', 'on'):
            return True
        if normalized in ('0', 'false', 'no', 'off'):
            return False
    return bool(value)


def build_optimizer(parameters, hp):
    optimizer_name = str(get_train_value(hp, 'optimizer', 'adam')).lower()
    learning_rate = hp.train.learning_rate
    weight_decay = float(get_train_value(hp, 'weight_decay', 0.0))

    if optimizer_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError('Unsupported optimizer: %s' % optimizer_name)


def build_scheduler(optimizer, hp):
    scheduler_name = str(get_train_value(hp, 'scheduler', 'none')).lower()
    if scheduler_name in ('none', ''):
        return None
    if scheduler_name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(get_train_value(hp, 'lr_factor', 0.5)),
            patience=int(get_train_value(hp, 'lr_patience', 3)),
            min_lr=float(get_train_value(hp, 'min_learning_rate', 0.0)),
        )
    raise ValueError('Unsupported scheduler: %s' % scheduler_name)


def current_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def load_checkpoint(checkpoint_path, model, adapter, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if adapter is not None and checkpoint.get('adapter') is not None:
        adapter.load_state_dict(checkpoint['adapter'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('global_step', 0),
        checkpoint.get('best_val_loss', None),
        checkpoint.get('best_val_si_sdr_improvement', None),
        checkpoint.get('best_val_negative_count', None),
        checkpoint.get('best_metric_val_loss', None),
        checkpoint.get('epochs_without_improvement', 0),
    )


def save_checkpoint(
    path,
    model,
    adapter,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_loss,
    best_val_si_sdr_improvement,
    best_val_negative_count,
    best_metric_val_loss,
    epochs_without_improvement,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'adapter': adapter.state_dict() if adapter is not None else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'best_val_si_sdr_improvement': best_val_si_sdr_improvement,
        'best_val_negative_count': best_val_negative_count,
        'best_metric_val_loss': best_metric_val_loss,
        'epochs_without_improvement': epochs_without_improvement,
    }, path)


def compute_loss(mask, enhanced_mag, clean_mag, mix_mag, hp):
    mag_l1 = F.l1_loss(enhanced_mag, clean_mag)
    mask_l1 = torch.zeros((), device=clean_mag.device)
    loss_type = str(get_train_value(hp, 'loss_type', 'mag_l1')).lower()

    if loss_type == 'mag_l1':
        total_loss = mag_l1
    elif loss_type == 'mag_l1_plus_mask_l1':
        ideal_mask = torch.clamp(clean_mag / torch.clamp(mix_mag, min=1e-6), 0.0, 1.0)
        mask_l1 = F.l1_loss(mask, ideal_mask)
        total_loss = mag_l1 + float(get_train_value(hp, 'mask_loss_weight', 0.2)) * mask_l1
    else:
        raise ValueError('Unsupported loss_type: %s' % loss_type)

    return {
        'total': total_loss,
        'mag_l1': mag_l1,
        'mask_l1': mask_l1,
    }


def run_epoch(model, adapter, loader, hp, optimizer, device, train_mode):
    if train_mode:
        model.train()
        if adapter is not None:
            adapter.train()
    else:
        model.eval()
        if adapter is not None:
            adapter.eval()

    total_metrics = {
        'loss': 0.0,
        'mag_l1': 0.0,
        'mask_l1': 0.0,
    }
    num_batches = 0
    grad_clip_norm = float(get_train_value(hp, 'grad_clip_norm', 0.0))
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            clean_mag = batch['clean_mag'].to(device)
            mix_mag = batch['mix_mag'].to(device)
            embedding = batch['embedding'].to(device)

            if adapter is not None:
                embedding = adapter(embedding)

            mask = model(mix_mag, embedding)
            enhanced_mag = mix_mag * mask
            loss_values = compute_loss(mask, enhanced_mag, clean_mag, mix_mag, hp)
            loss = loss_values['total']

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm > 0.0:
                    parameters = list(model.parameters())
                    if adapter is not None:
                        parameters += list(adapter.parameters())
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
                optimizer.step()

            total_metrics['loss'] += loss.item()
            total_metrics['mag_l1'] += loss_values['mag_l1'].item()
            total_metrics['mask_l1'] += loss_values['mask_l1'].item()
            num_batches += 1

    if num_batches == 0:
        return {key: 0.0 for key in total_metrics}
    return {key: value / float(num_batches) for key, value in total_metrics.items()}


def evaluate_manifest_metrics(model, adapter, items, hp, device, embedder_path=None):
    model.eval()
    if adapter is not None:
        adapter.eval()

    audio = Audio(hp)
    rows = []
    for item in items:
        row, _ = evaluate_item(
            item,
            model,
            adapter,
            audio,
            hp,
            device,
            embedder_path=embedder_path,
        )
        rows.append(row)
    return summarize_rows(rows)


def is_better_metric(summary, best_si_sdr_improvement, best_negative_count, fallback_val_loss, best_metric_val_loss):
    if best_si_sdr_improvement is None:
        return True

    current_si_sdr_improvement = summary['avg_si_sdr_improvement']
    if current_si_sdr_improvement > best_si_sdr_improvement:
        return True
    if current_si_sdr_improvement < best_si_sdr_improvement:
        return False

    current_negative_count = int(summary['negative_count'])
    best_negative_count = int(best_negative_count) if best_negative_count is not None else None
    if best_negative_count is None or current_negative_count < best_negative_count:
        return True
    if current_negative_count > best_negative_count:
        return False

    if best_metric_val_loss is None:
        return True
    return fallback_val_loss < best_metric_val_loss


def should_run_best_metric_eval(epoch_idx, total_epochs):
    if total_epochs <= 5:
        return True
    return (epoch_idx % 5 == 0) or (epoch_idx > total_epochs - 5)


def resolve_noise_count(hp, cli_noise_count):
    if cli_noise_count is not None:
        return normalize_noise_count(cli_noise_count)
    return get_data_noise_count(hp.data, default=1)


def apply_runtime_noise_mode(hp, noise_count):
    hp.data.noise_count = int(noise_count)
    for key in ['manifest_train', 'manifest_val', 'manifest_test']:
        if key in hp.data and hp.data.get(key):
            hp.data[key] = resolve_manifest_path(hp.data[key], noise_count)
    return hp


def main():
    parser = argparse.ArgumentParser(description='Train the SnoreFilter conditioned snore enhancement model with precomputed vowel embeddings')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    parser.add_argument('--checkpoint-path', default=None, help='Optional checkpoint path to resume from')
    parser.add_argument('--noise-count', type=int, default=None, help='Noise mode to train: 1, 2, or 3')
    args = parser.parse_args()

    hp = HParam(args.config)
    noise_count = resolve_noise_count(hp, args.noise_count)
    hp = apply_runtime_noise_mode(hp, noise_count)
    device = build_device(args.device)
    set_seed(hp.train.seed)

    save_dir = hp.train.save_dir
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger('train_enhancement')
    writer = SummaryWriter(log_dir)
    logger.info('Using device: %s', device)
    logger.info('noise_count=%d', noise_count)
    d_vector_enabled = use_d_vector(hp)
    logger.info('use_d_vector=%s', d_vector_enabled)
    logger.info('vowel_embedding_mode=%s', hp.data.get('vowel_embedding_mode', 'avg'))
    logger.info('manifest_train=%s', hp.data.manifest_train)
    logger.info('manifest_val=%s', hp.data.manifest_val)
    logger.info('manifest_test=%s', hp.data.manifest_test)
    logger.info('Torch version: %s', torch.__version__)
    if device.type == 'cuda':
        logger.info('CUDA available: %s', torch.cuda.is_available())
        logger.info('CUDA device count: %d', torch.cuda.device_count())
        logger.info('CUDA device name: %s', torch.cuda.get_device_name(device))

    enable_best_metric_eval = get_train_bool(hp, 'enable_best_metric_eval', True)
    logger.info('enable_best_metric_eval=%s', enable_best_metric_eval)

    train_dataset = EnhancementDataset(hp.data.manifest_train, hp, train=True)
    val_dataset = EnhancementDataset(hp.data.manifest_val, hp, train=False)
    val_manifest_items = load_jsonl(hp.data.manifest_val) if enable_best_metric_eval else []
    embedder_path = resolve_embedder_path(required=False) if enable_best_metric_eval and d_vector_enabled else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.train.batch_size,
        shuffle=True,
        num_workers=hp.train.num_workers,
        collate_fn=enhancement_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.train.batch_size,
        shuffle=False,
        num_workers=max(0, hp.train.num_workers // 2),
        collate_fn=enhancement_collate_fn,
        drop_last=False,
    )

    model = SnoreFilter(hp).to(device)
    adapter = None
    if d_vector_enabled and hp.model.use_embedding_adapter:
        adapter = EmbeddingAdapter(hp.embedder.emb_dim, hp.model.adapter_hidden_dim).to(device)

    parameters = list(model.parameters())
    if adapter is not None:
        parameters += list(adapter.parameters())
    optimizer = build_optimizer(parameters, hp)
    scheduler = build_scheduler(optimizer, hp)
    logger.info('optimizer=%s lr=%.8f weight_decay=%.8f', get_train_value(hp, 'optimizer', 'adam'), hp.train.learning_rate, float(get_train_value(hp, 'weight_decay', 0.0)))
    logger.info('loss_type=%s mask_loss_weight=%.4f', get_train_value(hp, 'loss_type', 'mag_l1'), float(get_train_value(hp, 'mask_loss_weight', 0.0)))
    if scheduler is not None:
        logger.info(
            'scheduler=%s factor=%.4f patience=%d min_lr=%.8f',
            get_train_value(hp, 'scheduler', 'none'),
            float(get_train_value(hp, 'lr_factor', 0.5)),
            int(get_train_value(hp, 'lr_patience', 3)),
            float(get_train_value(hp, 'min_learning_rate', 0.0)),
        )

    start_epoch = 0
    global_step = 0
    best_val_loss = None
    best_val_si_sdr_improvement = None
    best_val_negative_count = None
    best_metric_val_loss = None
    epochs_without_improvement = 0
    if args.checkpoint_path:
        (
            start_epoch,
            global_step,
            best_val_loss,
            best_val_si_sdr_improvement,
            best_val_negative_count,
            best_metric_val_loss,
            epochs_without_improvement,
        ) = load_checkpoint(
            args.checkpoint_path,
            model,
            adapter,
            optimizer,
            scheduler,
            device,
        )
        logger.info('Resumed from %s', args.checkpoint_path)

    early_stop_patience = int(get_train_value(hp, 'early_stop_patience', 0))
    for epoch in range(start_epoch, hp.train.num_epochs):
        epoch_idx = epoch + 1
        train_metrics = run_epoch(model, adapter, train_loader, hp, optimizer, device, train_mode=True)
        val_metrics = run_epoch(model, adapter, val_loader, hp, optimizer, device, train_mode=False)
        train_loss = train_metrics['loss']
        val_loss = val_metrics['loss']
        global_step += 1
        ran_best_metric_eval = enable_best_metric_eval and should_run_best_metric_eval(epoch_idx, hp.train.num_epochs)
        val_metric_summary = None
        val_metric_si_sdr_improvement = 'N/A'
        val_metric_si_sdr = 'N/A'
        val_metric_negative_count = 'N/A'

        if ran_best_metric_eval:
            val_metric_summary = evaluate_manifest_metrics(
                model,
                adapter,
                val_manifest_items,
                hp,
                device,
                embedder_path=embedder_path,
            )
            val_metric_si_sdr_improvement = '%.6f' % val_metric_summary['avg_si_sdr_improvement']
            val_metric_si_sdr = '%.6f' % val_metric_summary['avg_si_sdr']
            val_metric_negative_count = '%d' % int(val_metric_summary['negative_count'])

        writer.add_scalar('loss/train', train_loss, epoch_idx)
        writer.add_scalar('loss/val', val_loss, epoch_idx)
        writer.add_scalar('loss_mag_l1/train', train_metrics['mag_l1'], epoch_idx)
        writer.add_scalar('loss_mag_l1/val', val_metrics['mag_l1'], epoch_idx)
        writer.add_scalar('loss_mask_l1/train', train_metrics['mask_l1'], epoch_idx)
        writer.add_scalar('loss_mask_l1/val', val_metrics['mask_l1'], epoch_idx)
        writer.add_scalar('learning_rate', current_learning_rate(optimizer), epoch_idx)
        if ran_best_metric_eval:
            writer.add_scalar('val_metric/si_sdr_improvement', val_metric_summary['avg_si_sdr_improvement'], epoch_idx)
            writer.add_scalar('val_metric/si_sdr', val_metric_summary['avg_si_sdr'], epoch_idx)
            writer.add_scalar('val_metric/negative_count', val_metric_summary['negative_count'], epoch_idx)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = False
        improved_metric = False
        should_save_best_loss = best_val_loss is None or val_loss < best_val_loss
        should_save_best_metric = False
        if ran_best_metric_eval:
            should_save_best_metric = is_better_metric(
                val_metric_summary,
                best_val_si_sdr_improvement,
                best_val_negative_count,
                val_loss,
                best_metric_val_loss,
            )
        if should_save_best_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1

        if should_save_best_metric:
            best_val_si_sdr_improvement = val_metric_summary['avg_si_sdr_improvement']
            best_val_negative_count = int(val_metric_summary['negative_count'])
            best_metric_val_loss = val_loss
            improved_metric = True

        if improved:
            best_loss_path = os.path.join(ckpt_dir, 'best_loss.pt')
            save_checkpoint(
                best_loss_path,
                model,
                adapter,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                best_val_loss,
                best_val_si_sdr_improvement,
                best_val_negative_count,
                best_metric_val_loss,
                epochs_without_improvement,
            )
            logger.info('Saved best loss checkpoint to %s', best_loss_path)

        if improved_metric:
            best_metric_path = os.path.join(ckpt_dir, 'best_metric.pt')
            save_checkpoint(
                best_metric_path,
                model,
                adapter,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                best_val_loss,
                best_val_si_sdr_improvement,
                best_val_negative_count,
                best_metric_val_loss,
                epochs_without_improvement,
            )
            logger.info('Saved best metric checkpoint to %s', best_metric_path)

        latest_path = os.path.join(ckpt_dir, 'latest.pt')
        save_checkpoint(
            latest_path,
            model,
            adapter,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            best_val_loss,
            best_val_si_sdr_improvement,
            best_val_negative_count,
            best_metric_val_loss,
            epochs_without_improvement,
        )

        logger.info(
            'epoch=%d train_loss=%.6f val_loss=%.6f train_mag_l1=%.6f val_mag_l1=%.6f '
            'train_mask_l1=%.6f val_mask_l1=%.6f val_avg_si_sdr_improvement=%s val_avg_si_sdr=%s '
            'val_negative_count=%s ran_best_metric_eval=%s lr=%.8f best_val_loss=%s '
            'best_val_si_sdr_improvement=%s best_val_negative_count=%s '
            'saved_best_loss=%s saved_best_metric=%s stale_epochs=%d',
            epoch_idx,
            train_loss,
            val_loss,
            train_metrics['mag_l1'],
            val_metrics['mag_l1'],
            train_metrics['mask_l1'],
            val_metrics['mask_l1'],
            val_metric_si_sdr_improvement,
            val_metric_si_sdr,
            val_metric_negative_count,
            str(ran_best_metric_eval),
            current_learning_rate(optimizer),
            'None' if best_val_loss is None else '%.6f' % best_val_loss,
            'None' if best_val_si_sdr_improvement is None else '%.6f' % best_val_si_sdr_improvement,
            'None' if best_val_negative_count is None else '%d' % int(best_val_negative_count),
            str(improved),
            str(improved_metric),
            epochs_without_improvement,
        )

        if early_stop_patience > 0 and not improved and epochs_without_improvement >= early_stop_patience:
            logger.info('Early stopping at epoch=%d after %d epochs without improvement', epoch + 1, epochs_without_improvement)
            break

    writer.close()


if __name__ == '__main__':
    main()
