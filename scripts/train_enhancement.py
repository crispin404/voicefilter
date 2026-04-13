import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.enhancement_dataset import EnhancementDataset, enhancement_collate_fn
from model.embedding_adapter import EmbeddingAdapter
from model.model import VoiceFilter
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


def load_checkpoint(checkpoint_path, model, adapter, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if adapter is not None and checkpoint.get('adapter') is not None:
        adapter.load_state_dict(checkpoint['adapter'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0), checkpoint.get('best_val_loss', None)


def save_checkpoint(path, model, adapter, optimizer, epoch, global_step, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'adapter': adapter.state_dict() if adapter is not None else None,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
    }, path)


def run_epoch(model, adapter, loader, criterion, optimizer, device, train_mode):
    if train_mode:
        model.train()
        if adapter is not None:
            adapter.train()
    else:
        model.eval()
        if adapter is not None:
            adapter.eval()

    total_loss = 0.0
    num_batches = 0
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
            loss = criterion(enhanced_mag, clean_mag)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / float(num_batches)


def main():
    parser = argparse.ArgumentParser(description='Train conditioned snore enhancement with precomputed vowel embeddings')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    parser.add_argument('--checkpoint-path', default=None, help='Optional checkpoint path to resume from')
    args = parser.parse_args()

    hp = HParam(args.config)
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

    train_dataset = EnhancementDataset(hp.data.manifest_train, hp, train=True)
    val_dataset = EnhancementDataset(hp.data.manifest_val, hp, train=False)
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

    model = VoiceFilter(hp).to(device)
    adapter = EmbeddingAdapter(hp.embedder.emb_dim, hp.model.adapter_hidden_dim).to(device) if hp.model.use_embedding_adapter else None

    parameters = list(model.parameters())
    if adapter is not None:
        parameters += list(adapter.parameters())
    optimizer = torch.optim.Adam(parameters, lr=hp.train.learning_rate)
    criterion = nn.L1Loss()

    start_epoch = 0
    global_step = 0
    best_val_loss = None
    if args.checkpoint_path:
        start_epoch, global_step, best_val_loss = load_checkpoint(args.checkpoint_path, model, adapter, optimizer, device)
        logger.info('Resumed from %s', args.checkpoint_path)

    for epoch in range(start_epoch, hp.train.num_epochs):
        train_loss = run_epoch(model, adapter, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss = run_epoch(model, adapter, val_loader, criterion, optimizer, device, train_mode=False)
        global_step += 1

        writer.add_scalar('loss/train', train_loss, epoch + 1)
        writer.add_scalar('loss/val', val_loss, epoch + 1)
        logger.info('epoch=%d train_loss=%.6f val_loss=%.6f', epoch + 1, train_loss, val_loss)

        latest_path = os.path.join(ckpt_dir, 'latest.pt')
        save_checkpoint(latest_path, model, adapter, optimizer, epoch + 1, global_step, best_val_loss)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, 'best.pt')
            save_checkpoint(best_path, model, adapter, optimizer, epoch + 1, global_step, best_val_loss)
            logger.info('Saved best checkpoint to %s', best_path)

    writer.close()


if __name__ == '__main__':
    main()
