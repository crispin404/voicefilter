import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tqdm import tqdm

from model.embedding_adapter import EmbeddingAdapter
from model.model import SnoreFilter
from utils.audio import Audio, save_wav
from utils.dataset_index import (
    build_manifest_rows,
    ensure_dir,
    get_data_noise_count,
    load_jsonl,
    load_subjects,
    normalize_noise_count,
    resolve_manifest_path,
)
from utils.dvector import use_d_vector
from utils.enhancement_eval import evaluate_item, print_summary
from utils.embedder_checkpoint import DEFAULT_EMBEDDER_PATH, resolve_embedder_path
from utils.hparams import HParam


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SnoreFilter enhanced outputs on a manifest and write CSV metrics')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--checkpoint-path', required=True, help='Trained enhancement checkpoint')
    parser.add_argument('--noise-count', type=int, default=None, help='Noise mode to evaluate: 1, 2, or 3')
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--manifest', default=None, help='Manifest path, defaults to config test manifest')
    source_group.add_argument('--subject-ids-file', default=None, help='Text file containing one subject_id per line for custom evaluation')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path used with --subject-ids-file')
    parser.add_argument('--embedder-path', default=None, help='Embedder checkpoint used when embeddings must be computed online, defaults to %s' % DEFAULT_EMBEDDER_PATH)
    parser.add_argument('--output-csv', default=os.path.join('outputs', 'platformax', 'eval', 'metrics.csv'), help='CSV output path')
    parser.add_argument('--save-wavs-dir', default=os.path.join('outputs', 'platformax', 'eval', 'enhanced_wavs'), help='Optional directory for enhanced wavs')
    parser.add_argument('--no-save-wavs', action='store_true', help='Do not write enhanced wav files')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    return parser.parse_args()


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


def load_selected_subject_ids(path):
    abs_path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Subject id file not found: %s' % abs_path)

    subject_ids = []
    seen = set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            subject_id = line.strip()
            if not subject_id or subject_id in seen:
                continue
            seen.add(subject_id)
            subject_ids.append(subject_id)

    if not subject_ids:
        raise ValueError('Subject id file is empty or contains no valid subject_id entries: %s' % abs_path)
    return subject_ids


def load_evaluation_items(args, hp, noise_count):
    if args.subject_ids_file:
        subject_ids = load_selected_subject_ids(args.subject_ids_file)
        subjects = load_subjects(args.subjects)
        available_subject_ids = {subject['subject_id'] for subject in subjects}
        missing_subject_ids = [subject_id for subject_id in subject_ids if subject_id not in available_subject_ids]
        if missing_subject_ids:
            raise ValueError(
                'Unknown subject_id(s) in %s: %s'
                % (os.path.abspath(args.subject_ids_file), ', '.join(missing_subject_ids))
            )

        processed_root = hp.data.get('processed_root')
        vowel_embedding_mode = hp.data.get('vowel_embedding_mode', 'avg')
        items = build_manifest_rows(
            subjects,
            subject_ids,
            processed_root=processed_root,
            vowel_embedding_mode=vowel_embedding_mode,
            noise_count=noise_count,
        )
        if not items:
            raise ValueError(
                'No evaluable samples were found for the selected subject_id(s): %s'
                % ', '.join(subject_ids)
            )

        print(
            'Loaded %d evaluation items from %d selected subjects in %s (vowel_embedding_mode=%s)'
            % (len(items), len(subject_ids), os.path.abspath(args.subject_ids_file), vowel_embedding_mode)
        )
        return items

    manifest_path = args.manifest or hp.data.manifest_test
    items = load_jsonl(manifest_path)
    if not items:
        raise ValueError('Manifest is empty: %s' % os.path.abspath(manifest_path))

    print('Loaded %d evaluation items from manifest %s' % (len(items), os.path.abspath(manifest_path)))
    return items


def main():
    args = parse_args()

    hp = HParam(args.config)
    noise_count = resolve_noise_count(hp, args.noise_count)
    hp = apply_runtime_noise_mode(hp, noise_count)
    device = build_device(args.device)
    d_vector_enabled = use_d_vector(hp)
    embedder_path = resolve_embedder_path(args.embedder_path, required=False) if d_vector_enabled else None
    audio = Audio(hp)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = SnoreFilter(hp).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    adapter = None
    if d_vector_enabled and hp.model.use_embedding_adapter:
        adapter = EmbeddingAdapter(hp.embedder.emb_dim, hp.model.adapter_hidden_dim).to(device)
    if adapter is not None and checkpoint.get('adapter') is not None:
        adapter.load_state_dict(checkpoint['adapter'])
        adapter.eval()

    items = load_evaluation_items(args, hp, noise_count)
    ensure_dir(os.path.dirname(args.output_csv))
    if not args.no_save_wavs:
        ensure_dir(args.save_wavs_dir)

    rows = []
    for item in tqdm(items, desc='evaluate'):
        row, enhanced_wav = evaluate_item(
            item,
            model,
            adapter,
            audio,
            hp,
            device,
            embedder_path=embedder_path,
        )

        mix_name = os.path.splitext(os.path.basename(item['mix_path']))[0]
        if args.no_save_wavs:
            wav_out_path = ''
        else:
            wav_out_path = os.path.join(args.save_wavs_dir, '%s_%s.wav' % (item['subject_id'], mix_name))
            save_wav(wav_out_path, enhanced_wav, hp.audio.sample_rate)
        row['enhanced_path'] = wav_out_path
        rows.append(row)

    fieldnames = [
        'subject_id', 'noise_type', 'snore_index',
        'input_snr', 'snr_improvement', 'enhanced_snr',
        'input_si_sdr', 'si_sdr_improvement', 'si_sdr', 
    ]
    # 'noise_count',  'mix_path','clean_path', 'enhanced_path', 'mag_l1','clean_active_ratio', 'clean_active_windows', 'clean_total_windows',
    
    decimal_fields = {
        'input_snr',
        'snr_improvement',
        'enhanced_snr',
        'input_si_sdr',
        'si_sdr',
        'si_sdr_improvement',
        'mag_l1',
        'clean_active_ratio',
    }
    csv_rows = []
    for row in rows:
        csv_row = {}
        for name in fieldnames:
            value = row.get(name, '')
            if name in decimal_fields and value != '':
                csv_row[name] = '%.2f' % float(value)
            else:
                csv_row[name] = value
        csv_rows.append(csv_row)

    with open(args.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    if rows:
        print('Saved %d evaluation rows to %s' % (len(rows), os.path.abspath(args.output_csv)))
        print_summary('all', rows)
        active_rows = [row for row in rows if row['clean_active_ratio'] > 0.0]
        zero_active_count = len(rows) - len(active_rows)
        if active_rows:
            print_summary('active', active_rows)
        print('zero_active_count=%d' % zero_active_count)


if __name__ == '__main__':
    main()
