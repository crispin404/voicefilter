import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import (
    build_manifest_rows,
    build_snr_lookup,
    default_mix_dir_name,
    get_data_noise_count,
    load_csv_rows,
    load_jsonl,
    load_subject_ids,
    load_subjects,
    normalize_noise_count,
    resolve_manifest_dir,
    resolve_mode_filepath,
    safe_float,
    save_jsonl,
)
from utils.hparams import HParam


DEFAULT_SPLITS_DIR = 'splits'
DEFAULT_OUTPUT_DIR = 'manifests'
DEFAULT_SNR_STATS_CSV = os.path.join('metadata', 'preprocess_snr_stats.csv')
DEFAULT_MIX_DIR_NAME = '合成声'


def parse_args():
    parser = argparse.ArgumentParser(description='Build enhancement manifests from subjects and subject splits')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--splits-dir', default=DEFAULT_SPLITS_DIR, help='Directory containing *_subjects.txt')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory for output jsonl files')
    parser.add_argument('--processed-root', default=None, help='Optional processed root used to resolve audio and embedding paths')
    parser.add_argument('--mix-dir-name', default=DEFAULT_MIX_DIR_NAME, help='Raw mix subdirectory name')
    parser.add_argument('--noise-count', type=int, default=None, help='Noise mode to build: 1, 2, or 3')
    parser.add_argument('--min-snr-db', type=float, default=None, help='Optional lower bound for actual_snr_db filtering')
    parser.add_argument('--max-snr-db', type=float, default=None, help='Optional upper bound for actual_snr_db filtering')
    parser.add_argument('--snr-stats-csv', default=None, help='Optional CSV or JSONL exported by preprocess_audio.py')
    return parser.parse_args()


def load_snr_lookup(path):
    if not path:
        return None
    if path.lower().endswith('.jsonl'):
        rows = load_jsonl(path)
    elif path.lower().endswith('.csv'):
        rows = load_csv_rows(path)
    else:
        raise ValueError('Unsupported SNR stats format: %s' % path)
    return build_snr_lookup(rows)


def snr_in_range(row, min_snr_db=None, max_snr_db=None):
    actual_snr_db = safe_float(row.get('actual_snr_db'))
    if actual_snr_db is None:
        return min_snr_db is None and max_snr_db is None
    if min_snr_db is not None and actual_snr_db < min_snr_db:
        return False
    if max_snr_db is not None and actual_snr_db > max_snr_db:
        return False
    return True


def resolve_noise_count(hp, cli_noise_count):
    if cli_noise_count is not None:
        return normalize_noise_count(cli_noise_count)
    return get_data_noise_count(hp.data, default=1)


def resolve_mode_mix_dir_name(mix_dir_name, noise_count):
    if mix_dir_name == DEFAULT_MIX_DIR_NAME:
        return default_mix_dir_name(noise_count)
    return mix_dir_name


def resolve_mode_snr_stats_path(path, noise_count):
    if not path:
        return path
    if path == DEFAULT_SNR_STATS_CSV:
        return resolve_mode_filepath(path, noise_count)
    return path


def resolve_mode_output_dir(output_dir, noise_count):
    if output_dir == DEFAULT_OUTPUT_DIR:
        return resolve_manifest_dir(output_dir, noise_count)
    return output_dir


def main():
    args = parse_args()
    hp = HParam(args.config)
    noise_count = resolve_noise_count(hp, args.noise_count)
    snr_stats_csv = resolve_mode_snr_stats_path(args.snr_stats_csv, noise_count)
    snr_lookup = load_snr_lookup(snr_stats_csv)
    processed_root = args.processed_root if args.processed_root is not None else hp.data.get('processed_root')
    vowel_embedding_mode = hp.data.get('vowel_embedding_mode', 'avg')
    mix_dir_name = resolve_mode_mix_dir_name(args.mix_dir_name, noise_count)
    output_dir = resolve_mode_output_dir(args.output_dir, noise_count)

    subjects = load_subjects(args.subjects)
    for split_name in ['train', 'val', 'test']:
        subject_ids = load_subject_ids(os.path.join(args.splits_dir, '%s_subjects.txt' % split_name))
        rows = build_manifest_rows(
            subjects,
            subject_ids,
            processed_root=processed_root,
            mix_dir_name=mix_dir_name,
            snr_lookup=snr_lookup,
            vowel_embedding_mode=vowel_embedding_mode,
            noise_count=noise_count,
        )
        raw_count = len(rows)
        rows = [
            row for row in rows
            if snr_in_range(row, min_snr_db=args.min_snr_db, max_snr_db=args.max_snr_db)
        ]

        out_path = os.path.join(output_dir, 'enhancement_manifest_%s.jsonl' % split_name)
        save_jsonl(rows, out_path)
        print(
            '%s: %d samples (%d after SNR filter, vowel_embedding_mode=%s, noise_count=%d) -> %s'
            % (split_name, raw_count, len(rows), vowel_embedding_mode, noise_count, os.path.abspath(out_path))
        )


if __name__ == '__main__':
    main()
