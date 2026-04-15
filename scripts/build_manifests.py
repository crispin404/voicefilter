import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import (
    build_manifest_rows,
    build_snr_lookup,
    load_csv_rows,
    load_jsonl,
    load_subject_ids,
    load_subjects,
    safe_float,
    save_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Build enhancement manifests from subjects and subject splits')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--splits-dir', default='splits', help='Directory containing *_subjects.txt')
    parser.add_argument('--output-dir', default='manifests', help='Directory for output jsonl files')
    parser.add_argument('--processed-root', default=None, help='Optional processed root used to resolve audio and embedding paths')
    parser.add_argument('--mix-dir-name', default=None, help='Optional raw mix subdirectory name, e.g. 合成声_2')
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


def main():
    args = parse_args()
    snr_lookup = load_snr_lookup(args.snr_stats_csv)

    subjects = load_subjects(args.subjects)
    for split_name in ['train', 'val', 'test']:
        subject_ids = load_subject_ids(os.path.join(args.splits_dir, '%s_subjects.txt' % split_name))
        rows = build_manifest_rows(
            subjects,
            subject_ids,
            processed_root=args.processed_root,
            mix_dir_name=args.mix_dir_name,
            snr_lookup=snr_lookup,
        )
        raw_count = len(rows)
        rows = [
            row for row in rows
            if snr_in_range(row, min_snr_db=args.min_snr_db, max_snr_db=args.max_snr_db)
        ]

        out_path = os.path.join(args.output_dir, 'enhancement_manifest_%s.jsonl' % split_name)
        save_jsonl(rows, out_path)
        print('%s: %d samples (%d after SNR filter) -> %s' % (split_name, raw_count, len(rows), os.path.abspath(out_path)))


if __name__ == '__main__':
    main()
