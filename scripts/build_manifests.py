import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import build_manifest_rows, load_subject_ids, load_subjects, save_jsonl


def main():
    parser = argparse.ArgumentParser(description='Build enhancement manifests from subjects and subject splits')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--splits-dir', default='splits', help='Directory containing *_subjects.txt')
    parser.add_argument('--output-dir', default='manifests', help='Directory for output jsonl files')
    parser.add_argument('--processed-root', default=None, help='Optional processed root used to resolve audio and embedding paths')
    args = parser.parse_args()

    subjects = load_subjects(args.subjects)
    for split_name in ['train', 'val', 'test']:
        subject_ids = load_subject_ids(os.path.join(args.splits_dir, '%s_subjects.txt' % split_name))
        rows = build_manifest_rows(subjects, subject_ids, processed_root=args.processed_root)
        out_path = os.path.join(args.output_dir, 'enhancement_manifest_%s.jsonl' % split_name)
        save_jsonl(rows, out_path)
        print('%s: %d samples -> %s' % (split_name, len(rows), os.path.abspath(out_path)))


if __name__ == '__main__':
    main()
