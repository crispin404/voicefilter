import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import save_json, scan_subjects


def main():
    parser = argparse.ArgumentParser(description='Scan subject folders and create subjects.json')
    parser.add_argument('--data-root', required=True, help='Root directory containing subject folders')
    parser.add_argument('--output', default=os.path.join('metadata', 'subjects.json'), help='Output JSON path')
    args = parser.parse_args()

    subjects = scan_subjects(args.data_root)
    save_json(subjects, args.output)

    total = len(subjects)
    valid = sum(1 for subject in subjects if subject['exists'])
    print('Scanned %d subjects, %d valid, %d with missing files.' % (total, valid, total - valid))
    print('Saved to %s' % os.path.abspath(args.output))


if __name__ == '__main__':
    main()
