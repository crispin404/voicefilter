import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import load_subjects, save_subject_ids, split_subjects


def main():
    parser = argparse.ArgumentParser(description='Build reproducible subject-level train/val/test splits')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--output-dir', default='splits', help='Directory for split txt files')
    parser.add_argument('--train-count', type=int, default=29, help='Number of train subjects')
    parser.add_argument('--val-count', type=int, default=6, help='Number of val subjects')
    parser.add_argument('--test-count', type=int, default=6, help='Number of test subjects')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    subjects = load_subjects(args.subjects)
    splits = split_subjects(
        subjects,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    for split_name, split_subjects_list in splits.items():
        out_path = os.path.join(args.output_dir, '%s_subjects.txt' % split_name)
        save_subject_ids(split_subjects_list, out_path)
        print('%s: %d subjects -> %s' % (split_name, len(split_subjects_list), os.path.abspath(out_path)))


if __name__ == '__main__':
    main()
