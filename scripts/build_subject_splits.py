import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import load_subjects, save_subject_ids, split_subjects


def build_ratio_counts(total_subjects, train_ratio, val_ratio, test_ratio):
    ratios = {
        'train': train_ratio,
        'val': val_ratio,
        'test': test_ratio,
    }
    if total_subjects <= 0:
        raise ValueError('No valid subjects were found.')
    if any(value < 0 for value in ratios.values()):
        raise ValueError('Split ratios must be non-negative.')

    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError('At least one split ratio must be greater than zero.')

    exact_counts = {
        name: float(total_subjects) * (ratio / ratio_sum)
        for name, ratio in ratios.items()
    }
    counts = {
        name: int(math.floor(value))
        for name, value in exact_counts.items()
    }

    remaining = total_subjects - sum(counts.values())
    by_fraction = sorted(
        exact_counts,
        key=lambda name: (exact_counts[name] - counts[name], ratios[name], name),
        reverse=True,
    )
    for index in range(remaining):
        counts[by_fraction[index % len(by_fraction)]] += 1

    positive_splits = [name for name, ratio in ratios.items() if ratio > 0]
    if total_subjects >= len(positive_splits):
        for name in positive_splits:
            if counts[name] > 0:
                continue
            donor = max(
                (candidate for candidate in positive_splits if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[name] += 1

    return counts['train'], counts['val'], counts['test']


def main():
    parser = argparse.ArgumentParser(description='Build reproducible subject-level train/val/test splits')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--output-dir', default='splits', help='Directory for split txt files')
    parser.add_argument('--train-count', type=int, default=29, help='Number of train subjects')
    parser.add_argument('--val-count', type=int, default=6, help='Number of val subjects')
    parser.add_argument('--test-count', type=int, default=6, help='Number of test subjects')
    parser.add_argument('--train-ratio', type=float, default=None, help='Optional train ratio, e.g. 0.8')
    parser.add_argument('--val-ratio', type=float, default=None, help='Optional validation ratio, e.g. 0.1')
    parser.add_argument('--test-ratio', type=float, default=None, help='Optional test ratio, e.g. 0.1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    subjects = load_subjects(args.subjects)
    use_ratios = any(
        value is not None
        for value in (args.train_ratio, args.val_ratio, args.test_ratio)
    )
    if use_ratios:
        train_ratio = 0.8 if args.train_ratio is None else args.train_ratio
        val_ratio = 0.1 if args.val_ratio is None else args.val_ratio
        test_ratio = 0.1 if args.test_ratio is None else args.test_ratio
        train_count, val_count, test_count = build_ratio_counts(
            len(subjects),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        print(
            'Using ratio split train=%.4g val=%.4g test=%.4g over %d subjects -> %d/%d/%d'
            % (
                train_ratio,
                val_ratio,
                test_ratio,
                len(subjects),
                train_count,
                val_count,
                test_count,
            )
        )
    else:
        train_count = args.train_count
        val_count = args.val_count
        test_count = args.test_count

    splits = split_subjects(
        subjects,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=args.seed,
    )

    for split_name, split_subjects_list in splits.items():
        out_path = os.path.join(args.output_dir, '%s_subjects.txt' % split_name)
        save_subject_ids(split_subjects_list, out_path)
        print('%s: %d subjects -> %s' % (split_name, len(split_subjects_list), os.path.abspath(out_path)))


if __name__ == '__main__':
    main()
