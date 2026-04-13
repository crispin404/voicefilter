import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import load_subjects, parse_info_file, write_csv


def main():
    parser = argparse.ArgumentParser(description='Parse subject info.txt files into a CSV table')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--output', default=os.path.join('metadata', 'subject_info.csv'), help='Output CSV path')
    args = parser.parse_args()

    subjects = load_subjects(args.subjects)
    rows = []
    for subject in subjects:
        info = parse_info_file(subject['info_path'])
        row = {
            'subject_id': subject['subject_id'],
            'class_index': subject['class_index'],
            'subject_dir': subject['subject_dir'],
            'info_path': subject['info_path'],
        }
        row.update(info)
        rows.append(row)

    fieldnames = [
        'subject_id', 'class_index', 'name', 'gender', 'age',
        'height_cm', 'weight_kg', 'neck_cm', 'waist_cm', 'bmi',
        'subject_dir', 'info_path',
    ]
    write_csv(rows, args.output, fieldnames)
    print('Parsed %d info.txt files -> %s' % (len(rows), os.path.abspath(args.output)))


if __name__ == '__main__':
    main()
