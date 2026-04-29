import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.audio import compute_rms, load_wav
from utils.dataset_index import SNORE_PATTERN, ensure_dir, load_subjects, normalize_path, scan_subjects


def parse_args():
    parser = argparse.ArgumentParser(description='Audit raw clean snore recordings with a minimal 3-metric quality system.')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--subjects', default=None, help='subjects.json path created by scan_dataset.py')
    source_group.add_argument('--data-root', default=None, help='Root directory containing subject folders')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate for analysis')
    parser.add_argument('--output-csv', default=os.path.join('metadata', 'snore_audit.csv'), help='CSV report path')
    parser.add_argument('--output-json', default=os.path.join('metadata', 'snore_audit.json'), help='JSON report path')
    parser.add_argument('--top-k-per-subject', type=int, default=3, help='How many lowest-scoring snores to print per subject')
    return parser.parse_args()


def load_subject_entries(args):
    if args.subjects:
        return load_subjects(args.subjects)
    return [subject for subject in scan_subjects(args.data_root) if subject.get('exists', True)]


def fixed_frame_params(sample_rate):
    frame_length = max(256, int(round(0.025 * sample_rate)))
    hop_length = max(128, int(round(0.010 * sample_rate)))
    return frame_length, hop_length


def compute_frame_rms(wav, frame_length, hop_length):
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length, center=False)
    return rms[0].astype(np.float32)


def compute_active_ratio(frame_rms, full_rms):
    if frame_rms.size == 0:
        return 0.0
    active_threshold = max(0.003, float(full_rms) * 0.5)
    return float(np.mean(frame_rms >= active_threshold))


def compute_clipping_ratio(wav):
    if wav.size == 0:
        return 0.0
    return float(np.mean(np.abs(wav) >= 0.98))


def parse_snore_index(path):
    match = SNORE_PATTERN.match(os.path.basename(path))
    if match is None:
        return ''
    return int(match.group(2))


def analyze_snore(subject, snore_path, sample_rate):
    row = {
        'subject_id': subject['subject_id'],
        'snore_index': parse_snore_index(snore_path),
        'snore_path': normalize_path(snore_path),
        'active_ratio': 0.0,
        'rms': 0.0,
        'clipping_ratio': 0.0,
        'quality_score': 0,
        'risk_flags': '',
        'decision': 'keep',
        'read_error': '',
    }

    try:
        wav, _ = load_wav(snore_path, sample_rate=sample_rate, mono=True)
    except Exception as exc:
        row['read_error'] = str(exc)
        return row

    wav = wav.astype(np.float32)
    if wav.size == 0:
        row['read_error'] = 'empty_audio'
        return row

    rms = compute_rms(wav)
    frame_length, hop_length = fixed_frame_params(sample_rate)
    frame_rms = compute_frame_rms(wav, frame_length, hop_length)
    active_ratio = compute_active_ratio(frame_rms, rms)
    clipping_ratio = compute_clipping_ratio(wav)

    row.update({
        'active_ratio': active_ratio,
        'rms': rms,
        'clipping_ratio': clipping_ratio,
    })
    return row


def apply_quality_rules(row):
    if row['read_error']:
        row['quality_score'] = 0
        row['risk_flags'] = ''
        row['decision'] = 'drop_candidate'
        return row

    score = 100
    flags = []

    active_ratio = row['active_ratio']
    rms = row['rms']
    clipping_ratio = row['clipping_ratio']

    if active_ratio < 0.15:
        flags.append('too_much_silence')
        score -= 45
    elif active_ratio < 0.30:
        flags.append('too_much_silence')
        score -= 25
    elif active_ratio < 0.45:
        flags.append('too_much_silence')
        score -= 10

    if rms < 0.002:
        flags.append('too_quiet')
        score -= 40
    elif rms < 0.005:
        flags.append('too_quiet')
        score -= 20
    elif rms < 0.008:
        flags.append('too_quiet')
        score -= 8

    if clipping_ratio >= 0.01:
        flags.append('possible_clipping')
        score -= 50
    elif clipping_ratio >= 0.001:
        flags.append('possible_clipping')
        score -= 25

    row['quality_score'] = max(0, int(round(score)))
    row['risk_flags'] = '|'.join(flags)
    if (
        active_ratio < 0.15
        or rms < 0.002
        or clipping_ratio >= 0.01
        or row['quality_score'] < 60
    ):
        row['decision'] = 'drop_candidate'
    elif (
        active_ratio < 0.30
        or (0.002 <= rms < 0.005)
        or (0.001 <= clipping_ratio < 0.01)
        or (60 <= row['quality_score'] < 80)
    ):
        row['decision'] = 'review'
    else:
        row['decision'] = 'keep'
    return row


def summarize_rows(rows):
    avg_score = float(np.mean([row['quality_score'] for row in rows])) if rows else 0.0
    keep_count = sum(1 for row in rows if row['decision'] == 'keep')
    review_count = sum(1 for row in rows if row['decision'] == 'review')
    drop_candidate_count = sum(1 for row in rows if row['decision'] == 'drop_candidate')
    flag_counter = Counter()
    for row in rows:
        if not row['risk_flags']:
            continue
        for flag in row['risk_flags'].split('|'):
            if flag:
                flag_counter[flag] += 1
    return {
        'total_rows': len(rows),
        'keep_count': keep_count,
        'review_count': review_count,
        'drop_candidate_count': drop_candidate_count,
        'average_quality_score': avg_score,
        'risk_flag_counts': dict(sorted(flag_counter.items())),
    }


def print_summary(rows, top_k_per_subject):
    summary = summarize_rows(rows)
    print('Audited %d snore files.' % summary['total_rows'])
    print(
        'keep=%d review=%d drop_candidate=%d avg_quality_score=%.2f'
        % (
            summary['keep_count'],
            summary['review_count'],
            summary['drop_candidate_count'],
            summary['average_quality_score'],
        )
    )
    print('Risk flag counts:')
    if summary['risk_flag_counts']:
        for flag, count in summary['risk_flag_counts'].items():
            print('  %s: %d' % (flag, count))
    else:
        print('  none')

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['subject_id']].append(row)

    for subject_id in sorted(grouped):
        raw_subject_rows = grouped[subject_id]
        keep_count = sum(1 for row in raw_subject_rows if row['decision'] == 'keep')
        review_count = sum(1 for row in raw_subject_rows if row['decision'] == 'review')
        drop_candidate_count = sum(1 for row in raw_subject_rows if row['decision'] == 'drop_candidate')
        mean_quality_score = float(np.mean([row['quality_score'] for row in raw_subject_rows])) if raw_subject_rows else 0.0
        min_quality_score = min((row['quality_score'] for row in raw_subject_rows), default=0)
        print(
            'Subject %s: total_snores=%d keep=%d review=%d drop_candidate=%d mean_quality_score=%.2f min_quality_score=%d'
            % (
                subject_id,
                len(raw_subject_rows),
                keep_count,
                review_count,
                drop_candidate_count,
                mean_quality_score,
                min_quality_score,
            )
        )
        subject_rows = sorted(
            raw_subject_rows,
            key=lambda item: (
                item['quality_score'],
                {'drop_candidate': 0, 'review': 1, 'keep': 2}.get(item['decision'], 3),
                item['snore_index'] if item['snore_index'] != '' else 10 ** 9,
                item['snore_path'],
            ),
        )
        print('Worst %d for %s:' % (min(top_k_per_subject, len(subject_rows)), subject_id))
        for row in subject_rows[:top_k_per_subject]:
            print(
                '  score=%d decision=%s snore_index=%s flags=%s active_ratio=%.4f rms=%.6f clipping_ratio=%.6f'
                % (
                    row['quality_score'],
                    row['decision'],
                    row['snore_index'],
                    row['risk_flags'] or 'none',
                    row['active_ratio'],
                    row['rms'],
                    row['clipping_ratio'],
                )
            )


def write_csv(rows, output_csv):
    ensure_dir(os.path.dirname(output_csv))
    fieldnames = [
        'subject_id',
        'snore_index',
        'active_ratio',
        'rms',
        'clipping_ratio',
        'quality_score',
        'risk_flags',
        'decision',
        'read_error',
    ]
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                'subject_id': row['subject_id'],
                'snore_index': row['snore_index'],
                'active_ratio': row['active_ratio'],
                'rms': row['rms'],
                'clipping_ratio': row['clipping_ratio'],
                'quality_score': row['quality_score'],
                'risk_flags': row['risk_flags'],
                'decision': row['decision'],
                'read_error': row['read_error'],
            }
            for row in rows
        ])


def write_json(rows, output_json):
    ensure_dir(os.path.dirname(output_json))
    json_rows = [
        {
            'subject_id': row['subject_id'],
            'snore_index': row['snore_index'],
            'active_ratio': row['active_ratio'],
            'rms': row['rms'],
            'clipping_ratio': row['clipping_ratio'],
            'quality_score': row['quality_score'],
            'risk_flags': row['risk_flags'],
            'decision': row['decision'],
            'read_error': row['read_error'],
        }
        for row in rows
    ]
    payload = {
        'summary': summarize_rows(rows),
        'rows': json_rows,
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    subjects = load_subject_entries(args)

    rows = []
    for subject in subjects:
        for snore_path in subject.get('snore_paths', []):
            rows.append(analyze_snore(subject, snore_path, sample_rate=args.sample_rate))

    rows = [apply_quality_rules(row) for row in rows]
    rows.sort(key=lambda item: (item['subject_id'], item['snore_index'] if item['snore_index'] != '' else 10 ** 9, item['snore_path']))

    write_csv(rows, args.output_csv)
    write_json(rows, args.output_json)
    print_summary(rows, top_k_per_subject=args.top_k_per_subject)
    print('Saved CSV report to %s' % os.path.abspath(args.output_csv))
    print('Saved JSON report to %s' % os.path.abspath(args.output_json))


if __name__ == '__main__':
    main()
