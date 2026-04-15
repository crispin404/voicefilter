import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.audio import (
    compute_snr_db,
    load_wav,
    paired_peak_normalize,
    peak_normalize,
    repeat_pad_wav,
    save_wav,
)
from utils.dataset_index import (
    VOWEL_FILES,
    build_clean_index,
    build_snr_lookup,
    ensure_dir,
    list_subject_mix_paths,
    load_csv_rows,
    load_jsonl,
    load_subjects,
    normalize_path,
    parse_mix_filename,
    safe_float,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess vowel, clean snore, and mix audio with pair-wise normalization.')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--processed-root', default='processed', help='Processed output root')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--vowel-seconds', type=float, default=1.0, help='Target vowel duration after repeat-padding')
    parser.add_argument('--pair-peak', type=float, default=0.95, help='Peak target used for pair-wise clean/mix normalization')
    parser.add_argument('--mix-dir-name', default=None, help='Optional raw mix subdirectory name, e.g. 合成声_2')
    parser.add_argument(
        '--synthesis-metadata',
        default=None,
        help='Optional synthesis metadata CSV/JSONL containing pair_scale/target_snr_db',
    )
    parser.add_argument(
        '--snr-stats-csv',
        default=os.path.join('metadata', 'preprocess_snr_stats.csv'),
        help='CSV path for per-pair SNR statistics',
    )
    return parser.parse_args()


def load_metadata_lookup(path):
    if not path:
        return {}
    if path.lower().endswith('.jsonl'):
        rows = load_jsonl(path)
    elif path.lower().endswith('.csv'):
        rows = load_csv_rows(path)
    else:
        raise ValueError('Unsupported metadata format: %s' % path)
    return build_snr_lookup(rows)


def preprocess_vowel_file(src_path, dst_path, sample_rate, repeat_seconds):
    wav, _ = load_wav(src_path, sample_rate=sample_rate, mono=True)
    wav = peak_normalize(wav)
    wav = repeat_pad_wav(wav, sample_rate=sample_rate, target_seconds=repeat_seconds)
    save_wav(dst_path, wav, sample_rate)


def warn(message):
    print('WARNING: %s' % message)


def resolve_pair_metadata(lookup, subject_id, mix_path):
    if not lookup:
        return None
    item = lookup.get((subject_id, os.path.basename(mix_path)))
    if item is not None:
        return item
    return lookup.get(normalize_path(mix_path))


def preprocess_pair(clean_path, mix_path, dst_clean_path, dst_mix_path, sample_rate, pair_peak, pair_metadata=None):
    clean_wav, _ = load_wav(clean_path, sample_rate=sample_rate, mono=True)
    mix_wav, _ = load_wav(mix_path, sample_rate=sample_rate, mono=True)

    if clean_wav.size == 0:
        raise ValueError('Clean waveform is empty: %s' % clean_path)
    if mix_wav.size == 0:
        raise ValueError('Mix waveform is empty: %s' % mix_path)

    synthesis_pair_scale = 1.0
    if pair_metadata:
        parsed_pair_scale = safe_float(pair_metadata.get('pair_scale'))
        if parsed_pair_scale is not None and parsed_pair_scale > 0:
            synthesis_pair_scale = parsed_pair_scale
            clean_wav = clean_wav * np.float32(synthesis_pair_scale)

    length_warning = ''
    if clean_wav.size != mix_wav.size:
        length_warning = 'length_mismatch:%d_vs_%d' % (clean_wav.size, mix_wav.size)
    pair_length = min(clean_wav.size, mix_wav.size)
    clean_wav = clean_wav[:pair_length]
    mix_wav = mix_wav[:pair_length]

    estimated_noise = mix_wav - clean_wav
    actual_snr_db = compute_snr_db(clean_wav, estimated_noise)
    processed_clean, processed_mix, preprocess_scale = paired_peak_normalize(clean_wav, mix_wav, peak=pair_peak)

    save_wav(dst_clean_path, processed_clean, sample_rate)
    save_wav(dst_mix_path, processed_mix, sample_rate)

    warnings = []
    if length_warning:
        warnings.append(length_warning)
    if actual_snr_db is None:
        warnings.append('invalid_snr')
    if np.max(np.abs(mix_wav)) > 1.01:
        warnings.append('mix_peak_gt_1_before_preprocess')
    target_snr_db = safe_float(pair_metadata.get('target_snr_db')) if pair_metadata else None
    if target_snr_db is not None and actual_snr_db is not None and abs(actual_snr_db - target_snr_db) > 0.75:
        warnings.append('snr_deviation_gt_0.75db')

    return {
        'target_snr_db': target_snr_db,
        'actual_snr_db': actual_snr_db,
        'duration_seconds': float(pair_length) / float(sample_rate),
        'synthesis_pair_scale': float(synthesis_pair_scale),
        'preprocess_pair_scale': float(preprocess_scale),
        'warning': ';'.join(warnings),
    }


def preprocess_subject(subject, processed_root, sample_rate, vowel_seconds, pair_peak, mix_dir_name, metadata_lookup):
    subject_id = subject['subject_id']
    counts = {'vowel': 0, 'clean': 0, 'mix': 0}
    snr_rows = []

    vowel_out_dir = os.path.join(processed_root, 'vowel', subject_id)
    clean_out_dir = os.path.join(processed_root, 'clean', subject_id)
    mix_out_dir = os.path.join(processed_root, 'mix', subject_id)
    ensure_dir(vowel_out_dir)
    ensure_dir(clean_out_dir)
    ensure_dir(mix_out_dir)

    for vowel_name in VOWEL_FILES:
        src_path = os.path.join(subject['vowel_dir'], vowel_name)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(vowel_out_dir, vowel_name)
        preprocess_vowel_file(src_path, dst_path, sample_rate=sample_rate, repeat_seconds=vowel_seconds)
        counts['vowel'] += 1

    mix_paths = list_subject_mix_paths(subject, mix_dir_name=mix_dir_name)
    clean_index = build_clean_index(subject['snore_dir'])

    for mix_path in mix_paths:
        mix_meta = parse_mix_filename(mix_path)
        if mix_meta is None:
            warn('skip unrecognized mix filename: %s' % mix_path)
            continue

        clean_path = clean_index.get((mix_meta['inner_id'], mix_meta['snore_index']))
        if clean_path is None:
            warn('missing paired clean snore for mix: %s' % mix_path)
            continue

        mix_stem = os.path.splitext(os.path.basename(mix_path))[0]
        dst_clean_path = os.path.join(clean_out_dir, '%s_clean.wav' % mix_stem)
        dst_mix_path = os.path.join(mix_out_dir, os.path.basename(mix_path))
        pair_metadata = resolve_pair_metadata(metadata_lookup, subject_id, mix_path)

        try:
            pair_stats = preprocess_pair(
                clean_path=clean_path,
                mix_path=mix_path,
                dst_clean_path=dst_clean_path,
                dst_mix_path=dst_mix_path,
                sample_rate=sample_rate,
                pair_peak=pair_peak,
                pair_metadata=pair_metadata,
            )
        except Exception as exc:
            warn('failed to preprocess subject=%s mix=%s: %s' % (subject_id, os.path.basename(mix_path), exc))
            continue

        counts['clean'] += 1
        counts['mix'] += 1

        row = {
            'subject_id': subject_id,
            'clean_file': normalize_path(clean_path),
            'mix_file': normalize_path(mix_path),
            'noise_type': mix_meta['noise_type'],
            'output_clean_file': normalize_path(dst_clean_path),
            'output_mix_file': normalize_path(dst_mix_path),
            'target_snr_db': pair_stats['target_snr_db'],
            'actual_snr_db': pair_stats['actual_snr_db'],
            'duration_seconds': pair_stats['duration_seconds'],
            'synthesis_pair_scale': pair_stats['synthesis_pair_scale'],
            'preprocess_pair_scale': pair_stats['preprocess_pair_scale'],
            'warning': pair_stats['warning'],
        }
        snr_rows.append(row)

        if row['warning']:
            warn('subject=%s mix=%s %s' % (subject_id, os.path.basename(mix_path), row['warning']))

    return counts, snr_rows


def main():
    args = parse_args()

    metadata_lookup = load_metadata_lookup(args.synthesis_metadata)
    subjects = load_subjects(args.subjects)
    totals = {'vowel': 0, 'clean': 0, 'mix': 0}
    all_snr_rows = []

    for subject in tqdm(subjects, desc='preprocess'):
        counts, snr_rows = preprocess_subject(
            subject=subject,
            processed_root=args.processed_root,
            sample_rate=args.sample_rate,
            vowel_seconds=args.vowel_seconds,
            pair_peak=args.pair_peak,
            mix_dir_name=args.mix_dir_name,
            metadata_lookup=metadata_lookup,
        )
        for key, value in counts.items():
            totals[key] += value
        all_snr_rows.extend(snr_rows)

    write_csv(
        all_snr_rows,
        args.snr_stats_csv,
        fieldnames=[
            'subject_id',
            'clean_file',
            'mix_file',
            'noise_type',
            'output_clean_file',
            'output_mix_file',
            'target_snr_db',
            'actual_snr_db',
            'duration_seconds',
            'synthesis_pair_scale',
            'preprocess_pair_scale',
            'warning',
        ],
    )

    print('Processed %d subjects into %s' % (len(subjects), os.path.abspath(args.processed_root)))
    print('vowel=%d clean=%d mix=%d' % (totals['vowel'], totals['clean'], totals['mix']))
    print('SNR stats CSV: %s' % os.path.abspath(args.snr_stats_csv))


if __name__ == '__main__':
    main()
