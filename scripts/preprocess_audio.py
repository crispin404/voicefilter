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
    VOWEL_CANONICAL_FILENAMES,
    build_clean_index,
    build_snr_lookup,
    default_mix_dir_name,
    default_processed_mix_subdir,
    ensure_dir,
    get_data_noise_count,
    iter_subject_vowel_items,
    list_subject_mix_paths,
    load_csv_rows,
    load_jsonl,
    load_subjects,
    normalize_noise_count,
    normalize_path,
    parse_mix_filename,
    resolve_mode_filepath,
    safe_float,
    write_csv,
)
from utils.hparams import HParam


DEFAULT_SUBJECTS_PATH = os.path.join('metadata', 'subjects.json')
DEFAULT_MIX_DIR_NAME = '合成声'
DEFAULT_SYNTHESIS_METADATA_JSONL = os.path.join('metadata', 'synthesized_mix_metadata.jsonl')
DEFAULT_SYNTHESIS_METADATA_CSV = os.path.join('metadata', 'synthesized_mix_metadata.csv')
DEFAULT_SNR_STATS_CSV = os.path.join('metadata', 'preprocess_snr_stats.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Incrementally preprocess vowel, clean snore, and mix audio.')
    parser.add_argument('-c', '--config', default=None, help='Optional YAML config path used to read data.noise_count')
    parser.add_argument('--subjects', default=DEFAULT_SUBJECTS_PATH, help='subjects.json path')
    parser.add_argument('--processed-root', default='processed', help='Processed output root')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--vowel-seconds', type=float, default=1.0, help='Target vowel duration after repeat-padding')
    parser.add_argument('--pair-peak', type=float, default=0.95, help='Peak target used for pair-wise clean/mix normalization')
    parser.add_argument('--mix-dir-name', default=DEFAULT_MIX_DIR_NAME, help='Raw mix subdirectory name')
    parser.add_argument('--noise-count', type=int, default=None, help='Noise mode to preprocess: 1, 2, or 3')
    parser.add_argument('--force', action='store_true', help='Reprocess all expected outputs even when they are up to date')
    parser.add_argument(
        '--synthesis-metadata',
        default=None,
        help='Optional synthesis metadata CSV/JSONL containing pair_scale/target_snr_db',
    )
    parser.add_argument(
        '--snr-stats-csv',
        default=DEFAULT_SNR_STATS_CSV,
        help='CSV path for per-pair SNR statistics',
    )
    return parser.parse_args()


def resolve_noise_count(config_path, cli_noise_count):
    if cli_noise_count is not None:
        return normalize_noise_count(cli_noise_count)
    if config_path:
        hp = HParam(config_path)
        return get_data_noise_count(hp.data, default=1)
    return 1


def resolve_mode_mix_dir_name(mix_dir_name, noise_count):
    if mix_dir_name == DEFAULT_MIX_DIR_NAME:
        return default_mix_dir_name(noise_count)
    return mix_dir_name


def resolve_mode_file_path(path, default_path, noise_count):
    if not path:
        return path
    if path == default_path:
        return resolve_mode_filepath(path, noise_count)
    return path


def load_metadata_lookup(path):
    if not path or not os.path.isfile(path):
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


def output_up_to_date(output_paths, source_paths):
    if any(not os.path.isfile(path) for path in output_paths):
        return False
    output_mtime = min(os.path.getmtime(path) for path in output_paths)
    return all(os.path.isfile(path) and output_mtime >= os.path.getmtime(path) for path in source_paths)


def stats_match_metadata(stats_row, pair_metadata, noise_count):
    if not stats_row:
        return False
    stats_noise_count = safe_float(stats_row.get('noise_count'))
    if stats_noise_count is not None and stats_noise_count != float(noise_count):
        return False
    if not pair_metadata:
        return True
    expected_target = safe_float(pair_metadata.get('target_snr_db'))
    expected_pair_scale = safe_float(pair_metadata.get('pair_scale'))
    current_target = safe_float(stats_row.get('target_snr_db'))
    current_pair_scale = safe_float(stats_row.get('synthesis_pair_scale'))
    if expected_target is not None and current_target != expected_target:
        return False
    if expected_pair_scale is not None and current_pair_scale != expected_pair_scale:
        return False
    return True


def normalize_stats_row(row, subject_id, clean_path, mix_path, dst_clean_path, dst_mix_path, mix_meta, noise_count):
    updated = dict(row)
    updated['subject_id'] = subject_id
    updated['clean_file'] = normalize_path(clean_path)
    updated['mix_file'] = normalize_path(mix_path)
    updated['noise_type'] = mix_meta['noise_type']
    updated['noise_count'] = int(noise_count)
    updated['output_clean_file'] = normalize_path(dst_clean_path)
    updated['output_mix_file'] = normalize_path(dst_mix_path)
    return updated


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


def current_mode_clean_filename(path, noise_count):
    basename = os.path.basename(path)
    if not basename.lower().endswith('_clean.wav'):
        return False
    mix_basename = basename[:-10] + '.wav'
    mix_meta = parse_mix_filename(mix_basename)
    if mix_meta is None:
        return False
    return mix_meta['noise_type'].count('+') + 1 == int(noise_count)


def remove_stale_processed_files(mix_out_dir, clean_out_dir, expected_mix_filenames, expected_clean_filenames, noise_count):
    removed = 0
    for path in list_subject_processed_wavs(mix_out_dir):
        if os.path.basename(path) not in expected_mix_filenames:
            os.remove(path)
            removed += 1
    for path in list_subject_processed_wavs(clean_out_dir):
        basename = os.path.basename(path)
        if not current_mode_clean_filename(path, noise_count):
            continue
        if basename not in expected_clean_filenames:
            os.remove(path)
            removed += 1
    return removed


def list_subject_processed_wavs(path):
    if not os.path.isdir(path):
        return []
    return [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith('.wav') and os.path.isfile(os.path.join(path, name))
    ]


def preprocess_subject(subject, processed_root, sample_rate, vowel_seconds, pair_peak, mix_dir_name, metadata_lookup, stats_lookup, noise_count, force=False):
    subject_id = subject['subject_id']
    counts = {'vowel': 0, 'vowel_skipped': 0, 'clean': 0, 'mix': 0, 'skipped': 0, 'removed': 0}
    snr_rows = []

    vowel_out_dir = os.path.join(processed_root, 'vowel', subject_id)
    clean_out_dir = os.path.join(processed_root, 'clean', subject_id)
    mix_out_dir = os.path.join(processed_root, default_processed_mix_subdir(noise_count), subject_id)
    ensure_dir(vowel_out_dir)
    ensure_dir(clean_out_dir)
    ensure_dir(mix_out_dir)

    for vowel_key, src_path in iter_subject_vowel_items(subject):
        dst_path = os.path.join(vowel_out_dir, VOWEL_CANONICAL_FILENAMES[vowel_key])
        if not src_path or not os.path.isfile(src_path):
            warn('subject=%s missing vowel=%s' % (subject_id, vowel_key))
            continue
        if not force and output_up_to_date([dst_path], [src_path]):
            counts['vowel_skipped'] += 1
            continue
        preprocess_vowel_file(src_path, dst_path, sample_rate=sample_rate, repeat_seconds=vowel_seconds)
        counts['vowel'] += 1

    for vowel_key, candidates in sorted((subject.get('vowel_candidates') or {}).items()):
        if len(candidates) > 1:
            warn(
                'subject=%s vowel=%s multiple candidates=%s selected=%s'
                % (subject_id, vowel_key, candidates, candidates[0])
            )

    mix_paths = list_subject_mix_paths(subject, mix_dir_name=mix_dir_name, noise_count=noise_count)
    clean_index = build_clean_index(subject['snore_dir'])
    expected_mix_filenames = set()
    expected_clean_filenames = set()
    valid_pairs = []

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
        expected_mix_filenames.add(os.path.basename(dst_mix_path))
        expected_clean_filenames.add(os.path.basename(dst_clean_path))
        valid_pairs.append((mix_path, mix_meta, clean_path, dst_clean_path, dst_mix_path))

    counts['removed'] += remove_stale_processed_files(
        mix_out_dir,
        clean_out_dir,
        expected_mix_filenames,
        expected_clean_filenames,
        noise_count,
    )

    for mix_path, mix_meta, clean_path, dst_clean_path, dst_mix_path in valid_pairs:
        pair_metadata = resolve_pair_metadata(metadata_lookup, subject_id, mix_path)
        previous_stats = resolve_pair_metadata(stats_lookup, subject_id, mix_path)
        if previous_stats is None:
            previous_stats = resolve_pair_metadata(stats_lookup, subject_id, dst_mix_path)

        can_skip = (
            not force
            and output_up_to_date([dst_clean_path, dst_mix_path], [clean_path, mix_path])
            and stats_match_metadata(previous_stats, pair_metadata, noise_count)
        )
        if can_skip:
            snr_rows.append(
                normalize_stats_row(
                    previous_stats,
                    subject_id,
                    clean_path,
                    mix_path,
                    dst_clean_path,
                    dst_mix_path,
                    mix_meta,
                    noise_count,
                )
            )
            counts['skipped'] += 1
            continue

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
            'noise_count': int(noise_count),
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
    noise_count = resolve_noise_count(args.config, args.noise_count)
    mix_dir_name = resolve_mode_mix_dir_name(args.mix_dir_name, noise_count)
    synthesis_metadata = args.synthesis_metadata
    if synthesis_metadata in (DEFAULT_SYNTHESIS_METADATA_JSONL, DEFAULT_SYNTHESIS_METADATA_CSV):
        synthesis_metadata = resolve_mode_filepath(synthesis_metadata, noise_count)
    snr_stats_csv = resolve_mode_file_path(args.snr_stats_csv, DEFAULT_SNR_STATS_CSV, noise_count)

    metadata_lookup = load_metadata_lookup(synthesis_metadata)
    stats_lookup = load_metadata_lookup(snr_stats_csv)
    subjects = load_subjects(args.subjects)
    totals = {'vowel': 0, 'vowel_skipped': 0, 'clean': 0, 'mix': 0, 'skipped': 0, 'removed': 0}
    all_snr_rows = []

    for subject in tqdm(subjects, desc='preprocess'):
        counts, snr_rows = preprocess_subject(
            subject=subject,
            processed_root=args.processed_root,
            sample_rate=args.sample_rate,
            vowel_seconds=args.vowel_seconds,
            pair_peak=args.pair_peak,
            mix_dir_name=mix_dir_name,
            metadata_lookup=metadata_lookup,
            stats_lookup=stats_lookup,
            noise_count=noise_count,
            force=args.force,
        )
        for key, value in counts.items():
            totals[key] += value
        all_snr_rows.extend(snr_rows)

    write_csv(
        all_snr_rows,
        snr_stats_csv,
        fieldnames=[
            'subject_id',
            'clean_file',
            'mix_file',
            'noise_type',
            'noise_count',
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
    print('noise_count=%d raw_mix_dir=%s processed_mix_subdir=%s' % (
        noise_count,
        mix_dir_name,
        default_processed_mix_subdir(noise_count),
    ))
    print(
        'vowel=%d vowel_skipped=%d clean=%d mix=%d pair_skipped=%d removed_stale=%d'
        % (
            totals['vowel'],
            totals['vowel_skipped'],
            totals['clean'],
            totals['mix'],
            totals['skipped'],
            totals['removed'],
        )
    )
    print('SNR stats CSV: %s' % os.path.abspath(snr_stats_csv))


if __name__ == '__main__':
    main()
