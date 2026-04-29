import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

from utils.audio import (
    compute_rms,
    compute_snr_db,
    load_wav,
    match_length_with_random_crop,
    save_wav,
    scale_noise_to_target_snr,
)
from utils.dataset_index import (
    SNORE_PATTERN,
    default_mix_dir_name,
    ensure_dir,
    get_data_noise_count,
    list_wavs_if_dir,
    load_jsonl,
    load_subject_ids,
    load_subjects,
    normalize_noise_count,
    normalize_path,
    parse_mix_filename,
    resolve_mode_filepath,
    safe_float,
    save_jsonl,
    strip_wav_extension,
    write_csv,
)
from utils.hparams import HParam


DEFAULT_SUBJECTS_PATH = os.path.join('metadata', 'subjects.json')
DEFAULT_CONFIG_PATH = os.path.join('config', 'enhancement.yaml')
DEFAULT_OUTPUT_SUBDIR = '合成声'
DEFAULT_METADATA_PATH = os.path.join('metadata', 'synthesized_mix_metadata.jsonl')
DEFAULT_METADATA_CSV = os.path.join('metadata', 'synthesized_mix_metadata.csv')
DEFAULT_NOISE_SPLICE_PATH = os.path.join('data', 'noise_splice.txt')


def parse_args():
    parser = argparse.ArgumentParser(description='Incrementally synthesize snore+noise mixtures with a fixed target SNR.')
    parser.add_argument('-c', '--config', default=None, help='Optional YAML config path used to read data.noise_count')
    parser.add_argument('--subjects', default=DEFAULT_SUBJECTS_PATH, help='subjects.json path')
    parser.add_argument('--subject-ids-file', default=None, help='Optional text file with one subject_id per line')
    parser.add_argument('--subject-id', action='append', default=None, help='Optional subject_id filter, can be repeated')
    parser.add_argument('--noise-root', required=True, help='Directory containing flat noise wav files')
    parser.add_argument('--noise-count', type=int, default=None, help='Noise mode to synthesize: 1, 2, or 3')
    parser.add_argument('--noise-splice-file', default=DEFAULT_NOISE_SPLICE_PATH, help='Noise recipe file used when noise_count is 2 or 3')
    parser.add_argument('--output-subdir', default=DEFAULT_OUTPUT_SUBDIR, help='Per-subject output subdirectory for synthesized mixtures')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--target-snr-db', type=float, default=8.0, help='Target clean-over-noise SNR in dB')
    parser.add_argument('--peak-limit', type=float, default=0.99, help='Max allowed absolute peak after mixing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Re-synthesize all expected mixtures even when outputs are up to date')
    parser.add_argument(
        '--metadata-path',
        default=DEFAULT_METADATA_PATH,
        help='Output JSONL path for synthesis metadata',
    )
    parser.add_argument('--metadata-csv', default=None, help='Optional CSV path for synthesis metadata')
    return parser.parse_args()


def select_subjects(subjects, subject_ids_file=None, subject_ids=None):
    selected_ids = set()
    if subject_ids_file:
        selected_ids.update(load_subject_ids(subject_ids_file))
    if subject_ids:
        selected_ids.update(subject_ids)
    if not selected_ids:
        return subjects
    return [subject for subject in subjects if subject['subject_id'] in selected_ids]


def resolve_noise_count(config_path, cli_noise_count):
    if cli_noise_count is not None:
        return normalize_noise_count(cli_noise_count)
    if config_path:
        hp = HParam(config_path)
        return get_data_noise_count(hp.data, default=1)
    return 1


def resolve_mode_output_subdir(output_subdir, noise_count):
    if output_subdir == DEFAULT_OUTPUT_SUBDIR:
        return default_mix_dir_name(noise_count)
    return output_subdir


def resolve_mode_file_path(path, default_path, noise_count):
    if not path:
        return path
    if path == default_path:
        return resolve_mode_filepath(path, noise_count)
    return path


def normalize_noise_spec(noise_type, noise_paths):
    normalized_paths = [normalize_path(path) for path in noise_paths]
    noise_file = normalized_paths[0] if len(normalized_paths) == 1 else '|'.join(normalized_paths)
    return {
        'noise_type': noise_type,
        'noise_paths': normalized_paths,
        'noise_file': noise_file,
    }


def list_single_noise_specs(noise_root):
    noise_paths = list_wavs_if_dir(noise_root)
    if not noise_paths:
        raise FileNotFoundError('No wav files were found in noise root: %s' % os.path.abspath(noise_root))
    return [normalize_noise_spec(strip_wav_extension(path), [path]) for path in noise_paths]


def load_noise_specs_from_splice(noise_root, noise_splice_path, noise_count):
    abs_noise_splice_path = os.path.abspath(noise_splice_path)
    if not os.path.isfile(noise_splice_path):
        raise FileNotFoundError(
            'noise_splice.txt not found for noise_count=%d: %s' % (noise_count, abs_noise_splice_path)
        )

    specs = []
    seen_types = set()
    with open(noise_splice_path, 'r', encoding='utf-8-sig') as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [part.strip() for part in line.split('+')]
            if any(not part for part in parts):
                raise ValueError('Invalid noise recipe at %s:%d -> %s' % (abs_noise_splice_path, line_number, raw_line.rstrip()))
            if len(parts) not in (2, 3):
                raise ValueError(
                    'Noise recipe at %s:%d has %d part(s), expected 2 or 3: %s'
                    % (abs_noise_splice_path, line_number, len(parts), line)
                )
            if len(parts) != noise_count:
                continue

            missing = []
            noise_paths = []
            for part in parts:
                candidate = os.path.join(noise_root, '%s.wav' % part)
                if not os.path.isfile(candidate):
                    missing.append(part)
                else:
                    noise_paths.append(candidate)
            if missing:
                raise FileNotFoundError(
                    'Missing noise wav(s) for %s:%d -> %s (missing: %s)'
                    % (abs_noise_splice_path, line_number, line, ', '.join(missing))
                )

            noise_type = '+'.join(parts)
            if noise_type in seen_types:
                raise ValueError('Duplicate noise recipe in %s: %s' % (abs_noise_splice_path, noise_type))
            seen_types.add(noise_type)
            specs.append(normalize_noise_spec(noise_type, noise_paths))

    if not specs:
        raise ValueError('No valid noise recipes were found in %s for noise_count=%d' % (abs_noise_splice_path, noise_count))
    return specs


def list_noise_specs(noise_root, noise_count, noise_splice_path):
    if noise_count == 1:
        return list_single_noise_specs(noise_root)
    return load_noise_specs_from_splice(noise_root, noise_splice_path, noise_count)


def build_output_filename(clean_path, noise_type):
    basename = os.path.basename(clean_path)
    match = SNORE_PATTERN.match(basename)
    if match is None:
        raise ValueError('Clean snore filename does not match hs_{inner_id}_{snore_index}.wav: %s' % basename)

    inner_id_raw = match.group(1)
    snore_index = int(match.group(2))
    return 'hs_%s_%s_%02d.wav' % (inner_id_raw, noise_type, snore_index)


def metadata_key(subject_id, output_filename):
    return subject_id, output_filename


def load_existing_metadata(path):
    if not path or not os.path.isfile(path):
        return {}
    return {
        metadata_key(row.get('subject_id'), os.path.basename(row.get('output_mix_file', ''))): row
        for row in load_jsonl(path)
        if row.get('subject_id') and row.get('output_mix_file')
    }


def is_output_up_to_date(output_path, source_paths):
    if not os.path.isfile(output_path):
        return False
    output_mtime = os.path.getmtime(output_path)
    return all(os.path.isfile(path) and output_mtime >= os.path.getmtime(path) for path in source_paths)


def row_matches_current_run(row, sample_rate, target_snr_db, noise_count):
    if not row:
        return False
    row_sample_rate = safe_float(row.get('sample_rate'))
    row_target_snr = safe_float(row.get('target_snr_db'))
    row_noise_count = safe_float(row.get('noise_count'))
    return (
        row_sample_rate == float(sample_rate)
        and row_target_snr == float(target_snr_db)
        and row_noise_count == float(noise_count)
    )


def normalize_metadata_row(row, subject_id, clean_path, noise_spec, output_path, target_snr_db, sample_rate, noise_count):
    updated = dict(row)
    updated['subject_id'] = subject_id
    updated['clean_file'] = normalize_path(clean_path)
    updated['noise_file'] = noise_spec['noise_file']
    updated['noise_type'] = noise_spec['noise_type']
    updated['noise_count'] = int(noise_count)
    updated['output_mix_file'] = normalize_path(output_path)
    updated['target_snr_db'] = float(target_snr_db)
    updated['sample_rate'] = int(sample_rate)
    return updated


def remove_stale_mix_files(output_dir, expected_filenames):
    removed = 0
    for path in list_wavs_if_dir(output_dir):
        basename = os.path.basename(path)
        if parse_mix_filename(path) is None:
            continue
        if basename not in expected_filenames:
            os.remove(path)
            removed += 1
    return removed


def concatenate_noise_paths(noise_paths, sample_rate):
    parts = []
    for noise_path in noise_paths:
        noise_wav, _ = load_wav(noise_path, sample_rate=sample_rate, mono=True)
        if noise_wav.size == 0:
            raise ValueError('Noise is empty: %s' % noise_path)
        parts.append(noise_wav)
    if not parts:
        raise ValueError('Noise recipe resolved to zero waveforms.')
    return np.concatenate(parts, axis=0).astype(np.float32)


def repeat_to_length(wav, target_length):
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if wav.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    repeats = int(np.ceil(float(target_length) / float(wav.size)))
    return np.tile(wav, repeats)[:target_length].astype(np.float32)


def align_noise_wav(noise_wav, target_length, rng, noise_count):
    if normalize_noise_count(noise_count) == 1:
        return match_length_with_random_crop(noise_wav, target_length, rng)
    return repeat_to_length(noise_wav, target_length)


def synthesize_pair(clean_path, noise_spec, output_path, sample_rate, target_snr_db, peak_limit, rng, noise_count):
    clean_wav, _ = load_wav(clean_path, sample_rate=sample_rate, mono=True)
    if clean_wav.size == 0:
        raise ValueError('Clean snore is empty: %s' % clean_path)

    noise_wav = concatenate_noise_paths(noise_spec['noise_paths'], sample_rate=sample_rate)
    aligned_noise = align_noise_wav(noise_wav, clean_wav.size, rng, noise_count)
    scaled_noise, noise_scale, actual_snr_db = scale_noise_to_target_snr(clean_wav, aligned_noise, target_snr_db)
    mix_wav = clean_wav + scaled_noise

    peak_before_scale = float(np.max(np.abs(mix_wav))) if mix_wav.size > 0 else 0.0
    pair_scale = 1.0
    if peak_before_scale > peak_limit:
        pair_scale = peak_limit / peak_before_scale
        mix_wav = mix_wav * pair_scale

    save_wav(output_path, mix_wav, sample_rate, subtype='FLOAT')

    return {
        'duration_seconds': float(clean_wav.size) / float(sample_rate),
        'clean_rms': compute_rms(clean_wav),
        'noise_rms': compute_rms(scaled_noise),
        'actual_snr_db': actual_snr_db if actual_snr_db is not None else compute_snr_db(clean_wav, scaled_noise),
        'noise_scale': noise_scale,
        'pair_scale': float(pair_scale),
        'peak_before_scale': peak_before_scale,
        'peak_after_scale': float(peak_before_scale * pair_scale),
    }


def synthesize_subject(subject, noise_specs, output_subdir, sample_rate, target_snr_db, peak_limit, rng, metadata_lookup, noise_count, force=False):
    clean_paths = subject.get('snore_paths') or [normalize_path(path) for path in list_wavs_if_dir(subject['snore_dir'])]
    if not clean_paths:
        raise FileNotFoundError('No clean snore wav files found for subject %s' % subject['subject_id'])

    output_dir = os.path.join(subject['subject_dir'], output_subdir)
    ensure_dir(output_dir)

    expected_filenames = set()
    for clean_path in clean_paths:
        for noise_spec in noise_specs:
            expected_filenames.add(build_output_filename(clean_path, noise_spec['noise_type']))
    removed = remove_stale_mix_files(output_dir, expected_filenames)

    rows = []
    synthesized = 0
    skipped = 0
    for clean_path in clean_paths:
        clean_name = os.path.basename(clean_path)
        if SNORE_PATTERN.match(clean_name) is None:
            raise ValueError('Unexpected clean snore filename for subject %s: %s' % (subject['subject_id'], clean_name))

        for noise_spec in noise_specs:
            output_filename = build_output_filename(clean_path, noise_spec['noise_type'])
            output_path = os.path.join(output_dir, output_filename)
            existing_row = metadata_lookup.get(metadata_key(subject['subject_id'], output_filename))
            source_paths = [clean_path] + list(noise_spec['noise_paths'])
            up_to_date = (
                not force
                and is_output_up_to_date(output_path, source_paths)
                and row_matches_current_run(existing_row, sample_rate, target_snr_db, noise_count)
            )

            if up_to_date:
                rows.append(
                    normalize_metadata_row(
                        existing_row,
                        subject['subject_id'],
                        clean_path,
                        noise_spec,
                        output_path,
                        target_snr_db,
                        sample_rate,
                        noise_count,
                    )
                )
                skipped += 1
                continue

            try:
                pair_meta = synthesize_pair(
                    clean_path=clean_path,
                    noise_spec=noise_spec,
                    output_path=output_path,
                    sample_rate=sample_rate,
                    target_snr_db=target_snr_db,
                    peak_limit=peak_limit,
                    rng=rng,
                    noise_count=noise_count,
                )
            except Exception as exc:
                raise RuntimeError(
                    'Failed to synthesize subject=%s clean=%s noise=%s: %s'
                    % (subject['subject_id'], clean_name, noise_spec['noise_type'], exc)
                ) from exc

            rows.append({
                'subject_id': subject['subject_id'],
                'clean_file': normalize_path(clean_path),
                'noise_file': noise_spec['noise_file'],
                'noise_type': noise_spec['noise_type'],
                'noise_count': int(noise_count),
                'output_mix_file': normalize_path(output_path),
                'target_snr_db': float(target_snr_db),
                'actual_snr_db': pair_meta['actual_snr_db'],
                'duration_seconds': pair_meta['duration_seconds'],
                'sample_rate': int(sample_rate),
                'noise_scale': pair_meta['noise_scale'],
                'pair_scale': pair_meta['pair_scale'],
                'peak_before_scale': pair_meta['peak_before_scale'],
                'peak_after_scale': pair_meta['peak_after_scale'],
            })
            synthesized += 1

    return rows, {'synthesized': synthesized, 'skipped': skipped, 'removed': removed}


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    noise_count = resolve_noise_count(args.config, args.noise_count)
    output_subdir = resolve_mode_output_subdir(args.output_subdir, noise_count)
    metadata_path = resolve_mode_file_path(args.metadata_path, DEFAULT_METADATA_PATH, noise_count)
    metadata_csv = resolve_mode_file_path(args.metadata_csv, DEFAULT_METADATA_CSV, noise_count)

    subjects = load_subjects(args.subjects)
    selected_subjects = select_subjects(subjects, subject_ids_file=args.subject_ids_file, subject_ids=args.subject_id)
    if not selected_subjects:
        raise ValueError('No subjects were selected.')

    noise_specs = list_noise_specs(args.noise_root, noise_count, args.noise_splice_file)
    metadata_lookup = load_existing_metadata(metadata_path)

    metadata_rows = []
    totals = {'synthesized': 0, 'skipped': 0, 'removed': 0}
    for subject in tqdm(selected_subjects, desc='synthesize'):
        rows, counts = synthesize_subject(
            subject=subject,
            noise_specs=noise_specs,
            output_subdir=output_subdir,
            sample_rate=args.sample_rate,
            target_snr_db=args.target_snr_db,
            peak_limit=args.peak_limit,
            rng=rng,
            metadata_lookup=metadata_lookup,
            noise_count=noise_count,
            force=args.force,
        )
        metadata_rows.extend(rows)
        for key, value in counts.items():
            totals[key] += value

    save_jsonl(metadata_rows, metadata_path)
    if metadata_csv:
        write_csv(
            metadata_rows,
            metadata_csv,
            fieldnames=[
                'subject_id',
                'clean_file',
                'noise_file',
                'noise_type',
                'noise_count',
                'output_mix_file',
                'target_snr_db',
                'actual_snr_db',
                'duration_seconds',
                'sample_rate',
                'noise_scale',
                'pair_scale',
                'peak_before_scale',
                'peak_after_scale',
            ],
        )

    print('noise_count=%d output_subdir=%s noise_recipes=%d' % (noise_count, output_subdir, len(noise_specs)))
    print('Current mixture rows: %d for %d subjects.' % (len(metadata_rows), len(selected_subjects)))
    print('synthesized=%d skipped=%d removed_stale=%d' % (
        totals['synthesized'],
        totals['skipped'],
        totals['removed'],
    ))
    print('JSONL metadata: %s' % os.path.abspath(metadata_path))
    if metadata_csv:
        print('CSV metadata: %s' % os.path.abspath(metadata_csv))


if __name__ == '__main__':
    main()
