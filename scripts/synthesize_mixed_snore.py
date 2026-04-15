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
    ensure_dir,
    list_wavs_if_dir,
    load_subject_ids,
    load_subjects,
    normalize_path,
    save_jsonl,
    strip_wav_extension,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Synthesize snore+noise mixtures with a fixed target SNR.')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--subject-ids-file', default=None, help='Optional text file with one subject_id per line')
    parser.add_argument('--subject-id', action='append', default=None, help='Optional subject_id filter, can be repeated')
    parser.add_argument('--noise-root', required=True, help='Directory containing flat noise wav files')
    parser.add_argument('--output-subdir', default='合成声_2', help='Per-subject output subdirectory for synthesized mixtures')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--target-snr-db', type=float, default=8.0, help='Target clean-over-noise SNR in dB')
    parser.add_argument('--peak-limit', type=float, default=0.99, help='Max allowed absolute peak after mixing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--metadata-path',
        default=os.path.join('metadata', 'synthesized_mix_metadata.jsonl'),
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


def list_noise_files(noise_root):
    noise_paths = list_wavs_if_dir(noise_root)
    if not noise_paths:
        raise FileNotFoundError('No wav files were found in noise root: %s' % os.path.abspath(noise_root))
    return noise_paths


def build_output_filename(clean_path, noise_type):
    basename = os.path.basename(clean_path)
    match = SNORE_PATTERN.match(basename)
    if match is None:
        raise ValueError('Clean snore filename does not match hs_{inner_id}_{snore_index}.wav: %s' % basename)

    inner_id_raw = match.group(1)
    snore_index = int(match.group(2))
    return 'hs_%s_%s_%02d.wav' % (inner_id_raw, noise_type, snore_index)


def synthesize_pair(clean_path, noise_path, output_path, sample_rate, target_snr_db, peak_limit, rng):
    clean_wav, _ = load_wav(clean_path, sample_rate=sample_rate, mono=True)
    noise_wav, _ = load_wav(noise_path, sample_rate=sample_rate, mono=True)

    if clean_wav.size == 0:
        raise ValueError('Clean snore is empty: %s' % clean_path)
    if noise_wav.size == 0:
        raise ValueError('Noise is empty: %s' % noise_path)

    aligned_noise = match_length_with_random_crop(noise_wav, clean_wav.size, rng)
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


def synthesize_subject(subject, noise_paths, output_subdir, sample_rate, target_snr_db, peak_limit, rng):
    clean_paths = subject.get('snore_paths') or [normalize_path(path) for path in list_wavs_if_dir(subject['snore_dir'])]
    if not clean_paths:
        raise FileNotFoundError('No clean snore wav files found for subject %s' % subject['subject_id'])

    output_dir = os.path.join(subject['subject_dir'], output_subdir)
    ensure_dir(output_dir)

    rows = []
    for clean_path in clean_paths:
        clean_name = os.path.basename(clean_path)
        if SNORE_PATTERN.match(clean_name) is None:
            raise ValueError('Unexpected clean snore filename for subject %s: %s' % (subject['subject_id'], clean_name))

        for noise_path in noise_paths:
            noise_type = strip_wav_extension(noise_path)
            output_filename = build_output_filename(clean_path, noise_type)
            output_path = os.path.join(output_dir, output_filename)

            try:
                pair_meta = synthesize_pair(
                    clean_path=clean_path,
                    noise_path=noise_path,
                    output_path=output_path,
                    sample_rate=sample_rate,
                    target_snr_db=target_snr_db,
                    peak_limit=peak_limit,
                    rng=rng,
                )
            except Exception as exc:
                raise RuntimeError(
                    'Failed to synthesize subject=%s clean=%s noise=%s: %s'
                    % (subject['subject_id'], clean_name, os.path.basename(noise_path), exc)
                ) from exc

            rows.append({
                'subject_id': subject['subject_id'],
                'clean_file': normalize_path(clean_path),
                'noise_file': normalize_path(noise_path),
                'noise_type': noise_type,
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
    return rows


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    subjects = load_subjects(args.subjects)
    selected_subjects = select_subjects(subjects, subject_ids_file=args.subject_ids_file, subject_ids=args.subject_id)
    if not selected_subjects:
        raise ValueError('No subjects were selected.')

    noise_paths = list_noise_files(args.noise_root)

    metadata_rows = []
    for subject in tqdm(selected_subjects, desc='synthesize'):
        metadata_rows.extend(
            synthesize_subject(
                subject=subject,
                noise_paths=noise_paths,
                output_subdir=args.output_subdir,
                sample_rate=args.sample_rate,
                target_snr_db=args.target_snr_db,
                peak_limit=args.peak_limit,
                rng=rng,
            )
        )

    save_jsonl(metadata_rows, args.metadata_path)
    if args.metadata_csv:
        write_csv(
            metadata_rows,
            args.metadata_csv,
            fieldnames=[
                'subject_id',
                'clean_file',
                'noise_file',
                'noise_type',
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

    print('Synthesized %d mixtures for %d subjects.' % (len(metadata_rows), len(selected_subjects)))
    print('JSONL metadata: %s' % os.path.abspath(args.metadata_path))
    if args.metadata_csv:
        print('CSV metadata: %s' % os.path.abspath(args.metadata_csv))


if __name__ == '__main__':
    main()
