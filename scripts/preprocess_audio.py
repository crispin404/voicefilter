import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

from utils.audio import load_wav, peak_normalize, repeat_pad_wav, save_wav
from utils.dataset_index import VOWEL_FILES, ensure_dir, load_subjects


def preprocess_file(src_path, dst_path, sample_rate, repeat_seconds=None):
    wav, _ = load_wav(src_path, sample_rate=sample_rate, mono=True)
    wav = peak_normalize(wav)
    if repeat_seconds is not None:
        wav = repeat_pad_wav(wav, sample_rate=sample_rate, target_seconds=repeat_seconds)
    save_wav(dst_path, wav, sample_rate)
    return len(wav)


def preprocess_subject(subject, processed_root, sample_rate, vowel_seconds):
    subject_id = subject['subject_id']

    counts = {
        'vowel': 0,
        'clean': 0,
        'mix': 0,
    }

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
        preprocess_file(src_path, dst_path, sample_rate=sample_rate, repeat_seconds=vowel_seconds)
        counts['vowel'] += 1

    for src_path in subject.get('snore_paths', []):
        dst_path = os.path.join(clean_out_dir, os.path.basename(src_path))
        preprocess_file(src_path, dst_path, sample_rate=sample_rate)
        counts['clean'] += 1

    for src_path in subject.get('mix_paths', []):
        dst_path = os.path.join(mix_out_dir, os.path.basename(src_path))
        preprocess_file(src_path, dst_path, sample_rate=sample_rate)
        counts['mix'] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description='Preprocess vowel, clean snore, and mix audio to 16k mono wavs')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--processed-root', default='processed', help='Processed output root')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--vowel-seconds', type=float, default=1.0, help='Target vowel duration after repeat-padding')
    args = parser.parse_args()

    subjects = load_subjects(args.subjects)
    totals = {'vowel': 0, 'clean': 0, 'mix': 0}
    for subject in tqdm(subjects, desc='preprocess'):
        counts = preprocess_subject(
            subject,
            processed_root=args.processed_root,
            sample_rate=args.sample_rate,
            vowel_seconds=args.vowel_seconds,
        )
        for key, value in counts.items():
            totals[key] += value

    print('Processed %d subjects into %s' % (len(subjects), os.path.abspath(args.processed_root)))
    print('vowel=%d clean=%d mix=%d' % (totals['vowel'], totals['clean'], totals['mix']))


if __name__ == '__main__':
    main()
