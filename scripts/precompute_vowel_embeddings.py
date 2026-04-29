import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from tqdm import tqdm

from model.vowel_encoder import VowelEmbeddingEncoder
from utils.audio import Audio, load_wav
from utils.dataset_index import (
    discover_vowel_files,
    ensure_dir,
    get_vowel_embedding_keys,
    load_subjects,
    normalize_vowel_embedding_mode,
    resolve_embedding_dir,
)
from utils.embedder_checkpoint import DEFAULT_EMBEDDER_PATH, resolve_embedder_path
from utils.hparams import HParam


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def load_vowel_mels(audio, subject, processed_root, vowel_keys):
    subject_id = subject['subject_id']
    vowel_dir = os.path.join(processed_root, 'vowel', subject_id)
    vowel_info = discover_vowel_files(vowel_dir)
    missing = [key for key in vowel_keys if key not in vowel_info['selected']]
    if missing:
        raise FileNotFoundError('Missing vowel files in %s for subject=%s: %s' % (vowel_dir, subject_id, ', '.join(missing)))

    for vowel_key, candidates in sorted(vowel_info['conflicts'].items()):
        if vowel_key in vowel_keys:
            print(
                'WARNING: multiple vowel candidates in %s for %s, using %s'
                % (vowel_dir, vowel_key, candidates[0])
            )

    mels = []
    for vowel_key in vowel_keys:
        wav_path = vowel_info['selected'][vowel_key]
        wav, _ = load_wav(wav_path, sample_rate=audio.hp.audio.sample_rate, mono=True)
        mel = audio.get_mel(wav)
        mels.append(torch.from_numpy(mel).float())
    return mels


def main():
    parser = argparse.ArgumentParser(description='Precompute vowel embeddings for each subject')
    parser.add_argument('-c', '--config', required=True, help='YAML config path')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--processed-root', default=None, help='Processed root containing vowel wavs')
    parser.add_argument('--embedder-path', default=None, help='Pre-trained embedder weights, defaults to %s' % DEFAULT_EMBEDDER_PATH)
    parser.add_argument('--output-dir', default=None, help='Optional output directory for .npy embeddings')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    args = parser.parse_args()

    hp = HParam(args.config)
    device = build_device(args.device)
    embedder_path = resolve_embedder_path(args.embedder_path)
    subjects = load_subjects(args.subjects)
    processed_root = args.processed_root if args.processed_root is not None else hp.data.get('processed_root', 'processed')
    vowel_embedding_mode = normalize_vowel_embedding_mode(hp.data.get('vowel_embedding_mode', 'avg'))
    vowel_keys = get_vowel_embedding_keys(vowel_embedding_mode)
    output_dir = args.output_dir or resolve_embedding_dir(processed_root, vowel_embedding_mode=vowel_embedding_mode)
    audio = Audio(hp)

    encoder = VowelEmbeddingEncoder(hp)
    loaded = encoder.load_embedder(embedder_path)
    encoder.eval()
    encoder.to(device)

    ensure_dir(output_dir)
    print(
        'Precomputing vowel embeddings: mode=%s keys=%s output_dir=%s'
        % (vowel_embedding_mode, ','.join(vowel_keys), os.path.abspath(output_dir))
    )

    with torch.no_grad():
        for subject in tqdm(subjects, desc='embeddings'):
            vowel_mels = [mel.to(device) for mel in load_vowel_mels(audio, subject, processed_root, vowel_keys)]
            embedding = encoder(vowel_mels).detach().cpu().numpy().astype(np.float32)
            output_path = os.path.join(output_dir, '%s.npy' % subject['subject_id'])
            np.save(output_path, embedding)

    print('Saved %d embeddings to %s' % (len(subjects), os.path.abspath(output_dir)))
    print('Embedder checkpoint loaded: %s' % ('yes' if loaded else 'no'))


if __name__ == '__main__':
    main()
