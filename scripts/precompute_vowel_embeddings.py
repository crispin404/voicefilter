import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from tqdm import tqdm

from model.vowel_encoder import VowelEmbeddingEncoder
from utils.audio import Audio, load_wav
from utils.dataset_index import VOWEL_FILES, ensure_dir, load_subjects
from utils.embedder_checkpoint import DEFAULT_EMBEDDER_PATH, resolve_embedder_path
from utils.hparams import HParam


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def load_vowel_mels(audio, subject, processed_root):
    subject_id = subject['subject_id']
    vowel_dir = os.path.join(processed_root, 'vowel', subject_id)
    mels = []
    for vowel_name in VOWEL_FILES:
        wav_path = os.path.join(vowel_dir, vowel_name)
        wav, _ = load_wav(wav_path, sample_rate=audio.hp.audio.sample_rate, mono=True)
        mel = audio.get_mel(wav)
        mels.append(torch.from_numpy(mel).float())
    return mels


def main():
    parser = argparse.ArgumentParser(description='Precompute aggregated vowel embeddings for each subject')
    parser.add_argument('-c', '--config', required=True, help='YAML config path')
    parser.add_argument('--subjects', default=os.path.join('metadata', 'subjects.json'), help='subjects.json path')
    parser.add_argument('--processed-root', default='processed', help='Processed root containing vowel wavs')
    parser.add_argument('--embedder-path', default=None, help='Pre-trained embedder weights, defaults to %s' % DEFAULT_EMBEDDER_PATH)
    parser.add_argument('--output-dir', default=os.path.join('processed', 'embeddings'), help='Output directory for .npy embeddings')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    args = parser.parse_args()

    hp = HParam(args.config)
    device = build_device(args.device)
    embedder_path = resolve_embedder_path(args.embedder_path)
    subjects = load_subjects(args.subjects)
    audio = Audio(hp)

    encoder = VowelEmbeddingEncoder(hp)
    loaded = encoder.load_embedder(embedder_path)
    encoder.eval()
    encoder.to(device)

    ensure_dir(args.output_dir)
    with torch.no_grad():
        for subject in tqdm(subjects, desc='embeddings'):
            vowel_mels = [mel.to(device) for mel in load_vowel_mels(audio, subject, args.processed_root)]
            embedding = encoder(vowel_mels).detach().cpu().numpy().astype(np.float32)
            output_path = os.path.join(args.output_dir, '%s.npy' % subject['subject_id'])
            np.save(output_path, embedding)

    print('Saved %d embeddings to %s' % (len(subjects), os.path.abspath(args.output_dir)))
    print('Embedder checkpoint loaded: %s' % ('yes' if loaded else 'no'))


if __name__ == '__main__':
    main()
