import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from model.embedding_adapter import EmbeddingAdapter
from model.model import VoiceFilter
from model.vowel_encoder import VowelEmbeddingEncoder
from utils.audio import Audio, load_wav, pad_or_trim_wav, peak_normalize, repeat_pad_wav, save_wav
from utils.dataset_index import VOWEL_KEYS, discover_vowel_files, ensure_dir
from utils.hparams import HParam


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def load_vowel_embedding(hp, device, vowel_dir, embedder_path=None):
    audio = Audio(hp)
    encoder = VowelEmbeddingEncoder(hp).to(device)
    if embedder_path:
        encoder.load_embedder(embedder_path)
    encoder.eval()

    vowel_info = discover_vowel_files(vowel_dir)
    if vowel_info['missing']:
        raise FileNotFoundError('Missing vowel files in %s: %s' % (vowel_dir, ', '.join(vowel_info['missing'])))
    for vowel_key, candidates in sorted(vowel_info['conflicts'].items()):
        print(
            'WARNING: multiple vowel candidates in %s for %s, using %s'
            % (vowel_dir, vowel_key, candidates[0])
        )

    vowel_mels = []
    for vowel_key in VOWEL_KEYS:
        wav_path = vowel_info['selected'][vowel_key]
        wav, _ = load_wav(wav_path, sample_rate=hp.audio.sample_rate, mono=True)
        wav = peak_normalize(repeat_pad_wav(wav, hp.audio.sample_rate, 1.0))
        mel = torch.from_numpy(audio.get_mel(wav)).float().to(device)
        vowel_mels.append(mel)

    with torch.no_grad():
        embedding = encoder(vowel_mels)
    return embedding.unsqueeze(0)


def overlap_add(windows, total_length, window_length, hop_length):
    output = np.zeros(total_length, dtype=np.float32)
    weights = np.zeros(total_length, dtype=np.float32)
    window = np.hanning(window_length).astype(np.float32)
    if np.max(window) <= 0:
        window = np.ones(window_length, dtype=np.float32)

    for start, wav in windows:
        end = min(total_length, start + window_length)
        valid_length = end - start
        if valid_length <= 0:
            continue
        output[start:end] += wav[:valid_length] * window[:valid_length]
        weights[start:end] += window[:valid_length]

    weights = np.maximum(weights, 1e-6)
    return output / weights


def main():
    parser = argparse.ArgumentParser(description='Run sliding-window long-form enhancement on a mixed snore wav')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--checkpoint-path', required=True, help='Trained enhancement checkpoint')
    parser.add_argument('--mixed-file', required=True, help='Path to the 30-second mixed wav')
    parser.add_argument('--vowel-dir', required=True, help='Directory containing 5 vowel wavs')
    parser.add_argument('--embedder-path', default=None, help='Optional embedder checkpoint used for vowel encoding')
    parser.add_argument('--output-path', required=True, help='Output enhanced wav path')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    args = parser.parse_args()

    hp = HParam(args.config)
    device = build_device(args.device)
    audio = Audio(hp)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = VoiceFilter(hp).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    adapter = EmbeddingAdapter(hp.embedder.emb_dim, hp.model.adapter_hidden_dim).to(device) if hp.model.use_embedding_adapter else None
    if adapter is not None and checkpoint.get('adapter') is not None:
        adapter.load_state_dict(checkpoint['adapter'])
        adapter.eval()

    embedding = load_vowel_embedding(hp, device, args.vowel_dir, embedder_path=args.embedder_path)
    if adapter is not None:
        with torch.no_grad():
            embedding = adapter(embedding)

    mixed_wav, _ = load_wav(args.mixed_file, sample_rate=hp.audio.sample_rate, mono=True)
    mixed_wav = peak_normalize(mixed_wav)

    window_length = int(round(hp.data.inference_window_seconds * hp.audio.sample_rate))
    hop_length = int(round(hp.data.inference_hop_seconds * hp.audio.sample_rate))
    starts = list(range(0, max(len(mixed_wav) - window_length, 0) + 1, hop_length))
    if not starts or starts[-1] + window_length < len(mixed_wav):
        starts.append(max(0, len(mixed_wav) - window_length))

    windows = []
    with torch.no_grad():
        for start in starts:
            chunk = pad_or_trim_wav(mixed_wav[start:start + window_length], window_length)
            mag, phase = audio.wav2spec(chunk)
            mag_tensor = torch.from_numpy(mag).float().unsqueeze(0).to(device)
            mask = model(mag_tensor, embedding)
            enhanced_mag = (mag_tensor * mask)[0].cpu().numpy()
            enhanced_wav = audio.spec2wav(enhanced_mag, phase)
            enhanced_wav = pad_or_trim_wav(enhanced_wav, window_length)
            windows.append((start, enhanced_wav))

    enhanced = overlap_add(windows, len(mixed_wav), window_length, hop_length)
    enhanced = peak_normalize(enhanced)
    ensure_dir(os.path.dirname(args.output_path))
    save_wav(args.output_path, enhanced, hp.audio.sample_rate)
    print('Saved enhanced wav to %s' % os.path.abspath(args.output_path))


if __name__ == '__main__':
    main()
