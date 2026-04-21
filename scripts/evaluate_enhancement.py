import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from tqdm import tqdm

from model.embedding_adapter import EmbeddingAdapter
from model.model import VoiceFilter
from model.vowel_encoder import VowelEmbeddingEncoder
from utils.audio import Audio, load_wav, pad_or_trim_wav, peak_normalize, repeat_pad_wav, save_wav
from utils.dataset_index import VOWEL_FILES, ensure_dir, load_jsonl
from utils.embedder_checkpoint import DEFAULT_EMBEDDER_PATH, resolve_embedder_path
from utils.hparams import HParam
from utils.metrics import sdr, si_sdr, snr_improvement


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def overlap_add(windows, total_length, window_length):
    output = np.zeros(total_length, dtype=np.float32)
    weights = np.zeros(total_length, dtype=np.float32)
    window = np.hanning(window_length).astype(np.float32)
    if np.max(window) <= 0:
        window = np.ones(window_length, dtype=np.float32)

    for start, wav in windows:
        end = min(total_length, start + window_length)
        valid_length = end - start
        output[start:end] += wav[:valid_length] * window[:valid_length]
        weights[start:end] += window[:valid_length]

    return output / np.maximum(weights, 1e-6)


def load_subject_embedding(item, hp, device, embedder_path=None):
    if os.path.isfile(item['embedding_path']):
        return torch.from_numpy(np.load(item['embedding_path']).astype(np.float32)).unsqueeze(0).to(device)

    encoder = VowelEmbeddingEncoder(hp).to(device)
    encoder.load_embedder(embedder_path)
    encoder.eval()
    audio = Audio(hp)

    vowel_mels = []
    for wav_path in item['vowel_paths']:
        wav, _ = load_wav(wav_path, sample_rate=hp.audio.sample_rate, mono=True)
        wav = peak_normalize(repeat_pad_wav(wav, hp.audio.sample_rate, 1.0))
        vowel_mels.append(torch.from_numpy(audio.get_mel(wav)).float().to(device))

    with torch.no_grad():
        embedding = encoder(vowel_mels)
    return embedding.unsqueeze(0)


def enhance_wav(model, adapter, audio, hp, device, mixed_wav, embedding):
    mixed_wav = peak_normalize(mixed_wav)
    window_length = int(round(hp.data.inference_window_seconds * hp.audio.sample_rate))
    hop_length = int(round(hp.data.inference_hop_seconds * hp.audio.sample_rate))
    starts = list(range(0, max(len(mixed_wav) - window_length, 0) + 1, hop_length))
    if not starts or starts[-1] + window_length < len(mixed_wav):
        starts.append(max(0, len(mixed_wav) - window_length))

    windows = []
    with torch.no_grad():
        conditioned_embedding = adapter(embedding) if adapter is not None else embedding
        for start in starts:
            chunk = pad_or_trim_wav(mixed_wav[start:start + window_length], window_length)
            mag, phase = audio.wav2spec(chunk)
            mag_tensor = torch.from_numpy(mag).float().unsqueeze(0).to(device)
            mask = model(mag_tensor, conditioned_embedding)
            enhanced_mag = (mag_tensor * mask)[0].cpu().numpy()
            enhanced_chunk = audio.spec2wav(enhanced_mag, phase)
            windows.append((start, pad_or_trim_wav(enhanced_chunk, window_length)))
    return peak_normalize(overlap_add(windows, len(mixed_wav), window_length))


def spectral_l1(audio, clean_wav, enhanced_wav):
    clean_mag, _ = audio.wav2spec(clean_wav)
    enhanced_mag, _ = audio.wav2spec(enhanced_wav)
    length = min(clean_mag.shape[0], enhanced_mag.shape[0])
    return float(np.mean(np.abs(clean_mag[:length] - enhanced_mag[:length])))


def main():
    parser = argparse.ArgumentParser(description='Evaluate enhanced outputs on a manifest and write CSV metrics')
    parser.add_argument('-c', '--config', default=os.path.join('config', 'enhancement.yaml'), help='YAML config path')
    parser.add_argument('--checkpoint-path', required=True, help='Trained enhancement checkpoint')
    parser.add_argument('--manifest', default=None, help='Manifest path, defaults to config test manifest')
    parser.add_argument('--embedder-path', default=None, help='Embedder checkpoint used when embeddings must be computed online, defaults to %s' % DEFAULT_EMBEDDER_PATH)
    parser.add_argument('--output-csv', default=os.path.join('outputs', 'eval', 'metrics.csv'), help='CSV output path')
    parser.add_argument('--save-wavs-dir', default=os.path.join('outputs', 'eval', 'enhanced_wavs'), help='Optional directory for enhanced wavs')
    parser.add_argument('--device', default='auto', help='cpu, cuda, or auto')
    args = parser.parse_args()

    hp = HParam(args.config)
    manifest_path = args.manifest or hp.data.manifest_test
    device = build_device(args.device)
    embedder_path = resolve_embedder_path(args.embedder_path)
    audio = Audio(hp)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = VoiceFilter(hp).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    adapter = EmbeddingAdapter(hp.embedder.emb_dim, hp.model.adapter_hidden_dim).to(device) if hp.model.use_embedding_adapter else None
    if adapter is not None and checkpoint.get('adapter') is not None:
        adapter.load_state_dict(checkpoint['adapter'])
        adapter.eval()

    items = load_jsonl(manifest_path)
    ensure_dir(os.path.dirname(args.output_csv))
    ensure_dir(args.save_wavs_dir)

    rows = []
    for item in tqdm(items, desc='evaluate'):
        mixed_wav, _ = load_wav(item['mix_path'], sample_rate=hp.audio.sample_rate, mono=True)
        clean_wav, _ = load_wav(item['clean_path'], sample_rate=hp.audio.sample_rate, mono=True)
        clean_wav = peak_normalize(clean_wav)

        pair_length = min(len(mixed_wav), len(clean_wav))
        mixed_wav = mixed_wav[:pair_length]
        clean_wav = clean_wav[:pair_length]

        embedding = load_subject_embedding(item, hp, device, embedder_path=embedder_path)
        enhanced_wav = enhance_wav(model, adapter, audio, hp, device, mixed_wav, embedding)
        enhanced_wav = pad_or_trim_wav(enhanced_wav, pair_length)

        mix_name = os.path.splitext(os.path.basename(item['mix_path']))[0]
        wav_out_path = os.path.join(args.save_wavs_dir, '%s_%s.wav' % (item['subject_id'], mix_name))
        save_wav(wav_out_path, enhanced_wav, hp.audio.sample_rate)

        rows.append({
            'subject_id': item['subject_id'],
            'noise_type': item['noise_type'],
            'snore_index': item['snore_index'],
            'mix_path': item['mix_path'],
            'clean_path': item['clean_path'],
            'enhanced_path': wav_out_path,
            'sdr': sdr(clean_wav, enhanced_wav),
            'si_sdr': si_sdr(clean_wav, enhanced_wav),
            'snr_improvement': snr_improvement(clean_wav, mixed_wav, enhanced_wav),
            'mag_l1': spectral_l1(audio, clean_wav, enhanced_wav),
        })

    fieldnames = [
        'subject_id', 'noise_type', 'snore_index',
        'mix_path', 'clean_path', 'enhanced_path',
        'sdr', 'si_sdr', 'snr_improvement', 'mag_l1',
    ]
    with open(args.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        avg_sdr = np.mean([row['sdr'] for row in rows])
        avg_si_sdr = np.mean([row['si_sdr'] for row in rows])
        avg_snr_imp = np.mean([row['snr_improvement'] for row in rows])
        avg_mag_l1 = np.mean([row['mag_l1'] for row in rows])
        print('Saved %d evaluation rows to %s' % (len(rows), os.path.abspath(args.output_csv)))
        print('avg_sdr=%.4f avg_si_sdr=%.4f avg_snr_improvement=%.4f avg_mag_l1=%.6f' % (
            avg_sdr, avg_si_sdr, avg_snr_imp, avg_mag_l1
        ))


if __name__ == '__main__':
    main()
