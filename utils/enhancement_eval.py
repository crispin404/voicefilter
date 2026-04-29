import os

import numpy as np
import torch

from model.vowel_encoder import VowelEmbeddingEncoder
from utils.audio import Audio, compute_rms, load_wav, pad_or_trim_wav, peak_normalize, repeat_pad_wav
from utils.dataset_index import VOWEL_KEYS, get_vowel_embedding_keys, normalize_vowel_embedding_mode
from utils.dvector import use_d_vector, zero_embedding
from utils.metrics import si_sdr, snr


def get_data_value(hp, name, default):
    return hp.data.get(name, default)


def build_window_starts(total_length, window_length, hop_length):
    starts = list(range(0, max(total_length - window_length, 0) + 1, hop_length))
    if not starts or starts[-1] + window_length < total_length:
        starts.append(max(0, total_length - window_length))
    return starts


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
    if not use_d_vector(hp):
        return zero_embedding(1, hp, device=device)

    if os.path.isfile(item['embedding_path']):
        return torch.from_numpy(np.load(item['embedding_path']).astype(np.float32)).unsqueeze(0).to(device)

    encoder = VowelEmbeddingEncoder(hp).to(device)
    encoder.load_embedder(embedder_path)
    encoder.eval()
    audio = Audio(hp)

    vowel_mels = []
    vowel_embedding_mode = normalize_vowel_embedding_mode(get_data_value(hp, 'vowel_embedding_mode', 'avg'))
    selected_vowel_keys = get_vowel_embedding_keys(vowel_embedding_mode)
    vowel_path_map = {}
    for index, key in enumerate(VOWEL_KEYS):
        if index < len(item['vowel_paths']):
            vowel_path_map[key] = item['vowel_paths'][index]
    missing = [key for key in selected_vowel_keys if not vowel_path_map.get(key)]
    if missing:
        raise FileNotFoundError(
            'Missing required vowel path(s) for subject=%s mode=%s: %s'
            % (item.get('subject_id', 'unknown'), vowel_embedding_mode, ', '.join(missing))
        )

    repeat_seconds = float(get_data_value(hp, 'vowel_seconds', 1.0))
    for key in selected_vowel_keys:
        wav_path = vowel_path_map[key]
        wav, _ = load_wav(wav_path, sample_rate=hp.audio.sample_rate, mono=True)
        wav = peak_normalize(repeat_pad_wav(wav, hp.audio.sample_rate, repeat_seconds))
        vowel_mels.append(torch.from_numpy(audio.get_mel(wav)).float().to(device))

    with torch.no_grad():
        embedding = encoder(vowel_mels)
    return embedding.unsqueeze(0)


def enhance_wav(model, adapter, audio, hp, device, mixed_wav, embedding):
    window_length = int(round(hp.data.inference_window_seconds * hp.audio.sample_rate))
    hop_length = int(round(hp.data.inference_hop_seconds * hp.audio.sample_rate))
    starts = build_window_starts(len(mixed_wav), window_length, hop_length)

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
    return overlap_add(windows, len(mixed_wav), window_length)


def clean_activity_stats(clean_wav, hp):
    window_length = int(round(hp.data.inference_window_seconds * hp.audio.sample_rate))
    hop_length = int(round(hp.data.inference_hop_seconds * hp.audio.sample_rate))
    starts = build_window_starts(len(clean_wav), window_length, hop_length)
    full_clean_rms = compute_rms(clean_wav)
    min_rms = float(get_data_value(hp, 'active_crop_min_rms', 1e-4))
    relative_rms = float(get_data_value(hp, 'active_crop_relative_rms', 0.5))
    active_threshold = max(min_rms, full_clean_rms * relative_rms)

    active_windows = 0
    for start in starts:
        chunk = pad_or_trim_wav(clean_wav[start:start + window_length], window_length)
        if compute_rms(chunk) >= active_threshold:
            active_windows += 1

    total_windows = len(starts)
    active_ratio = float(active_windows) / float(total_windows) if total_windows > 0 else 0.0
    return active_ratio, active_windows, total_windows


def spectral_l1(audio, clean_wav, enhanced_wav):
    clean_mag, _ = audio.wav2spec(clean_wav)
    enhanced_mag, _ = audio.wav2spec(enhanced_wav)
    length = min(clean_mag.shape[0], enhanced_mag.shape[0])
    return float(np.mean(np.abs(clean_mag[:length] - enhanced_mag[:length])))


def evaluate_item(item, model, adapter, audio, hp, device, embedder_path=None):
    mixed_wav, _ = load_wav(item['mix_path'], sample_rate=hp.audio.sample_rate, mono=True)
    clean_wav, _ = load_wav(item['clean_path'], sample_rate=hp.audio.sample_rate, mono=True)

    pair_length = min(len(mixed_wav), len(clean_wav))
    mixed_wav = mixed_wav[:pair_length]
    clean_wav = clean_wav[:pair_length]

    embedding = load_subject_embedding(item, hp, device, embedder_path=embedder_path)
    enhanced_wav = enhance_wav(model, adapter, audio, hp, device, mixed_wav, embedding)
    enhanced_wav = pad_or_trim_wav(enhanced_wav, pair_length)

    input_snr = snr(clean_wav, mixed_wav)
    input_si_sdr = si_sdr(clean_wav, mixed_wav)
    enhanced_snr = snr(clean_wav, enhanced_wav)
    enhanced_si_sdr = si_sdr(clean_wav, enhanced_wav)
    clean_active_ratio, clean_active_windows, clean_total_windows = clean_activity_stats(clean_wav, hp)

    return {
        'subject_id': item['subject_id'],
        'noise_type': item['noise_type'],
        'noise_count': int(item.get('noise_count', 1)),
        'snore_index': item['snore_index'],
        'mix_path': item['mix_path'],
        'clean_path': item['clean_path'],
        'enhanced_path': '',
        'input_snr': input_snr,
        'snr_improvement': enhanced_snr - input_snr,
        'enhanced_snr': enhanced_snr,
        'input_si_sdr': input_si_sdr,
        'si_sdr': enhanced_si_sdr,
        'si_sdr_improvement': enhanced_si_sdr - input_si_sdr,
        'mag_l1': spectral_l1(audio, clean_wav, enhanced_wav),
        'clean_active_ratio': clean_active_ratio,
        'clean_active_windows': clean_active_windows,
        'clean_total_windows': clean_total_windows,
    }, enhanced_wav


def summarize_rows(rows):
    count = len(rows)
    if count == 0:
        return {
            'count': 0,
            'avg_input_snr': 0.0,
            'avg_snr_improvement': 0.0,
            'avg_enhanced_snr': 0.0,
            'avg_input_si_sdr': 0.0,
            'avg_si_sdr': 0.0,
            'avg_si_sdr_improvement': 0.0,
            'avg_mag_l1': 0.0,
            'avg_clean_active_ratio': 0.0,
            'negative_count': 0,
            'negative_rate': 0.0,
        }

    negative_count = sum(1 for row in rows if row['si_sdr_improvement'] < 0.0)
    return {
        'count': count,
        'avg_input_snr': float(np.mean([row['input_snr'] for row in rows])),
        'avg_snr_improvement': float(np.mean([row['snr_improvement'] for row in rows])),
        'avg_enhanced_snr': float(np.mean([row['enhanced_snr'] for row in rows])),
        'avg_input_si_sdr': float(np.mean([row['input_si_sdr'] for row in rows])),
        'avg_si_sdr': float(np.mean([row['si_sdr'] for row in rows])),
        'avg_si_sdr_improvement': float(np.mean([row['si_sdr_improvement'] for row in rows])),
        'avg_mag_l1': float(np.mean([row['mag_l1'] for row in rows])),
        'avg_clean_active_ratio': float(np.mean([row['clean_active_ratio'] for row in rows])),
        'negative_count': negative_count,
        'negative_rate': float(negative_count) / float(count),
    }


def format_summary(label, summary):
    return (
        '%s_count=%d avg_input_snr=%.4f avg_snr_improvement=%.4f avg_enhanced_snr=%.4f '
        'avg_input_si_sdr=%.4f avg_si_sdr=%.4f avg_si_sdr_improvement=%.4f avg_mag_l1=%.6f '
        'avg_clean_active_ratio=%.6f negative_improvement_rate=%.6f negative_count=%d'
        % (
            label,
            summary['count'],
            summary['avg_input_snr'],
            summary['avg_snr_improvement'],
            summary['avg_enhanced_snr'],
            summary['avg_input_si_sdr'],
            summary['avg_si_sdr'],
            summary['avg_si_sdr_improvement'],
            summary['avg_mag_l1'],
            summary['avg_clean_active_ratio'],
            summary['negative_rate'],
            summary['negative_count'],
        )
    )


def print_summary(label, rows):
    print(format_summary(label, summarize_rows(rows)))
