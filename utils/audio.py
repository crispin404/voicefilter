# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np
import soundfile as sf


def load_wav(path, sample_rate=None, mono=True):
    wav, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return wav.astype(np.float32), sr


def to_mono(wav):
    if wav.ndim == 1:
        return wav.astype(np.float32)
    return librosa.to_mono(wav).astype(np.float32)


def peak_normalize(wav, peak=0.95):
    wav = wav.astype(np.float32)
    max_abs = np.max(np.abs(wav)) if wav.size > 0 else 0.0
    if max_abs < 1e-8:
        return wav
    return wav / max_abs * peak


def repeat_pad_wav(wav, sample_rate, target_seconds):
    target_length = int(round(sample_rate * target_seconds))
    if target_length <= 0:
        return wav.astype(np.float32)
    if wav.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    repeats = int(np.ceil(float(target_length) / float(wav.size)))
    padded = np.tile(wav, repeats)[:target_length]
    return padded.astype(np.float32)


def pad_or_trim_wav(wav, target_length):
    wav = wav.astype(np.float32)
    if target_length <= 0:
        return wav
    if wav.size >= target_length:
        return wav[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:wav.size] = wav
    return padded


def compute_rms(wav, eps=1e-12):
    wav = wav.astype(np.float32)
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav), dtype=np.float64) + eps))


def compute_snr_db(clean_wav, noise_wav, eps=1e-12):
    clean_rms = compute_rms(clean_wav, eps=eps)
    noise_rms = compute_rms(noise_wav, eps=eps)
    if clean_rms <= eps or noise_rms <= eps:
        return None
    return float(20.0 * np.log10(clean_rms / noise_rms))


def match_length_with_random_crop(wav, target_length, rng):
    wav = wav.astype(np.float32)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if wav.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if wav.size < target_length:
        repeats = int(np.ceil(float(target_length) / float(wav.size)))
        wav = np.tile(wav, repeats)
    if wav.size == target_length:
        return wav.astype(np.float32)

    max_start = wav.size - target_length
    start = rng.randint(0, max_start) if max_start > 0 else 0
    end = start + target_length
    return wav[start:end].astype(np.float32)


def scale_noise_to_target_snr(clean_wav, noise_wav, target_snr_db, eps=1e-12):
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = noise_wav.astype(np.float32)
    clean_rms = compute_rms(clean_wav, eps=eps)
    noise_rms = compute_rms(noise_wav, eps=eps)
    if clean_rms <= eps:
        raise ValueError('Clean waveform RMS is too small for SNR scaling.')
    if noise_rms <= eps:
        raise ValueError('Noise waveform RMS is too small for SNR scaling.')

    target_noise_rms = clean_rms / (10.0 ** (target_snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    scaled_noise = noise_wav * np.float32(scale)
    actual_snr_db = compute_snr_db(clean_wav, scaled_noise, eps=eps)
    return scaled_noise.astype(np.float32), float(scale), actual_snr_db


def paired_peak_normalize(clean_wav, mix_wav, peak=0.95, eps=1e-8):
    clean_wav = clean_wav.astype(np.float32)
    mix_wav = mix_wav.astype(np.float32)
    if clean_wav.size == 0 and mix_wav.size == 0:
        return clean_wav, mix_wav, 1.0

    peak_value = max(
        float(np.max(np.abs(clean_wav))) if clean_wav.size > 0 else 0.0,
        float(np.max(np.abs(mix_wav))) if mix_wav.size > 0 else 0.0,
    )
    if peak_value < eps:
        return clean_wav, mix_wav, 1.0

    scale = peak / peak_value
    return clean_wav * np.float32(scale), mix_wav * np.float32(scale), float(scale)


def save_wav(path, wav, sample_rate, subtype=None):
    kwargs = {}
    if subtype is not None:
        kwargs['subtype'] = subtype
    sf.write(path, wav.astype(np.float32), sample_rate, **kwargs)


class Audio():
    def __init__(self, hp):
        self.hp = hp
        self.mel_basis = librosa.filters.mel(sr=hp.audio.sample_rate,
                                             n_fft=hp.embedder.n_fft,
                                             n_mels=hp.embedder.num_mels)

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=self.hp.embedder.n_fft,
                              hop_length=self.hp.audio.hop_length,
                              win_length=self.hp.audio.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j*phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
