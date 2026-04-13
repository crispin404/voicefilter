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


def save_wav(path, wav, sample_rate):
    sf.write(path, wav.astype(np.float32), sample_rate)


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
