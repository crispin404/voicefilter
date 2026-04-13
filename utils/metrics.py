import numpy as np
from mir_eval.separation import bss_eval_sources


def safe_metric(value):
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def sdr(clean_wav, enhanced_wav):
    score = bss_eval_sources(
        np.expand_dims(clean_wav, axis=0),
        np.expand_dims(enhanced_wav, axis=0),
        False,
    )[0][0]
    return safe_metric(score)


def si_sdr(clean_wav, enhanced_wav):
    clean_wav = clean_wav.astype(np.float64)
    enhanced_wav = enhanced_wav.astype(np.float64)
    target = np.dot(enhanced_wav, clean_wav) / (np.dot(clean_wav, clean_wav) + 1e-8) * clean_wav
    noise = enhanced_wav - target
    ratio = (np.sum(target ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return safe_metric(10.0 * np.log10(ratio))


def snr(clean_wav, test_wav):
    noise = clean_wav - test_wav
    ratio = (np.sum(clean_wav ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return safe_metric(10.0 * np.log10(ratio))


def snr_improvement(clean_wav, mixed_wav, enhanced_wav):
    return safe_metric(snr(clean_wav, enhanced_wav) - snr(clean_wav, mixed_wav))
