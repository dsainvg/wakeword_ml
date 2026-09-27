"""
Physics-Based Acoustic Augmentation Engine for Robust Keyword Spotting
----------------------------------------------------------------------
Simulates physical environmental and transducer phenomena:
1. Room Impulse Response (RIR) Reverberation:
   - Schroeder exponentially decaying reverberation with frequency-dependent absorption
   - Simulates diverse room sizes and distances (0.5m to 5.0m, T60 from 0.15s to 0.75s)
2. Atmospheric Air Absorption:
   - High-frequency damping due to acoustic propagation through air
3. Microphone Transducer Emulation:
   - Low-cut high-pass filter (<120 Hz) modeling MEMS microphone roll-off
   - Formant resonance peak (2.2 kHz - 3.2 kHz) modeling acoustic cavity resonance
   - Soft tanh harmonic saturation modeling analog pre-amp distortion
4. Speed & Pitch Perturbation:
   - Speed changes (0.85x to 1.15x) using linear interpolation resampling
   - Pitch shifting without altering clip duration
5. SpecAugment (Time & Frequency Masking):
   - Frequency channel masking (zeroing bands)
   - Time frame masking (zeroing frames)
6. INMP441 I2S MEMS microphone (see INMP441_SPEC):
   - Absolute electronic self-noise floor derived from sensitivity and SNR
   - 60 Hz - 15 kHz response, which sits entirely OUTSIDE the 100-7500 Hz feature
     band, so it is essentially transparent in-band apart from the LDC low shelf
   - 24-bit I2S quantization, which is far below the self-noise floor
"""

import numpy as np
from scipy import signal

SAMPLE_RATE = 16000

# ==============================================================================
# Target transducer: TDK InvenSense INMP441
# ==============================================================================
# Datasheet figures and what each one actually implies for this model.
INMP441_SPEC = {
    "sensitivity_dbfs": -26.0,   # at 94 dB SPL (1 Pa)
    "snr_dba": 61.0,             # A-weighted
    "f_low_hz": 60.0,
    "f_high_hz": 15000.0,
    "ldc_shelf_db": 3.0,         # Low-Noise-Compensation gain, ~+6 dB at 100 Hz
    "unit_spread_db": 5.0,       # unit-to-unit sensitivity spread
    "full_scale_spl_db": 94.0,
    "i2s_bits": 24,
}

# Electronic noise floor at the capsule, referenced to full scale: sensitivity - SNR.
# This is the number before any board preamp gain is applied.
INMP441_NOISE_FLOOR_DBFS = INMP441_SPEC["sensitivity_dbfs"] - INMP441_SPEC["snr_dba"]

# Feature front-end band (features.py): f_min=100 Hz, f_max=7500 Hz.
FEATURE_F_MIN_HZ = 100.0
FEATURE_F_MAX_HZ = 7500.0

# Conversational speech at various distances in free field (inverse-square from 1 m,
# ~60 dB SPL at 1 m for a talker at normal volume), plus a vocal-effort offset.
SPEAKING_SPL_DB = {0.3: 70.0, 0.5: 66.0, 1.0: 60.0, 2.0: 54.0, 3.0: 50.5}
SPEAKING_DISTANCES_M = tuple(SPEAKING_SPL_DB)
SPEAKING_SPL_REF_DB = 60.0          # the 1 m reference
VOCAL_EFFORT_DB = (-15.0, 12.0)    # whisper .. shouting


def spl_to_dbfs(spl_db: float) -> float:
    """Raw sensitivity mapping: 94 dB SPL (1 Pa) -> -26 dBFS, with no board gain.

    Note the consequence: a talker at 60 dB SPL lands near -60 dBFS RMS. In practice a
    board applies preamp/software gain to lift that into a usable range, so the absolute
    digital level depends on the design while the *ratio* between speech and the capsule
    noise floor does not. That is why the level is treated as a free variable below and
    the SNR is derived from distance instead.
    """
    return spl_db - INMP441_SPEC["full_scale_spl_db"] + INMP441_SPEC["sensitivity_dbfs"]


def effective_snr_db(spl_db: float) -> float:
    """Mic SNR as the deployed system sees it.

    The datasheet's 61 dBA is quoted at maximum sensitivity (i.e. for a sound pressure
    at the top of the useful range). Once the speech level drops -- because the user is
    3 m away, or whispering -- the capsule noise stays where it is and the achievable
    SNR falls with it. This is the effect that actually matters and it is invisible if
    the noise floor is modelled as a fixed offset.
    """
    return INMP441_SPEC["snr_dba"] + (spl_db - SPEAKING_SPL_REF_DB)


def draw_talker_conditions(rng) -> tuple:
    """Draws a talker distance, vocal effort and the resulting deployed conditions.

    Returns (level_dbfs, snr_db, spl_db): where the board's gain puts the speech in the
    digital domain, and the SNR the capsule delivers at that distance and loudness.
    """
    d = float(rng.choice(SPEAKING_DISTANCES_M))
    effort = float(rng.uniform(*VOCAL_EFFORT_DB))
    spl = SPEAKING_SPL_DB[d] + effort
    return float(rng.uniform(-50.0, -14.0)), effective_snr_db(spl), spl


def apply_inmp441_response(audio: np.ndarray) -> np.ndarray:
    """INMP441 acoustic response as seen inside the 100-7500 Hz feature band.

    The capsule is specified flat from 60 Hz to 15 kHz. Our feature band sits wholly
    inside that, so there is NO meaningful roll-off to apply: the only in-band effects
    are the high-pass corner just below f_min and the LDC low-frequency shelf, which
    lifts roughly 100-200 Hz by a few dB. Applying a generic MEMS darkening curve here
    would misrepresent the hardware.
    """
    a = np.asarray(audio, dtype=np.float32)
    # 2nd-order high-pass at the 60 Hz capsule corner (well below the 100 Hz f_min)
    b, c = signal.butter(2, INMP441_SPEC["f_low_hz"] / (SAMPLE_RATE / 2.0), btype="high")
    a = signal.lfilter(b, c, a).astype(np.float32)
    # LDC low-frequency lift. Needs a 2nd-order lowpass here: a 1-pole roll-off is only
    # -0.7 dB by 4 kHz, which turns the "shelf" into a flat gain boost across the whole
    # feature band instead of a shelf.
    b2, c2 = signal.butter(2, 250.0 / (SAMPLE_RATE / 2.0), btype="low")
    low = signal.lfilter(b2, c2, a).astype(np.float32)
    g = 10 ** (INMP441_SPEC["ldc_shelf_db"] / 20.0)
    return (a + (g - 1.0) * low).astype(np.float32)


def add_mic_self_noise(audio: np.ndarray, rng, snr_db: float = 61.0,
                       extra_db: float = 0.0) -> np.ndarray:
    """Adds the capsule's own electronic noise at a given SNR below the signal.

    Modelled as white noise with a mild 1/f tilt (real capsule self-noise is not white)
    plus per-unit sensitivity spread. The noise is placed relative to the signal level,
    not at a fixed absolute level, because the board's preamp lifts speech and self-noise
    together -- what degrades the microphone is distance and loudness, which is what
    `snr_db` already encodes.
    """
    a = np.asarray(audio, dtype=np.float32)
    n = len(a)
    sig_rms = float(np.sqrt(np.mean(a ** 2)))
    if sig_rms < 1e-9:
        sig_rms = 1e-9
    sig_dbfs = 20.0 * float(np.log10(sig_rms))
    floor_dbfs = (sig_dbfs - snr_db
                  + float(rng.normal(0.0, INMP441_SPEC["unit_spread_db"] / 3.0))
                  + extra_db)
    rms = float(10 ** (floor_dbfs / 20.0))
    noise = rng.standard_normal(n).astype(np.float32)
    # 1/f tilt: one-pole lowpass blend, keeps the floor from being pure white
    b, c = signal.butter(1, min(0.99, 2000.0 / (SAMPLE_RATE / 2.0)), btype="low")
    noise = 0.7 * signal.lfilter(b, c, noise) + 0.3 * noise
    cur = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    noise *= rms / cur
    return (a + noise).astype(np.float32)


def apply_i2s_quantization(audio: np.ndarray, bits: int = 24) -> np.ndarray:
    """I2S word quantization. At 24 bits the floor is about -146 dBFS, i.e. roughly
    59 dB below the capsule self-noise, so it is a no-op in practice. Kept because a
    deployment may read the 24-bit word through a 16-bit I2S peripheral, which throws
    away the low byte."""
    q = 2 ** bits / 2.0 - 0.5
    return (np.round((np.clip(audio, -1.0, 1.0) + 1.0) * q) / q - 1.0).astype(np.float32)


def apply_i2s_16bit_truncation(audio: np.ndarray) -> np.ndarray:
    """24-bit I2S word read by a 16-bit peripheral: the low byte is dropped, so the
    effective depth is 16 bits (floor about -98 dBFS, still under the capsule)."""
    return apply_i2s_quantization(audio, bits=16)



def generate_synthetic_rir(t60_sec: float = 0.35, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generates a synthetic Room Impulse Response (RIR) using exponentially
    decaying filtered noise with direct-path impulse and early reflections.
    """
    length = int(t60_sec * sr)
    if length < 100:
        length = 100

    # Time vector
    t = np.arange(length) / sr

    # Decay rate: -60 dB over t60 seconds -> decay envelope = exp(-6.91 * t / t60)
    decay = np.exp(-6.91 * t / t60_sec)

    # Gaussian noise for diffuse tail
    tail = np.random.randn(length) * decay

    # Low-pass filter the tail to simulate frequency-dependent wall absorption
    # (high frequencies decay faster in real rooms)
    b, a = signal.butter(1, 4000.0 / (sr / 2.0), btype='low')
    tail_filtered = signal.lfilter(b, a, tail)

    # Direct sound and early discrete reflections (image-source approximation)
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0  # Direct path impulse

    # Add 4-6 early reflections with random delays and attenuation
    num_early = np.random.randint(4, 8)
    for _ in range(num_early):
        delay_ms = np.random.uniform(5.0, 45.0)
        delay_idx = int(delay_ms * 1e-3 * sr)
        if delay_idx < length:
            gain = np.random.uniform(0.15, 0.45) * (-1.0 if np.random.random() > 0.5 else 1.0)
            rir[delay_idx] += gain

    # Combine early reflections and diffuse late reverberation
    rir = rir + (tail_filtered * 0.35)

    # Normalize energy
    norm = np.sqrt(np.sum(rir ** 2)) + 1e-8
    return (rir / norm).astype(np.float32)


def apply_reverberation(audio: np.ndarray, t60_sec: float = None) -> np.ndarray:
    """Convolves audio with a synthetic Room Impulse Response."""
    if t60_sec is None:
        t60_sec = np.random.uniform(0.15, 0.65)
    rir = generate_synthetic_rir(t60_sec)
    rev = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    # Match RMS of original audio
    orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    rev_rms = np.sqrt(np.mean(rev ** 2)) + 1e-8
    return (rev * (orig_rms / rev_rms)).astype(np.float32)


def apply_mems_mic_response(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Emulates hardware MEMS microphone frequency response and mild saturation."""
    # 1. High-pass filter (cut below 120 Hz to mimic MEMS DC roll-off)
    b_hp, a_hp = signal.butter(2, 120.0 / (sr / 2.0), btype='high')
    filtered = signal.lfilter(b_hp, a_hp, audio)

    # 2. Resonant band boost around 2.5 kHz (mic cavity acoustic resonance)
    center_freq = np.random.uniform(2200.0, 3200.0)
    q = 1.5
    b_peak, a_peak = signal.iirpeak(center_freq / (sr / 2.0), q)
    boosted = signal.lfilter(b_peak, a_peak, filtered)
    mixed = filtered * 0.75 + boosted * 0.25

    # 3. Non-linear analog saturation (soft clipping)
    saturated = np.tanh(mixed * 1.1) * 0.95
    return saturated.astype(np.float32)


def apply_distance_attenuation(audio: np.ndarray, distance_m: float = None, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulates speech spoken at distance (0.5m to 5.0m) with high-frequency air loss."""
    if distance_m is None:
        distance_m = np.random.uniform(0.5, 4.5)

    # Air absorption dampens frequencies above 3.5 kHz progressively with distance
    cutoff = max(2000.0, 7500.0 - (distance_m * 900.0))
    b, a = signal.butter(1, cutoff / (sr / 2.0), btype='low')
    damped = signal.lfilter(b, a, audio)

    # Inverse square distance gain attenuation (normalized with dynamic range limit)
    gain = 1.0 / (1.0 + (distance_m - 0.5) * 0.3)
    return (damped * gain).astype(np.float32)


def apply_speed_perturbation(audio: np.ndarray, speed_factor: float = None) -> np.ndarray:
    """Changes speed of audio (0.85x to 1.15x) while maintaining 1.0s target length."""
    if speed_factor is None:
        speed_factor = np.random.uniform(0.85, 1.15)
    if abs(speed_factor - 1.0) < 0.02:
        return audio

    n = len(audio)
    resampled_len = int(round(n / speed_factor))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, resampled_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)

    # Fit back into original length with center padding / trimming
    if len(resampled) < n:
        pad_l = (n - len(resampled)) // 2
        pad_r = n - len(resampled) - pad_l
        resampled = np.pad(resampled, (pad_l, pad_r), mode='constant')
    else:
        start = (len(resampled) - n) // 2
        resampled = resampled[start:start + n]

    return resampled


def apply_spec_augment(spectrogram: np.ndarray, max_freq_mask: int = 6, max_time_mask: int = 8) -> np.ndarray:
    """
    Applies SpecAugment directly on a (49, 40) Log-Mel spectrogram:
    - Frequency channel masking
    - Time frame masking
    """
    spec = spectrogram.copy()
    num_time, num_mel = spec.shape

    # 1. Frequency masking
    f_width = np.random.randint(1, max_freq_mask + 1)
    f_start = np.random.randint(0, num_mel - f_width) if (num_mel - f_width > 0) else 0
    spec[:, f_start:f_start + f_width] = np.min(spec)

    # 2. Time masking
    t_width = np.random.randint(1, max_time_mask + 1)
    t_start = np.random.randint(0, num_time - t_width) if (num_time - t_width > 0) else 0
    spec[t_start:t_start + t_width, :] = np.min(spec)

    return spec


def apply_full_acoustic_chain(audio: np.ndarray, noise_slice: np.ndarray = None, snr_db: float = None) -> np.ndarray:
    """
    Applies the full physical acoustic propagation chain:
    Original -> Speed Jitter -> Room Reverberation -> Distance Loss -> Mic Emulation -> Noise Mix
    """
    # 1. Speed perturbation (80% probability)
    if np.random.random() < 0.80:
        audio = apply_speed_perturbation(audio)

    # 2. Room reverberation (70% probability)
    if np.random.random() < 0.70:
        audio = apply_reverberation(audio)

    # 3. Distance decay & air absorption (70% probability)
    if np.random.random() < 0.70:
        audio = apply_distance_attenuation(audio)

    # 4. Hardware MEMS mic filtering & saturation (90% probability)
    if np.random.random() < 0.90:
        audio = apply_mems_mic_response(audio)

    # 5. Background noise mixing if provided
    if noise_slice is not None and len(noise_slice) == len(audio):
        if snr_db is None:
            snr_db = np.random.uniform(-4.0, 22.0)
        sig_pow = np.mean(audio ** 2) + 1e-8
        noise_pow = np.mean(noise_slice ** 2) + 1e-8
        target_noise_pow = sig_pow / (10 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_pow / noise_pow)
        audio = audio + (noise_slice * scale)

    # Peak normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio / max_val * 0.95

    return audio.astype(np.float32)


def mix_dual_independent_noises(speech: np.ndarray, noise1: np.ndarray, noise2: np.ndarray, snr1_db: float, snr2_db: float) -> np.ndarray:
    """Cocktail party mixture: speech + stationary ambient noise1 + transient noise2."""
    sig_pow = np.mean(speech ** 2) + 1e-8

    p1 = np.mean(noise1 ** 2) + 1e-8
    target_p1 = sig_pow / (10 ** (snr1_db / 10.0))
    scale1 = np.sqrt(target_p1 / p1)

    p2 = np.mean(noise2 ** 2) + 1e-8
    target_p2 = sig_pow / (10 ** (snr2_db / 10.0))
    scale2 = np.sqrt(target_p2 / p2)

    mixed = speech + (noise1 * scale1) + (noise2 * scale2)
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    return mixed.astype(np.float32)
