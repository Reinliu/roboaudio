# noise_reduction_librosa.py
import numpy as np
import librosa

def spectral_gating_denoise(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    noise_percentile: float = 20.0,
    n_std_thresh: float = 1.5,
    prop_decrease: float = 1.0
):

    # STFT
    stft = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    mag, phase = np.abs(stft), np.exp(1j * np.angle(stft))

    # Frame-wise energy to pick noise frames
    frame_energy = np.mean(mag, axis=0)
    thresh_energy = np.percentile(frame_energy, noise_percentile)
    noise_frames = mag[:, frame_energy <= thresh_energy]

    if noise_frames.shape[1] < 1:
        # fallback: nothing looks like noise
        return y

    # Noise profile per frequency bin
    noise_mean = np.mean(noise_frames, axis=1, keepdims=True)
    noise_std = np.std(noise_frames, axis=1, keepdims=True)

    # Frequency-dependent threshold
    noise_thresh = noise_mean + n_std_thresh * noise_std

    # Mask: keep bins above threshold
    mask = mag >= noise_thresh
    mask = mask.astype(np.float32)

    # Soft gating
    gated_mag = mag * (mask + (1 - mask) * (1 - prop_decrease))

    # Reconstruct
    stft_clean = gated_mag * phase
    y_clean = librosa.istft(stft_clean, hop_length=hop_length, win_length=win_length)

    return y_clean.astype(np.float32)


def denoise_mono_wrapper(mono, sr):

    return spectral_gating_denoise(
        mono, sr,
        n_fft=1024, hop_length=256, win_length=1024,
        noise_percentile=20.0,
        n_std_thresh=1.5,
        prop_decrease=0.9
    )
