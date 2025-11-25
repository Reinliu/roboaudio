# main.py (pure Python version with noise reduction + diarisation)

import argparse
import json
import numpy as np

from audio_io import WavStream
from doa import DoAEstimator
from asr import WhisperASR
from utils import normalize_command, direction_is_valid

from noise_reduction import denoise_mono_wrapper
from diarization import run_diarization


def run_processor(wav_path, frame_len=16000, hop_len=8000,
                  min_conf=0.5, doa_algo="SRP", whisper_model="base"):

    # 1. Audio stream
    stream = WavStream(
        wav_path=wav_path,
        frame_len=frame_len,
        hop_len=hop_len,
        realtime=True
    )

    # 2. 4-mic circular array geometry
    radius = 0.05
    angles = np.deg2rad([0, 90, 180, 270])
    mic_positions = np.vstack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros_like(angles)
    ])  # (3, 4)

    # 3. Initialize DOA + ASR
    doa_estimator = DoAEstimator(
        mic_positions=mic_positions,
        sample_rate=stream.sr,
        nfft=512,
        doa_algo=doa_algo
    )

    asr = WhisperASR(model_size=whisper_model)

    # ---- Diarisation (offline, pyAudioAnalysis) ----
    print("Running diarisation (offline)...")
    flags, classes, mt_step = run_diarization(wav_path, n_speakers=2)
    print("Diarisation complete.")

    sample_idx = 0    # tracks position in stream

    # 4. Process frames
    for frame, sr in stream:

        # ---------------- DOA ----------------
        direction_deg = doa_estimator.estimate_deg(frame)

        # OPTIONAL: add delay-and-sum beamforming here
        # Currently: use simple mono (channel average)
        mono = np.mean(frame, axis=0).astype(np.float32)

        # ---------------- Noise Reduction ----------------
        mono_denoised = denoise_mono_wrapper(mono, sr)

        # ---------------- ASR ----------------
        # WhisperASR expects shape (M, N), so wrap 1 channel
        text, conf = asr.transcribe(mono_denoised[None, :], sr, min_conf=min_conf)

        if text is None:
            sample_idx += frame.shape[1]
            continue

        cmd = normalize_command(text)

        # ---------------- Diarisation lookup ----------------
        t_sec = sample_idx / sr
        diar_idx = int(t_sec / mt_step)
        speaker_id = (
            int(flags[diar_idx])
            if diar_idx < len(flags)
            else -1
        )

        sample_idx += frame.shape[1]

        # ---------------- Spatial Gate ----------------
        if not direction_is_valid(cmd, direction_deg):
            print(f"Ignored '{cmd}' from {direction_deg:.1f}° (invalid direction)")
            continue

        # ---------------- Output Payload ----------------
        payload = {
            "command": cmd,
            "confidence": round(conf, 3),
            "direction": round(direction_deg, 1),
            "speaker": speaker_id
        }

        print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default='../test_audio/generated_synthetic_sig_spliced.wav', type=str)
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument("--frame_len", type=int, default=16000)
    parser.add_argument("--hop_len", type=int, default=8000)
    parser.add_argument("--doa_algo", type=str, default="SRP")
    parser.add_argument("--whisper_model", type=str, default="base")
    args = parser.parse_args()

    run_processor(
        args.wav_path,
        frame_len=args.frame_len,
        hop_len=args.hop_len,
        min_conf=args.min_conf,
        doa_algo=args.doa_algo,
        whisper_model=args.whisper_model
    )
