# main.py (pure Python version)

import argparse
import json
import numpy as np
from audio_io import WavStream
from doa import DoAEstimator
from asr import WhisperASR
from utils import normalize_command, direction_is_valid


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
    ])

    # 3. Initialize DOA + ASR
    doa_estimator = DoAEstimator(
        mic_positions=mic_positions,
        sample_rate=stream.sr,
        nfft=512,
        doa_algo=doa_algo
    )

    asr = WhisperASR(model_size=whisper_model)

    # 4. Process frames
    for frame, sr in stream:
        direction_deg = doa_estimator.estimate_deg(frame)
        text, conf = asr.transcribe(frame, sr, min_conf=min_conf)

        if text is None:
            continue

        cmd = normalize_command(text)

        if not direction_is_valid(cmd, direction_deg):
            print(f"Ignored '{cmd}' from {direction_deg:.1f}° (invalid direction)")
            continue

        payload = {
            "command": cmd,
            "confidence": round(conf, 3),
            "direction": round(direction_deg, 1)
        }

        print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", required=True, type=str)
    args = parser.parse_args()

    run_processor(args.wav_path)
