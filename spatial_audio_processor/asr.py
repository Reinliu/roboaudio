# spatial_audio_processor/asr.py
import numpy as np
import whisper

TARGET_SR = 16000

def simple_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight resampler using linear interpolation.
    No extra dependencies. Good enough for speech chunks.
    """
    if orig_sr == target_sr:
        return x.astype(np.float32)

    x = x.astype(np.float32)
    n_orig = x.shape[0]
    n_target = int(np.round(n_orig * target_sr / orig_sr))

    # time positions
    orig_idx = np.arange(n_orig)
    target_idx = np.linspace(0, n_orig - 1, n_target)

    return np.interp(target_idx, orig_idx, x).astype(np.float32)


class WhisperASR:
    def __init__(self, model_size="base", language="en", device=None):
        self.model = whisper.load_model(model_size, device=device)
        self.language = language

    def _is_speech(self, mono, energy_thresh=1e-4):
        """Basic energy gate to skip silent chunks."""
        return np.mean(mono**2) > energy_thresh

    def transcribe(self, frame, sr, min_conf=0.5):
        """
        frame: (M, N) multi-channel
        returns (text, confidence) or (None, 0.0)
        """
        mono = np.mean(frame, axis=0)

        if not self._is_speech(mono):
            return None, 0.0

        # Resample to 16k if needed
        if sr != TARGET_SR:
            mono = simple_resample(mono, sr, TARGET_SR)
            sr = TARGET_SR
        else:
            mono = mono.astype(np.float32)

        result = self.model.transcribe(
            mono,
            language=self.language,
            fp16=False,
            verbose=False
        )

        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        avg_logprob = (
            np.mean([s.get("avg_logprob", -10) for s in segments])
            if segments else -10.0
        )

        # Map logprob ~ [-10, 0] -> [0, 1] rough confidence
        confidence = float(np.clip((avg_logprob + 10) / 10, 0, 1))

        if confidence < min_conf or not text:
            return None, confidence

        return text, confidence
