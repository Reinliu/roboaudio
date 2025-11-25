# spatial_audio_processor/audio_io.py
import numpy as np
import soundfile as sf
import time

class WavStream:
    """
    Streams a multichannel wav file chunk-by-chunk.
    Produces frames shaped (num_channels, frame_len).
    """
    def __init__(self, wav_path, frame_len=16000, hop_len=8000, realtime=True):
        self.wav_path = wav_path
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.realtime = realtime

        self.audio, self.sr = sf.read(wav_path, always_2d=True)  # shape (T, C)
        self.audio = self.audio.T  # -> (C, T)
        self.num_channels, self.num_samples = self.audio.shape
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx + self.frame_len > self.num_samples:
            raise StopIteration

        frame = self.audio[:, self.idx:self.idx + self.frame_len]
        self.idx += self.hop_len

        if self.realtime:
            # sleep hop duration to simulate real-time
            time.sleep(self.hop_len / self.sr)

        return frame, self.sr

# Optional: real mic array stub
class MicArrayStream:
    """
    Placeholder if you move to real mic array.
    Use sounddevice or pyaudio to capture (C, frame_len).
    """
    def __init__(self, device=None, sr=16000, frame_len=16000, hop_len=8000):
        self.device = device
        self.sr = sr
        self.frame_len = frame_len
        self.hop_len = hop_len

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("Implement real mic streaming here.")
