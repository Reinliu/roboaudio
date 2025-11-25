# diarization.py
import librosa
from pyAudioAnalysis import audioSegmentation as aS

def run_diarization(wav_path: str, n_speakers: int = 2):
    """
    Offline diarisation with forced resampling to 16 kHz mono.
    This avoids reshape errors in pyAudioAnalysis.
    """

    # Load audio at 16k mono for compatibility
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Save to temp WAV for pyAudioAnalysis
    import soundfile as sf
    tmp_path = "tmp_diarisation_16k.wav"
    sf.write(tmp_path, y, 16000)

    # Call pyAudioAnalysis diarizer on the resampled file
    flags, classes, _ = aS.speaker_diarization(
        tmp_path,
        n_speakers=n_speakers,
        lda_dim=0
    )

    # pyAudioAnalysis uses 0.2s step by default
    mt_step = 0.2

    return flags, classes, mt_step
