# spatial_audio_processor (Python)

This repository provides a package for:
- multichannel audio input 
- DoA localisation using pyroomacoustics 
- speech recognition using Whisper
- speaker diarization with pyAudioAnalysis
- noise reduction with spectral gatting using librosa
- publishing spatially-aware commands to `/recognised_commands`

## Requirements
- Python 3.9+

## Install

### 1. Clone the Repo
```
git clone https://github.com/Reinliu/roboaudio.git
```

### 2. Configure Python Environment
```
conda create -n roboaudio python==3.10
conda activate roboaudio
conda install -c conda-forge "libstdcxx-ng>=11" "libgcc-ng>=11"
pip install -r requirements.txt
```

### 3. Run the main.py from spatial_audio_processor to load audio and print out actions.
```
python main.py
```
