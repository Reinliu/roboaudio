# spatial_audio_processor (Python)

This repository provides a package for:
- multichannel audio input (simulated wav)
- DoA localisation using pyroomacoustics (SRP-PHAT)
- speech recognition using Whisper
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
conda create -n roboaudio python==3.9
conda activate roboaudio
conda install -c conda-forge "libstdcxx-ng>=11" "libgcc-ng>=11"
pip install -r requirements.txt
```

### 3. Run the Script
```
python main.py --wav /path_to_audio.wav
```
