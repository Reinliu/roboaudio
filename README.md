# spatial_audio_processor (Python)

ROS 2 package for:
- multichannel audio input (simulated wav)
- DoA localisation using pyroomacoustics (SRP-PHAT)
- speech recognition using Whisper
- publishing spatially-aware commands to `/recognised_commands`

## Requirements
- ROS 2 Humble (or Iron)
- Python 3.10+
- pip

## Install

### 1. Create workspace
```bash
mkdir -p ~/ros2_audio_ws/src
cd ~/ros2_audio_ws/src
git clone https://github.com/<yourname>/spatial_audio_processor.git

conda create -n roboaudio python==3.9
conda activate roboaudio
conda install -c conda-forge "libstdcxx-ng>=11" "libgcc-ng>=11"
