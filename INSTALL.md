# DJ-R3X Installation Guide

## System Dependencies
Install these with apt before setting up Python dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  python3-dev \
  python3-venv \
  portaudio19-dev \
  libportaudio2 \
  libasound2-dev \
  ffmpeg \
  sox \
  libsox-fmt-all \
  git \
  curl \
  pipewire-alsa
```

Reboot after installing pipewire-alsa:
```bash
sudo reboot
```

## Python Dependencies
```bash
cd ~/djr3x
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-raspberry-pi.txt
```

## Configuration
Copy .env.example to .env and fill in your API keys and device settings:
```bash
cp .env.example .env
```

## Hardware
- Raspberry Pi 4
- ReSpeaker Lite USB audio (or similar USB audio device)
- Logitech C920 webcam (or similar for vision)
- Pololu Maestro Mini for servos
- Arduino Nanos for LED control
