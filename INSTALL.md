# DJ-R3X Installation Guide

## System Dependencies
Run the Pi bootstrap script:
```bash
./scripts/setup_pi.sh
```

`pulseaudio-utils` provides `pactl`, which the Bluetooth audio path uses to
switch the PipeWire/Pulse default sink. `libspa-0.2-bluetooth` enables
PipeWire's Bluetooth audio support.

Reboot after installing the PipeWire packages:
```bash
sudo reboot
```

## Python Dependencies
If you prefer to run the Python steps manually after the script:
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
- Raspberry Pi 4 or 5
- ReSpeaker Lite USB audio (or similar USB audio device)
- Logitech C920 webcam (or similar for vision)
- Pololu Maestro Mini for servos
- Arduino Nanos for LED control
