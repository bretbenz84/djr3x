# DJ-R3X Controller

An interactive animatronic controller for a DJ-R3X (Rex) build, running on Raspberry Pi 4 or macOS Apple Silicon.
Rex responds to voice commands, engages in AI-powered conversation, reacts to music,
uses computer vision to greet and roast people, and remembers who you are.

Raspberry Pi 4 builds send transcription, images, and text to OpenAI. Apple Silicon builds run transcription and LLM locally. Text to Speech is handled by ElevenLabs on both platforms.

## Hardware

| Component | Description |
|-----------|-------------|
| Raspberry Pi 4 | Main controller |
| Pololu Maestro Mini 18 | Servo controller via USB serial (/dev/ttyACM0) |
| Arduino Uno | Head LED controller — mouth PCB + eye LEDs (/dev/ttyACM2) |
| Arduino Nano | Chest light panel controller (/dev/ttyUSB0) |
| ReSpeaker Lite | USB microphone array for wake word and transcription |
| Speakers + Stereo Amp | Audio output via 3.5mm jack |
| ELP-USBFHD01M-L21 | 1080p wide angle camera, mounted in head (use a stable udev path like `/dev/camera_main` on Pi when available) |
| Custom Mouth PCB | 80x WS2812B NeoPixels — emotion-based center-out pulse animation |
| Eye PCB | 2x WS2812B NeoPixels with natural random blink animation |

## Servo Channels (Pololu Maestro Mini 18)

| Channel | Name | Notes |
|---------|------|-------|
| 0 | Neck | Left/right rotation |
| 1 | Headlift | Up/down — higher values = head up |
| 2 | Headtilt | Forward/back tilt — inverted, lower = up |
| 3 | Visor | Open/close — inverted, lower = closed/eyes covered |
| 4 | Elbow | Left arm |
| 5 | Hand | Left hand |
| 6 | Pokerarm | Right arm |
| 7 | Heroarm | Right hand |

## Architecture

### AI Backend Selection (Automatic by Platform)

| Component | Raspberry Pi | macOS Apple Silicon |
|-----------|-------------|---------------------|
| Transcription | OpenAI Whisper API | Local mlx-whisper (whisper-small-mlx) |
| LLM | OpenAI GPT-4o-mini | Local Ollama llama3.2 |
| Vision queries | OpenAI GPT-4o | OpenAI GPT-4o (always cloud) |
| TTS | ElevenLabs streaming | ElevenLabs streaming |

### Audio Pipeline
- Wake word detected by OpenWakeWord (4 models: `Dee-Jay_Rex`, `Hey_DJ_Rex`, `Hey_rex`, `Yo_robot`)
- Audio captured via ReSpeaker Lite (PipeWire device, stereo mixdown to mono)
- Two-phase silence detection: 5s wait for speech to start, 1.5s sustained silence to end recording
- Speech transcribed via Whisper API (Pi) or local mlx-whisper (Mac)
- Hallucination filtering with phrase blocklist
- Command parser: exact match → prefix match → fuzzy match (0.82 threshold) → LLM fallback
- Semantic exclusion: "my name" never matches "your name" regardless of fuzzy score
- Multiple response variations per command (5 variations, anti-repeat shuffle)
- If matched: execute local command or speak canned response
- If no match: stream to ChatGPT/Ollama → ElevenLabs → speakers
- Mouth PCB emotion pulse driven by speech audio RMS level in real time

### Wake Greeting Pipeline
- Wake word fires → camera pose preparation (visor opens, neck centers, 0.5s settle) → capture image → run face recognition
- **Known person**: greet by name with GPT-4o roast referencing appearance and visit count
- **Unknown person**: roast line → ask name → enroll face + name in database → GPT-4o roast as new person
- **Empty database**: enrollment flow

### Vision Pipeline
- Vision intent detection on transcribed text
- If visual query detected: camera pose preparation → capture fresh frame → send to GPT-4o with query
- Vision always uses GPT-4o cloud API regardless of platform
- Rex answers naturally without narrating that he is looking at an image

### Face Recognition
- dlib ResNet model generates 128-dimension face encodings
- SQLite database stores people, encodings, visit counts, first/last seen
- "who am I" / "what's my name" → face DB lookup → GPT-4o roast using name + appearance
- Voice commands: `call me [name]`, `forget me`, `rename me`, `my name is [name]`
- Refusal detection: anonymous responses handled with roast lines
- Command detection during name capture: shutdown commands work mid-enrollment

### Servo Behavior
- **Idle**: independent random movements — neck pan, headlift, visor drift, arms
- **Speech reactive**: head, visor, elbow, hand move based on audio RMS intensity
- **Camera pose**: visor to max, neck to neutral before any image capture
- **Emotion states**: excited (head up, visor open, fast), sad (head down, visor closed, slow)
- **Wake greeting**: hand wave animation concurrent with greeting audio
- **Startup animation**: slumped pose → neck looks around → head rises → visor opens
- **Shutdown animation**: gradual droop to slumped pose → visor closes

### LED System
- Arduino Uno (head) receives ASCII serial commands from Pi via USB
- **Mouth**: 80x WS2812B NeoPixels — emotion-based center-outward pulse animation
  - `SPEAK:{emotion}` sets color scheme (neutral=amber, happy=cyan, excited=yellow, sad=blue, angry=red)
  - `SPEAK_LEVEL:{0-255}` drives pulse speed and brightness from audio RMS
  - Pulse radiates from center pixels outward through 5 zone rings
- **Eyes**: 2x WS2812B NeoPixels with natural random blink (100-400ms blink, 2-8s interval, 10% double blink)
- Mouth only illuminates during speech — never during music
- Arduino Nano (chest): 98 WS2811 LEDs — RandomBlocks2 default, emotion-based modes via serial commands

### State Machine

| State | Behavior |
|-------|----------|
| IDLE | Idle servo movements, optional idle audio clips, wake word listening |
| QUIET | Wake word listening + face tracking stay active, but Rex will not speak or face-greet until resumed |
| ACTIVE | Full pipeline — face scan, greet, transcribe, parse, respond, animate |
| SLEEP | Sleep animation + sleep-only wake word |
| SHUTDOWN | Shutdown speech, hyperdrive audio + slumped animation concurrent, clean exit |

### Startup Sequence
1. USB device enumeration (skipped in interactive mode)
2. `light_speed.mp3` + servo startup animation (concurrent)
3. `Roger Control.mp3` spoken intro with mouth LEDs and servo animation
4. Startup chime
5. IDLE — wake word listening begins

### Shutdown Sequence
1. Shutdown speech phrase (ElevenLabs)
2. `hyperdrive_down.mp3` + shutdown servo animation (concurrent)
3. Servos settle in slumped pose
4. Optional OS halt (controlled by `ENABLE_OS_SHUTDOWN`)

## Software Stack

| Component | Technology |
|-----------|------------|
| Wake word | OpenWakeWord (4 custom trained .onnx models) |
| Transcription | OpenAI Whisper API (Pi) / local mlx-whisper (Mac) |
| LLM | OpenAI GPT-4o-mini (Pi) / local Ollama llama3.2 (Mac) |
| Vision | OpenCV + GPT-4o (always cloud) |
| Voice synthesis | ElevenLabs streaming TTS (Rex voice clone from Star Tours audio) |
| Face recognition | dlib ResNet + SQLite via FaceDB |
| Servo control | Pololu compact serial protocol |
| LED control | FastLED on Arduino Uno/Nano via serial |
| Audio | PipeWire / sounddevice (Debian Trixie) |

## Voice Commands

| Command | Phrases | Action |
|---------|---------|--------|
| Recall name | "who am I", "what's my name", "do you know me" | Face DB lookup + GPT-4o roast |
| Rename | "call me [name]", "my name is [name]", "rename me to [name]" | Updates face database |
| Forget me | "forget me", "delete me", "forget my face" | Removes from database (with confirmation) |
| Cancel | "cancel", "nevermind", "forget it" | Returns to IDLE |
| Quiet mode | "shut up", "be quiet", "stop talking", "silence" | Enters QUIET until a wake word or resume command |
| Shutdown | "shut down", "exit program", "shut down rex" | Stops Python program |
| Power down | "power down", "turn off", "goodbye forever" | OS shutdown (if enabled) |
| Vision | "what do you see", "what am I wearing", "take a picture" | Captures image → GPT-4o |

## Project Structure

```bash
djr3x/
├── main.py                 # Entry point
├── config.py               # All constants and environment variables
├── platform_utils.py       # Platform detection (Pi vs macOS Apple Silicon)
├── setup_assets.py         # Downloads required model files
├── audio/
│   ├── player.py           # Speech and music playback, RMS tracking
│   └── recorder.py         # Mic input
├── speech/
│   ├── wake_word.py        # OpenWakeWord detection (4 models)
│   ├── transcriber.py      # Whisper API or local mlx-whisper transcription
│   └── synthesizer.py      # ElevenLabs streaming TTS
├── commands/
│   ├── parser.py           # Exact, prefix, fuzzy matching + semantic exclusions
│   └── command_list.py     # 25+ commands, 5 variations each
├── llm/
│   ├── chatgpt.py          # Streaming GPT-4o-mini / Ollama
│   ├── greeter.py          # Personalized vision-based wake greetings
│   └── vision_intent.py    # Detects visually oriented queries
├── hardware/
│   ├── servos.py           # Maestro serial control
│   └── leds.py             # Arduino serial LED commands
├── states/
│   └── state_machine.py    # IDLE/QUIET/ACTIVE/SLEEP/SHUTDOWN state machine
├── sequences/
│   └── animations.py       # Startup, shutdown, emotion sequences
├── vision/
│   ├── camera.py           # OpenCV webcam capture
│   ├── face_recognizer.py  # dlib face detection and encoding
│   └── face_db.py          # SQLite face database
├── arduino/
│   ├── head_nano/          # Arduino Uno sketch — 82 NeoPixels
│   └── chest_nano/         # Arduino Nano sketch — 98 WS2811 LEDs
└── assets/
    ├── audio/              # Sound effects, idle clips, startup audio
    ├── music/              # Idle music tracks (gitignored)
    └── models/             # Wake word .onnx files + dlib model files
```

## Installation

### Quick Setup (all platforms)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Choose one:

# Raspberry Pi
pip install -r requirements-raspberry-pi.txt

# macOS Apple Silicon
pip install -r requirements-macos-apple-silicon.txt

# Then on either platform
python3 setup_assets.py
```

`setup_assets.py` downloads required model files (~120MB) and patches `face_recognition_models` automatically if that optional package is installed.

### Raspberry Pi — additional dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev portaudio19-dev libportaudio2 libasound2-dev \
  ffmpeg sox libsox-fmt-all git curl cmake

# Recommended Python install on Pi (uses piwheels for faster dlib installs)
pip install -r requirements-raspberry-pi.txt
```

### macOS (Apple Silicon) — additional dependencies
```bash
brew install portaudio ffmpeg cmake libpng
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
```

Local transcription dependencies are installed automatically by the Apple
Silicon environment markers in `requirements.txt`. The macOS requirements
wrapper also installs `dlib`; make sure you upgrade `pip`, `setuptools`, and
`wheel` first in a fresh venv before installing.

For local LLM on macOS:
```bash
brew install ollama
ollama serve
ollama pull llama3.2
```

### Arduino sketches
```bash
arduino-cli core install arduino:avr
arduino-cli lib install "FastLED"

# Head Uno
arduino-cli compile --fqbn arduino:avr:uno arduino/head_nano
arduino-cli upload --fqbn arduino:avr:uno --port /dev/ttyACM2 arduino/head_nano

# Chest Nano
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328 arduino/chest_nano
arduino-cli upload --fqbn arduino:avr:nano:cpu=atmega328 --port /dev/ttyUSB0 arduino/chest_nano
```

## Environment Variables (.env)

```env
# OpenAI
OPENAI_API_KEY=your_key_here

# TTS
TTS_PROVIDER=elevenlabs      # elevenlabs | piper

# ElevenLabs (required when TTS_PROVIDER=elevenlabs)
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Piper (required when TTS_PROVIDER=piper)
PIPER_MODEL_PATH=assets/models/rexvoice.onnx
PIPER_CONFIG_PATH=assets/models/rexvoice.onnx.json

# Audio devices
AUDIO_INPUT_DEVICE=3        # Pi: ReSpeaker=3 | Mac: MacBook mic=1
AUDIO_OUTPUT_DEVICE=0       # Pi: 3.5mm jack=0 | Mac: leave blank
AUDIO_OUTPUT_MODE=device    # default | device | bluetooth
AUDIO_BLUETOOTH_DEVICE=auto # auto | BT MAC | alias substring

# Wake word models
WAKE_WORD_MODEL_1=assets/models/Dee-Jay_Rex.onnx
WAKE_WORD_MODEL_2=assets/models/Hey_DJ_Rex.onnx
WAKE_WORD_MODEL_3=assets/models/Hey_rex.onnx
WAKE_WORD_MODEL_4=assets/models/Yo_robot.onnx
WAKE_WORD_THRESHOLD=0.60

# Hardware ports (leave blank to skip gracefully)
MAESTRO_PORT=/dev/ttyACM0
NANO_HEAD_PORT=/dev/ttyACM2
NANO_CHEST_PORT=/dev/ttyUSB0

# Camera
CAMERA_DEVICE=/dev/camera_main   # preferred on Pi; accepts /dev path or numeric string
# CAMERA_DEVICE_INDEX=0          # legacy fallback if CAMERA_DEVICE is blank

# Feature flags
ENABLE_OS_SHUTDOWN=false
SERVO_SAFE_MODE=true

# Tuning
FACE_RECOGNITION_TOLERANCE=0.6
SYNTHESIZER_VOLUME_GAIN=1.5
NOISE_FLOOR_MULTIPLIER=2.0

# Local AI (macOS Apple Silicon — auto-detected, override if needed)
# USE_LOCAL_TRANSCRIPTION=true
# USE_LOCAL_LLM=true
# LOCAL_WHISPER_MODEL=mlx-community/whisper-small-mlx
# LOCAL_LLM_MODEL=llama3.2
# LOCAL_LLM_BASE_URL=http://localhost:11434/v1
```

### Bluetooth Audio On Raspberry Pi

To use a paired Bluetooth playback device without hard-coding a changing
PipeWire/PortAudio index:

```env
AUDIO_OUTPUT_DEVICE=
AUDIO_OUTPUT_MODE=bluetooth
AUDIO_BLUETOOTH_DEVICE=auto
```

You can also target a specific paired device by MAC address or alias substring:

```env
AUDIO_OUTPUT_MODE=bluetooth
AUDIO_BLUETOOTH_DEVICE=5C:2C:FF:05:70:B1
# or
# AUDIO_BLUETOOTH_DEVICE=BT-WUZHI
```

In Bluetooth mode, Rex will ask `bluetoothctl` for paired devices, prefer an
already-connected audio sink, otherwise connect the requested paired sink, then
scan the available PortAudio/PipeWire playback devices and choose the matching
Bluetooth output dynamically. If Bluetooth is unavailable, playback falls back
to `AUDIO_OUTPUT_DEVICE` when set, then to the system default output.

### Camera Device On Raspberry Pi

If your camera gets a stable udev symlink, point Rex at that path instead of a
changing numeric index:

```env
CAMERA_DEVICE=/dev/camera_main
```

`CAMERA_DEVICE` accepts either a `/dev/...` path or a numeric string like `0`.
If it is blank, Rex falls back to the legacy `CAMERA_DEVICE_INDEX` setting.

## Running

```bash
# Manual
cd ~/djr3x
source venv/bin/activate
python3 main.py

# As a service (Pi only)
sudo systemctl enable djr3x
sudo systemctl start djr3x

# Monitor logs
journalctl -u djr3x -f
```

## Current Status

### Working
- [x] Wake word detection (4 models)
- [x] Two-phase transcription silence detection
- [x] Whisper API transcription with hallucination filtering
- [x] Local mlx-whisper transcription on macOS Apple Silicon
- [x] Command parser (exact + prefix + fuzzy matching, semantic exclusions)
- [x] Multiple response variations per command (5 variations, anti-repeat shuffle)
- [x] ChatGPT streaming responses with Rex roaster personality
- [x] Local Ollama llama3.2 on macOS Apple Silicon
- [x] Platform detection — automatic backend selection Pi vs Mac
- [x] ElevenLabs voice synthesis with volume gain
- [x] Computer vision — intent based photo capture with GPT-4o
- [x] Camera pose preparation before image capture
- [x] Personalized wake greeting — GPT-4o roasts based on appearance
- [x] Face recognition with dlib ResNet + SQLite database
- [x] Known person greetings with visit counter and roast tiers
- [x] Unknown person enrollment with roast on first meeting
- [x] Recall name command — face DB lookup + GPT-4o roast
- [x] Name enrollment, rename, forget me voice commands
- [x] Cancel/nevermind returns to IDLE
- [x] Servo idle animations (neck, headlift, visor, arms)
- [x] Speech reactive servo movement with emotions
- [x] Startup and shutdown animations with concurrent audio
- [x] Idle audio clips with mouth LED and servo sync
- [x] Mouth LED emotion-based center-out pulse animation
- [x] Eye LEDs with natural random blink timing
- [x] Chest LED Arduino sketch with emotion-based lighting
- [x] ELP 1080p camera installed and working
- [x] Fast startup when hardware ports are blank
- [x] setup_assets.py — downloads all required model files
- [x] systemd service with boot retry logic
- [x] PipeWire audio on Debian Trixie
- [x] macOS Apple Silicon development environment

### Pending
- [ ] Head tracking with ELP camera (face position → neck servo)
- [ ] Conversation memory per person (per-person GPT summary in SQLite)
- [ ] udev rules for fixed USB device names on Pi
- [ ] Dance mode (beat-synced servo sequences)
- [ ] Mecanum wheel base (future)
- [ ] Local TTS voice cloning (revisit when MLX TTS matures)

## Platform Notes

### Raspberry Pi
- Audio via PipeWire on Debian Trixie — set `AUDIO_INPUT_DEVICE=3` for ReSpeaker Lite
- dlib installs from piwheels as a prebuilt wheel — no compilation needed
- Serial ports: Maestro=`/dev/ttyACM0`, Head Nano=`/dev/ttyACM2`, Chest Nano=`/dev/ttyUSB0`
- Prefer `CAMERA_DEVICE=/dev/camera_main` (or another udev symlink) over a changing camera index

### macOS (Apple Silicon)
- Run `python3 setup_assets.py` after pip install — it also patches `face_recognition_models` if you installed that optional package
- Set `AUDIO_INPUT_DEVICE=1` (MacBook Air Microphone) and leave `AUDIO_OUTPUT_DEVICE=` blank
- Serial ports left blank — Rex runs in software-only mode without hardware
- Ollama must be running before starting Rex: `ollama serve`
- `requirements.txt` installs the MLX stack automatically on Apple Silicon
- On macOS, install `libpng` and export `PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig` before building `dlib`
- Camera index 0 = built-in FaceTime camera (or USB webcam if FaceTime is unavailable)
