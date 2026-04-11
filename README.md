# DJ-R3X Controller

An interactive animatronic controller for a DJ-R3X (Rex) build, running on Raspberry Pi 4.
Rex responds to voice commands, engages in AI-powered conversation, reacts to music,
uses computer vision to greet and roast people, and remembers who you are.

## Hardware

| Component | Description |
|-----------|-------------|
| Raspberry Pi 4 | Main controller (user: bbenziger) |
| Pololu Maestro Mini 18 | Servo controller via USB serial (/dev/ttyACM0) |
| Arduino Uno | Head LED controller — mouth PCB + eye LEDs (/dev/ttyACM2) |
| Arduino Nano (pending) | Chest light panel controller |
| ReSpeaker Lite | USB microphone array for wake word and transcription |
| Speakers + Stereo Amp | Audio output via 3.5mm jack |
| Webcam (j5 JVU430) | Computer vision for person detection and greetings |
| ELP-USBFHD01M-L21 | 1080p wide angle camera (ordered — for face recognition) |
| Custom Mouth PCB | 80x WS2812B NeoPixels — emotion-based center-out pulse animation |
| Eye PCB | 2x WS2812B NeoPixels with natural random blink animation |

## Servo Channels (Pololu Maestro Mini 18)

| Channel | Name | Notes |
|---------|------|-------|
| 0 | Neck | Left/right rotation |
| 1 | Headlift | Up/down — lower values = head down |
| 2 | Headtilt | Forward/back tilt — lower values = tilt up |
| 3 | Visor | Open/close — lower values = closed/eyes covered |
| 4 | Elbow | Left arm |
| 5 | Hand | Left hand |
| 6 | Pokerarm | Right arm |
| 7 | Heroarm | Right hand |

## Architecture

### Audio Pipeline
- Wake word detected by OpenWakeWord (4 models: `Dee-Jay_Rex`, `Hey_DJ_Rex`, `Hey_rex`, `Yo_robot`)
- Audio captured via ReSpeaker Lite (PipeWire device, stereo mixdown to mono)
- Speech transcribed via OpenAI Whisper API with hallucination filtering
- Single-word command whitelist bypasses minimum word filter
- Command parser checks transcription: exact match → prefix match → fuzzy match (0.82 threshold) → LLM fallback
- If matched: execute local command or speak canned response
- If no match: stream to ChatGPT gpt-4o-mini → ElevenLabs → speakers
- Mouth PCB emotion pulse driven by speech audio RMS level in real time

### Wake Greeting Pipeline
- Wake word fires → capture image → run face recognition
- **Known person**: greet by name with roast tier based on visit count
- **Unknown person**: run scanning line concurrently with dlib processing → GPT-4o roast greeting → ask name → enroll face + name in database
- **Empty database**: scanning line + enrollment flow
- Alternating pattern for unknown: hi-there audio clip / GPT-4o personalized roast

### Vision Pipeline
- Vision intent detection on transcribed text
- If visual query detected: capture fresh frame → send to GPT-4o with query
- Rex answers naturally without narrating that he is looking at an image

### Face Recognition
- dlib ResNet model generates 128-dimension face encodings
- SQLite database stores people, encodings, visit counts, first/last seen
- Voice commands: `call me [name]`, `forget me`, `rename me`, `my name is [name]`
- Refusal detection: anonymous responses handled with roast lines
- Command detection during name capture: shutdown commands work mid-enrollment

### Servo Behavior
- **Idle**: independent random movements — neck pan, headlift, visor drift, arms
- **Speech reactive**: head, visor, elbow, hand move based on audio RMS intensity
- **Emotion states**: excited (head up, visor open, fast), sad (head down, visor closed, slow)
- **Wake greeting**: hand wave animation concurrent with greeting audio
- **Startup animation**: slumped pose → neck looks around → head rises → visor opens
- **Shutdown animation**: gradual droop to slumped pose → visor closes

### LED System
- Arduino Uno receives ASCII serial commands from Pi via USB
- **Mouth**: 80x WS2812B NeoPixels — emotion-based center-outward pulse animation
  - `SPEAK:{emotion}` sets color scheme (neutral=amber, happy=cyan, excited=yellow, sad=blue, angry=red)
  - `SPEAK_LEVEL:{0-255}` drives pulse speed and brightness from audio RMS
  - Pulse radiates from center pixels (D36/D37/D44/D45) outward through 5 zone rings
- **Eyes**: 2x WS2812B NeoPixels with natural random blink (100-400ms blink, 2-8s interval, 10% double blink)
- Mouth only illuminates during speech — never during music

### State Machine

| State | Behavior |
|-------|----------|
| IDLE | Idle servo movements, random audio clips with mouth sync, wake word listening |
| ACTIVE | Full pipeline — face scan, greet, transcribe, parse, respond, animate |
| SHUTDOWN | Shutdown speech, hyperdrive audio + slumped animation concurrent, clean exit |

### Startup Sequence
1. 5s USB device enumeration wait
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
| Transcription | OpenAI Whisper API (whisper-1) |
| LLM | OpenAI GPT-4o-mini (text) / GPT-4o (vision + face description) |
| Voice synthesis | ElevenLabs streaming TTS (Rex voice clone from Star Tours audio) |
| Face recognition | dlib ResNet + SQLite via FaceDB |
| Servo control | Pololu compact serial protocol |
| LED control | FastLED on Arduino Uno via serial |
| Audio | PipeWire / sounddevice (Debian Trixie) |
| Vision | OpenCV + GPT-4o |

## Voice Commands

| Command | Phrases | Action |
|---------|---------|--------|
| Rename | "call me [name]", "my name is [name]", "rename me to [name]" | Updates face database |
| Forget me | "forget me", "delete me", "forget my face" | Removes from database (with confirmation) |
| Cancel | "cancel", "nevermind", "forget it", "i was talking to someone else" | Returns to IDLE |
| Shutdown | "shut down", "exit program", "shut down rex" | Stops Python program |
| Power down | "power down", "turn off", "goodbye forever" | OS shutdown (if enabled) |
| Vision | "what do you see", "what am I wearing", "take a picture" | Captures image → GPT-4o |

## Project Structure

```bash
djr3x/
├── main.py                 # Entry point
├── config.py               # All constants and environment variables
├── audio/
│   ├── player.py           # Speech and music playback, RMS tracking
│   └── recorder.py         # Mic input
├── speech/
│   ├── wake_word.py        # OpenWakeWord dual model detection
│   ├── transcriber.py      # Whisper API transcription
│   └── synthesizer.py      # ElevenLabs streaming TTS
├── commands/
│   ├── parser.py           # Exact and fuzzy command matching
│   └── command_list.py     # 25 commands, 119 phrases
├── llm/
│   ├── chatgpt.py          # Streaming GPT-4o-mini/GPT-4o
│   ├── greeter.py          # Personalized vision-based wake greetings
│   └── vision_intent.py    # Detects visually oriented queries
├── hardware/
│   ├── servos.py           # Maestro serial control
│   └── leds.py             # Arduino Nano serial LED commands
├── states/
│   └── state_machine.py    # IDLE/ACTIVE/SHUTDOWN state machine
├── sequences/
│   └── animations.py       # Startup, shutdown, emotion sequences
├── vision/
│   └── camera.py           # OpenCV webcam capture
└── assets/
    ├── audio/              # Cached responses, music, chime
    └── models/             # Wake word .onnx models
```

## Installation

### Quick Setup (all platforms)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 setup_assets.py
```

`setup_assets.py` downloads required model files (~100MB) and fixes known Python 3.11+ compatibility issues automatically.

### Raspberry Pi — additional dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev portaudio19-dev libportaudio2 libasound2-dev \
  ffmpeg sox libsox-fmt-all git curl cmake

# dlib compiles from source on Pi — use piwheels to save ~30 minutes
pip install dlib --extra-index-url https://www.piwheels.org/simple
```

### macOS (Apple Silicon) — additional dependencies
```bash
brew install portaudio ffmpeg cmake
```

### Arduino sketch
```bash
arduino-cli core install arduino:avr
arduino-cli lib install "FastLED"
arduino-cli compile --fqbn arduino:avr:uno arduino/head_nano
arduino-cli upload --fqbn arduino:avr:uno --port /dev/ttyACM2 arduino/head_nano
```

## Environment Variables (.env)

```env
# OpenAI
OPENAI_API_KEY=your_key_here

# ElevenLabs
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Audio devices
AUDIO_INPUT_DEVICE=3
AUDIO_OUTPUT_DEVICE=0
MIC_CHANNELS=2

# Wake word models
WAKE_WORD_MODEL_1=assets/models/Dee-Jay_Rex.onnx
WAKE_WORD_MODEL_2=assets/models/Hey_DJ_Rex.onnx
WAKE_WORD_MODEL_3=assets/models/Hey_rex.onnx
WAKE_WORD_MODEL_4=assets/models/Yo_robot.onnx
WAKE_WORD_THRESHOLD=0.60

# Hardware ports
MAESTRO_PORT=/dev/ttyACM0
NANO_HEAD_PORT=/dev/ttyACM2
#NANO_CHEST_PORT=/dev/ttyUSB1

# Feature flags
ENABLE_OS_SHUTDOWN=false
SERVO_SAFE_MODE=true

# Tuning
FACE_RECOGNITION_TOLERANCE=0.6
SYNTHESIZER_VOLUME_GAIN=1.5
NOISE_FLOOR_MULTIPLIER=2.0
```

## Running

```bash
# Manual
cd ~/djr3x
source venv/bin/activate
python3 main.py

# As a service
sudo systemctl enable djr3x
sudo systemctl start djr3x

# Monitor logs
journalctl -u djr3x -f
```

## Current Status

### Working
- [x] Wake word detection (4 models)
- [x] Whisper transcription with hallucination filtering
- [x] Command parser (exact + prefix + fuzzy matching)
- [x] ChatGPT streaming responses with Rex roaster personality
- [x] ElevenLabs voice synthesis with volume gain
- [x] Computer vision — intent based photo capture
- [x] Personalized wake greeting — GPT-4o roasts based on appearance
- [x] Face recognition with SQLite database
- [x] Known person greetings with visit counter roast tiers
- [x] Name enrollment, rename, forget me voice commands
- [x] Cancel/nevermind returns to IDLE immediately
- [x] Servo idle animations (neck, headlift, visor, arms)
- [x] Speech reactive servo movement with emotions
- [x] Startup animation — neck looks around, head rises
- [x] Shutdown animation — gradual slumped pose
- [x] Startup audio: light_speed.mp3 + Roger Control.mp3
- [x] Shutdown audio: hyperdrive_down.mp3 concurrent with animation
- [x] Idle audio clips with mouth LED and servo sync
- [x] Mouth LED emotion-based center-out pulse animation
- [x] Eye LEDs with natural random blink timing
- [x] systemd service with boot retry logic
- [x] PipeWire audio on Debian Trixie

### In Progress
- [ ] Face enrollment frame capture timing fix
- [ ] ELP camera installation (camera ordered)

### Pending
- [ ] Head tracking with ELP camera
- [ ] Conversation memory per person
- [ ] Chest Nano Arduino sketch
- [ ] udev rules for fixed USB device names

## Platform Notes

### Raspberry Pi
- Audio via PipeWire on Debian Trixie — set `AUDIO_INPUT_DEVICE=3` for ReSpeaker Lite
- dlib installs from piwheels as a prebuilt wheel — no compilation needed
- Serial ports: Maestro=`/dev/ttyACM0`, Head Nano=`/dev/ttyACM2`, Chest Nano=`/dev/ttyUSB0`

### macOS (Apple Silicon)
- Run `python3 setup_assets.py` after pip install — it patches `face_recognition_models` for Python 3.11+ automatically
- Set `AUDIO_INPUT_DEVICE=1` (MacBook Air Microphone) and leave `AUDIO_OUTPUT_DEVICE=` blank
- Serial ports will be blank when hardware is not connected — Rex runs in software-only mode
- Camera index 0 = built-in FaceTime camera (or USB webcam if FaceTime is unavailable)