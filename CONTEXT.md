# DJ-R3X Controller Project

## Platforms

### Raspberry Pi 4 (Primary Hardware Platform)
- Username: bbenziger, project path: /home/bbenziger/djr3x
- OS: Debian GNU/Linux 13 (Trixie), Kernel: 6.12.75 aarch64
- Python: 3.13.5
- Virtual environment: /home/bbenziger/djr3x/venv
- Audio system: PipeWire 1.4.2
- All hardware connected here (servos, LEDs, mic, camera)

### macOS Apple Silicon (Development / Alternate Brain)
- M1 MacBook Air (broken screen, external display + SSH)
- Username: bbenziger, project path: /Users/bbenziger/djr3x
- macOS 15.7.5, Python: 3.11.9 via pyenv
- Virtual environment: /Users/bbenziger/djr3x/venv
- Runs Rex in software-only mode (no servos or LEDs connected)
- Local AI models installed at /Users/bbenziger/models/

## Hardware

### Raspberry Pi Connected Devices
| Device | Port | Notes |
|--------|------|-------|
| Pololu Maestro Mini 18 | /dev/ttyACM0 | Servo controller, 9600 baud |
| Arduino Uno (head LEDs) | /dev/ttyACM2 | 82 NeoPixels, 115200 baud |
| Arduino Nano (chest LEDs) | /dev/ttyUSB0 | 98 WS2811 LEDs, 115200 baud, CH340 |
| ReSpeaker Lite | PipeWire device 3 | USB mic, stereo mixdown to mono |
| ELP-USBFHD01M-L21 | /dev/video0 (index 0) | 1080p wide angle, mounted in head |
| Speakers | 3.5mm jack device 0 | Via stereo amp, passive resistor mixer |

### Servo Channels (Pololu Maestro Mini 18, quarter-microseconds)
| Ch | Name | Min | Max | Neutral | Notes |
|----|------|-----|-----|---------|-------|
| 0 | Neck | 1984 | 9984 | 6000 | Pan left/right |
| 1 | Headlift | 1984 | 7744 | 6000 | Higher = up |
| 2 | Headtilt | 3904 | 5504 | 4320 | Inverted, lower = up |
| 3 | Visor | 4544 | 6976 | 6000 | Inverted, lower = closed |
| 4 | Elbow | 6300 | 7560 | 6720 | Left arm |
| 5 | Hand | 1984 | 9984 | 6000 | Left hand |
| 6 | Pokerarm | 3968 | 8000 | 6000 | Right arm |
| 7 | Heroarm | 3968 | 8000 | 6000 | Right hand |

HEAD_CHANNELS=[0,1,2,3], ARM_CHANNELS=[4,5,6,7], IDLE_HEAD_CHANNELS=[0,1]

## Environment
- API keys in .env (excluded from git)
- Dependencies in requirements.txt
- Run `python3 setup_assets.py` after pip install to download model files

## AI Backend Selection (Platform Automatic)

### Raspberry Pi
- Transcription: OpenAI Whisper API (whisper-1)
- LLM: OpenAI GPT-4o-mini streaming
- Vision/image queries: OpenAI GPT-4o
- TTS: ElevenLabs streaming (Rex voice clone)

### macOS Apple Silicon
- Transcription: Local mlx-whisper (mlx-community/whisper-small-mlx, ~0.7s)
- LLM: Local Ollama llama3.2 via OpenAI-compatible API at http://localhost:11434/v1
- Vision/image queries: OpenAI GPT-4o (always cloud regardless of platform)
- TTS: ElevenLabs streaming (same as Pi)

Fallback: if local models unavailable on macOS, falls back to cloud APIs with a warning.

## Architecture Decisions
- Platform detection at startup selects AI backends automatically
- OpenAI Whisper API on Pi / local mlx-whisper on Mac for transcription
- Exact phrase matching for local commands (no API needed)
- ChatGPT 4o-mini / Ollama fallback for unmatched/open ended input
- Streaming LLM → ElevenLabs to reduce latency
- ElevenLabs basic voice clone based on Star Tours Rex audio
- Mouth LED brightness driven by speech audio level in software
- Pi controls all servos via serial to Maestro (no onboard scripts)
- Nanos act as dumb executors, Pi sends serial LED commands
- Two audio channels mixed passively — speech and music separate
- Serial port retries skipped when port is blank — fast startup without hardware

## State Machine
- IDLE: slow breathing LED pulse, listening for wake word, idle audio clips
- ACTIVE: full interactivity, LLM responses, reactive movement
- SHUTDOWN: triggered by voice command or signal

## Servo Behaviors
- Background thread: random arm/hand movements continuously
- Speech thread: head/visor movement overlaid during speech
- Camera pose: visor opens to max, neck centers before any image capture (0.5s settle)
- Emotion states bias servo position ranges:
  - Excitement: head up, faster movement, higher visor
  - Sad: head down, slower movement, lower visor
  - Neutral: centered ranges

## Audio Pipeline
- Wake word detected (OpenWakeWord — 4 models: Dee-Jay_Rex, Hey_DJ_Rex, Hey_rex, Yo_robot)
- Two-phase silence detection: Phase 1 waits up to 5s for speech to start, Phase 2 uses 1.5s sustained silence to end recording
- Whisper API (Pi) or local mlx-whisper (Mac) transcribes audio
- Hallucination filtering with phrase blocklist
- Command parser: exact match → prefix match → fuzzy match (0.82 threshold) → LLM fallback
- Semantic exclusion: "my name" never matches "your name" regardless of fuzzy score
- Multiple response variations per command (5 variations, anti-repeat shuffle)
- If matched: execute locally, speak canned response (random variation)
- If no match: stream to ChatGPT/Ollama → ElevenLabs → speakers
- Mouth PCB brightness driven by speech audio buffer level
- Audio input: ReSpeaker Lite via PipeWire device index 3 (Pi) / MacBook mic device 1 (Mac)
- Audio output: bcm2835 Headphones device 0 (Pi) / system default (Mac)

## LED System
- Arduino Nanos receive simple serial commands from Pi
- Pi decides all logic, Nano executes
- Head Nano (Arduino Uno, /dev/ttyACM2): 82 WS2812B NeoPixels
  - Pixels 0-1: eyes with natural blink state machine
  - Pixels 2-81: mouth trapezoid PCB, emotion-based center-outward pulse
  - Commands: SPEAK:{emotion}, SPEAK_LEVEL:{0-255}, SPEAK_STOP, IDLE, ACTIVE, EYE:{r,g,b}, OFF, SLEEP
- Chest Nano (Arduino Nano, /dev/ttyUSB0): 98 WS2811 LEDs
  - Default pattern: RandomBlocks2 (random colored blocks and bars)
  - Commands: STARTUP, IDLE, ACTIVE, SPEAK:{emotion}, SLEEP, OFF, NEXT
  - Emotion colors: excited=red, sad=blue, angry=rapid flash, happy=confetti

## Vision
- OpenCV captures frame at wake word detection moment
- Camera pose preparation: visor to max, neck to neutral, 0.5s settle before capture
- Frame sent as base64 JPEG to GPT-4o (always cloud, both platforms)
- Vision only on LLM fallthrough — local commands never send images
- detail:low keeps vision cost ~65 tokens per interaction
- Face recognition: dlib ResNet generates 128-dim encodings, stored in SQLite

## Face Recognition
- dlib ResNet model generates 128-dimension face encodings
- SQLite database at assets/memory/faces.db — people, encodings, visit counts, first/last seen
- On wake: captures frame, checks DB
  - Known person: roast by name + appearance via GPT-4o, referencing visit count
  - Unknown person: random unknown-face roast line → ask name → enroll face → roast as new person
- "who am I" / "what's my name" → face DB lookup → GPT-4o roast using name + appearance
- Voice commands: call me [name], forget me, rename me, my name is [name]
- Required model files (downloaded by setup_assets.py):
  - assets/models/shape_predictor_68_face_landmarks.dat (95MB)
  - assets/models/dlib_face_recognition_resnet_model_v1.dat (21MB)
  - assets/models/mmod_human_face_detector.dat

## Current Status

### Completed
- [x] Wake word detection (4 models)
- [x] Two-phase transcription silence detection
- [x] Whisper API transcription with hallucination filtering
- [x] Local mlx-whisper transcription on macOS Apple Silicon
- [x] Command parser (exact + prefix + fuzzy matching, semantic exclusions)
- [x] Multiple response variations per command (5 variations, anti-repeat shuffle)
- [x] ChatGPT gpt-4o-mini streaming responses with Rex roaster personality
- [x] Local Ollama llama3.2 on macOS Apple Silicon
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
- [x] Speech reactive servo movement with emotion states
- [x] Startup animation — neck looks around, head rises
- [x] Shutdown animation — gradual slumped pose
- [x] Startup audio: light_speed.mp3 + Roger Control.mp3 intro
- [x] Shutdown audio: hyperdrive_down.mp3 concurrent with animation
- [x] Idle audio clips with mouth LED and servo sync
- [x] Mouth LED emotion-based center-out pulse animation
- [x] Eye LEDs with natural random blink timing
- [x] Chest LED Arduino sketch with emotion-based lighting
- [x] ELP 1080p camera installed and working
- [x] Platform detection — automatic backend selection Pi vs Mac
- [x] Fast startup when hardware ports are blank
- [x] setup_assets.py — downloads all required model files
- [x] systemd service with boot retry logic
- [x] PipeWire audio on Debian Trixie
- [x] macOS Apple Silicon development environment

### Pending
- [ ] Head tracking with ELP camera (face position → neck servo)
- [ ] Conversation memory per person (SQLite per-person summary injected into GPT system prompt)
- [ ] udev rules for fixed USB device names on Pi
- [ ] Dance mode (beat-synced servo sequences)
- [ ] Mecanum wheel base (future — JGB37-520 motors, BTS7960 drivers, Arduino Mega)
- [ ] Local TTS voice cloning (revisit when MLX TTS matures — currently too slow on M1)