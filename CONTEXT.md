# DJ-R3X Controller Project

## Hardware
- Raspberry Pi 4 (username: bbenziger, project path: /home/bbenziger/djr3x)
- Pololu Maestro Mini — servo control via serial
- Arduino Nanos — LED control via serial commands
- Custom mouth PCB — lights up with speech audio level
- USB microphone
- Speakers connected via 3.5mm jack through stereo amp

## Environment
- OS: Debian GNU/Linux 13 (Trixie)
- Kernel: 6.12.75 aarch64
- Python: 3.13.5
- Virtual environment: /home/bbenziger/djr3x/venv
- Audio system: PipeWire 1.4.2
- API keys in .env (excluded from git)
- Dependencies in requirements.txt

## Architecture Decisions
- OpenAI Whisper for speech transcription
- Exact phrase matching for local commands (no API needed)
- ChatGPT 4o-mini API fallback for unmatched/open ended input
- Streaming ChatGPT → ElevenLabs to reduce latency
- ElevenLabs basic voice clone based on Star Tours Rex audio
- Mouth LED brightness driven by speech audio level in software
- Pi controls all servos via serial to Maestro (no onboard scripts)
- Nanos act as dumb executors, Pi sends serial LED commands
- Two audio channels mixed passively — speech and music separate

## State Machine
- IDLE: slow breathing LED pulse, listening for wake word
- ACTIVE: full interactivity, LLM responses, reactive movement
- SHUTDOWN: triggered by voice command or physical button

## Servo Behaviors
- Background thread: random arm/hand movements continuously
- Speech thread: head/visor movement overlaid during speech
- Emotion states bias servo position ranges:
  - Excitement: head up, faster movement, higher visor
  - Sad: head down, slower movement, lower visor
  - Neutral: centered ranges

## Audio Pipeline
- Wake word detected (already trained with OpenWakeWord)
- Short audio captured
- Vosk transcribes locally
- Command parser checks against predefined command list
- If matched: execute locally, play cached response audio
- If no match: stream to ChatGPT → ElevenLabs → speakers
- Mouth PCB brightness driven by speech audio buffer level
- Audio input: ReSpeaker Lite via PipeWire device index 3 (NOT direct ALSA hw:4,0)
- Audio output: bcm2835 Headphones device index 0
- PipeWire is the audio manager — all audio must go through it

## LED System
- Arduino Nanos receive simple serial commands from Pi
- Pi decides all logic, Nano

## Vision
- OpenCV captures 640x480 frame at wake word detection moment
- Frame sent as base64 JPEG to gpt-4o (auto-upgrades from gpt-4o-mini when image present)
- Vision only on LLM fallthrough — local commands never send images
- detail:low keeps vision cost ~65 tokens per interaction

# Speech
- Whisper API (whisper-1) for speech transcription — replaces Vosk
- Same silence gating logic for recording, audio sent as WAV to OpenAI
- WHISPER_LANGUAGE=en skips language detection for ~200ms speedup

## Current Status

### Completed
- [x] Wake word detection (4 models: Dee-Jay_Rex, Hey_DJ_Rex, Hey_rex, Yo_robot)
- [x] Whisper API transcription with hallucination filtering
- [x] Command parser (exact + fuzzy + prefix matching, 0.82 threshold)
- [x] ChatGPT gpt-4o-mini streaming responses with Rex roaster personality
- [x] ElevenLabs voice synthesis with volume gain
- [x] Computer vision - intent based photo capture with gpt-4o
- [x] Personalized wake greeting - alternating hi-there and GPT-4o roast
- [x] Face recognition database (dlib ResNet, SQLite)
- [x] Known person greetings with visit counter and roast tiers
- [x] Name enrollment with natural language extraction
- [x] Rename and forget me voice commands
- [x] Cancel/nevermind command returns to IDLE
- [x] Servo idle animations (neck, headlift, visor, arms)
- [x] Speech reactive servo movement with emotion states
- [x] Startup animation with neck looking around
- [x] Shutdown animation with slumped pose
- [x] Startup audio: light_speed.mp3 + Roger Control.mp3 intro
- [x] Shutdown audio: hyperdrive_down.mp3 concurrent with animation
- [x] Idle music playback with mouth LED and servo sync
- [x] Mouth LED emotion-based center-out pulse animation
- [x] Eye LEDs with natural random blink timing
- [x] systemd service with boot retry logic
- [x] PipeWire audio on Debian Trixie
- [x] ReSpeaker Lite mic with stereo mixdown

### Hardware Connected
- [x] Pololu Maestro Mini 18 (/dev/ttyACM0)
- [x] Arduino Uno head LEDs (/dev/ttyACM2) - 82 NeoPixels
- [ ] Arduino chest LEDs (not yet configured)
- [ ] ELP camera (ordered, not yet arrived)

### Pending
- [ ] Face enrollment fix - frame capture timing
- [ ] Head tracking with ELP camera
- [ ] Conversation memory per person
- [ ] Chest Nano Arduino sketch
- [ ] udev rules for fixed USB device names