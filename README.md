# DJ-R3X Controller

An interactive animatronic controller for a DJ-R3X (Rex) build, running on Raspberry Pi 4.  
Rex responds to voice commands, engages in AI-powered conversation, reacts to music,  
and uses computer vision to greet and interact with people.

## Hardware

| Component | Description |
|-----------|-------------|
| Raspberry Pi 4 | Main controller (user: bbenziger) |
| Pololu Maestro Mini 18 | Servo controller via USB serial (/dev/ttyACM0) |
| Arduino Nano (x2) | LED controllers — chest panel and head (mouth + eyes) |
| ReSpeaker Lite | USB microphone array for wake word and transcription |
| Speakers + Stereo Amp | Audio output via 3.5mm jack |
| Webcam (j5 JVU430) | Computer vision for person detection and greetings |
| Custom Mouth PCB | 80x WS2812B NeoPixels driven by head Arduino Nano |
| Eye PCB | 2x RGB LEDs driven by head Arduino Nano |

## Servo Channels (Pololu Maestro)

| Channel | Name | Notes |
|---------|------|-------|
| 0 | Neck | Left/right rotation |
| 1 | Headlift | Up/down, lower values = up |
| 2 | Headtilt | Forward/back tilt, lower values = up |
| 3 | Visor | Open/close, lower values = open |
| 4 | Elbow | Left arm |
| 5 | Hand | Left hand |
| 6 | Pokerarm | Right arm |
| 7 | Heroarm | Right hand |

## Architecture

### Audio Pipeline
- Wake word detected by OpenWakeWord (two models: `Dee-Jay_Rex` and `Hey_DJ_Rex`)
- Audio captured via ReSpeaker Lite
- Speech transcribed locally via OpenAI Whisper API
- Whisper hallucination filtering applied
- Command parser checks transcription against predefined command list (exact + fuzzy match)
- If matched: execute local command or play cached response
- If no match: stream to ChatGPT → ElevenLabs → speakers
- Mouth PCB brightness driven by speech audio RMS level in real time

### Vision Pipeline
- Vision intent detection on transcribed text
- If visual query detected: capture fresh frame from webcam → send to GPT-4o with query
- Personalized wake greeting: 40% chance Rex takes photo and roasts the person
- Two-step greeter: GPT-4o describes person → GPT-4o generates Rex-style greeting

### Servo Behavior
- Background thread: random slow idle movements (neck, headlift, visor)
- Arms move randomly and independently during idle
- Speech reactive: head/visor move based on audio RMS level during speech
- Emotion states bias servo ranges (excited = head up/visor open, sad = head down/visor closed)

### LED System
- Arduino Nanos receive ASCII serial commands from Pi
- Chest Nano: chest light panel effects
- Head Nano: mouth NeoPixels (80x WS2812B) + eye RGB LEDs
- Mouth brightness driven by speech RMS — music never triggers mouth LEDs

### State Machine

| State | Behavior |
|-------|----------|
| IDLE | Slow idle servo movements, music clips, wake word listening |
| ACTIVE | Full pipeline — transcribe, parse, respond, animate |
| SHUTDOWN | Servo home, LEDs off, optional OS shutdown |

## Software Stack

| Component | Technology |
|-----------|------------|
| Wake word | OpenWakeWord |
| Transcription | OpenAI Whisper API |
| LLM | OpenAI GPT-4o-mini (text) / GPT-4o (vision) |
| Voice synthesis | ElevenLabs streaming TTS |
| Servo control | Pololu compact serial protocol |
| LED control | FastLED via Arduino Nano serial |
| Audio | PipeWire / sounddevice |
| Vision | OpenCV + GPT-4o |

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

## Environment Variables (.env)

```bash
# OpenAI
OPENAI_API_KEY=your_key_here

# ElevenLabs
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Audio devices
AUDIO_INPUT_DEVICE=2
AUDIO_OUTPUT_DEVICE=0

# Wake word models
WAKE_WORD_MODEL_1=assets/models/Dee-Jay_Rex.onnx
WAKE_WORD_MODEL_2=assets/models/Hey_DJ_Rex.onnx
WAKE_WORD_THRESHOLD=0.60

# Hardware ports
MAESTRO_PORT=/dev/ttyACM0
NANO_CHEST_PORT=/dev/ttyUSB1
NANO_HEAD_PORT=/dev/ttyUSB2

# Feature flags
ENABLE_OS_SHUTDOWN=false
SERVO_SAFE_MODE=true
GREETER_PROBABILITY=0.4
```

## Setup

### Requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run manually
```bash
cd ~/djr3x
source venv/bin/activate
python3 main.py
```

### Run as a service
```bash
sudo systemctl enable djr3x
sudo systemctl start djr3x
```

### Monitor logs
```bash
journalctl -u djr3x -f
```

## Current Status

- [x] Wake word detection (dual model)
- [x] Whisper transcription with hallucination filtering
- [x] Command parser (exact + fuzzy matching)
- [x] ChatGPT streaming responses with Rex personality
- [x] ElevenLabs voice synthesis
- [x] Computer vision — intent based photo capture
- [x] Personalized wake greeting (roasts people 40% of the time)
- [x] Servo idle animations (neck, headlift, visor, arms)
- [x] Speech reactive servo movement
- [x] Emotion states (excited, sad, neutral)
- [x] Startup and shutdown animations
- [x] Idle music playback
- [x] Mouth LED RMS sync (hardware/leds.py ready, Arduino sketch pending)
- [x] Arduino Nano sketches for mouth and chest LEDs
- [ ] udev rules for fixed USB device names
- [x] Physical hardware fully installed in Rex body
