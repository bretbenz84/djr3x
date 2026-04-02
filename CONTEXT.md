# DJ-R3X Controller Project

## Hardware
- Raspberry Pi 4 (username: bbenziger, project path: /home/bbenziger/djr3x)
- Pololu Maestro Mini — servo control via serial
- Arduino Nanos — LED control via serial commands
- Custom mouth PCB — lights up with speech audio level
- USB microphone (bringing from work tomorrow)
- Speakers connected via 3.5mm jack through stereo amp
- Two audio sources mixed via passive resistor mixer (speech + music)

## Environment
- OS: Debian GNU/Linux 13 (Trixie)
- Kernel: 6.12.75 aarch64
- Python: 3.13.5
- Virtual environment: /home/bbenziger/djr3x/venv
- Audio system: PipeWire 1.4.2
- API keys in .env (excluded from git)
- Dependencies in requirements.txt

## Architecture Decisions
- Vosk for local speech transcription (fast, no API call)
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

## LED System
- Arduino Nanos receive simple serial commands from Pi
- Pi decides all logic, Nano

## VIBE CODING DECISIONS MADE BY CLAUDE
Here's what the file covers and why each decision was made:
Env loading
* _require() / _optional() / _optional_int() helpers — fail fast at startup with a clear message for missing required keys rather than a cryptic None error deep in the code. MIC/SPEAKER device index default to None so PyAudio picks the system default when not set.
Servo constants
* Positions in Pololu quarter-microseconds (6000 = 1500µs neutral). Per-channel SERVO_LIMITS enforce software travel limits. SERVO_EMOTION_LIMITS lets servos.py just call config.SERVO_EMOTION_LIMITS["excited"] and merge/override only the affected channels — no duplicated magic numbers.
Audio constants
* Sample rate hard-coded to 16000 Hz — both Vosk and OpenWakeWord require exactly this. AUDIO_FORMAT = 8 is pyaudio.paInt16's integer value, kept as an int to avoid importing PyAudio here (keeps config importable anywhere).
LED commands
* Simple \n-terminated ASCII strings — matches the "dumb executor" design where the Nano just pattern-matches incoming serial bytes.
Mouth LED
* MOUTH_LED_SMOOTHING + MOUTH_LED_GAIN give leds.py tunable knobs to control how the real-time RMS level maps to brightness without hard-coding the math in two places.

Key decisions worth knowing:
SPEECH_SAMPLE_RATE = 22050 — separate from config.AUDIO_SAMPLE_RATE (16000, for mic input). The synthesizer must request pcm_22050 from ElevenLabs to match.
Speech queue uses _EndMarker(done=threading.Event) — the per-invocation Event is what makes play_file() block correctly. No race: the event belongs to this call only, so a previous sentinel firing can't accidentally unblock it.
stop_speech() drains the queue and fires any pending done events — so play_file() unblocks immediately rather than timing out when the state machine interrupts mid-response.
Music uses an explicit sd.OutputStream — avoids relying on sd.play()'s internal global state, which would conflict with our speech stream. sd.CallbackStop() from inside the callback triggers finished_callback, which unblocks finished.wait() cleanly on both natural end and forced stop.
RMS only computed on filled frames, not zero-padding — without this, padding silence would drag the smoothed level down artificially between TTS chunks, causing the mouth to flicker off mid-word.
Exponential smoothing (MOUTH_LED_SMOOTHING) — prevents harsh LED flicker on plosive consonants. The gain and smoothing constants live in config.py so they can be tuned without touching this file.

Key decisions:

stream() vs convert() — both return Iterator[bytes] in this SDK version. stream() hits the /stream endpoint which starts sending audio immediately; convert() may buffer server-side. stream() with optimize_streaming_latency=4 is the lowest-latency path for complete text.

convert_realtime() for ChatGPT tokens — takes text: Iterator[str] directly, opens a WebSocket and sends tokens as the generator yields them. Audio comes back before the full response is generated. This is the real latency win in the pipeline: ChatGPT token 1 triggers the WebSocket, audio starts arriving by the time ChatGPT produces token 10.

try/finally around the chunk loop — guarantees end_speech() is called even if the ElevenLabs connection drops mid-stream. Without this, the player would stay in "speech active" state and the state machine would never hear the silence timeout.

Cache only on clean exit — the _write_wav_cache call is after the finally block, so a mid-stream error leaves no partial .wav file that would be replayed on the next call.

wait_for_speech(timeout=30.0) in the finally — speak() blocks until audio drains from the player queue, not just until the last HTTP chunk arrives. A very long response at slow network could have significant audio still buffered when the last byte comes down.