"""
audio/recorder.py — Microphone capture for DJ-R3X.

Responsibilities:
- Open the USB microphone via PipeWire/PyAudio at the configured sample
  rate and chunk size required by both OpenWakeWord and Vosk
- Provide a continuous audio stream to the wake word detector while in
  IDLE state
- On activation, capture a bounded audio clip (silence-gated or
  fixed-duration) and return it as a PCM buffer for transcription
- Handle device enumeration and reconnection if the mic is unplugged
"""
