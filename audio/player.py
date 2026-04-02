"""
audio/player.py — Audio playback for DJ-R3X.

Responsibilities:
- Play cached .wav/.mp3 response files from assets/audio/
- Play background music on the music channel (separate from speech channel)
- Accept a raw PCM audio stream (from ElevenLabs TTS) and push it to the
  speech output device in real time
- Expose a per-frame RMS level so the mouth PCB brightness can be driven
  by the current speech audio buffer level
- Support stop/pause/resume so the state machine can interrupt playback
  on shutdown or wake word detection during music
"""
