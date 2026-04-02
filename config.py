"""
config.py — Configuration loader for DJ-R3X.

Responsibilities:
- Load environment variables from .env (API keys, serial ports, device paths)
- Define project-wide constants: serial baud rates, servo channel mappings,
  audio sample rates, wake word model path, Vosk model path, ElevenLabs
  voice ID, ChatGPT model name, volume levels, etc.
- Expose a single config object or module-level constants consumed by all
  other modules — no other module reads .env directly.
"""
