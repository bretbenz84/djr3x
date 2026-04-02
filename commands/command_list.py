"""
commands/command_list.py — Predefined commands and canned responses for DJ-R3X.

Responsibilities:
- Define every recognized voice command as a trigger phrase (or set of
  synonyms) mapped to:
    * A canned response string (spoken via TTS or played from cache)
    * An optional hardware action key (servo animation, LED sequence)
    * An optional audio file path in assets/audio/ for pre-rendered replies
- Categories include: greetings, Star Wars / Star Tours lore, DJ requests,
  "what song is this", system commands (volume, shutdown, sleep), and
  character-specific catchphrases
- This file is the single source of truth for all offline-capable responses;
  adding a new command here is enough to make it work without touching other modules
"""
