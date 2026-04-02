"""
states/state_machine.py — State machine for DJ-R3X.

States:
- IDLE     : slow breathing LED pulse on all zones, wake word listener active,
             music playback optional, servos at neutral, no LLM calls
- ACTIVE   : wake word detected; full interactivity enabled — transcription,
             command parsing, LLM fallback, TTS, reactive servo movement,
             mouth LED driven by audio level; returns to IDLE after silence
             timeout or explicit "go to sleep" command
- SHUTDOWN : triggered by voice command ("shut down", "goodbye") or physical
             button press; plays farewell sequence, moves servos to home
             position, fades LEDs, closes serial ports, exits cleanly

Responsibilities:
- Own the current state enum and enforce valid transitions
- Call into sequences/animations.py for entry/exit choreography per state
- Notify registered subsystems of state changes via callbacks or events
- Be the single authority that other modules query for current state
"""
