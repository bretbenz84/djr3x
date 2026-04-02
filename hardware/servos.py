"""
hardware/servos.py — Pololu Maestro Mini servo control for DJ-R3X.

Responsibilities:
- Open serial connection to the Maestro Mini at the configured port/baud
- Send Pololu compact protocol commands to set individual servo positions
- Manage channel assignments: head tilt, head pan, visor, left arm,
  right arm, left hand, right hand (mapped in config)
- Run a background thread for continuous random idle arm/hand movements
  while in ACTIVE state, with per-emotion position range biases:
    * Excitement: head up, faster movement, higher visor
    * Sad: head down, slower movement, lower visor
    * Neutral: centered ranges
- Overlay speech-synchronized head/visor movements during TTS playback
  without interrupting the background random motion thread
- Provide safe home/neutral position method called on shutdown
"""
