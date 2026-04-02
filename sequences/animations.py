"""
sequences/animations.py — Multi-system choreographed sequences for DJ-R3X.

Responsibilities:
- Define and execute named sequences that coordinate servo positions,
  LED patterns, and audio playback simultaneously using timed keyframes
- Built-in sequences:
    * startup      : boot light sweep, servo home → neutral, welcome audio
    * shutdown     : farewell audio, servos to home, LEDs fade to off
    * idle_enter   : transition servos to neutral, begin breathing LED pulse
    * active_enter : perk up head, flash LEDs, optional activation sound
    * excited      : rapid arm wave, bright LED flash, enthusiastic audio sting
    * sad          : head droops, dim warm LEDs, dejected audio sting
    * listening    : subtle visor raise, LED color shift to indicate attention
- Each sequence is a list of timestamped steps; the player dispatches them
  to ServoController and LEDController with the correct timing offsets
- Sequences can be interrupted cleanly if the state machine transitions
  before a sequence completes
"""
