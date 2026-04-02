"""
hardware/leds.py — Arduino Nano LED control for DJ-R3X.

Responsibilities:
- Open serial connection(s) to one or more Arduino Nanos at configured
  port(s)/baud rate
- Send simple ASCII or binary serial command strings; Nanos are dumb
  executors — all logic lives here on the Pi
- Commands include: set pattern (idle pulse, active, rainbow, etc.),
  set brightness (0–255), set color (R,G,B), trigger one-shot animations
- Drive the custom mouth PCB brightness in real time from the audio RMS
  level provided by the AudioPlayer during speech playback
- Support named LED zones if multiple Nanos control different body regions
- Provide a safe off/idle state method called on shutdown
"""
