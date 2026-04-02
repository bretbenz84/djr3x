"""
speech/wake_word.py — Wake word detection using OpenWakeWord.

Responsibilities:
- Load the pre-trained wake word model (already trained for DJ-R3X / "Hey Rex")
- Accept a continuous stream of audio frames from the recorder
- Fire a callback when the wake word is detected above the confidence threshold
- Run in a background thread so it does not block the main loop
- Suppress false positives during active speech playback (echo suppression)
"""
