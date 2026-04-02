"""
speech/transcriber.py — Local speech-to-text using Vosk.

Responsibilities:
- Load the Vosk model from the path specified in config
- Accept a PCM audio buffer captured after wake word detection
- Return the transcribed text string synchronously (fast, no API call)
- Handle partial/final result modes for low-latency feedback if needed
- Log transcription confidence for debugging
"""
