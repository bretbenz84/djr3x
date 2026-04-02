"""
main.py — Entry point for the DJ-R3X controller.

Responsibilities:
- Initialize all subsystems (audio, speech, hardware, state machine)
- Start the main event loop
- Wire together the audio pipeline: wake word → transcription →
  command parsing or LLM fallback → TTS → servo/LED reactions
- Handle graceful shutdown on voice command, button press, or SIGINT
"""
