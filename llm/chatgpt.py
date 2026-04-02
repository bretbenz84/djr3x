"""
llm/chatgpt.py — Streaming ChatGPT integration for DJ-R3X.

Responsibilities:
- Maintain a system prompt that establishes DJ-R3X's Star Tours persona:
  enthusiastic droid DJ, speaks in short punchy sentences, occasional
  beeps/droid sounds in text form, knows Star Wars lore
- Accept a user utterance and stream the response token-by-token via the
  OpenAI ChatCompletion streaming API (gpt-4o-mini)
- Yield text chunks to the caller so the Synthesizer can begin TTS before
  the full response is complete (pipeline overlap for low latency)
- Maintain a short rolling conversation history for context continuity
  within an ACTIVE session; clear history on return to IDLE
- Handle API errors, timeouts, and token budget limits gracefully
"""
