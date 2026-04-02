"""
speech/synthesizer.py — Text-to-speech via ElevenLabs (streaming).

Responsibilities:
- Accept a text string (or streaming text chunks from ChatGPT) and call
  the ElevenLabs API using the Star Tours Rex voice clone
- Stream the returned audio PCM directly to the AudioPlayer to minimize
  end-to-end latency (text arrives → audio starts playing ASAP)
- Yield audio chunks to the caller so mouth LED brightness can be updated
  in real time from the buffer level
- Cache completed responses to assets/audio/ keyed by content hash so
  repeated phrases skip the API call entirely
- Handle API errors and rate limits with graceful fallback messaging
"""
