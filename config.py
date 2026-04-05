"""
config.py — Configuration loader for DJ-R3X.

Loads .env and exposes all project-wide constants. No other module reads
.env directly — import what you need from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT       = Path(__file__).parent.resolve()
ASSETS_DIR         = PROJECT_ROOT / "assets"
AUDIO_CACHE_DIR    = ASSETS_DIR / "audio"
MODELS_DIR         = ASSETS_DIR / "models"
STARTUP_CHIME_PATH = ASSETS_DIR / "audio" / "startup_chime.mp3"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

def _require(key: str) -> str:
    """Return env var value or raise at startup with a clear message."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set in .env")
    return val

def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _optional_int(key: str, default: int | None = None) -> int | None:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return int(val)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

OPENAI_API_KEY      = _require("OPENAI_API_KEY")
ELEVENLABS_API_KEY  = _require("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _require("ELEVENLABS_VOICE_ID")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

OPENAI_MODEL        = "gpt-4o-mini"
MAX_HISTORY_TURNS   = 6      # number of user/assistant pairs kept in context

# ---------------------------------------------------------------------------
# Serial — Pololu Maestro Mini
# ---------------------------------------------------------------------------

MAESTRO_PORT = _optional("MAESTRO_PORT", "/dev/ttyACM0")
MAESTRO_BAUD = 9600          # Pololu default; must match Maestro USB settings

# ---------------------------------------------------------------------------
# Serial — Arduino Nano LED controller(s)
# ---------------------------------------------------------------------------

NANO_CHEST_PORT = _optional("NANO_CHEST_PORT", "/dev/ttyUSB0")
NANO_HEAD_PORT  = _optional("NANO_HEAD_PORT",  "/dev/ttyUSB1")
LED_NANO_BAUD   = 9600

# ---------------------------------------------------------------------------
# Servo channel definitions (Pololu Maestro, 0-based channel numbers)
#
# All position values are in quarter-microseconds (qµs).
# Maestro GUI shows microseconds — multiply by 4 for the serial protocol.
# ---------------------------------------------------------------------------

SERVO_CHANNELS: dict[int, dict] = {
    0: {"name": "neck",     "min": 1984,  "max": 9984,  "acceleration": 15, "neutral": 6000},
    1: {"name": "headlift", "min": 1984,  "max": 7744,  "acceleration": 20, "neutral": 6000},
    2: {"name": "headtilt", "min": 4032,  "max": 5824,  "acceleration": 25, "neutral": 5824},
    3: {"name": "visor",    "min": 4544,  "max": 6976,  "acceleration": 10, "neutral": 6000},
    4: {"name": "elbow",    "min": 3968,  "max": 8000,  "acceleration": 10, "neutral": 6000},
    5: {"name": "hand",     "min": 3968,  "max": 8000,  "acceleration":  5, "neutral": 6000},
    6: {"name": "pokerarm", "min": 3968,  "max": 8000,  "acceleration":  4, "neutral": 6000},
    7: {"name": "heroarm",  "min": 3968,  "max": 8000,  "acceleration":  4, "neutral": 6000},
}

# Channel groups used by the servo controller
HEAD_CHANNELS = [0, 1, 2, 3]   # neck, headlift, headtilt, visor
ARM_CHANNELS  = [4, 5, 6, 7]   # elbow, hand, pokerarm, heroarm

# Emotion bias: each emotion overrides (min, max) for affected channels (qµs).
# Only channels that differ from SERVO_CHANNELS limits need to be listed.
SERVO_EMOTION_LIMITS = {
    "excited": {
        0: (5500, 7500),   # neck up
        3: (5500, 6976),   # visor high
    },
    "sad": {
        0: (4500, 5500),   # neck down
        3: (4544, 5500),   # visor low
    },
    "neutral": {},  # use default SERVO_CHANNELS limits
}

# Random idle motion timing (seconds)
SERVO_IDLE_MOVE_INTERVAL_MIN = 1.5   # minimum time between random moves
SERVO_IDLE_MOVE_INTERVAL_MAX = 4.0   # maximum time between random moves

# Maestro move speed (0 = unlimited; units = (0.25µs) / (10ms))
SERVO_DEFAULT_SPEED        = 20
SERVO_EXCITED_SPEED        = 40
SERVO_SAD_SPEED            = 8

# ---------------------------------------------------------------------------
# Audio — recording / wake word
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE   = 16000   # Hz — required by OpenWakeWord; also used for mic capture
SPEECH_SAMPLE_RATE  = int(_optional("SPEECH_SAMPLE_RATE", "16000"))  # Hz — TTS output; ReSpeaker Lite only supports 16000
AUDIO_CHANNELS      = 1       # mono mic input (legacy alias — prefer AUDIO_INPUT_CHANNELS)
AUDIO_INPUT_CHANNELS  = 1     # microphone input channels (mono)
AUDIO_OUTPUT_CHANNELS = 2     # speaker output channels (stereo — ReSpeaker Lite requires 2)
AUDIO_CHUNK_SIZE   = 1024    # frames per buffer read
AUDIO_FORMAT       = 8       # pyaudio.paInt16 == 8 (avoids importing pyaudio here)

AUDIO_INPUT_DEVICE  = _optional_int("AUDIO_INPUT_DEVICE")   # None → system default
AUDIO_OUTPUT_DEVICE = _optional_int("AUDIO_OUTPUT_DEVICE")  # None → system default

# Software volume: 0.0 (mute) – 1.0 (full). ReSpeaker Lite has no hardware
# mixer, so all output paths scale samples by this factor before writing.
AUDIO_VOLUME: float = max(0.0, min(1.0, float(_optional("AUDIO_VOLUME", "0.5"))))

# Silence-gated recording: stop capturing when RMS drops below threshold
# for SILENCE_DURATION consecutive seconds (or MAX_RECORD_SECONDS elapses)
SILENCE_THRESHOLD            = 300   # RMS amplitude (0–32767) — legacy, kept for reference
TRANSCRIBE_SPEECH_THRESHOLD  = 500  # RMS threshold to detect speech start (raise if hallucinating on noise)
TRANSCRIBE_MIN_SPEECH_CHUNKS = 3    # consecutive chunks above threshold required before speech is confirmed
SILENCE_DURATION             = 1.2  # seconds of silence to end capture
MAX_RECORD_SECONDS           = 12.0 # hard cap on a single utterance

# ---------------------------------------------------------------------------
# Audio — mouth LED brightness
# ---------------------------------------------------------------------------

MOUTH_LED_SMOOTHING = 0.25   # exponential smoothing factor (0=frozen, 1=raw)
MOUTH_LED_GAIN      = 3.0    # multiplier applied to normalised RMS before
                             # mapping to 0–255 brightness

# ---------------------------------------------------------------------------
# Speech — wake word (OpenWakeWord)
# ---------------------------------------------------------------------------

# Paths to the two .onnx wake word model files.  Override in .env if needed.
# Put custom-trained models in assets/models/ and point these at them.
WAKE_WORD_MODEL_1 = Path(
    _optional("WAKE_WORD_MODEL_1",
              str(MODELS_DIR / "hey_rex.onnx"))
)
WAKE_WORD_MODEL_2 = Path(
    _optional("WAKE_WORD_MODEL_2",
              str(MODELS_DIR / "hey_r3x.onnx"))
)

WAKE_WORD_THRESHOLD = 0.6    # minimum score (0–1) to count as a detection
WAKE_WORD_COOLDOWN  = 2.0    # seconds to ignore further detections after one fires

# OpenWakeWord requires exactly 1280 samples (80 ms at 16 kHz) per predict() call.
WAKE_WORD_CHUNK_SIZE = 1280

# ---------------------------------------------------------------------------
# Speech — transcription (OpenAI Whisper)
# ---------------------------------------------------------------------------

# BCP-47 language hint passed to whisper-1.  "en" skips language detection
# and makes the first API call ~200 ms faster.  Set to "" to auto-detect.
WHISPER_LANGUAGE = _optional("WHISPER_LANGUAGE", "en")

# Minimum number of words in a Whisper result to be treated as real speech.
# Results shorter than this are discarded as likely hallucinations.
WHISPER_MIN_WORDS = 2

# Whisper-specific recording limits (tighter than the generic caps in the
# "recording / wake word" section above — shorter recording = lower latency).
WHISPER_MAX_RECORD_SECONDS = float(_optional("WHISPER_MAX_RECORD_SECONDS", "8.0"))
WHISPER_SILENCE_DURATION   = float(_optional("WHISPER_SILENCE_DURATION",   "0.4"))

# Substrings (lowercase) that identify known Whisper hallucination phrases.
# Any result whose lowercased text contains one of these is silently dropped.
WHISPER_HALLUCINATION_FILTER: list[str] = [
    "pissedconsumer",
    "please see review",
    "cc by",
    "subscribe",
    "like and subscribe",
    "work of fiction",
    "living or dead",
    "purely coincidental",
    "thank you for watching",
    "thanks for watching",
    "financial advice",
    "not investment advice",
    "derivative work",
    "touhou project",
    "no relation to the original",
]

# ---------------------------------------------------------------------------
# Speech — synthesis (ElevenLabs)
# ---------------------------------------------------------------------------

ELEVENLABS_MODEL_ID   = "eleven_turbo_v2"   # lowest-latency streaming model
ELEVENLABS_STABILITY  = 0.45
ELEVENLABS_SIMILARITY = 0.80

# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

# Minimum difflib SequenceMatcher ratio (0–1) for a fuzzy phrase match to be
# accepted. 0.72 catches one-word transcription errors ("louder" → "loudr")
# while rejecting accidental matches between short unrelated phrases.
# Lower = more permissive; raise toward 0.85 if you get false positives.
COMMAND_FUZZY_THRESHOLD = 0.72

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

IDLE_CLIP_INTERVAL_MIN = 6.0   # minimum seconds between idle atmosphere clips
IDLE_CLIP_INTERVAL_MAX = 30.0  # maximum seconds between idle atmosphere clips

ACTIVE_IDLE_TIMEOUT  = 20.0  # legacy — superseded by ACTIVE_TIMEOUT_SECONDS
ACTIVE_TIMEOUT_SECONDS   = 8.0   # seconds of silence after a response before returning to IDLE
WAKE_NO_SPEECH_TIMEOUT   = 5.0   # seconds to wait for first speech after wake word
WAKE_GOODBYE_TIMEOUT     = 4.0   # seconds to wait after "are you there?" before saying goodbye

# ---------------------------------------------------------------------------
# Vision — webcam capture
# ---------------------------------------------------------------------------

CAMERA_DEVICE_INDEX  = int(_optional("CAMERA_DEVICE_INDEX", "0"))
VISION_JPEG_QUALITY  = int(_optional("VISION_JPEG_QUALITY", "85"))

# Set to True only on the production Pi so the "shutdown" voice command and
# the physical shutdown button actually halt the OS.  Leave False during
# development — Ctrl-C (and even the voice command) will exit Python cleanly
# but will NOT run `sudo shutdown -h now`.
ENABLE_OS_SHUTDOWN: bool = _optional("ENABLE_OS_SHUTDOWN", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# LED serial command strings sent to Arduino Nanos
# Nanos are dumb executors; all logic lives on the Pi.
# ---------------------------------------------------------------------------

LED_CMD_IDLE      = "IDLE\n"      # slow breathing pulse
LED_CMD_ACTIVE    = "ACTIVE\n"    # full-brightness steady
LED_CMD_LISTENING = "LISTEN\n"    # color shift indicating attention
LED_CMD_SPEAKING  = "SPEAK\n"     # driven per-frame by audio level
LED_CMD_OFF       = "OFF\n"

# Parameterised commands — format() before sending
LED_CMD_BRIGHTNESS = "BRIGHT:{}\n"      # 0–255 integer
LED_CMD_EYE_COLOR  = "EYE:{},{},{}\n"   # R,G,B integers 0–255
