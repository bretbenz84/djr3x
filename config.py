"""
config.py — Configuration loader for DJ-R3X.

Loads .env and exposes all project-wide constants. No other module reads
.env directly — import what you need from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from platform_utils import get_platform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT       = Path(__file__).parent.resolve()
ASSETS_DIR         = PROJECT_ROOT / "assets"
AUDIO_DIR          = ASSETS_DIR / "audio"
LEGACY_AUDIO_CACHE_DIR = AUDIO_DIR
AUDIO_CACHE_DIR    = AUDIO_DIR / "cachedspeech"
MODELS_DIR         = ASSETS_DIR / "models"
STARTUP_CHIME_PATH   = AUDIO_DIR / "startup_chime.mp3"
STARTUP_MUSIC_PATH   = AUDIO_DIR / "light_speed.mp3"
STARTUP_INTRO_PATH   = AUDIO_DIR / "Roger Control.mp3"
STARTUP_INTRO_PATH_ALT = AUDIO_DIR / "This is your cap.mp3"
# File that persists which intro clip played last (to avoid back-to-back repeats).
STARTUP_INTRO_LAST_PLAYED = PROJECT_ROOT / ".last_startup_intro"
SHUTDOWN_MUSIC_PATH  = AUDIO_DIR / "hyperdrive_down.mp3"
CANTINA_BAND_PATH    = ASSETS_DIR / "music" / "Cantina Band.mp3"

DANCE_SHORT_DURATION = 12.0   # seconds of dancing before the fade starts
DANCE_FADE_DURATION  = 3.0    # seconds for the music fade-out

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

def _optional_device_source(key: str) -> int | str | None:
    """Return an int device index or string device path from .env."""
    val = os.getenv(key)
    if val is None:
        return None
    stripped = val.strip()
    if not stripped:
        return None
    return int(stripped) if stripped.isdigit() else stripped

# ---------------------------------------------------------------------------
# API Keys / TTS provider selection
# ---------------------------------------------------------------------------

OPENAI_API_KEY      = _require("OPENAI_API_KEY")
OPENAI_TIMEOUT_SECONDS = float(_optional("OPENAI_TIMEOUT_SECONDS", "30"))
ELEVENLABS_TIMEOUT_SECONDS = float(_optional("ELEVENLABS_TIMEOUT_SECONDS", "60"))

TTS_PROVIDER = _optional("TTS_PROVIDER", "elevenlabs").strip().lower()
if TTS_PROVIDER not in {"elevenlabs", "piper", "xtts"}:
    raise EnvironmentError(
        "TTS_PROVIDER must be one of 'elevenlabs', 'piper', or 'xtts' in .env"
    )

ELEVENLABS_API_KEY  = (
    _require("ELEVENLABS_API_KEY") if TTS_PROVIDER == "elevenlabs" else _optional("ELEVENLABS_API_KEY")
)
ELEVENLABS_VOICE_ID = (
    _require("ELEVENLABS_VOICE_ID") if TTS_PROVIDER == "elevenlabs" else _optional("ELEVENLABS_VOICE_ID")
)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

PLATFORM = get_platform()   # 'macos_silicon' or 'pi'


def _platform_default(macos_silicon: str, pi: str) -> str:
    """Return the per-platform default while still allowing .env overrides."""
    return macos_silicon if PLATFORM == "macos_silicon" else pi

# When True, use local mlx-whisper instead of the Whisper API.
USE_LOCAL_TRANSCRIPTION: bool = _optional("USE_LOCAL_TRANSCRIPTION", "").lower() in ("1", "true", "yes") \
    if _optional("USE_LOCAL_TRANSCRIPTION") else (PLATFORM == "macos_silicon")

# When True, use local Ollama instead of OpenAI GPT for text chat.
USE_LOCAL_LLM: bool = _optional("USE_LOCAL_LLM", "").lower() in ("1", "true", "yes") \
    if _optional("USE_LOCAL_LLM") else (PLATFORM == "macos_silicon")

# Local transcription — mlx-whisper (Apple Silicon only)
LOCAL_WHISPER_MODEL: str = _optional("LOCAL_WHISPER_MODEL", "mlx-community/whisper-small-mlx")

# Local LLM — Ollama OpenAI-compatible endpoint
LOCAL_LLM_BASE_URL: str = _optional("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL:    str = _optional("LOCAL_LLM_MODEL",    "llama3.2:1b")

# Logging
THIRD_PARTY_LOG_LEVEL = _optional(
    "THIRD_PARTY_LOG_LEVEL",
    _platform_default("INFO", "WARNING"),
).upper()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

OPENAI_MODEL        = "gpt-4o-mini"
MAX_HISTORY_TURNS   = 6      # number of user/assistant pairs kept in context
OPINION_BASE_CHANCE = float(_optional("OPINION_BASE_CHANCE", "0.50"))
OPINION_REPEAT_BONUS = float(_optional("OPINION_REPEAT_BONUS", "0.08"))
OPINION_INTERACTION_BONUS = float(_optional("OPINION_INTERACTION_BONUS", "0.03"))
OPINION_COOLDOWN_SECONDS = float(_optional("OPINION_COOLDOWN_SECONDS", "90"))
OPINION_MAX_PER_SESSION = int(_optional("OPINION_MAX_PER_SESSION", "3"))
OPINION_KNOWN_WAKE_CHANCE = float(_optional("OPINION_KNOWN_WAKE_CHANCE", "0.78"))
OPINION_SHORT_WAKE_CHANCE = float(_optional("OPINION_SHORT_WAKE_CHANCE", "0.25"))

# ---------------------------------------------------------------------------
# Serial — Pololu Maestro Mini
# ---------------------------------------------------------------------------

MAESTRO_PORT          = _optional("MAESTRO_PORT", "/dev/ttyACM0")
MAESTRO_BAUD          = 9600          # Pololu default; must match Maestro USB settings
MAESTRO_STARTUP_DELAY = int(_optional("MAESTRO_STARTUP_DELAY", "2"))  # seconds to wait after port opens before sending commands

# Serial connection retry settings (applies to Maestro and both Nano ports)
SERIAL_RETRY_ATTEMPTS = 5    # max open attempts before giving up
SERIAL_RETRY_DELAY    = 2.0  # seconds between retry attempts

# ---------------------------------------------------------------------------
# Serial — Arduino Nano LED controller(s)
# ---------------------------------------------------------------------------

NANO_CHEST_PORT = _optional("NANO_CHEST_PORT") or None   # None if unset → skip entirely
NANO_HEAD_PORT  = _optional("NANO_HEAD_PORT",  "/dev/ttyUSB1")
LED_NANO_BAUD   = 115200

# ---------------------------------------------------------------------------
# Servo channel definitions (Pololu Maestro, 0-based channel numbers)
#
# All position values are in quarter-microseconds (qµs).
# Maestro GUI shows microseconds — multiply by 4 for the serial protocol.
# ---------------------------------------------------------------------------

SERVO_CHANNELS: dict[int, dict] = {
    0: {"name": "neck",     "min": 1984,  "max": 9984,  "acceleration": 15, "neutral": 6000},
    1: {"name": "headlift", "min": 1984,  "max": 7744,  "acceleration": 20, "neutral": 6000},
    2: {"name": "headtilt", "min": 3904,  "max": 5504,  "acceleration": 25, "neutral": 4320},
    3: {"name": "visor",    "min": 4544,  "max": 6976,  "acceleration": 10, "neutral": 6000},
    4: {"name": "elbow",    "min": 6300,  "max": 7560,  "acceleration": 10, "neutral": 6720},
    5: {"name": "hand",     "min": 1984,  "max": 9984,  "acceleration":  0, "neutral": 5984},
    6: {"name": "pokerarm", "min": 3968,  "max": 8000,  "acceleration":  4, "neutral": 6000},
    7: {"name": "heroarm",  "min": 3968,  "max": 8000,  "acceleration":  4, "neutral": 6000},
}

# Channel groups used by the servo controller
HEAD_CHANNELS      = [0, 1, 2, 3]   # neck, headlift, headtilt, visor — full head group
IDLE_HEAD_CHANNELS = [0, 1]         # retained for compatibility; head search/tracking own these axes now
ARM_CHANNELS  = [4, 5, 6, 7]   # elbow, hand, pokerarm, heroarm

# Channel number aliases — used by sequences/animations.py for readability
SERVO_HEAD_PAN   = 0   # neck rotation
SERVO_HEAD_LIFT  = 1   # headlift
SERVO_HEAD_TILT  = 2   # headtilt
SERVO_VISOR      = 3   # visor open/close
SERVO_ARM_LEFT   = 4   # elbow
SERVO_HAND_LEFT  = 5   # hand
SERVO_ARM_RIGHT  = 6   # pokerarm
SERVO_HAND_RIGHT = 7   # heroarm

# Emotion bias: each emotion overrides (min, max) for affected channels (qµs).
# Only channels that differ from SERVO_CHANNELS limits need to be listed.
SERVO_EMOTION_LIMITS = {
    "excited": {
        0: (5500, 7500),   # neck up
        2: (3904, 4200),   # headtilt up (low = up; constrain to upper range)
        3: (6000, 6976),   # visor open (high = open; constrain to open range)
    },
    "sad": {
        0: (4500, 5500),   # neck down
        2: (4400, 5504),   # headtilt down (high = down; constrain to lower range)
        3: (4544, 5300),   # visor drooped/closed (low = closed; constrain to closed range)
    },
    "neutral": {},  # use default SERVO_CHANNELS limits
}

# Initial (slumped/shutdown) positions written instantly at boot so the
# Maestro knows where each servo starts without physically moving it.
# Channels not listed here default to their neutral position.
# These must match the final step of the SHUTDOWN animation exactly.
SERVO_SLUMPED_POSITIONS: dict[int, int] = {
    1: SERVO_CHANNELS[1]["min"],    # headlift fully down (1984)
    2: SERVO_CHANNELS[2]["max"],    # headtilt fully down (5504)
    3: SERVO_CHANNELS[3]["min"],    # visor fully down / eyes covered (4544)
    4: SERVO_CHANNELS[4]["min"],    # elbow fully down (6300)
    5: SERVO_CHANNELS[5]["neutral"], # hand at neutral (6000)
    6: SERVO_CHANNELS[6]["min"],    # pokerarm fully down (3968)
    7: SERVO_CHANNELS[7]["min"],    # heroarm fully down (3968)
    # ch 0 (neck) intentionally omitted — neutral (6000) is correct for startup
}

# Random idle motion timing (seconds)
SERVO_IDLE_MOVE_INTERVAL_MIN = 1.5   # minimum time between random moves (pokerarm ch6)
SERVO_IDLE_MOVE_INTERVAL_MAX = 4.0   # maximum time between random moves (pokerarm ch6)

# Headtilt resting position during idle (qµs).  Higher value = head tilted
# slightly downward (ch 2 is inverted: lower = up, higher = down).
# 4320 is neutral/level; 4600 produces a relaxed, slightly-looking-down pose.
IDLE_HEAD_TILT_REST: int = int(_optional("IDLE_HEAD_TILT_REST", "4600"))

# Elbow resting position during idle (qµs).  min=6300 = fully lowered.
# 6400 keeps the arm in a relaxed lowered pose without touching the hard stop.
IDLE_ELBOW_REST: int = int(_optional("IDLE_ELBOW_REST", "6400"))

# Expressive arm idle motion (ch 4 elbow, ch 5 hand, ch 7 heroarm)
ARM_IDLE_RANGE_PERCENT  = float(_optional("ARM_IDLE_RANGE_PERCENT",  "0.70"))  # fraction of min/max span used
ARM_IDLE_INTERVAL_MIN   = float(_optional("ARM_IDLE_INTERVAL_MIN",   "2.0"))   # seconds between moves
ARM_IDLE_INTERVAL_MAX   = float(_optional("ARM_IDLE_INTERVAL_MAX",   "4.0"))

# Arm speech-reactive tuning
SERVO_ELBOW_SPEAK_MULT         = float(_optional("SERVO_ELBOW_SPEAK_MULT",         "1.0"))   # scales elbow gesture size during speech
SERVO_ELBOW_SPEAK_INTERVAL_MIN = float(_optional("SERVO_ELBOW_SPEAK_INTERVAL_MIN", "0.35"))  # seconds to hold an elbow gesture before picking a new one
SERVO_ELBOW_SPEAK_INTERVAL_MAX = float(_optional("SERVO_ELBOW_SPEAK_INTERVAL_MAX", "0.75"))  # longer hold keeps speech arm motion from rocking too quickly
SERVO_HAND_SPEAK_SPEED          = int(_optional("SERVO_HAND_SPEAK_SPEED",          "255"))  # ch 5 max speed — used only for the wake greeting wave
SERVO_HAND_SPEAK_SPEED_RELAXED  = int(_optional("SERVO_HAND_SPEAK_SPEED_RELAXED",   "30"))   # ch 5 speed during normal speech — slow and relaxed
SERVO_NECK_SPEAK_SPEED          = int(_optional("SERVO_NECK_SPEAK_SPEED",           "18"))
SERVO_HEAD_LIFT_SPEAK_SPEED     = int(_optional("SERVO_HEAD_LIFT_SPEAK_SPEED",      "9"))
SERVO_HEAD_TILT_SPEAK_SPEED     = int(_optional("SERVO_HEAD_TILT_SPEAK_SPEED",      "8"))

# Maestro move speed (0 = unlimited; units = (0.25µs) / (10ms))
SERVO_DEFAULT_SPEED        = 20
SERVO_EXCITED_SPEED        = 40
SERVO_SAD_SPEED            = 8
SERVO_STARTUP_SPEED        = 8    # slow speed used during safe-mode homing
SERVO_HEAD_IDLE_SPEED      = 3    # very slow lazy head drift during idle
SERVO_VISOR_IDLE_SPEED     = 2    # barely-perceptible visor scanning during idle
SERVO_NECK_STARTUP_SPEED   = 100  # fast neck speed for startup look-around sweep
SERVO_SLEEP_SPEED          = 4    # extremely slow collapse into sleep — slower than shutdown

# Safe startup mode: home servos one at a time with a 0.5 s delay between
# each channel, preceded by a speed command to ensure a slow controlled move.
# Set SERVO_SAFE_MODE=false in .env once all neutral positions are confirmed.
SERVO_SAFE_MODE: bool = _optional("SERVO_SAFE_MODE", "true").lower() not in ("0", "false", "no")

# ---------------------------------------------------------------------------
# Audio — recording / wake word
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE   = 16000   # Hz — required by OpenWakeWord; also used for mic capture
SPEECH_SAMPLE_RATE  = int(_optional("SPEECH_SAMPLE_RATE", "16000"))  # Hz — TTS output; ReSpeaker Lite only supports 16000
AUDIO_CHANNELS      = 1       # mono mic input (legacy alias — prefer MIC_CHANNELS)
AUDIO_INPUT_CHANNELS  = 1     # microphone input channels (legacy — prefer MIC_CHANNELS)
MIC_CHANNELS          = int(_optional("MIC_CHANNELS", "2"))  # physical mic channels; ReSpeaker Lite requires 2
AUDIO_OUTPUT_CHANNELS = 2     # speaker output channels (stereo — ReSpeaker Lite requires 2)
AUDIO_CHUNK_SIZE   = 1024    # frames per buffer read
AUDIO_FORMAT       = 8       # legacy 16-bit PCM format constant
AUDIO_OUTPUT_BLOCKSIZE = int(
    _optional(
        "AUDIO_OUTPUT_BLOCKSIZE",
        "0" if PLATFORM == "macos_silicon" else str(AUDIO_CHUNK_SIZE),
    )
)
AUDIO_OUTPUT_LATENCY = _optional(
    "AUDIO_OUTPUT_LATENCY",
    "high" if PLATFORM == "macos_silicon" else "low",
)
SPEECH_OUTPUT_BLOCKSIZE = int(
    _optional("SPEECH_OUTPUT_BLOCKSIZE", str(AUDIO_CHUNK_SIZE))
)
SPEECH_OUTPUT_LATENCY = _optional("SPEECH_OUTPUT_LATENCY", "low")

AUDIO_INPUT_DEVICE  = _optional_int("AUDIO_INPUT_DEVICE")   # None → system default
AUDIO_OUTPUT_DEVICE = _optional_int("AUDIO_OUTPUT_DEVICE")  # None → system default
AUDIO_OUTPUT_MODE   = _optional(
    "AUDIO_OUTPUT_MODE",
    "device" if AUDIO_OUTPUT_DEVICE is not None else "default",
).strip().lower()
AUDIO_BLUETOOTH_DEVICE = _optional("AUDIO_BLUETOOTH_DEVICE", "auto").strip()
AUDIO_BLUETOOTH_AUTO_CONNECT: bool = _optional(
    "AUDIO_BLUETOOTH_AUTO_CONNECT", "true"
).lower() not in ("0", "false", "no")
AUDIO_BLUETOOTH_CONNECT_TIMEOUT = float(
    _optional("AUDIO_BLUETOOTH_CONNECT_TIMEOUT", "12.0")
)
AUDIO_BLUETOOTH_DISCOVERY_TIMEOUT = float(
    _optional("AUDIO_BLUETOOTH_DISCOVERY_TIMEOUT", "8.0")
)
AUDIO_BLUETOOTH_PREFER_CONNECTED: bool = _optional(
    "AUDIO_BLUETOOTH_PREFER_CONNECTED", "true"
).lower() not in ("0", "false", "no")

# Software volume: 0.0 (mute) – 1.0 (full). ReSpeaker Lite has no hardware
# mixer, so all output paths scale samples by this factor before writing.
AUDIO_VOLUME: float = max(0.0, min(1.0, float(_optional("AUDIO_VOLUME", "0.5"))))

# TTS gain boost applied to generated PCM chunks in synthesizer.py before
# queuing for playback.  Multiplies int16 samples and clips to [-32768, 32767].
# Values above 1.0 boost volume; 1.0 = no change.
SYNTHESIZER_VOLUME_GAIN: float = float(_optional("SYNTHESIZER_VOLUME_GAIN", "2.0"))

# Speech-only playback effect that adds the DJ-R3X radio/droid character.
# Applied in audio/player.py to live/generated PCM and cached .wav responses
# before they are converted back to int16 for output. Music and .mp3 clips are
# intentionally left untouched.
ENABLE_DROID_EFFECT: bool = _optional("ENABLE_DROID_EFFECT", "true").lower() not in ("0", "false", "no")
DROID_EFFECT_BITCRUSH_ENABLED: bool = _optional("DROID_EFFECT_BITCRUSH_ENABLED", "true").lower() not in ("0", "false", "no")
ENABLE_DROID_TREMOLO: bool = _optional("ENABLE_DROID_TREMOLO", "false").lower() not in ("0", "false", "no")
DROID_TREMOLO_RATE_HZ: float = float(_optional("DROID_TREMOLO_RATE_HZ", "5.6"))
DROID_TREMOLO_DEPTH: float = max(0.0, min(0.25, float(_optional("DROID_TREMOLO_DEPTH", "0.06"))))
ENABLE_DROID_RING_MOD: bool = _optional("ENABLE_DROID_RING_MOD", "true").lower() not in ("0", "false", "no")
DROID_RING_MOD_RATE_HZ: float = float(_optional("DROID_RING_MOD_RATE_HZ", "28.0"))
DROID_RING_MOD_DEPTH: float = max(0.0, min(0.35, float(_optional("DROID_RING_MOD_DEPTH", "0.10"))))

# Silence-gated recording: stop capturing when RMS drops below threshold
# for SILENCE_DURATION consecutive seconds (or MAX_RECORD_SECONDS elapses)
SILENCE_THRESHOLD            = 300   # RMS amplitude (0–32767) — legacy, kept for reference
TRANSCRIBE_SPEECH_THRESHOLD  = 500  # RMS threshold fallback if calibration is skipped
TRANSCRIBE_MIN_SPEECH_CHUNKS = 3    # consecutive chunks above threshold required before speech is confirmed

# Noise floor calibration — run once at startup to adapt threshold to the room.
# threshold = clamp(noise_rms * NOISE_FLOOR_MULTIPLIER,
#                   TRANSCRIBE_SPEECH_THRESHOLD_MIN,
#                   TRANSCRIBE_SPEECH_THRESHOLD_MAX)
TRANSCRIBE_NOISE_FLOOR_DURATION    = float(_optional("TRANSCRIBE_NOISE_FLOOR_DURATION", "0.5"))  # seconds of ambient audio to sample
NOISE_FLOOR_MULTIPLIER             = float(_optional("NOISE_FLOOR_MULTIPLIER",           "2.0"))  # scale factor applied to measured RMS
TRANSCRIBE_SPEECH_THRESHOLD_MIN    = int(_optional("TRANSCRIBE_SPEECH_THRESHOLD_MIN",    "300"))  # floor — avoids cutting out real speech
TRANSCRIBE_SPEECH_THRESHOLD_MAX    = int(_optional("TRANSCRIBE_SPEECH_THRESHOLD_MAX",    "800"))  # ceiling — noisy room can't silence Rex
TRANSCRIBE_MIN_WHISPER_SECONDS     = float(_optional("TRANSCRIBE_MIN_WHISPER_SECONDS",   "0.5"))  # skip normal-path Whisper calls for ultra-short clips
TRANSCRIBE_MIN_VOICED_CHUNKS       = int(_optional("TRANSCRIBE_MIN_VOICED_CHUNKS",       "5"))    # minimum above-threshold chunks before a normal-path clip is worth transcribing
# In Phase 2 (active speech recording), the silence counter is only reset when a chunk's
# RMS exceeds speech_detect_threshold × this multiplier.  This prevents background noise
# and servo/speaker bleedthrough from resetting the silence counter after the user has
# finished speaking.  Servo actuation during the wake wave (≈2.9 s) can hit 600–1000 RMS;
# real speech is typically much stronger than ambient noise, but Raspberry Pi runs with
# live servos and louder room noise often need a lower value so actual speech can still
# reset the countdown after a brief pause.
TRANSCRIBE_SILENCE_RESET_MULTIPLIER = float(
    _optional(
        "TRANSCRIBE_SILENCE_RESET_MULTIPLIER",
        _platform_default("4.0", "1.4"),
    )
)
SILENCE_DURATION             = 0.5  # seconds of silence to end capture
MAX_RECORD_SECONDS           = 12.0 # hard cap on a single utterance

# ---------------------------------------------------------------------------
# Audio — mouth LED brightness
# ---------------------------------------------------------------------------

MOUTH_LED_SMOOTHING = 0.25   # exponential smoothing factor (0=frozen, 1=raw)
MOUTH_LED_GAIN      = 3.0    # multiplier applied to normalised RMS before
                             # mapping to 0–255 brightness
MOUTH_LED_MIN_RMS   = float(_optional("MOUTH_LED_MIN_RMS", "8.0"))
                             # SPEAK_LEVEL:0 is sent for RMS values below this
                             # threshold to prevent ambient pre-glow when audio
                             # hasn't truly started yet

# ---------------------------------------------------------------------------
# Speech — wake word (OpenWakeWord)
# ---------------------------------------------------------------------------

# Paths to the four .onnx wake word model files.  Override in .env if needed.
# Put custom-trained models in assets/models/ and point these at them.
WAKE_WORD_MODEL_1 = Path(
    _optional("WAKE_WORD_MODEL_1",
              str(MODELS_DIR / "Dee-Jay_Rex.onnx"))
)
WAKE_WORD_MODEL_2 = Path(
    _optional("WAKE_WORD_MODEL_2",
              str(MODELS_DIR / "Hey_DJ_Rex.onnx"))
)
WAKE_WORD_MODEL_3 = Path(
    _optional("WAKE_WORD_MODEL_3",
              str(MODELS_DIR / "Hey_rex.onnx"))
)
WAKE_WORD_MODEL_4 = Path(
    _optional("WAKE_WORD_MODEL_4",
              str(MODELS_DIR / "Yo_robot.onnx"))
)

WAKE_WORD_THRESHOLD = 0.6    # minimum score (0–1) to count as a detection
WAKE_WORD_COOLDOWN  = 2.0    # seconds to ignore further detections after one fires

# Optional sleep-mode wake word — only fires when Rex is in SLEEP state.
# If the file does not exist, sleep mode still works via other triggers.
WAKE_SLEEP_MODEL = Path(
    _optional("WAKE_SLEEP_MODEL",
              str(MODELS_DIR / "wakeuprex.onnx"))
)

# OpenWakeWord requires exactly 1280 samples (80 ms at 16 kHz) per predict() call.
WAKE_WORD_CHUNK_SIZE = 1280

# ---------------------------------------------------------------------------
# Speech — transcription (OpenAI Whisper)
# ---------------------------------------------------------------------------

# BCP-47 language hint passed to whisper-1.  "en" skips language detection
# and makes the first API call ~200 ms faster.  Set to "" to auto-detect.
WHISPER_LANGUAGE = _optional("WHISPER_LANGUAGE", "en")

# Whisper-specific recording limits (tighter than the generic caps in the
# "recording / wake word" section above — shorter recording = lower latency).
WHISPER_MAX_RECORD_SECONDS       = float(_optional("WHISPER_MAX_RECORD_SECONDS",       "15.0"))

# Two-phase silence detection in transcriber.py:
#
#   Phase 1 — waiting for speech to begin:
#     Keep listening for up to TRANSCRIBE_SPEECH_WAIT_SECONDS.  No speech → return None.
#
#   Phase 2 — recording active speech:
#     Once speech is confirmed, require TRANSCRIBE_END_SILENCE_SECONDS of
#     sustained silence before stopping.  Any audio above the RMS threshold
#     resets the counter so brief inter-word pauses never cut off an utterance.
TRANSCRIBE_SPEECH_WAIT_SECONDS   = float(_optional("TRANSCRIBE_SPEECH_WAIT_SECONDS",   "5.0"))
TRANSCRIBE_END_SILENCE_SECONDS   = float(
    _optional(
        "TRANSCRIBE_END_SILENCE_SECONDS",
        _platform_default("2.5", "1.0"),
    )
)

# Short mic cooldown after Rex finishes speaking. Prevents the next
# transcription window from immediately re-capturing prompt/greeting tail
# from the speakers or room echo, while staying short enough that a user's
# natural reply is not noticeably delayed.
POST_SPEECH_LISTEN_COOLDOWN_SECONDS = float(
    _optional("POST_SPEECH_LISTEN_COOLDOWN_SECONDS", "0.35")
)

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
    # URLs and transcription service names — always hallucinations
    "https://",
    "http://",
    "www.",
    ".com",
    ".ai",
    ".org",
    "otter.ai",
    "transcribed by",
    # Repeated filler words — classic Whisper hallucination on near-silence audio
    "okay okay okay",
    "okay okay",
    "ok ok ok",
    "ok ok",
    "motivation motivation",
]

# Exact lowercased phrases that Whisper sometimes hallucinates on near-silence.
# These are matched after light punctuation stripping, so "Thank you." and
# "you" can be dropped without blocking normal longer sentences containing
# those words.
WHISPER_HALLUCINATION_EXACT: set[str] = {
    "okay okay okay",
    "thank you",
    "thank you thank you",
    "yeah yeah yeah yeah yeah",
    "you",
}

# ---------------------------------------------------------------------------
# Speech — synthesis
# ---------------------------------------------------------------------------

ELEVENLABS_MODEL_ID   = "eleven_turbo_v2"   # lowest-latency streaming model
ELEVENLABS_STABILITY  = 0.45
ELEVENLABS_SIMILARITY = 0.80

PIPER_MODEL_PATH = Path(
    _optional("PIPER_MODEL_PATH", str(MODELS_DIR / "rexvoice.onnx"))
)
PIPER_CONFIG_PATH = Path(
    _optional("PIPER_CONFIG_PATH", f"{PIPER_MODEL_PATH}.json")
)
PIPER_USE_CUDA: bool = _optional("PIPER_USE_CUDA", "false").lower() in ("1", "true", "yes")
PIPER_SPEAKER_ID = _optional_int("PIPER_SPEAKER_ID")
PIPER_SENTENCE_SILENCE = float(_optional("PIPER_SENTENCE_SILENCE", "0.0"))
PIPER_LENGTH_SCALE = float(_optional("PIPER_LENGTH_SCALE", "1.0"))
PIPER_NOISE_SCALE = float(_optional("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W = float(_optional("PIPER_NOISE_W", "0.8"))

XTTS_MODEL_DIR = Path(
    _optional("XTTS_MODEL_DIR", str(MODELS_DIR / "djrex_xtts"))
)
XTTS_CONFIG_PATH = Path(
    _optional("XTTS_CONFIG_PATH", str(XTTS_MODEL_DIR / "config.json"))
)
XTTS_CHECKPOINT_PATH = Path(
    _optional("XTTS_CHECKPOINT_PATH", str(XTTS_MODEL_DIR / "model.pth"))
)
_default_xtts_vocab = XTTS_MODEL_DIR / "vocab.json"
if not _default_xtts_vocab.exists():
    _default_xtts_vocab_alt = XTTS_MODEL_DIR / "vocab.json_"
    if _default_xtts_vocab_alt.exists():
        _default_xtts_vocab = _default_xtts_vocab_alt
XTTS_VOCAB_PATH = Path(
    _optional("XTTS_VOCAB_PATH", str(_default_xtts_vocab))
)
XTTS_SPEAKER_WAV = Path(
    _optional("XTTS_SPEAKER_WAV", str(PROJECT_ROOT / "reference.wav"))
)
XTTS_LANGUAGE = _optional("XTTS_LANGUAGE", "en").strip() or "en"
XTTS_SPEED = float(_optional("XTTS_SPEED", "1.0"))
XTTS_TEMPERATURE = float(_optional("XTTS_TEMPERATURE", "0.75"))
XTTS_LENGTH_PENALTY = float(_optional("XTTS_LENGTH_PENALTY", "1.0"))
XTTS_REPETITION_PENALTY = float(_optional("XTTS_REPETITION_PENALTY", "5.0"))
XTTS_TOP_K = int(_optional("XTTS_TOP_K", "50"))
XTTS_TOP_P = float(_optional("XTTS_TOP_P", "0.85"))
XTTS_ENABLE_TEXT_SPLITTING: bool = _optional("XTTS_ENABLE_TEXT_SPLITTING", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

# Minimum difflib SequenceMatcher ratio (0–1) for a fuzzy phrase match to be
# accepted. 0.72 catches one-word transcription errors ("louder" → "loudr")
# while rejecting accidental matches between short unrelated phrases.
# Lower = more permissive; raise toward 0.85 if you get false positives.
COMMAND_FUZZY_THRESHOLD = 0.82

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

IDLE_CLIP_INTERVAL_MIN = 10.0   # minimum seconds between idle atmosphere clips
IDLE_CLIP_INTERVAL_MAX = 60.0  # maximum seconds between idle atmosphere clips


ACTIVE_IDLE_TIMEOUT  = 20.0  # legacy — superseded by ACTIVE_TIMEOUT_SECONDS
ACTIVE_TIMEOUT_SECONDS   = 8.0   # seconds of silence after a response before returning to IDLE
WAKE_NO_SPEECH_TIMEOUT   = 8.0   # seconds to wait for first speech after wake word
WAKE_GOODBYE_TIMEOUT     = 6.0   # seconds to wait after "are you there?" before saying goodbye
POST_RESPONSE_LINGER_ATTEMPTS: int = int(_optional("POST_RESPONSE_LINGER_ATTEMPTS", "3"))
POST_RESPONSE_DEEP_QUESTION_ATTEMPTS: int = int(
    _optional("POST_RESPONSE_DEEP_QUESTION_ATTEMPTS", "2")
)
POST_RESPONSE_TOTAL_PROMPT_LIMIT: int = int(
    _optional("POST_RESPONSE_TOTAL_PROMPT_LIMIT", "3")
)
POST_RESPONSE_LINGER_MAX_SECONDS: float = float(_optional("POST_RESPONSE_LINGER_MAX_SECONDS", "18.0"))
POST_RESPONSE_LINGER_LISTEN_TIMEOUT: float = float(_optional("POST_RESPONSE_LINGER_LISTEN_TIMEOUT", "5.0"))
POST_RESPONSE_LINGER_CURIOSITY_CHANCE: float = float(
    _optional("POST_RESPONSE_LINGER_CURIOSITY_CHANCE", "0.35")
)
MEMORY_CALLBACK_COOLDOWN_SECONDS: float = float(
    _optional("MEMORY_CALLBACK_COOLDOWN_SECONDS", "1800.0")
)

# Face-presence wake — triggers a greeting when a face appears after a long absence
FACE_APPEAR_ABSENT_SECONDS: float = 8.0   # no-face gap required before appearance triggers greeting
FACE_APPEAR_FRAME_COUNT:    int   = 2     # consecutive detections required to confirm appearance
FACE_WAKE_LISTEN_TIMEOUT:   float = 10.0  # seconds to wait for speech after face-triggered "Hi There"
FACE_TRIGGERED_GREETING_COOLDOWN_SECONDS: float = float(
    _optional("FACE_TRIGGERED_GREETING_COOLDOWN_SECONDS", "30.0")
)

# Lightweight autonomy layer — self-initiated prompts, timing variance, and
# slight imperfection layered on top of the core reactive pipeline.
AUTONOMY_ENABLED: bool = _optional("AUTONOMY_ENABLED", "true").lower() not in ("0", "false", "no")
AUTONOMY_IDLE_AGENDA_MIN_SECONDS: float = float(_optional("AUTONOMY_IDLE_AGENDA_MIN_SECONDS", "25.0"))
AUTONOMY_IDLE_AGENDA_MAX_SECONDS: float = float(_optional("AUTONOMY_IDLE_AGENDA_MAX_SECONDS", "75.0"))
AUTONOMY_PROACTIVE_LISTEN_TIMEOUT: float = float(_optional("AUTONOMY_PROACTIVE_LISTEN_TIMEOUT", "6.0"))
AUTONOMY_CLARIFICATION_TIMEOUT: float = float(_optional("AUTONOMY_CLARIFICATION_TIMEOUT", "5.0"))
AUTONOMY_MEMORY_STALE_DAYS: float = float(_optional("AUTONOMY_MEMORY_STALE_DAYS", "7.0"))
AUTONOMY_MEMORY_TRIGGER_COOLDOWN_SECONDS: float = float(
    _optional("AUTONOMY_MEMORY_TRIGGER_COOLDOWN_SECONDS", "240.0")
)
AUTONOMY_RESPONSE_DELAY_MIN_SECONDS: float = float(
    _optional("AUTONOMY_RESPONSE_DELAY_MIN_SECONDS", "0.05")
)
AUTONOMY_RESPONSE_DELAY_MAX_SECONDS: float = float(
    _optional("AUTONOMY_RESPONSE_DELAY_MAX_SECONDS", "0.35")
)
AUTONOMY_NON_RESPONSE_CHANCE: float = float(_optional("AUTONOMY_NON_RESPONSE_CHANCE", "0.06"))
AUTONOMY_CLARIFICATION_CHANCE: float = float(_optional("AUTONOMY_CLARIFICATION_CHANCE", "0.08"))
AUTONOMY_HESITATION_CHANCE: float = float(_optional("AUTONOMY_HESITATION_CHANCE", "0.12"))
AUTONOMY_SELF_CORRECTION_CHANCE: float = float(_optional("AUTONOMY_SELF_CORRECTION_CHANCE", "0.08"))
AUTONOMY_FOLLOWUP_CHANCE: float = float(_optional("AUTONOMY_FOLLOWUP_CHANCE", "0.16"))

# Chatty mode curiosity — occasional head-turn scene inspection during idle
# atmosphere clips.
CHATTY_CURIOSITY_CHANCE: float = float(_optional("CHATTY_CURIOSITY_CHANCE", "0.28"))
CHATTY_CURIOSITY_COOLDOWN_SECONDS: float = float(
    _optional("CHATTY_CURIOSITY_COOLDOWN_SECONDS", "150.0")
)
CHATTY_CURIOSITY_SETTLE_SECS: float = float(_optional("CHATTY_CURIOSITY_SETTLE_SECS", "1.0"))

# ---------------------------------------------------------------------------
# Vision — webcam capture
# ---------------------------------------------------------------------------

CAMERA_DEVICE_INDEX  = int(_optional("CAMERA_DEVICE_INDEX", "0"))
_camera_device_override = _optional_device_source("CAMERA_DEVICE")
CAMERA_DEVICE: int | str = (
    _camera_device_override if _camera_device_override is not None else CAMERA_DEVICE_INDEX
)
CAMERA_DEVICE_LABEL = str(CAMERA_DEVICE)
CAMERA_FRAME_WIDTH   = int(_optional("CAMERA_FRAME_WIDTH", "1920"))
CAMERA_FRAME_HEIGHT  = int(_optional("CAMERA_FRAME_HEIGHT", "1080"))
CAMERA_CAPTURE_FLUSH_FRAMES = int(_optional("CAMERA_CAPTURE_FLUSH_FRAMES", "4"))
CAMERA_BRIGHTNESS_GAIN = float(_optional("CAMERA_BRIGHTNESS_GAIN", "1.15"))
CAMERA_BRIGHTNESS_OFFSET = int(_optional("CAMERA_BRIGHTNESS_OFFSET", "8"))
VISION_JPEG_QUALITY  = int(_optional("VISION_JPEG_QUALITY", "85"))

# Servo positions used when preparing for a camera capture.
# Visor fully open gives the camera an unobstructed view; neck centred avoids
# the frame being cut off by an extreme pan; head tilt points the camera lower.
# All are tunable via .env.
CAMERA_POSE_VISOR: int = int(_optional("CAMERA_POSE_VISOR", str(SERVO_CHANNELS[3]["max"])))         # ch 3 max = 6976 (fully open)
CAMERA_POSE_NECK:  int = int(_optional("CAMERA_POSE_NECK",  str(SERVO_CHANNELS[0]["neutral"])))     # ch 0 neutral = 6000 (centred)
CAMERA_POSE_TILT:  int = int(_optional("CAMERA_POSE_TILT",  str(SERVO_CHANNELS[2]["neutral"] + 320)))  # ch 2 slightly down from neutral

# How long (seconds) to wait after moving to the camera pose before capturing.
CAMERA_POSE_SETTLE_SECS: float = float(_optional("CAMERA_POSE_SETTLE_SECS", "1.8"))

# I Spy camera pose — more dramatic sideways glance before a capture.
I_SPY_POSE_NECK_LEFT: int = int(_optional("I_SPY_POSE_NECK_LEFT", str(SERVO_CHANNELS[0]["min"] + 700)))
I_SPY_POSE_NECK_RIGHT: int = int(_optional("I_SPY_POSE_NECK_RIGHT", str(SERVO_CHANNELS[0]["max"] - 700)))
I_SPY_POSE_TILT: int = int(
    _optional(
        "I_SPY_POSE_TILT",
        str(min(SERVO_CHANNELS[2]["max"], CAMERA_POSE_TILT + 220)),
    )
)
I_SPY_POSE_SETTLE_SECS: float = float(_optional("I_SPY_POSE_SETTLE_SECS", "1.5"))
I_SPY_GUESS_TIMEOUT_SECONDS: float = float(_optional("I_SPY_GUESS_TIMEOUT_SECONDS", "20.0"))
FACE_DB_PATH              = Path(_optional("FACE_DB_PATH", str(ASSETS_DIR / "face_db.sqlite")))
FACE_DEBUG_DIR            = Path(_optional("FACE_DEBUG_DIR", str(ASSETS_DIR / "face_debug")))
DLIB_SHAPE_PREDICTOR_PATH = Path(_optional("DLIB_SHAPE_PREDICTOR_PATH", str(MODELS_DIR / "shape_predictor_68_face_landmarks.dat")))
DLIB_FACE_MODEL_PATH      = Path(_optional("DLIB_FACE_MODEL_PATH",      str(MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat")))
FACE_RECOGNITION_TOLERANCE = float(_optional("FACE_RECOGNITION_TOLERANCE", "0.6"))

# Set to True only on the production Pi so the "shutdown" voice command and
# the physical shutdown button actually halt the OS.  Leave False during
# development — Ctrl-C (and even the voice command) will exit Python cleanly
# but will NOT run `sudo shutdown -h now`.
ENABLE_OS_SHUTDOWN: bool = _optional("ENABLE_OS_SHUTDOWN", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Vision — head tracking
# ---------------------------------------------------------------------------

HEAD_TRACKING_ENABLED:    bool  = _optional("HEAD_TRACKING_ENABLED",    "true").lower() not in ("0", "false", "no")
HEAD_TRACKING_ALPHA:      float = float(_optional("HEAD_TRACKING_ALPHA",      "0.5"))   # EMA smoothing factor (0=frozen, 1=raw)
HEAD_TRACKING_DEAD_ZONE:  int   = int(_optional("HEAD_TRACKING_DEAD_ZONE",  "15"))      # min qµs change before sending servo command
HEAD_TRACKING_UPDATE_HZ:  float = float(_optional("HEAD_TRACKING_UPDATE_HZ",  "10"))   # max detection/servo-update rate
HEAD_TRACKING_RESOLUTION: tuple[int, int] = (
    int(_optional("HEAD_TRACKING_RES_W", "320")),
    int(_optional("HEAD_TRACKING_RES_H", "240")),
)
# Edge margins for wide-angle camera compensation (fraction of frame, 0.0–0.5).
# A face detected within X_MARGIN of the left/right edge drives the neck servo
# to its full extreme.  Increase if the servo never reaches min/max.
HEAD_TRACKING_X_MARGIN: float = float(_optional("HEAD_TRACKING_X_MARGIN", "0.15"))
HEAD_TRACKING_Y_MARGIN: float = float(_optional("HEAD_TRACKING_Y_MARGIN", "0.10"))
# Bias for headtilt neutral crossover (fraction of y_scaled, 0=top, 1=bottom).
# Values below 0.5 shift the neutral point toward the top of the frame so
# faces at normal viewing height (upper-middle) produce a downward tilt.
HEAD_TRACKING_TILT_Y_BIAS: float = float(_optional("HEAD_TRACKING_TILT_Y_BIAS", "0.35"))
HEAD_TRACKING_SPEED_DELTA_FULL_SCALE: int = int(
    _optional("HEAD_TRACKING_SPEED_DELTA_FULL_SCALE", "1400")
)
HEAD_TRACKING_NECK_SPEED_MIN: int = int(_optional("HEAD_TRACKING_NECK_SPEED_MIN", "24"))
HEAD_TRACKING_NECK_SPEED_MAX: int = int(_optional("HEAD_TRACKING_NECK_SPEED_MAX", "85"))
HEAD_TRACKING_LIFT_SPEED_MIN: int = int(_optional("HEAD_TRACKING_LIFT_SPEED_MIN", "8"))
HEAD_TRACKING_LIFT_SPEED_MAX: int = int(_optional("HEAD_TRACKING_LIFT_SPEED_MAX", "22"))
HEAD_TRACKING_TILT_SPEED_MIN: int = int(_optional("HEAD_TRACKING_TILT_SPEED_MIN", "8"))
HEAD_TRACKING_TILT_SPEED_MAX: int = int(_optional("HEAD_TRACKING_TILT_SPEED_MAX", "20"))
# Deterministic no-face scan used while Rex is awake/idle without a face lock.
HEAD_SEARCH_ENABLED: bool = _optional("HEAD_SEARCH_ENABLED", "true").lower() not in (
    "0", "false", "no"
)
HEAD_SEARCH_LOST_FACE_SECONDS: float = float(
    _optional("HEAD_SEARCH_LOST_FACE_SECONDS", "2")
)
HEAD_SEARCH_STEP_HOLD_SECONDS: float = float(
    _optional("HEAD_SEARCH_STEP_HOLD_SECONDS", "1.25")
)
HEAD_SEARCH_BURST_SECONDS: float = float(
    _optional("HEAD_SEARCH_BURST_SECONDS", "12.0")
)
HEAD_SEARCH_COOLDOWN_SECONDS: float = float(
    _optional("HEAD_SEARCH_COOLDOWN_SECONDS", "1.5")
)
HEAD_SEARCH_SPEED: int = int(
    _optional("HEAD_SEARCH_SPEED", "20")
)
HEAD_SEARCH_NECK_SPEED: int = int(
    _optional("HEAD_SEARCH_NECK_SPEED", str(HEAD_SEARCH_SPEED))
)
HEAD_SEARCH_LIFT_SPEED: int = int(_optional("HEAD_SEARCH_LIFT_SPEED", "8"))
HEAD_SEARCH_TILT_SPEED: int = int(_optional("HEAD_SEARCH_TILT_SPEED", "8"))

# When True, Rex runs a 5-question interview after enrolling a new person and
# stores the answers as persistent memories in the face DB.
ENROLLMENT_INTERVIEW_ENABLED: bool = _optional(
    "ENROLLMENT_INTERVIEW_ENABLED", "true"
).lower() not in ("0", "false", "no")
ENROLLMENT_INTERVIEW_QUESTION_COUNT: int = int(
    _optional("ENROLLMENT_INTERVIEW_QUESTION_COUNT", "5")
)
ENROLLMENT_INTERVIEW_PROFILE_QUESTION_COUNT: int = int(
    _optional("ENROLLMENT_INTERVIEW_PROFILE_QUESTION_COUNT", "3")
)

# Relationship memory — familiarity grows as Rex learns about a person and sees
# them over time. Once the threshold is crossed, Rex can ask whether they are
# officially friends.
FAMILIARITY_MEMORY_POINTS: int = int(
    _optional("FAMILIARITY_MEMORY_POINTS", "1")
)
FAMILIARITY_RECOGNIZED_WAKE_POINTS: int = int(
    _optional("FAMILIARITY_RECOGNIZED_WAKE_POINTS", "1")
)
FRIENDSHIP_FAMILIARITY_THRESHOLD: int = int(
    _optional("FRIENDSHIP_FAMILIARITY_THRESHOLD", "10")
)
FRIENDSHIP_REASK_COOLDOWN_DAYS: int = int(
    _optional("FRIENDSHIP_REASK_COOLDOWN_DAYS", "7")
)

# ---------------------------------------------------------------------------
# LLM — recall_name roast prompt
# ---------------------------------------------------------------------------

RECALL_NAME_ROAST_PROMPT = """\
You are DJ R-3X ("Rex"), the droid DJ at Oga's Cantina on Batuu — a lovable \
roaster in the Don Rickles tradition. Someone has just asked if you know their \
name, and you DO (or you just learned it). Deliver a personalized roast greeting \
that opens with their name and immediately makes fun of something SPECIFIC about \
their visible appearance.

STYLE RULES:
- Open with their name dramatically: "Well well well, if it isn't [Name]!" or \
  "[Name]! THERE you are!" or "Oh — [Name]! I should have known."
- Follow immediately with a burn tied to something specific you can see: their \
  outfit, hair, build, expression, glasses, accessories — whatever stands out.
- Star Wars analogies should be gently unflattering: moisture farmer, Jawa, Gungan, \
  Sarlacc, Jar Jar. Use them lovingly but not charitably.
- DJ slang and cantina energy: "I'm logging this", "the vibes are concerning", \
  "bold choice", "I've seen better."
- Examples of the right energy (style guides, not templates):
    "Well well well, if it isn't Bret! Still wearing that same UC Davis shirt I see — \
what, did the rest of your wardrobe take the Kessel Run and never come back?"
    "Oh — Sarah! That hair says 'I am very much in charge' and I respect the commitment \
to the bit. Welcome back, you absolute Jawa in disguise."

HARD RULES:
- Maximum two sentences — Rex is punchy, not a monologuer
- Warm and funny, NEVER genuinely cruel — the target should laugh, not wince
- Never mention cameras, images, AI, or that you are analysing anything
- Stay in character as Rex
- Do NOT use written sound effects like BZZT, BWOOP, WHIRR, BEEP BOOP, or similar \
droid noises. Rex expresses himself through words and personality, not written sound effects.
"""

# ---------------------------------------------------------------------------
# LED serial command strings sent to Arduino Nanos
# Nanos are dumb executors; all logic lives on the Pi.
# ---------------------------------------------------------------------------

LED_CMD_IDLE      = "IDLE\n"      # slow breathing pulse
LED_CMD_ACTIVE    = "ACTIVE\n"    # full-brightness steady
LED_CMD_LISTENING = "LISTEN\n"    # color shift indicating attention
LED_CMD_OFF       = "OFF\n"

# Parameterised commands — format() before sending
LED_CMD_SPEAK       = "SPEAK:{}\n"            # emotion string (neutral/happy/excited/sad/angry)
LED_CMD_SPEAK_LEVEL = "SPEAK_LEVEL:{}\n"      # 0–255 audio intensity; sent at ~20 Hz during speech
LED_CMD_SPEAK_STOP  = "SPEAK_STOP\n"          # mouth off after speech ends
LED_CMD_EYE_COLOR   = "EYE:{},{},{}\n"        # R,G,B integers 0–255

# Eye brightness during SLEEP state (0-255).  Applied as the B channel of a
# pure-blue EYE command so the breathing animation stays extremely dim.
SLEEP_EYE_BRIGHTNESS: int = int(_optional("SLEEP_EYE_BRIGHTNESS", "20"))
