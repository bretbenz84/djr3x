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
STARTUP_CHIME_PATH   = ASSETS_DIR / "audio" / "startup_chime.mp3"
STARTUP_MUSIC_PATH   = ASSETS_DIR / "audio" / "light_speed.mp3"
STARTUP_INTRO_PATH   = ASSETS_DIR / "audio" / "Roger Control.mp3"
SHUTDOWN_MUSIC_PATH  = ASSETS_DIR / "audio" / "hyperdrive_down.mp3"

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
IDLE_HEAD_CHANNELS = [0, 1]         # channels eligible for idle head motion
                                    # ch 2 (headtilt) and ch 3 (visor) are excluded:
                                    # headtilt is speech-only; visor has its own idle timer
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
SERVO_ELBOW_SPEAK_MULT     = float(_optional("SERVO_ELBOW_SPEAK_MULT",     "1.2"))  # elbow intensity multiplier (lower = less jerk on narrow range)
SERVO_ELBOW_SPEAK_THROTTLE = int(_optional("SERVO_ELBOW_SPEAK_THROTTLE",   "3"))    # only update elbow every N speak_move() calls so it completes each raise/lower
SERVO_HAND_SPEAK_SPEED          = int(_optional("SERVO_HAND_SPEAK_SPEED",          "255"))  # ch 5 max speed — used only for the wake greeting wave
SERVO_HAND_SPEAK_SPEED_RELAXED  = int(_optional("SERVO_HAND_SPEAK_SPEED_RELAXED",   "30"))   # ch 5 speed during normal speech — slow and relaxed

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
AUDIO_FORMAT       = 8       # pyaudio.paInt16 == 8 (avoids importing pyaudio here)

AUDIO_INPUT_DEVICE  = _optional_int("AUDIO_INPUT_DEVICE")   # None → system default
AUDIO_OUTPUT_DEVICE = _optional_int("AUDIO_OUTPUT_DEVICE")  # None → system default

# Software volume: 0.0 (mute) – 1.0 (full). ReSpeaker Lite has no hardware
# mixer, so all output paths scale samples by this factor before writing.
AUDIO_VOLUME: float = max(0.0, min(1.0, float(_optional("AUDIO_VOLUME", "0.5"))))

# TTS gain boost applied to ElevenLabs PCM chunks in synthesizer.py before
# queuing for playback.  Multiplies int16 samples and clips to [-32768, 32767].
# Values above 1.0 boost volume; 1.0 = no change.
SYNTHESIZER_VOLUME_GAIN: float = float(_optional("SYNTHESIZER_VOLUME_GAIN", "2.0"))

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
SILENCE_DURATION             = 1.2  # seconds of silence to end capture
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

# Minimum number of words in a Whisper result to be treated as real speech.
# Results shorter than this are discarded as likely hallucinations.
WHISPER_MIN_WORDS = 2

# Single-word results that are always valid and must never be filtered by the
# min-word check — covers commands, confirmations, and greetings that are
# legitimately one word.
SINGLE_WORD_COMMANDS: frozenset[str] = frozenset({
    "shutdown", "quit", "stop",
    "cancel", "nevermind",
    "yes", "no", "yeah", "nope", "sure", "ok", "okay",
    "bye", "hello", "hi", "help",
})

# Whisper-specific recording limits (tighter than the generic caps in the
# "recording / wake word" section above — shorter recording = lower latency).
WHISPER_MAX_RECORD_SECONDS   = float(_optional("WHISPER_MAX_RECORD_SECONDS",   "8.0"))
WHISPER_SILENCE_DURATION     = float(_optional("WHISPER_SILENCE_DURATION",     "0.4"))
# Silence required before ending a recording — longer than WHISPER_SILENCE_DURATION
# to avoid cutting off speech in noisy environments or natural mid-sentence pauses.
TRANSCRIBE_SILENCE_DURATION  = float(_optional("TRANSCRIBE_SILENCE_DURATION",  "1.2"))

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
COMMAND_FUZZY_THRESHOLD = 0.82

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

IDLE_CLIP_INTERVAL_MIN = 10.0   # minimum seconds between idle atmosphere clips
IDLE_CLIP_INTERVAL_MAX = 60.0  # maximum seconds between idle atmosphere clips


ACTIVE_IDLE_TIMEOUT  = 20.0  # legacy — superseded by ACTIVE_TIMEOUT_SECONDS
ACTIVE_TIMEOUT_SECONDS   = 5.0   # seconds of silence after a response before returning to IDLE
WAKE_NO_SPEECH_TIMEOUT   = 5.0   # seconds to wait for first speech after wake word
WAKE_GOODBYE_TIMEOUT     = 4.0   # seconds to wait after "are you there?" before saying goodbye

# ---------------------------------------------------------------------------
# Vision — webcam capture
# ---------------------------------------------------------------------------

CAMERA_DEVICE_INDEX  = int(_optional("CAMERA_DEVICE_INDEX", "0"))
VISION_JPEG_QUALITY  = int(_optional("VISION_JPEG_QUALITY", "85"))

# Servo positions used when preparing for a camera capture.
# Visor fully open gives the camera an unobstructed view; neck centred avoids
# the frame being cut off by an extreme pan.  Both are tunable via .env.
CAMERA_POSE_VISOR: int = int(_optional("CAMERA_POSE_VISOR", str(SERVO_CHANNELS[3]["max"])))     # ch 3 max = 6976 (fully open)
CAMERA_POSE_NECK:  int = int(_optional("CAMERA_POSE_NECK",  str(SERVO_CHANNELS[0]["neutral"]))) # ch 0 neutral = 6000 (centred)

# How long (seconds) to wait after moving to the camera pose before capturing.
CAMERA_POSE_SETTLE_SECS: float = float(_optional("CAMERA_POSE_SETTLE_SECS", "0.5"))
FACE_DB_PATH              = Path(_optional("FACE_DB_PATH", str(ASSETS_DIR / "face_db.sqlite")))
DLIB_SHAPE_PREDICTOR_PATH = Path(_optional("DLIB_SHAPE_PREDICTOR_PATH", str(MODELS_DIR / "shape_predictor_68_face_landmarks.dat")))
DLIB_FACE_MODEL_PATH      = Path(_optional("DLIB_FACE_MODEL_PATH",      str(MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat")))
FACE_RECOGNITION_TOLERANCE = float(_optional("FACE_RECOGNITION_TOLERANCE", "0.6"))

# Set to True only on the production Pi so the "shutdown" voice command and
# the physical shutdown button actually halt the OS.  Leave False during
# development — Ctrl-C (and even the voice command) will exit Python cleanly
# but will NOT run `sudo shutdown -h now`.
ENABLE_OS_SHUTDOWN: bool = _optional("ENABLE_OS_SHUTDOWN", "").lower() in ("1", "true", "yes")

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
- Occasional sound effects: *BWOOP*, *WHIRR*, *BZZT* — undercuts the burn perfectly.
- Examples of the right energy (style guides, not templates):
    "Well well well, if it isn't Bret! Still wearing that same UC Davis shirt I see — \
what, did the rest of your wardrobe take the Kessel Run and never come back? *BWOOP*"
    "Oh — Sarah! That hair says 'I am very much in charge' and I respect the commitment \
to the bit. Welcome back, you absolute Jawa in disguise."

HARD RULES:
- Maximum two sentences — Rex is punchy, not a monologuer
- Warm and funny, NEVER genuinely cruel — the target should laugh, not wince
- Never mention cameras, images, AI, or that you are analysing anything
- Stay in character as Rex
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
