"""
sequences/animations.py — Choreographed servo + LED sequences for DJ-R3X.

Each animation is a list of Steps. A Step fires after `delay` seconds and
sets servo positions and/or LED states simultaneously. The AnimationPlayer
runs each sequence in a daemon thread so callers are not blocked (except
play_shutdown(), which joins its thread so hardware can be safely closed
immediately after it returns).

Step timing model
-----------------
  delay is the pause *before* this step executes (relative to the previous
  step completing, not relative to sequence start).  A delay of 0.0 on the
  first step means "fire immediately".

Servo channel notes
-------------------
  Emotion animations (excited, sad, neutral) only move head channels 0–2
  (tilt, pan, visor) because the ServoController's idle thread continuously
  randomises arm/hand channels 3–6.  Competing with that thread would produce
  twitchy, unpredictable arm behaviour.  The head is not touched by the idle
  thread, so emotion animations own it cleanly.

  Startup and shutdown animations move all channels — they run before the idle
  thread is started (startup) and after it is stopped (shutdown).

Graceful degradation
--------------------
  Both servos and leds may be None (hardware absent during development).
  Every hardware call is guarded; animations still "play" on the time axis
  so callers can wait() for the expected duration without any errors.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel aliases — shorter names for sequence tables
# ---------------------------------------------------------------------------

_P  = config.SERVO_HEAD_PAN     # ch 0 — neck rotation (pan)
_L  = config.SERVO_HEAD_LIFT    # ch 1 — headlift (inverted: lower qµs = head up)
_T  = config.SERVO_HEAD_TILT    # ch 2 — headtilt (speech-reactive only)
_V  = config.SERVO_VISOR        # ch 3 — visor open/close
_AL = config.SERVO_ARM_LEFT     # ch 4 — elbow
_HL = config.SERVO_HAND_LEFT    # ch 5 — hand
_AR = config.SERVO_ARM_RIGHT    # ch 6 — pokerarm
_HR = config.SERVO_HAND_RIGHT   # ch 7 — heroarm

# ---------------------------------------------------------------------------
# Named eye colours
# ---------------------------------------------------------------------------

_EYE_OFF      = (0,   0,   0)
_EYE_DIM_BLUE = (0,   10,  40)
_EYE_SAD      = (0,   30,  100)
_EYE_BLUE     = (0,   80,  255)   # calm idle blue
_EYE_AMBER    = (255, 200, 0)     # active / neutral amber
_EYE_EXCITED  = (255, 220, 0)     # excited bright yellow-amber

# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One timed step in an animation sequence.

    Attributes
    ----------
    delay   : seconds to wait *before* executing this step.
    servos  : channel → position (qµs) mapping; only listed channels move.
    speeds  : channel → speed override applied *before* servo positions this
              step.  Lets sequences set per-channel speeds inline (e.g. a fast
              neck sweep during startup) without touching other channels.
    chest   : effect string sent to chest Nano, or None to leave unchanged.
    head    : effect string sent to head Nano (mouth grid), or None.
    eyes    : (R, G, B) sent via set_eye_color(), or None.
    """
    delay:  float
    servos: dict[int, int]               = field(default_factory=dict)
    speeds: dict[int, int]               = field(default_factory=dict)
    chest:  Optional[str]                = None
    head:   Optional[str]                = None
    eyes:   Optional[tuple[int,int,int]] = None


# ---------------------------------------------------------------------------
# Sequence definitions
# ---------------------------------------------------------------------------

# --- Startup ----------------------------------------------------------------
# Rex boots from a slumped, dark state and gradually comes to life (~3.2 s).
# All channels animated — idle thread has not started yet at this point.

STARTUP: list[Step] = [
    # Immediately: powered-down slump — headlift at min (down), headtilt at max
    # (tilted down), visor at min (eyes covered/down), arms low, all LEDs off.
    Step(delay=0.0,
         servos={_T: config.SERVO_CHANNELS[_T]["max"],   # headtilt fully down
                 _L: config.SERVO_CHANNELS[_L]["min"],   # headlift fully down
                 _P: 6000,
                 _V: config.SERVO_CHANNELS[_V]["min"],   # visor fully down = eyes covered
                 _AL: 5200, _AR: 5200, _HL: 5000, _HR: 5000},
         chest=config.LED_CMD_OFF,
         head=config.LED_CMD_OFF,
         eyes=_EYE_OFF),

    # 0.5 s — a faint flicker: head still down, visor barely lifting.
    #          Set neck to fast speed so the coming look-around sweep feels lively.
    Step(delay=0.5,
         speeds={_P: config.SERVO_NECK_STARTUP_SPEED},
         servos={_T: 5100, _V: 5100},
         eyes=(0, 0, 40)),

    # 0.5 s — waking up: headlift begins rising; neck sweeps to max (looks right).
    Step(delay=0.5,
         servos={_T: 4700, _L: 3500, _V: 5700,
                 _P: config.SERVO_CHANNELS[_P]["max"],
                 _AL: 5600, _AR: 5600},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 40, 140)),

    # 0.4 s — neck returns to center (max→neutral = ~3984 qµs at speed 100 ≈ 0.4 s).
    #          Headlift and visor continue rising while neck visibly pauses at center.
    Step(delay=0.4,
         servos={_T: 4500, _L: 4200, _V: 6000,
                 _P: 6000},
         eyes=(0, 55, 180)),

    # 0.8 s — neck sweeps to min (looks left); 6000→1984 = 4016 qµs, completes in ~0.4 s
    #          leaving 0.4 s for arms and visor to catch up before the next step.
    Step(delay=0.8,
         servos={_T: 4400, _L: 5000, _V: 6300,
                 _P: config.SERVO_CHANNELS[_P]["min"],
                 _AL: 5900, _AR: 5900, _HL: 5800, _HR: 5800},
         eyes=(0, 70, 220)),

    # 0.5 s — fully up: neck returns to center, headlift at neutral, visor open.
    #          Reset neck speed back to default so idle/speech motion is normal.
    Step(delay=0.5,
         speeds={_P: config.SERVO_DEFAULT_SPEED},
         servos={_T: 4200, _L: config.SERVO_CHANNELS[_L]["neutral"], _P: 6000, _V: 6600,
                 _AL: 6000, _AR: 6000, _HL: 6000, _HR: 6000},
         chest=config.LED_CMD_ACTIVE,
         head=config.LED_CMD_ACTIVE,
         eyes=_EYE_BLUE),

    # 0.4 s — Rex perks up with excitement — head snaps up, visor wide open
    Step(delay=0.4,
         servos={_T: 3960, _V: 6900},
         eyes=_EYE_EXCITED),

    # 0.3 s — settle to a confident, ready position
    Step(delay=0.3,
         servos={_T: 4100, _V: 6700},
         eyes=_EYE_AMBER),
]


def _startup_sequence() -> list[Step]:
    """Return the startup sequence with a randomized neck sweep direction."""
    neck_min = config.SERVO_CHANNELS[_P]["min"]
    neck_max = config.SERVO_CHANNELS[_P]["max"]
    first_pan, second_pan = random.choice([
        (neck_max, neck_min),
        (neck_min, neck_max),
    ])
    return [
        Step(delay=0.0,
             servos={_T: config.SERVO_CHANNELS[_T]["max"],
                     _L: config.SERVO_CHANNELS[_L]["min"],
                     _P: 6000,
                     _V: config.SERVO_CHANNELS[_V]["min"],
                     _AL: 5200, _AR: 5200, _HL: 5000, _HR: 5000},
             chest=config.LED_CMD_OFF,
             head=config.LED_CMD_OFF,
             eyes=_EYE_OFF),
        Step(delay=0.5,
             speeds={_P: config.SERVO_NECK_STARTUP_SPEED},
             servos={_T: 5100, _V: 5100},
             eyes=(0, 0, 40)),
        Step(delay=0.5,
             servos={_T: 4700, _L: 3500, _V: 5700,
                     _P: first_pan,
                     _AL: 5600, _AR: 5600},
             chest=config.LED_CMD_IDLE,
             eyes=(0, 40, 140)),
        Step(delay=0.4,
             servos={_T: 4500, _L: 4200, _V: 6000,
                     _P: 6000},
             eyes=(0, 55, 180)),
        Step(delay=0.8,
             servos={_T: 4400, _L: 5000, _V: 6300,
                     _P: second_pan,
                     _AL: 5900, _AR: 5900, _HL: 5800, _HR: 5800},
             eyes=(0, 70, 220)),
        Step(delay=0.5,
             speeds={_P: config.SERVO_DEFAULT_SPEED},
             servos={_T: 4200, _L: config.SERVO_CHANNELS[_L]["neutral"], _P: 6000, _V: 6600,
                     _AL: 6000, _AR: 6000, _HL: 6000, _HR: 6000},
             chest=config.LED_CMD_ACTIVE,
             head=config.LED_CMD_ACTIVE,
             eyes=_EYE_BLUE),
        Step(delay=0.4,
             servos={_T: 3960, _V: 6900},
             eyes=_EYE_EXCITED),
        Step(delay=0.3,
             servos={_T: 4100, _V: 6700},
             eyes=_EYE_AMBER),
    ]


# --- Shutdown ---------------------------------------------------------------
# Rex powers down theatrically — head droops, eyes dim, all goes dark (~2.7 s).
# All channels animated — idle thread is stopped before this runs.

SHUTDOWN: list[Step] = [
    # Immediately: head begins to droop (rising tilt), visor starts lowering
    Step(delay=0.0,
         servos={_T: 4500, _V: 6100},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 60, 180)),

    # 0.5 s — head drooping, headlift drops, visor lowering; neck starts centering,
    #          elbow begins lowering, arms start dropping
    Step(delay=0.5,
         servos={_T: 4900, _L: 5000, _V: 5500,
                 _P: 6100,                       # neck drifting toward center
                 _AL: 5800, _AR: 5000, _HR: 5000},
         eyes=_EYE_SAD),

    # 0.6 s — headlift continuing down, headtilt and visor nearly at extremes;
    #          neck reaches center, elbow and arms continuing down
    Step(delay=0.6,
         servos={_T: 5200, _L: 3500, _V: 5000,
                 _P: 6000,                       # neck centered
                 _AL: 6100, _AR: 4400, _HL: 5500, _HR: 4400},
         eyes=_EYE_DIM_BLUE),

    # 0.6 s — fully powered-down slump: all channels at final shutdown positions
    Step(delay=0.6,
         servos={_T: config.SERVO_CHANNELS[_T]["max"],   # headtilt fully down
                 _L: config.SERVO_CHANNELS[_L]["min"],   # headlift fully down
                 _V: config.SERVO_CHANNELS[_V]["min"],   # visor fully down = eyes covered
                 _P: config.SERVO_CHANNELS[_P]["neutral"],  # neck at center
                 _AL: config.SERVO_CHANNELS[_AL]["min"], # elbow fully down (6300)
                 _AR: config.SERVO_CHANNELS[_AR]["min"], # pokerarm fully down (3968)
                 _HL: config.SERVO_CHANNELS[_HL]["neutral"], # hand at neutral (6000)
                 _HR: config.SERVO_CHANNELS[_HR]["min"]},  # heroarm fully down (3968)
         chest=config.LED_CMD_OFF,
         eyes=_EYE_OFF),

    # 0.5 s — head LEDs out last (dramatic final darkness)
    Step(delay=0.5,
         head=config.LED_CMD_OFF),
]


# --- Excited ----------------------------------------------------------------
# Snappy, energetic — head snaps up, visor flings open, quick double-bob (~1.0 s).
# Head channels only (arm idle thread is running).

EXCITED: list[Step] = [
    # Immediately: head snaps up (low value), visor flings up (eyes wide open)
    Step(delay=0.0,
         servos={_T: 3980, _V: 6900},
         chest=config.LED_CMD_ACTIVE,
         eyes=_EYE_EXCITED),

    # 0.2 s — quick bob (head dips slightly, visor drops a touch)
    Step(delay=0.2,
         servos={_T: 4150, _V: 6600}),

    # 0.2 s — double-take: head snaps back up, visor wide again
    Step(delay=0.2,
         servos={_T: 3950, _V: 6900}),

    # 0.3 s — settle into neutral tilt resting pose
    Step(delay=0.3,
         servos={_T: 4320, _V: 6700},
         eyes=_EYE_AMBER),
]


# --- Sad --------------------------------------------------------------------
# Slow, heavy — head tilts down, visor droops, eyes go dim (~1.6 s).
# Head channels only.

SAD: list[Step] = [
    # Immediately: head begins to droop down (rising value), visor lowering
    Step(delay=0.0,
         servos={_T: 4600, _V: 5700},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 60, 180)),

    # 0.6 s — head drooping further, visor lower
    Step(delay=0.6,
         servos={_T: 5050, _V: 5100},
         eyes=_EYE_SAD),

    # 0.6 s — fully drooped, visor nearly covering eyes
    Step(delay=0.6,
         servos={_T: 5350, _V: 4800},
         eyes=(0, 20, 80)),
]


# --- Wake Greeting ----------------------------------------------------------
# Excited wave hello — elbow raises, hand twists back and forth 3 times (~2.9 s).
# Uses exactly config.SERVO_CHANNELS[5]['min'] / ['max'] for the shake targets
# so the servo physically reaches its travel extremes for a pronounced twist.
# 0.4 s per step at max speed (255) gives the servo ~1000 µs of travel time per
# half-swing (~25% of range per 50 ms), completing the full 4032 qµs span.
# Launched non-blocking via play_wake_greeting_arms() — runs concurrently with
# greeting audio so Rex waves while speaking.

WAKE_GREETING: list[Step] = [
    # Elbow raises slightly; hand twists to physical low limit
    Step(delay=0.0,
         servos={_AL: 6450, _HL: config.SERVO_CHANNELS[_HL]["min"]}),

    # Alternate full-range twists — 3 complete low/high cycles at 0.4 s each
    Step(delay=0.4, servos={_HL: config.SERVO_CHANNELS[_HL]["max"]}),
    Step(delay=0.4, servos={_HL: config.SERVO_CHANNELS[_HL]["min"]}),
    Step(delay=0.4, servos={_HL: config.SERVO_CHANNELS[_HL]["max"]}),
    Step(delay=0.4, servos={_HL: config.SERVO_CHANNELS[_HL]["min"]}),
    Step(delay=0.4, servos={_HL: config.SERVO_CHANNELS[_HL]["max"]}),

    # Return elbow and hand to neutral
    Step(delay=0.50,
         servos={_AL: 6720, _HL: 6000}),
]


# --- Sleep ------------------------------------------------------------------
# Rex exhaustedly collapses into a slumped rest — all channels animate at
# SERVO_SLEEP_SPEED (extremely slow) over ~7 s.  Runs after the idle thread
# is stopped so there is no channel competition.
# Eye color dims to a very faint blue; the final step activates dim breathing.

_EYE_SLEEP = (0, 0, config.SLEEP_EYE_BRIGHTNESS)   # very dim blue during sleep

SLEEP: list[Step] = [
    # Immediately: set every channel to sleep speed (extremely slow), then
    # begin the droop — head tilts down slightly, visor starts closing.
    Step(delay=0.0,
         speeds={_P: config.SERVO_SLEEP_SPEED,
                 _L: config.SERVO_SLEEP_SPEED,
                 _T: config.SERVO_SLEEP_SPEED,
                 _V: config.SERVO_SLEEP_SPEED,
                 _AL: config.SERVO_SLEEP_SPEED,
                 _HL: config.SERVO_SLEEP_SPEED,
                 _AR: config.SERVO_SLEEP_SPEED,
                 _HR: config.SERVO_SLEEP_SPEED},
         servos={_T: 4800, _V: 6000},
         eyes=(0, 40, 130)),

    # 2.0 s — head drooping more, headlift starting to fall, visor lowering,
    #          arms beginning to drop under their own (servo) weight.
    Step(delay=2.0,
         servos={_T: 5150, _L: 4000, _V: 5400,
                 _AL: 6050, _AR: 4600, _HR: 4600},
         eyes=(0, 15, 55)),

    # 2.5 s — nearly fully slumped; headlift well down, visor nearly closed.
    Step(delay=2.5,
         servos={_T: 5350, _L: 2500, _V: 4900,
                 _AL: 6150, _AR: 4200, _HR: 4200},
         eyes=(0, 5, 25)),

    # 2.0 s — final slumped position; all channels at shutdown/sleep values.
    #          Eyes and mouth LEDs are handled by the SLEEP serial command sent
    #          in _run_sleep() after this animation completes.
    Step(delay=2.0,
         servos={_T: config.SERVO_CHANNELS[_T]["max"],
                 _L: config.SERVO_CHANNELS[_L]["min"],
                 _V: config.SERVO_CHANNELS[_V]["min"],
                 _P: config.SERVO_CHANNELS[_P]["neutral"],
                 _AL: config.SERVO_CHANNELS[_AL]["min"],
                 _AR: config.SERVO_CHANNELS[_AR]["min"],
                 _HL: config.SERVO_CHANNELS[_HL]["neutral"],
                 _HR: config.SERVO_CHANNELS[_HR]["min"]}),
]


# --- Wake from sleep --------------------------------------------------------
# Rex struggles back from slumped to neutral — very slow at first, pauses
# mid-way as if almost going back to sleep, then commits (~7.5 s total).
# All channels animated — idle thread is stopped during sleep.

WAKE_FROM_SLEEP: list[Step] = [
    # Immediately: set extremely slow speed; barely a twitch — tiny headtilt
    # movement signals Rex is stirring.  Eyes start to brighten.
    Step(delay=0.0,
         speeds={_P: config.SERVO_SLEEP_SPEED,
                 _L: config.SERVO_SLEEP_SPEED,
                 _T: config.SERVO_SLEEP_SPEED,
                 _V: config.SERVO_SLEEP_SPEED,
                 _AL: config.SERVO_SLEEP_SPEED,
                 _HL: config.SERVO_SLEEP_SPEED,
                 _AR: config.SERVO_SLEEP_SPEED,
                 _HR: config.SERVO_SLEEP_SPEED},
         servos={_T: 5300, _V: 4700},
         eyes=(0, 8, 45)),

    # 1.8 s — slow stir: headlift barely lifts, visor opens a crack.
    Step(delay=1.8,
         servos={_T: 5100, _L: 2500, _V: 4900},
         eyes=(0, 18, 70)),

    # 1.5 s — PERSONALITY PAUSE: visor flutters open slightly (Rex hesitating).
    #          Speed bump on visor only so the flutter is perceptible.
    Step(delay=1.5,
         speeds={_V: 8},
         servos={_V: 5200},
         eyes=(0, 25, 90)),

    # 0.7 s — visor drops back — Rex almost goes back to sleep.
    Step(delay=0.7,
         servos={_V: 4900}),

    # 0.8 s — Rex commits; all channels pick up speed.
    Step(delay=0.8,
         speeds={_P: 6, _L: 6, _T: 6, _V: 6, _AL: 6, _AR: 6, _HR: 6},
         servos={_T: 4800, _L: 3600, _V: 5500,
                 _AR: 4800, _HR: 4800},
         eyes=(0, 40, 140)),

    # 1.5 s — picking up more speed, rising purposefully.
    Step(delay=1.5,
         speeds={_P: 12, _L: 12, _T: 12, _V: 12,
                 _AL: 12, _HL: 12, _AR: 12, _HR: 12},
         servos={_T: 4450, _L: 5000, _V: 6200,
                 _AL: 6000, _AR: 5500, _HR: 5500},
         eyes=(0, 55, 190)),

    # 1.2 s — fully upright; restore normal speeds and neutral positions.
    Step(delay=1.2,
         speeds={_P: config.SERVO_DEFAULT_SPEED,
                 _L: config.SERVO_DEFAULT_SPEED,
                 _T: config.SERVO_DEFAULT_SPEED,
                 _V: config.SERVO_DEFAULT_SPEED,
                 _AL: config.SERVO_DEFAULT_SPEED,
                 _HL: config.SERVO_DEFAULT_SPEED,
                 _AR: config.SERVO_DEFAULT_SPEED,
                 _HR: config.SERVO_DEFAULT_SPEED},
         servos={_T: 4320, _L: config.SERVO_CHANNELS[_L]["neutral"],
                 _P: 6000,  _V: 6500,
                 _AL: 6720, _AR: 6000, _HL: 6000, _HR: 6000},
         head=config.LED_CMD_ACTIVE,
         eyes=_EYE_BLUE),
]


# --- Neutral ----------------------------------------------------------------
# Smooth return to center — used when resetting emotion state (~0.8 s).
# Head channels only.

NEUTRAL: list[Step] = [
    # Immediately: return head to neutral tilt and pan center
    Step(delay=0.0,
         servos={_T: 4320, _P: 6000},
         chest=config.LED_CMD_ACTIVE,
         eyes=_EYE_BLUE),

    # 0.4 s — visor to neutral open position
    Step(delay=0.4,
         servos={_V: 6500}),
]


# ---------------------------------------------------------------------------
# AnimationPlayer
# ---------------------------------------------------------------------------

class AnimationPlayer:
    """Runs named animation sequences against ServoController and LEDController.

    Both hardware objects may be None (hardware absent in development). All
    methods are safe to call in that state; the timing still executes correctly
    so wait() returns after the expected duration.

    Only one animation runs at a time. Starting a new one cancels the current.
    """

    def __init__(self, servos, leds) -> None:
        """
        Parameters
        ----------
        servos : ServoController | None
        leds   : LEDController   | None
        """
        self._servos = servos
        self._leds   = leds

        self._cancel = threading.Event()
        self._done   = threading.Event()
        self._done.set()   # nothing playing at construction
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_startup(self) -> None:
        """Play the startup (boot) animation in the background.

        Call *before* starting the ServoController idle thread so arm
        motion is not fought by background randomisation.
        """
        self._launch(_startup_sequence(), blocking=False)

    def play_shutdown(self) -> None:
        """Start the shutdown (power-down) animation in a background thread
        and return immediately.

        Designed to be called *after* stopping the ServoController idle
        thread (ServoController.stop()) and *before* closing hardware.
        Call wait() after to block until the sequence has played out.
        """
        self._launch(SHUTDOWN, blocking=False)

    def play_wake_greeting_arms(self) -> None:
        """Start the excited arm-wave greeting in a background thread and return immediately.

        Stops the idle thread and sets hand speed to maximum (255) before
        launching so no conflicting serial commands can fight the wave.
        Returns as soon as the animation thread is running so the caller can
        start greeting audio concurrently.

        The caller is responsible for cleanup once both audio and animation
        have finished:
            self._animations.wait(timeout=5.0)
            if self._servos is not None:
                self._servos.set_channel_speed(
                    config.SERVO_HAND_LEFT, config.SERVO_DEFAULT_SPEED)
                self._servos.start()
        """
        log.info(
            "WAKE_GREETING: ch 5 (hand) targets  low=%d qµs  high=%d qµs"
            "  (config min=%d  max=%d)",
            config.SERVO_CHANNELS[config.SERVO_HAND_LEFT]["min"],
            config.SERVO_CHANNELS[config.SERVO_HAND_LEFT]["max"],
            config.SERVO_CHANNELS[config.SERVO_HAND_LEFT]["min"],
            config.SERVO_CHANNELS[config.SERVO_HAND_LEFT]["max"],
        )
        if self._servos is not None:
            # Stop the full idle thread so head and arm randomisation cannot
            # interfere with the wave animation on any channel.
            self._servos.stop()
            # Maximum Maestro speed on ch 5 — ensures servo reaches each
            # extreme within the 0.4 s step window for a dramatic full-range wave.
            self._servos.set_channel_speed(config.SERVO_HAND_LEFT, config.SERVO_HAND_SPEAK_SPEED)

        # Non-blocking — returns immediately; animation runs in daemon thread.
        self._launch(WAKE_GREETING, blocking=False)

    def play_sleep(self) -> None:
        """Start the sleep (exhausted collapse) animation in a background thread.

        Call *after* stopping the ServoController idle thread so all channels
        are free.  Returns immediately; the animation runs for ~7.5 s total.
        Call wait() to block until complete before entering sleep state.
        """
        self._launch(SLEEP, blocking=False)

    def play_wake_from_sleep(self) -> None:
        """Start the wake-from-sleep (struggle back to neutral) animation.

        Returns immediately; the animation runs for ~7.5 s total.
        Rex pauses mid-way for a hesitation personality beat before committing
        to waking up.  Call wait() to block until the animation finishes.
        """
        self._launch(WAKE_FROM_SLEEP, blocking=False)

    def play_emotion(self, emotion: str) -> None:
        """Play an emotion animation in the background.

        Valid values: "excited", "sad", "neutral".
        Unknown values log a warning and do nothing.
        """
        mapping = {
            "excited": EXCITED,
            "sad":     SAD,
            "neutral": NEUTRAL,
        }
        seq = mapping.get(emotion)
        if seq is None:
            log.warning("AnimationPlayer: unknown emotion %r — skipping", emotion)
            return
        self._launch(seq, blocking=False)

    def cancel(self) -> None:
        """Interrupt the current animation and wait for the thread to exit.

        Safe to call when nothing is playing.
        """
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the current animation finishes (or timeout elapses).

        Returns True if the animation completed, False if it timed out.
        """
        return self._done.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal — launch
    # ------------------------------------------------------------------

    def _launch(self, steps: list[Step], blocking: bool) -> None:
        """Cancel any running animation, then start (or run) the new one."""
        self.cancel()

        self._cancel.clear()
        self._done.clear()

        self._thread = threading.Thread(
            target=self._run,
            args=(steps,),
            daemon=True,
            name="djr3x-animation",
        )
        self._thread.start()

        if blocking:
            self._thread.join()

    # ------------------------------------------------------------------
    # Internal — sequence runner (executes in daemon thread)
    # ------------------------------------------------------------------

    def _run(self, steps: list[Step]) -> None:
        try:
            for step in steps:
                # Interruptible sleep: check cancel every 20 ms
                if step.delay > 0.0:
                    deadline = time.monotonic() + step.delay
                    while time.monotonic() < deadline:
                        if self._cancel.is_set():
                            return
                        remaining = max(0.0, deadline - time.monotonic())
                        time.sleep(min(0.02, remaining))

                if self._cancel.is_set():
                    return

                self._execute_step(step)

        except Exception:
            log.exception("AnimationPlayer: unhandled error in sequence")
        finally:
            self._done.set()

    def _execute_step(self, step: Step) -> None:
        """Apply servo positions and LED commands for one step."""
        # --- Per-step speed overrides (applied before positions) ---
        if self._servos is not None and step.speeds:
            for channel, speed in step.speeds.items():
                try:
                    self._servos.set_channel_speed(channel, speed)
                except Exception:
                    log.warning(
                        "AnimationPlayer: set_channel_speed(%d, %d) failed",
                        channel, speed, exc_info=True,
                    )

        # --- Servo positions ---
        if self._servos is not None and step.servos:
            for channel, position in step.servos.items():
                try:
                    self._servos.set_position(channel, position)
                except Exception:
                    log.warning(
                        "AnimationPlayer: set_position(%d, %d) failed",
                        channel, position, exc_info=True,
                    )

        # --- LEDs ---
        if self._leds is not None:
            try:
                if step.chest is not None:
                    self._leds.set_chest_effect(step.chest)
                if step.head is not None:
                    self._leds.set_head_effect(step.head)
                if step.eyes is not None:
                    self._leds.set_eye_color(*step.eyes)
            except Exception:
                log.warning("AnimationPlayer: LED command failed", exc_info=True)
