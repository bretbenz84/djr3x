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
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel aliases — shorter names for sequence tables
# ---------------------------------------------------------------------------

_T  = config.SERVO_HEAD_TILT    # ch 0 — tilt  (higher = head up)
_P  = config.SERVO_HEAD_PAN     # ch 1 — pan   (center = 6000)
_V  = config.SERVO_VISOR        # ch 2 — visor (higher = more open)
_AL = config.SERVO_ARM_LEFT     # ch 3
_AR = config.SERVO_ARM_RIGHT    # ch 4
_HL = config.SERVO_HAND_LEFT    # ch 5
_HR = config.SERVO_HAND_RIGHT   # ch 6

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
    chest   : effect string sent to chest Nano, or None to leave unchanged.
    head    : effect string sent to head Nano (mouth grid), or None.
    eyes    : (R, G, B) sent via set_eye_color(), or None.
    """
    delay:  float
    servos: dict[int, int]               = field(default_factory=dict)
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
    # Immediately: start dark and slumped
    Step(delay=0.0,
         servos={_T: 4800, _P: 6000, _V: 4200,
                 _AL: 5200, _AR: 5200, _HL: 5000, _HR: 5000},
         chest=config.LED_CMD_OFF,
         head=config.LED_CMD_OFF,
         eyes=_EYE_OFF),

    # 0.5 s — a faint flicker: eyes barely glow, servo twitch
    Step(delay=0.5,
         servos={_T: 5000, _V: 4500},
         eyes=(0, 0, 40)),

    # 0.5 s — rising: arms start to lift, head comes up
    Step(delay=0.5,
         servos={_T: 5400, _V: 5000, _AL: 5600, _AR: 5600},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 40, 140)),

    # 0.5 s — almost there: visor opening, arms near home
    Step(delay=0.5,
         servos={_T: 5800, _V: 5400,
                 _AL: 5900, _AR: 5900, _HL: 5800, _HR: 5800},
         eyes=(0, 70, 220)),

    # 0.5 s — fully up: all home, LEDs active
    Step(delay=0.5,
         servos={_T: 6000, _P: 6000, _V: 5500,
                 _AL: 6000, _AR: 6000, _HL: 6000, _HR: 6000},
         chest=config.LED_CMD_ACTIVE,
         head=config.LED_CMD_ACTIVE,
         eyes=_EYE_BLUE),

    # 0.4 s — Rex perks up with excitement at being alive
    Step(delay=0.4,
         servos={_T: 6500, _V: 6200},
         eyes=_EYE_EXCITED),

    # 0.3 s — settle to a confident, ready position
    Step(delay=0.3,
         servos={_T: 6200, _V: 5800},
         eyes=_EYE_AMBER),
]


# --- Shutdown ---------------------------------------------------------------
# Rex powers down theatrically — head droops, eyes dim, all goes dark (~2.7 s).
# All channels animated — idle thread is stopped before this runs.

SHUTDOWN: list[Step] = [
    # Immediately: switch to slow breathing, head starts to lower
    Step(delay=0.0,
         servos={_T: 5600, _V: 5000},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 60, 180)),

    # 0.5 s — drooping further, arms beginning to fall
    Step(delay=0.5,
         servos={_T: 5200, _V: 4600, _AL: 5500, _AR: 5500},
         eyes=_EYE_SAD),

    # 0.6 s — nearly down, eyes barely glowing
    Step(delay=0.6,
         servos={_T: 4900, _V: 4300,
                 _AL: 5100, _AR: 5100, _HL: 5500, _HR: 5500},
         eyes=_EYE_DIM_BLUE),

    # 0.6 s — fully slumped, chest LED off
    Step(delay=0.6,
         servos={_T: 4800, _V: 4200,
                 _AL: 5000, _AR: 5000, _HL: 5000, _HR: 5000},
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
    # Immediately: snap up with energy
    Step(delay=0.0,
         servos={_T: 6800, _V: 6500},
         chest=config.LED_CMD_ACTIVE,
         eyes=_EYE_EXCITED),

    # 0.2 s — quick bob down
    Step(delay=0.2,
         servos={_T: 6300, _V: 6000}),

    # 0.2 s — back up even higher (the double-take)
    Step(delay=0.2,
         servos={_T: 6900, _V: 6600}),

    # 0.3 s — settle into a proud, head-up resting pose
    Step(delay=0.3,
         servos={_T: 6600, _V: 6200},
         eyes=_EYE_AMBER),
]


# --- Sad --------------------------------------------------------------------
# Slow, heavy — head tilts down, visor droops, eyes go dim (~1.6 s).
# Head channels only.

SAD: list[Step] = [
    # Immediately: begin slow drop, chest dims to idle breathing
    Step(delay=0.0,
         servos={_T: 5600, _V: 5000},
         chest=config.LED_CMD_IDLE,
         eyes=(0, 60, 180)),

    # 0.6 s — continuing down, eyes cooling to sad blue
    Step(delay=0.6,
         servos={_T: 5100, _V: 4600},
         eyes=_EYE_SAD),

    # 0.6 s — fully drooped
    Step(delay=0.6,
         servos={_T: 4800, _V: 4300},
         eyes=(0, 20, 80)),
]


# --- Neutral ----------------------------------------------------------------
# Smooth return to center — used when resetting emotion state (~0.8 s).
# Head channels only.

NEUTRAL: list[Step] = [
    # Immediately: start moving head back to center
    Step(delay=0.0,
         servos={_T: 6000, _P: 6000},
         chest=config.LED_CMD_ACTIVE,
         eyes=_EYE_BLUE),

    # 0.4 s — visor back to home
    Step(delay=0.4,
         servos={_V: 5500}),
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
        self._launch(STARTUP, blocking=False)

    def play_shutdown(self) -> None:
        """Play the shutdown (power-down) animation and block until complete.

        Designed to be called *after* stopping the ServoController idle
        thread (ServoController.stop()) and *before* closing hardware.
        Returns only when the full sequence has played out, guaranteeing
        that hardware can be safely closed immediately after.
        """
        self._launch(SHUTDOWN, blocking=True)

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
                        time.sleep(min(0.02, deadline - time.monotonic()))

                if self._cancel.is_set():
                    return

                self._execute_step(step)

        except Exception:
            log.exception("AnimationPlayer: unhandled error in sequence")
        finally:
            self._done.set()

    def _execute_step(self, step: Step) -> None:
        """Apply servo positions and LED commands for one step."""
        # --- Servos ---
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
