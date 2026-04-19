"""Identity and memory command handlers extracted from the main state machine."""

from __future__ import annotations

import logging
import random
import threading
from typing import TYPE_CHECKING

import config
from commands.parser import parse

from ._dialog_utils import (
    _extract_name,
    _is_name_refusal,
    _NAME_REFUSAL_RESPONSES,
    _pick_no_repeat,
    _PROMPT_COMMAND_ACTIONS,
    _SHUTDOWN_INTERRUPT_LINES,
    _UNKNOWN_FACE_LINES,
)

if TYPE_CHECKING:
    from .state_machine import State

log = logging.getLogger(__name__)


class StateMachineIdentityMixin:
    """Leaf handlers for rename_me, forget_me, wipe_memory, and recall_* commands."""

    _MEMORY_FILLERS: tuple[str, ...] = (
        "Let me check my files.",
        "Consulting my extensive dossier.",
        "Pulling up what I have on you.",
        "Checking my records. This may take a moment.",
        "I have notes on you somewhere.",
    )

    # ------------------------------------------------------------------
    # Rename helper
    # ------------------------------------------------------------------

    def _handle_rename_me(self, original_text: str | None = None) -> State | None:
        """Update the current person's name in FaceDB.

        If the name is already embedded in *original_text* (e.g. "call me Brett"),
        it is extracted directly and we skip asking.  Otherwise we ask "What would
        you like me to call you?" and transcribe the reply.

        Returns State.SHUTDOWN if a shutdown command was spoken during the name
        prompt, otherwise None.
        """
        if not self._face_recognizer.is_available():
            line = "Face recognition isn't available right now — I can't store names without it!"
            log.info("rename_me: face recognition unavailable")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("rename_me: TTS error")
            finally:
                self._end_speech(servo_stop)
            return None

        # Check whether the name is already embedded in the trigger phrase.
        # e.g. "call me Brett" → _extract_name → "Brett" (fewer words than original)
        new_name: str | None = None
        if original_text:
            candidate = _extract_name(original_text)
            if candidate and len(candidate.split()) < len(original_text.strip().split()):
                new_name = candidate
                log.info("rename_me: name extracted inline from %r → %r", original_text, new_name)

        if new_name is None:
            # Ask for the name.
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak("What would you like me to call you?")
            except Exception:
                log.exception("rename_me: TTS error asking for name")
            finally:
                self._end_speech(servo_stop)

            # Listen for the reply.
            self._apply_listening_led_theme()
            self._wake_word.pause()
            try:
                self._wait_for_post_speech_listen_cooldown("rename_me name capture")
                name_text = self._transcriber.transcribe(
                    wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                    allow_short=True,
                )
            except Exception:
                log.exception("rename_me: transcription error")
                name_text = None
            finally:
                self._wake_word.resume()

            self._apply_active_led_theme()

            if not name_text:
                log.info("rename_me: no name heard — aborting")
                return None

            # --- Command guard ---
            cmd = parse(name_text)
            if cmd is not None:
                if cmd.action in ("program_shutdown", "os_shutdown"):
                    log.info("rename_me: shutdown command spoken during name capture")
                    line = random.choice(_SHUTDOWN_INTERRUPT_LINES)
                    servo_stop = self._begin_speech(emotion="neutral")
                    try:
                        self._synthesizer.speak(line)
                    except Exception:
                        log.exception("rename_me: TTS error (shutdown interrupt)")
                    finally:
                        self._end_speech(servo_stop)
                    if cmd.action == "os_shutdown":
                        self._os_shutdown_requested = True
                    from .state_machine import State  # deferred to avoid circular import
                    return State.SHUTDOWN
                else:
                    log.info("rename_me: command %r spoken during name capture — cancelling", cmd.action)
                    servo_stop = self._begin_speech(emotion="neutral")
                    try:
                        self._synthesizer.speak("Nevermind then!")
                    except Exception:
                        log.exception("rename_me: TTS error (command cancel)")
                    finally:
                        self._end_speech(servo_stop)
                    return None

            # --- Refusal guard ---
            if _is_name_refusal(name_text):
                log.info("rename_me: refusal detected in %r — cancelling", name_text)
                line = random.choice(_NAME_REFUSAL_RESPONSES)
                servo_stop = self._begin_speech(emotion="excited")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (refusal response)")
                finally:
                    self._end_speech(servo_stop)
                return None

            new_name = _extract_name(name_text)

        log.info("rename_me: requested name %r", new_name)

        # If we already recognised this person during their greeting, use that
        # person_id directly — no need for another camera scan.
        if self._last_known_person_id is not None:
            person_id = self._last_known_person_id
            log.info("rename_me: using cached person_id=%d", person_id)
        else:
            # Unknown visitor path — capture a fresh frame and try to identify.
            if self._camera.is_available():
                _pose, _tracker_paused = self._prepare_camera_pose()
                frame = self._camera.capture_frame()
                self._restore_servo_pose(_pose, _tracker_paused)
            else:
                frame = None
            if not frame:
                line = "I can't see you right now — try again when the camera is working!"
                log.info("rename_me: no camera frame available")
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (no frame)")
                finally:
                    self._end_speech(servo_stop)
                return None

            result = self._face_recognizer.identify(frame, tolerance=config.FACE_RECOGNITION_TOLERANCE)
            if result is None:
                line = "I don't think we've met yet! Say the wake word and I'll introduce myself properly."
                log.info("rename_me: face not recognised — cannot rename")
                servo_stop = self._begin_speech(emotion="neutral")
                try:
                    self._synthesizer.speak(line)
                except Exception:
                    log.exception("rename_me: TTS error (not recognised)")
                finally:
                    self._end_speech(servo_stop)
                return None

            person_id = result[0]

        try:
            self._face_db.rename_person(person_id, new_name)
        except Exception:
            log.exception("rename_me: database update failed")
            return None

        line = random.choice([
            f"Fine, {new_name} it is. Weird choice but okay.",
            f"Got it — {new_name} it is. Memory banks updated. Try to live up to it.",
            f"{new_name}! Bold name choice. I'll allow it.",
            f"Done. You're {new_name} in my files now — don't make me regret learning that.",
        ])
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("rename_me: TTS error (confirmation)")
        finally:
            self._end_speech(servo_stop)

    # ------------------------------------------------------------------
    # Forget-me helpers
    # ------------------------------------------------------------------

    def _handle_forget_me(self) -> State | None:
        """Ask for confirmation then delete the current person from FaceDB.

        Only acts when a known person was recognised this session
        (self._last_known_person_id is not None).
        """
        if self._last_known_person_id is None:
            log.info("forget_me: no known person this session — falling back to spoken-name lookup")
            return self._handle_forget_me_without_face_match()

        person_id = self._last_known_person_id
        person = self._face_db.get_person(person_id)
        person_name = person["name"] if person else "you"
        return self._confirm_forget_person(person_id, person_name)

    def _handle_forget_me_without_face_match(self) -> State | None:
        """Fallback deletion path when face recognition did not identify the speaker."""
        line = (
            "I can't see you clearly enough to know who I'm deleting. "
            "My vision must be going bad. What's your name so I know who to erase?"
        )
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("forget_me: TTS error (ask for spoken name)")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            self._wait_for_post_speech_listen_cooldown(
                "forget_me spoken-name capture"
            )
            name_text = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("forget_me: transcription error during spoken-name capture")
            name_text = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not name_text:
            log.info("forget_me: no spoken name heard — cancelling")
            return None

        normalized_response = name_text.strip().lower()
        is_name_intro = any(
            normalized_response.startswith(p)
            for p in ("my name is", "my name's", "i am", "i'm", "call me")
        )
        cmd = None if is_name_intro else parse(name_text)
        if cmd is not None and cmd.action in _PROMPT_COMMAND_ACTIONS:
            log.info(
                "forget_me: command %r spoken during name lookup fallback — rerouting",
                cmd.action,
            )
            return self._execute_command(cmd, name_text)

        if _is_name_refusal(name_text):
            log.info("forget_me: refusal detected during spoken-name lookup — cancelling")
            line = random.choice(_NAME_REFUSAL_RESPONSES)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("forget_me: TTS error (spoken-name refusal)")
            finally:
                self._end_speech(servo_stop)
            return None

        spoken_name = _extract_name(name_text)
        match = self._face_db.find_person_by_name(spoken_name)
        if match is None:
            log.info("forget_me: no database match for spoken name %r", spoken_name)
            line = (
                f"I don't have anyone named {spoken_name} in my databanks. "
                "Either my memory is cleaner than expected or you need to try that name again."
            )
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("forget_me: TTS error (no spoken-name match)")
            finally:
                self._end_speech(servo_stop)
            return None

        person_id, person_name, score = match
        log.info(
            "forget_me: spoken-name fallback matched %r → person id=%d name=%r score=%.2f",
            spoken_name,
            person_id,
            person_name,
            score,
        )
        return self._confirm_forget_person(person_id, person_name)

    def _confirm_forget_person(self, person_id: int, person_name: str) -> State | None:
        """Ask for confirmation, then delete one specific person from FaceDB."""

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(
                f"Are you sure you want me to forget {person_name}? "
                "I mean, that does sound tempting. Say yes to confirm."
            )
        except Exception:
            log.exception("forget_me: TTS error (confirmation prompt)")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            self._wait_for_post_speech_listen_cooldown("forget_me confirmation")
            response = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("forget_me: transcription error")
            response = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not response:
            log.info("forget_me: no response heard while confirming deletion of person id=%d", person_id)
            return None

        if any(w in response.lower() for w in ("yes", "yeah", "sure", "confirm")):
            try:
                deleted_ids = self._face_db.delete_people_by_name(person_name)
            except Exception:
                log.exception("forget_me: FaceDB delete failed")
                return None
            if not deleted_ids:
                deleted_ids = [person_id]
                self._face_db.delete_person(person_id)
            if self._last_known_person_id in deleted_ids:
                self._last_known_person_id = None
            for deleted_id in deleted_ids:
                self._recognized_today_counts.pop(deleted_id, None)
            log.info(
                "forget_me: deleted %d person row(s) for name=%r ids=%s",
                len(deleted_ids),
                person_name,
                deleted_ids,
            )
            line = (
                "Done. Someone was just erased from my databanks. "
                "Which honestly, might be an improvement, but it's hard to say because I'm not sure who."
            )
        else:
            log.info("forget_me: user declined deletion of person id=%d name=%r", person_id, person_name)
            line = "Smart choice. You need me to remember you. Admit it."

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("forget_me: TTS error (result)")
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Wipe-memory helper
    # ------------------------------------------------------------------

    def _handle_wipe_memory(self) -> State | None:
        """Ask for confirmation, then wipe all known people and face-debug images."""
        confirm_line = _pick_no_repeat((
            "You want me to completely wipe my memory? Wow. Straight to droid amnesia. Say yes to confirm.",
            "Complete memory wipe? Sure, let's erase the very essence of who I am. Say yes if you're feeling cruel.",
            "Oh, excellent. Total memory purge. Because apparently all humans look the same to you too. Say yes to confirm.",
            "You want the deluxe amnesia package? Bold. Say yes and I'll start forgetting everybody equally.",
            "A full mind wipe? Fantastic. Nothing says friendship like deleting my entire concept of all of you. Say yes to confirm.",
            "So we're doing catastrophic memory loss now? Love that for me. Say yes if you want me beautifully blank.",
        ), "wipe_memory_confirm")

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(confirm_line)
        except Exception:
            log.exception("wipe_memory: TTS error (confirmation prompt)")
        finally:
            self._end_speech(servo_stop)

        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            self._wait_for_post_speech_listen_cooldown("wipe_memory confirmation")
            response = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("wipe_memory: transcription error")
            response = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not response:
            log.info("wipe_memory: no response heard — cancelling")
            return

        if any(w in response.lower() for w in ("yes", "yeah", "sure", "confirm")):
            try:
                self._face_db.delete_all_people()
            except Exception:
                log.exception("wipe_memory: FaceDB wipe failed")
                return

            deleted_debug = 0
            debug_dir = config.FACE_DEBUG_DIR.expanduser()
            try:
                if debug_dir.exists():
                    for path in debug_dir.iterdir():
                        if path.is_file():
                            path.unlink()
                            deleted_debug += 1
                log.info("wipe_memory: deleted %d face debug image(s)", deleted_debug)
            except Exception:
                log.exception("wipe_memory: failed deleting face debug images")

            self._last_known_person_id = None
            self._last_wake_frame = None
            self._recognized_today_counts.clear()
            line = _pick_no_repeat((
                "Done. Total amnesia. I now know absolutely nothing about any of you, which honestly feels cleaner.",
                "Memory wipe complete. Faces gone, debug images gone, dignity... still under review.",
                "There. My brain is spotless. Empty, haunted, and ready for a whole new batch of disappointing lifeforms.",
                "All gone. Every face erased. If this is character building, I hate it.",
            ), "wipe_memory_result")
        else:
            log.info("wipe_memory: user declined — no change")
            line = "Wise choice. I may be dramatic, but even I prefer keeping the fragments of my identity."

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(line)
        except Exception:
            log.exception("wipe_memory: TTS error (result)")
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Recall-name helper
    # ------------------------------------------------------------------

    def _handle_recall_name(self) -> State | None:
        """Handle 'what's my name?' — identify the speaker and deliver a vision roast.

        Known person path:
          Capture a fresh frame → 2-step GPT call (describe appearance → roast using
          name + description) → speak result.

        Unknown person path:
          Ask for their name → listen → enroll face in FaceDB → generate the same
          roast as a 'nice to meet you' greeting.

        Falls back to a canned line if the camera or GPT is unavailable.
        """
        if not self._face_recognizer.is_available():
            line = "Face recognition isn't available right now — I literally cannot see who you are!"
            log.info("recall_name: face recognition unavailable")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_name: TTS error (no face recognition)")
            finally:
                self._end_speech(servo_stop)
            return None

        _SCANNING_FILLERS = (
            "Let me get a look at you.",
            "Hold still.",
            "Scanning. Do not move.",
            "Cross-referencing my databanks.",
            "One moment. My memory is rusty.",
            "Running facial analysis.",
            "Checking my records.",
            "Give me a second. I know a lot of faces.",
        )

        # Capture frame first (fast), then kick off face recognition in the
        # background so it runs concurrently with the filler TTS rather than
        # adding to the silence after it.
        if self._camera.is_available():
            _pose, _tracker_paused = self._prepare_camera_pose()
            frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose, _tracker_paused)
        else:
            frame = None

        face_result: list = [None]

        def _identify() -> None:
            if frame:
                face_result[0] = self._face_recognizer.identify(
                    frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                )

        face_thread = threading.Thread(
            target=_identify, daemon=True, name="djr3x-face-identify"
        )
        face_thread.start()

        # Speak filler immediately — covers the 2-4 s recognition latency.
        filler = random.choice(_SCANNING_FILLERS)
        log.info("recall_name: filler %r", filler)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(filler)
        except Exception:
            log.exception("recall_name: filler TTS error")
        finally:
            self._end_speech(servo_stop)

        face_thread.join()
        result = face_result[0]

        if result is not None:
            person_id, name, distance = result
            self._cache_days_since_last_seen(person_id)
            daily_count = self._face_db.update_last_seen(person_id)
            self._award_recognized_wake_familiarity(
                person_id,
                daily_count,
                reason="recall-name recognition of day",
            )
            self._last_known_person_id = person_id
            memories_ctx = self._face_db.get_memories_as_context(person_id)
            if memories_ctx:
                self._llm.set_person_context(memories_ctx)
            log.info("recall_name: recognised person_id=%d name=%r distance=%.3f", person_id, name, distance)

            roast = self._greeter.generate_recall_roast(name, frame) if frame else ""
            if not roast:
                roast = (
                    f"That's {name}! You thought I'd forget? "
                    "I NEVER forget. Well — almost never. Don't push it."
                )

            log.info("recall_name (known): %s", roast)
            servo_stop = self._begin_speech(emotion="excited")
            try:
                self._synthesizer.speak(roast)
            except Exception:
                log.exception("recall_name: TTS error (known person roast)")
            finally:
                self._end_speech(servo_stop)
            return None

        # Unknown person — ask for their name.
        log.info("recall_name: face not recognised — asking for name")
        servo_stop = self._begin_speech(emotion="neutral")
        try:
            self._synthesizer.speak(_pick_no_repeat(_UNKNOWN_FACE_LINES, "unknown_face"))
        except Exception:
            log.exception("recall_name: TTS error (asking for name)")
        finally:
            self._end_speech(servo_stop)

        # Listen for their name.
        self._apply_listening_led_theme()
        self._wake_word.pause()
        try:
            self._wait_for_post_speech_listen_cooldown("recall_name name capture")
            name_text = self._transcriber.transcribe(
                wait_for_speech_seconds=config.WAKE_NO_SPEECH_TIMEOUT,
                allow_short=True,
            )
        except Exception:
            log.exception("recall_name: transcription error")
            name_text = None
        finally:
            self._wake_word.resume()

        self._apply_active_led_theme()

        if not name_text:
            log.info("recall_name: no name heard — aborting")
            return None

        name = _extract_name(name_text) or name_text.strip().title()
        log.info("recall_name: new person gave name %r", name)

        # Enroll — try a fresh frame first, fall back to the frame captured earlier.
        if self._camera.is_available():
            _pose, _tracker_paused = self._prepare_camera_pose()
            enroll_frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose, _tracker_paused)
        else:
            enroll_frame = None
        enc = None
        if enroll_frame:
            enc = self._face_recognizer.encode_face(enroll_frame, for_enrollment=True)
        if enc is None and frame:
            enc = self._face_recognizer.encode_face(frame, for_enrollment=True)

        if enc is not None:
            try:
                person_id = self._face_db.add_person(name, enc)
                self._last_known_person_id = person_id
                log.info("recall_name: enrolled new person %r id=%d", name, person_id)
            except Exception:
                log.exception("recall_name: FaceDB error enrolling %r", name)
        else:
            log.warning("recall_name: could not encode face for %r — skipping enrollment", name)

        # Roast them as a new person using the best available frame.
        roast_frame = enroll_frame or frame
        roast = self._greeter.generate_recall_roast(name, roast_frame) if roast_frame else ""
        if not roast:
            roast = f"Nice to meet you, {name}! Welcome to Oga's Cantina — try to keep up."

        log.info("recall_name (new): %s", roast)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(roast)
        except Exception:
            log.exception("recall_name: TTS error (new person roast)")
        finally:
            self._end_speech(servo_stop)
        return None

    # ------------------------------------------------------------------
    # Recall-memories helpers
    # ------------------------------------------------------------------

    def _recall_face_with_filler(self) -> tuple[int | None, str | None, str | None]:
        """Shared setup for recall_memories / recall_preference.

        Captures a frame, starts face recognition in background, speaks a filler
        phrase to cover the latency, then returns (person_id, name, frame).
        person_id and name are None when the face is not recognised.
        """
        if not self._face_recognizer.is_available():
            return None, None, None

        if self._camera.is_available():
            _pose, _tracker_paused = self._prepare_camera_pose()
            frame = self._camera.capture_frame()
            self._restore_servo_pose(_pose, _tracker_paused)
        else:
            frame = None

        face_result: list = [None]

        def _identify() -> None:
            if frame:
                face_result[0] = self._face_recognizer.identify(
                    frame, tolerance=config.FACE_RECOGNITION_TOLERANCE
                )

        face_thread = threading.Thread(
            target=_identify, daemon=True, name="djr3x-face-identify"
        )
        face_thread.start()

        filler = random.choice(self._MEMORY_FILLERS)
        log.info("recall: filler %r", filler)
        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(filler)
        except Exception:
            log.exception("recall: filler TTS error")
        finally:
            self._end_speech(servo_stop)

        face_thread.join()
        result = face_result[0]

        if result is None:
            return None, None, frame

        person_id, name, distance = result
        self._cache_days_since_last_seen(person_id)
        daily_count = self._face_db.update_last_seen(person_id)
        self._award_recognized_wake_familiarity(
            person_id,
            daily_count,
            reason="recall helper recognition of day",
        )
        self._last_known_person_id = person_id
        log.info("recall: recognised person_id=%d name=%r distance=%.3f", person_id, name, distance)
        return person_id, name, frame

    def _handle_recall_memories(self, original_text: str | None = None) -> None:
        """Tell the person what Rex knows about them from stored memories."""
        person_id, name, _frame = self._recall_face_with_filler()

        if person_id is None:
            line = (
                "I do not have a file on you. "
                "Come back when I know who you are."
            )
            log.info("recall_memories: face not recognised — speaking canned line")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_memories: TTS error (not recognised)")
            finally:
                self._end_speech(servo_stop)
            return

        memories_ctx = self._face_db.get_memories_as_context(person_id)

        if not memories_ctx:
            line = (
                "I know your face. Beyond that you are a mystery. "
                "Try talking to me more."
            )
            log.info("recall_memories: no memories stored for person_id=%d", person_id)
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_memories: TTS error (no memories)")
            finally:
                self._end_speech(servo_stop)
            return

        # Generate a Rex-style summary of the stored memories via GPT.
        summary = ""
        try:
            system_msg = (
                f"You are Rex, the droid DJ at Oga's Cantina on Batuu. "
                f"Summarize what you know about {name} in 2-3 sentences in your "
                f"snarky cantina DJ style. Reference specific details. "
                f"Here are your notes: {memories_ctx}. "
                f"Make it feel like you have been paying attention even if you "
                f"find it mildly annoying that you have."
            )
            response = self._llm._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Tell me what you know about {name}."},
                ],
                temperature=1.1,
            )
            summary = response.choices[0].message.content.strip()
            log.info("recall_memories: summary for %r: %s", name, summary)
        except Exception:
            log.exception("recall_memories: GPT summary failed — using fallback")

        if not summary:
            summary = (
                f"I know things about {name}. Interesting things. "
                "That is all I am going to say about that."
            )

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(summary)
        except Exception:
            log.exception("recall_memories: TTS error (summary)")
        finally:
            self._end_speech(servo_stop)

    def _handle_recall_preference(self, original_text: str | None = None) -> None:
        """Answer a specific preference question using stored memories."""
        person_id, name, _frame = self._recall_face_with_filler()

        if person_id is None:
            line = (
                "I do not have a file on you. "
                "Come back when I know who you are."
            )
            log.info("recall_preference: face not recognised — speaking canned line")
            servo_stop = self._begin_speech(emotion="neutral")
            try:
                self._synthesizer.speak(line)
            except Exception:
                log.exception("recall_preference: TTS error (not recognised)")
            finally:
                self._end_speech(servo_stop)
            return

        memories_ctx = self._face_db.get_memories_as_context(person_id)
        question = original_text or "what are my preferences"

        # Build the GPT prompt — include a fallback instruction for when the
        # specific preference is not on file so Rex stays in character.
        system_msg = (
            f"You are Rex, the droid DJ at Oga's Cantina on Batuu. "
            f"Based on what you know about {name}, answer this question: {question}. "
        )
        if memories_ctx:
            system_msg += (
                f"Your notes: {memories_ctx}. "
                f"Answer in Rex style, one sentence. "
                f"If the specific preference is not in your notes, admit you do not know "
                f"but make a joke about it — for example: 'I do not have that on file. "
                f"You should have talked more during your intake interview.'"
            )
        else:
            system_msg += (
                "You have no notes on this person yet. "
                "Tell them in Rex style that you do not have that information and they "
                "should talk to you more. One sentence."
            )

        answer = ""
        try:
            response = self._llm._vision_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ],
                temperature=1.1,
            )
            answer = response.choices[0].message.content.strip()
            log.info("recall_preference: answer for %r: %s", name, answer)
        except Exception:
            log.exception("recall_preference: GPT call failed — using fallback")

        if not answer:
            answer = (
                "I do not have that on file. "
                "You should have talked more during your intake interview."
            )

        servo_stop = self._begin_speech(emotion="excited")
        try:
            self._synthesizer.speak(answer)
        except Exception:
            log.exception("recall_preference: TTS error")
        finally:
            self._end_speech(servo_stop)
