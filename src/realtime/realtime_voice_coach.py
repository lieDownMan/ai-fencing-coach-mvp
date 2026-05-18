"""
Real-time Voice Coach — Zero-latency TTS feedback from coach_playbook.json.

This module bypasses the LLM API entirely.  When an error key is received
from the HeuristicsEngine it looks up the corresponding ``short_cue`` in
``coach_playbook.json`` and speaks it immediately via **pyttsx3** (offline
TTS, no network round-trip).

Usage
-----
    from realtime_voice_coach import RealtimeVoiceCoach

    coach = RealtimeVoiceCoach()
    coach.speak("guard_dropped")          # blocking: waits until utterance ends
    coach.speak_async("guard_dropped")    # non-blocking: queues & returns
    coach.shutdown()                      # stop worker thread (async mode)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import threading
import time
import base64
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLAYBOOK_PATH = Path(__file__).resolve().parents[2] / "coach_playbook.json"
_LEGACY_PLAYBOOK_PATH = Path(__file__).resolve().parent / "coach_playbook.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_playbook(path: Path | str | None = None) -> Dict[str, Any]:
    """Load and cache the playbook JSON from disk."""
    if path:
        candidates = [Path(path)]
    else:
        candidates = [_PLAYBOOK_PATH, _LEGACY_PLAYBOOK_PATH]

    target = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not target.exists():
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Coach playbook not found. Checked: {checked}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def _init_tts_engine():
    """Create a pyttsx3 engine with sensible defaults for coaching."""
    try:
        import pyttsx3
    except ImportError:
        logger.warning(
            "pyttsx3 is not installed. Will use macOS 'say' fallback if available."
        )
        return None

    try:
        engine = pyttsx3.init()
        # Slightly slower rate for clarity during physical activity
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        return engine
    except Exception as e:
        logger.warning("pyttsx3 init failed: %s. Will use macOS 'say' fallback.", e)
        return None


def _macos_say(text: str) -> bool:
    """Speak text using macOS built-in 'say' command (supports Chinese).
    
    This call BLOCKS until the speech finishes, so cues don't overlap.
    """
    if platform.system() != "Darwin":
        return False
    try:
        # Use Mei-Jia voice for Chinese; blocking call so cues play sequentially
        subprocess.run(
            ["say", "-v", "Mei-Jia", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        return False


def _windows_sapi_speak(text: str) -> bool:
    """Speak text through Windows SAPI in a short-lived PowerShell process."""
    if platform.system() != "Windows":
        return False

    script = f"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = -1
$speaker.Volume = 100
$voice = $speaker.GetInstalledVoices() |
  Where-Object {{ $_.VoiceInfo.Culture.Name -like 'zh*' }} |
  Select-Object -First 1
if ($voice) {{ $speaker.SelectVoice($voice.VoiceInfo.Name) }}
$speaker.Speak(@'
{text}
'@)
$speaker.Dispose()
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RealtimeVoiceCoach:
    """Look up ``short_cue`` by error key and speak it via pyttsx3.

    Parameters
    ----------
    playbook_path : str or Path, optional
        Override location of ``coach_playbook.json``.
    cooldown_seconds : float
        Minimum interval between repeating the *same* cue to avoid
        spamming the fencer with identical messages.
    """

    def __init__(
        self,
        playbook_path: Optional[str | Path] = None,
        cooldown_seconds: float = 3.0,
    ):
        self.playbook = _load_playbook(playbook_path)
        self.cooldown = cooldown_seconds

        # Track last-spoken time per key to enforce cooldown
        self._last_spoken: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Lazy-init TTS engine (created on first use or in worker thread)
        self._engine = None
        self._async_queue: Optional[Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API — Synchronous
    # ------------------------------------------------------------------

    def speak(self, error_key: str) -> bool:
        """Look up *error_key* and speak the short_cue **synchronously**.

        Returns ``True`` if the cue was spoken, ``False`` if skipped
        (unknown key, cooldown active, or TTS unavailable).
        """
        cue = self._resolve_cue(error_key)
        if cue is None:
            return False
        if not self._cooldown_ok(error_key):
            return False

        if _windows_sapi_speak(cue):
            logger.info("Voice coach (sync/windows): [%s] -> %s", error_key, cue)
            return True

        if self._engine is None:
            self._engine = _init_tts_engine()
        if self._engine is None:
            return False

        logger.info("Voice coach (sync): [%s] → %s", error_key, cue)
        self._engine.say(cue)
        self._engine.runAndWait()
        return True

    # ------------------------------------------------------------------
    # Public API — Asynchronous (non-blocking)
    # ------------------------------------------------------------------

    def speak_async(self, error_key: str) -> bool:
        """Queue the cue and return immediately.

        A background worker thread will speak queued cues in order.
        Returns ``True`` if enqueued, ``False`` if skipped.
        """
        cue = self._resolve_cue(error_key)
        if cue is None:
            return False
        if not self._cooldown_ok(error_key):
            return False

        self._ensure_worker()
        self._async_queue.put((error_key, cue))  # type: ignore[union-attr]
        logger.info("Voice coach (async): enqueued [%s]", error_key)
        return True

    def shutdown(self) -> None:
        """Stop the background worker and release TTS resources."""
        self._shutdown_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_cue(self, error_key: str) -> Optional[str]:
        """Return the short_cue text for *error_key* without speaking."""
        return self._resolve_cue(error_key)

    def get_diagnosis(self, error_key: str) -> Optional[str]:
        """Return the full diagnosis text for *error_key*."""
        entry = self.playbook.get(error_key)
        if entry is None:
            return None
        return entry.get("diagnosis")

    def get_error_name(self, error_key: str) -> Optional[str]:
        """Return the human-readable error name for *error_key*."""
        entry = self.playbook.get(error_key)
        if entry is None:
            return None
        return entry.get("error_name")

    def list_keys(self):
        """Return all valid error keys."""
        return list(self.playbook.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_cue(self, error_key: str) -> Optional[str]:
        entry = self.playbook.get(error_key)
        if entry is None:
            logger.warning("Unknown error key: %s", error_key)
            return None
        return entry.get("short_cue")

    def _cooldown_ok(self, error_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last_spoken.get(error_key, 0.0)
            if now - last < self.cooldown:
                logger.debug("Cooldown active for [%s], skipping.", error_key)
                return False
            self._last_spoken[error_key] = now
        return True

    def _ensure_worker(self) -> None:
        if self._async_queue is None:
            self._async_queue = Queue()
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._shutdown_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="voice-coach"
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background loop: pulls (error_key, cue) from the queue and speaks."""
        system = platform.system()
        if system == "Windows":
            engine = None
            logger.info("Voice worker: using Windows SAPI subprocess backend.")
        elif system == "Darwin":
            engine = None
            logger.info("Voice worker: using macOS 'say' command for TTS.")
        else:
            engine = _init_tts_engine()
            if engine is None:
                logger.error("Voice worker: no TTS backend available. Worker exiting.")
                return

        while not self._shutdown_event.is_set():
            try:
                error_key, cue = self._async_queue.get(timeout=0.5)  # type: ignore[union-attr]
            except Empty:
                continue
            logger.info("Voice worker speaking: [%s] → %s", error_key, cue)
            try:
                if system == "Windows":
                    if not _windows_sapi_speak(cue):
                        logger.error("Windows SAPI failed for [%s]", error_key)
                elif engine is not None:
                    engine.say(cue)
                    engine.runAndWait()
                else:
                    _macos_say(cue)
                # Brief pause between consecutive cues for clarity
                time.sleep(0.5)
            except Exception as e:
                logger.error("TTS error for [%s]: %s", error_key, e)

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coach = RealtimeVoiceCoach()
    print("Available error keys:", coach.list_keys())
    for key in coach.list_keys():
        cue = coach.get_cue(key)
        print(f"  [{key}] → {cue}")
    # Uncomment below to actually hear it (requires audio output)
    # coach.speak("guard_dropped")
    coach.shutdown()
