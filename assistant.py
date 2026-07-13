#!/usr/bin/env python3
"""
Enhanced Voice Assistant with:
- Async/aiohttp for non-blocking operations
- Connection pooling for HA API
- Streaming TTS playback
- Follow-up mode (listen briefly after response without wake word)
- Barge-in support (interrupt playback with wake word)
- Success/failure confirmation sounds
- Volume normalization
- WebRTC VAD for better speech detection
- Noise suppression
- Echo cancellation (basic)
- Entity state caching with TTL
- Fuzzy entity/area matching
- State-aware responses
- Scene suggestions
- Health monitoring
- Graceful degradation (offline responses)
- Watchdog timer
- Dataclass-based configuration
- Structured logging with request IDs
- MODULARIZED: Config extracted to config.py

PATCHES applied (see CHANGES below each affected section):
  #1  VAD record loop: consume frame even when echo-suppressing (prevents buffer drift)
  #2  VAD frame size: computed from actual frame duration, not hardcoded 480 samples
  #3  PyAudio stream: opened once at startup and reused across all TTS calls
  #4  Prewarm OpenAI: cheap max_tokens=1 chat call instead of models.list()
  #5  Health check OpenAI: same cheap ping instead of models.list()
  #6  Brightness regex: anchored so "3000k" no longer extracts "300" as brightness
  #7  trim_conversation: index arithmetic instead of list.index() (no ValueError risk)
  #8  Follow-up VAD threshold: uses ambient-calibrated value, not hardcoded *3.0
  #9  debug_tools / debug_ha defaults moved to False in config.py
  #10 SFX listening_wav: now actually played when recording begins
  #11 Thread pool: max_workers raised from 4 to 6
"""

import os
import sys
import wave
import time
import signal
import tempfile
import threading
import subprocess
import json
import re
import logging
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache

import socket

import numpy as np
import sounddevice as sd
import aiohttp
import requests
import websocket
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import queue

import fitness_plan as fp   # Geo's Summer Reset plan data + schedule helpers

# Wake word engines — openWakeWord (via wyoming-openwakeword container) is the
# default; Porcupine is optional and only needed when WAKE_ENGINE=porcupine.
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    pvporcupine = None

# Claude is the preferred conversation brain; OpenAI remains the fallback
# (and still handles STT + TTS either way).
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Import config from modular config.py
from config import (
    AssistantConfig, AudioConfig, WakeWordConfig, TTSConfig, SFXConfig,
    HomeAssistantConfig, ConversationConfig, HealthConfig,
    load_config
)

# pyaudio for low-latency streaming playback
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

# Optional imports with fallbacks
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    webrtcvad = None

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    nr = None

try:
    from rapidfuzz import fuzz, process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    fuzz = None
    fuzz_process = None


# ============================
# Structured Logging with Request IDs
# ============================
class RequestContextFilter(logging.Filter):
    """Add request_id to log records."""

    _context = threading.local()

    @classmethod
    def set_request_id(cls, request_id: str):
        cls._context.request_id = request_id

    @classmethod
    def get_request_id(cls) -> str:
        return getattr(cls._context, 'request_id', '-')

    @classmethod
    def clear_request_id(cls):
        cls._context.request_id = '-'

    def filter(self, record):
        record.request_id = self.get_request_id()
        return True


LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

logger = logging.getLogger("voice_assistant")
_handler = logging.StreamHandler(sys.stdout)
_filter = RequestContextFilter()
_handler.addFilter(_filter)
_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)


def new_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:8]


# ============================
# Helpers
# ============================
def _normalize_text(s: str) -> str:
    s = (s or "").replace("_", " ").strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == "#").strip()


# ============================
# Global State
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYBACK_LOCK = threading.Lock()
_shutdown_event = threading.Event()
# Set for the full duration of a wake conversation (realtime or legacy) so the
# proactive announcer stays silent while Geo is talking to Jarvis.
_conversation_active = threading.Event()
_wake_detector_ref: Optional[Any] = None
_playback_interrupt = threading.Event()
_last_activity_time = time.time()
_activity_lock = threading.Lock()

# Thread pool for parallel operations (transcription, TTS, HA calls)
_thread_pool: Optional[ThreadPoolExecutor] = None

# Connection pool session
_http_session: Optional[requests.Session] = None
_async_session: Optional[aiohttp.ClientSession] = None

# OpenAI clients
sync_client: Optional[OpenAI] = None
async_client: Optional[AsyncOpenAI] = None

# Anthropic client (None when no ANTHROPIC_API_KEY is configured)
anthropic_client: Optional[Any] = None

# PATCH #3: persistent PyAudio stream reused across all TTS calls
_pyaudio_instance: Optional[Any] = None
_pyaudio_stream: Optional[Any] = None
_pyaudio_stream_lock = threading.Lock()


def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the global thread pool."""
    global _thread_pool
    if _thread_pool is None:
        # PATCH #11: raised from 4 to 6 workers to avoid bottleneck with parallel ops
        _thread_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="assistant")
    return _thread_pool


# ============================
# Persistent memory
# ============================
# Long-term facts Jarvis keeps ACROSS conversations (preferences, names,
# recurring plans). Stored as a small JSON list beside this file; injected
# into every session's system prompt so the model always knows them, and
# mutated by the remember/forget tools. Kept deliberately small (a briefing
# fact sheet, not a database) — capped so the prompt stays lean.
MEMORY_PATH = os.path.join(BASE_DIR, "jarvis_memory.json")
_MEMORY_MAX = 40                # oldest dropped past this
_memory_lock = threading.Lock()


def _load_memories() -> List[Dict[str, Any]]:
    try:
        with open(MEMORY_PATH) as f:
            data = json.load(f)
        items = data.get("memories", []) if isinstance(data, dict) else data
        return [m for m in items if isinstance(m, dict) and m.get("text")]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_memories(items: List[Dict[str, Any]]) -> None:
    tmp = MEMORY_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"memories": items[-_MEMORY_MAX:]}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, MEMORY_PATH)          # atomic: never a half-written file
    except OSError as e:
        logger.warning(f"Memory save failed: {e}")


def add_memory(text: str) -> Dict[str, Any]:
    """Store one durable fact. De-dupes near-identical text."""
    text = " ".join((text or "").split())[:240]
    if not text:
        return {"error": "nothing to remember"}
    with _memory_lock:
        items = _load_memories()
        norm = _normalize_text(text)
        for m in items:
            if _normalize_text(m["text"]) == norm:
                return {"status": "already remembered", "text": m["text"]}
        items.append({"text": text, "ts": datetime.now().strftime("%Y-%m-%d")})
        _save_memories(items)
        logger.info(f"Memory added: {text!r} ({len(items)} total)")
        return {"status": "remembered", "text": text, "total": len(items)}


def forget_memory(query: str) -> Dict[str, Any]:
    """Remove memories matching `query` (word overlap / substring). Empty
    query is refused so 'forget it' can't wipe everything by accident."""
    q = _normalize_text(query)
    if not q:
        return {"error": "say what to forget (e.g. 'forget my coffee order')"}
    with _memory_lock:
        items = _load_memories()
        qwords = set(q.split())
        kept, removed = [], []
        for m in items:
            mn = _normalize_text(m["text"])
            hit = q in mn or mn in q or (qwords & set(mn.split()) and
                  len(qwords & set(mn.split())) >= max(1, len(qwords) // 2))
            (removed if hit else kept).append(m)
        if not removed:
            return {"error": f"nothing matches '{query}'",
                    "current_memories": [m["text"] for m in items]}
        _save_memories(kept)
        logger.info(f"Memory forgot {len(removed)}: {[m['text'] for m in removed]}")
        return {"status": "forgotten", "removed": [m["text"] for m in removed],
                "remaining": len(kept)}


def _memory_prompt_block() -> str:
    """Rendered into the system prompt each session so the model always has
    the facts. Empty string when there are none (adds nothing to the prompt)."""
    items = _load_memories()
    if not items:
        return ""
    lines = "\n".join(f"- {m['text']}" for m in items)
    return ("\n\nWHAT YOU KNOW ABOUT GEO (persistent memory; use naturally, "
            "don't recite unless asked):\n" + lines)


# ============================
# HUD event broadcaster (SSE)
# ============================
class HudBroadcaster:
    """Streams assistant state + live audio spectrum to the kiosk HUD.

    Serves Server-Sent Events on 127.0.0.1:<port>/events (stdlib only, no
    deps). The HUD page (served on :8099) connects with EventSource — SSE
    needs the CORS header but nothing else. While TTS plays, feed_pcm()
    turns each PCM chunk into log-spaced frequency-band levels so the HUD
    equalizer follows the actual voice."""

    BANDS = 16
    F_LO, F_HI = 90.0, 7500.0   # speech band
    SPAN_DB = 28.0               # dB of visible dynamic range below the AGC peak
    AGC_FLOOR = -46.0            # dB; AGC never chases quieter than this (noise)
    AGC_DECAY = 0.35             # dB per frame (~8 dB/s) the reference falls

    def __init__(self):
        self._clients: List["queue.Queue[str]"] = []
        self._lock = threading.Lock()
        self._state = "idle"
        self._started = False
        self.say_cb: Optional[Callable[[str], Any]] = None  # set in main()
        self._window: Dict[int, np.ndarray] = {}
        self._eq_ema: Optional[np.ndarray] = None  # temporal smoothing state
        self._agc_db = self.AGC_FLOOR              # slow auto-gain reference
        # broadcasts scheduled for the moment the audio actually leaves the
        # speaker (output buffers make the feed lead the ear by 100-400ms)
        self._delayed: deque = deque()
        self._delay_lock = threading.Lock()

    def start(self, port: int):
        if self._started:
            return
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_OPTIONS(self):
                """CORS preflight for the dashboard card's fetch() to /alarm
                (the HA page on :8123 is a different origin than :8765)."""
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                """POST /say {"message": "..."} — spoken announcement (HA
                automations via rest_command.jarvis_say). POST /alarm
                {"action": "set"|"cancel", ...} — dashboard alarm control,
                same AlarmManager the voice tools use. Localhost-only
                (server binds 127.0.0.1)."""
                path = self.path.rstrip("/")
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                except Exception:
                    body = {}
                if path == "/say":
                    message = str(body.get("message", "")).strip()[:500]
                    if message and hub.say_cb:
                        hub.say_cb(message)
                        payload = b'{"ok": true}'
                    else:
                        payload = b'{"ok": false, "error": "no message or announcer offline"}'
                elif path == "/alarm":
                    action = str(body.get("action", ""))
                    if action == "set":
                        result = alarms.set_alarm(str(body.get("at_time", "")),
                                                  label=str(body.get("label", ""))[:100],
                                                  repeat=str(body.get("repeat", "")))
                    elif action == "cancel":
                        result = alarms.cancel(str(body.get("which", "")))
                    elif action == "volume":
                        result = alarms.set_volume(body.get("level"))
                    else:
                        result = {"error": f"unknown action '{action}'"}
                    payload = json.dumps(result).encode()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path.rstrip("/") not in ("/events", ""):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q: "queue.Queue[str]" = queue.Queue(maxsize=64)
                with hub._lock:
                    hub._clients.append(q)
                try:
                    self.wfile.write(
                        f"data: {json.dumps({'t': 'state', 'v': hub._state})}\n\n".encode()
                    )
                    # snapshot of timers/alarms so a (re)connecting client
                    # renders current state without waiting for a mutation
                    for kind, items in (("timers", timers.snapshot()),
                                        ("alarms", alarms.snapshot()),
                                        ("alarmvol", alarms.volume),
                                        ("weight", weight_stats())):
                        self.wfile.write(
                            f"data: {json.dumps({'t': kind, 'v': items})}\n\n".encode()
                        )
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                        except queue.Empty:
                            msg = ": ping\n\n"  # keepalive comment
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with hub._lock:
                        if q in hub._clients:
                            hub._clients.remove(q)

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True,
                             name="hud-sse").start()
            threading.Thread(target=self._pacer, daemon=True,
                             name="hud-pacer").start()
            self._started = True
            logger.info(f"HUD event stream on http://127.0.0.1:{port}/events")
        except OSError as e:
            logger.warning(f"HUD event stream disabled (port {port}): {e}")

    def _broadcast(self, obj: Dict[str, Any]):
        if not self._started:
            return
        msg = f"data: {json.dumps(obj)}\n\n"
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # slow client; drop frame

    def _pacer(self):
        """Deliver delayed broadcasts at their due time (10ms resolution)."""
        while True:
            item = None
            with self._delay_lock:
                if self._delayed and self._delayed[0][0] <= time.monotonic():
                    item = self._delayed.popleft()[1]
            if item is not None:
                self._broadcast(item)
            else:
                time.sleep(0.01)

    def _broadcast_at(self, obj: Dict[str, Any], delay: float):
        if delay <= 0.0:
            self._broadcast(obj)
            return
        with self._delay_lock:
            self._delayed.append((time.monotonic() + delay, obj))

    def set_state(self, state: str):
        if state == self._state:
            return
        self._state = state
        if state == "idle":
            # let delayed tail-of-audio frames land before the clock returns
            self._broadcast_at({"t": "state", "v": "idle"}, 0.6)
        else:
            # a new interaction outruns any still-queued idle transition
            with self._delay_lock:
                self._delayed = deque(
                    (due, obj) for due, obj in self._delayed if obj.get("t") != "state"
                )
            self._broadcast({"t": "state", "v": state})

    def feed_pcm(self, pcm_bytes: bytes, rate: int, delay: float = 0.0):
        """Compute equalizer band levels from an int16 mono PCM chunk.

        `delay` shifts the broadcast to when this chunk becomes audible
        (output-buffer depth for PyAudio, startup+buffer for aplay), so the
        bars move in sync with the ear instead of leading it.

        Classic visualizer pipeline: band energies in dB (with a treble tilt
        so consonants read), mapped against a slow auto-gain reference so the
        full bar range is used without per-frame renormalization — loudness
        changes you can hear become height changes you can see. Neighbor
        blending + asymmetric temporal EMA (fast attack, slower release)
        happen here so every consumer gets smooth, coherent motion."""
        if not self._started or not self._clients or len(pcm_bytes) < 512:
            return
        try:
            x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(x)
            rms_v = float(np.sqrt(np.mean(x * x)))
            silent = 20.0 * np.log10(rms_v + 1e-9) < -44.0
            if silent:
                raw = np.zeros(self.BANDS, dtype=np.float32)
            else:
                win = self._window.get(n)
                if win is None:
                    win = np.hanning(n).astype(np.float32)
                    self._window[n] = win
                spec = np.abs(np.fft.rfft(x * win)) / n
                freqs = np.fft.rfftfreq(n, 1.0 / rate)
                edges = np.geomspace(self.F_LO, min(self.F_HI, rate / 2 - 1), self.BANDS + 1)
                dbs = np.empty(self.BANDS, dtype=np.float32)
                for i in range(self.BANDS):
                    sel = spec[(freqs >= edges[i]) & (freqs < edges[i + 1])]
                    mag = float(sel.mean()) if sel.size else 0.0
                    dbs[i] = 20.0 * np.log10(mag + 1e-9) + 12.0 * i / (self.BANDS - 1)
                # blend neighbors so the row reads as one curve, not spikes
                sm = dbs.copy()
                sm[1:-1] = 0.25 * dbs[:-2] + 0.5 * dbs[1:-1] + 0.25 * dbs[2:]
                # slow AGC: reference rides recent peaks, falls ~8 dB/s
                self._agc_db = max(float(sm.max()), self._agc_db - self.AGC_DECAY,
                                   self.AGC_FLOOR)
                raw = np.clip((sm - self._agc_db + self.SPAN_DB) / self.SPAN_DB, 0.0, 1.0)
            ema = self._eq_ema
            if ema is None or len(ema) != self.BANDS:
                ema = raw
            else:
                alpha = np.where(raw > ema, 0.80, 0.50)  # fast attack, slower release
                ema = ema + (raw - ema) * alpha
            self._eq_ema = ema
            self._broadcast_at({"t": "eq", "v": [round(float(v), 2) for v in ema]},
                               delay)
        except Exception:
            pass  # visualizer must never break playback

    def feed_wav_async(self, wav_path: str, delay: float = 0.08):
        """Stream a WAV file's levels in real time while aplay plays it
        (SFX chimes and non-streaming TTS bypass the PyAudio tap).
        `delay` covers aplay's device-open + buffer-prefill lead time."""
        if not self._started or not self._clients or not wav_path:
            return

        def run():
            try:
                with wave.open(wav_path, "rb") as wf:
                    if wf.getsampwidth() != 2:
                        return
                    rate = wf.getframerate()
                    ch = wf.getnchannels()
                    step = int(rate * 0.04)  # 40ms frames, paced to real time
                    t0 = time.monotonic()
                    sent = 0
                    while True:
                        data = wf.readframes(step)
                        if len(data) < 512 * ch:
                            break
                        x = np.frombuffer(data, dtype=np.int16)
                        if ch > 1:
                            x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
                        self.feed_pcm(x.tobytes(), rate, delay=delay)
                        sent += 1
                        wait = t0 + sent * 0.04 - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True, name="hud-wav").start()


hud = HudBroadcaster()


# ============================
# Voice timers & reminders
# ============================
class TimerManager:
    """Named in-memory timers/reminders set by voice. On fire: chime, speak
    the reminder, and update the HUD countdown. Deliberately not persisted —
    these are desk timers, not calendar events."""

    MAX_SECONDS = 24 * 3600

    def __init__(self):
        self._lock = threading.Lock()
        self._timers: Dict[str, Dict[str, Any]] = {}
        self._config: Optional[AssistantConfig] = None
        self._counter = 0

    def attach(self, config: AssistantConfig):
        self._config = config

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(
                ({"label": t["label"], "fires_at_ms": int(t["fires_at"] * 1000)}
                 for t in self._timers.values()),
                key=lambda x: x["fires_at_ms"],
            )

    def _push_hud(self):
        hud._broadcast({"t": "timers", "v": self.snapshot()})

    def set_timer(self, duration_seconds: Optional[float] = None,
                  at_time: str = "", label: str = "") -> Dict[str, Any]:
        if not duration_seconds and at_time:
            try:
                hh, mm = (int(p) for p in at_time.strip().split(":")[:2])
                now = datetime.now()
                target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                duration_seconds = (target - now).total_seconds()
            except Exception:
                return {"error": f"could not parse at_time '{at_time}' (want 24h HH:MM)"}
        try:
            delay = float(duration_seconds or 0)
        except (TypeError, ValueError):
            return {"error": "invalid duration"}
        if not 1 <= delay <= self.MAX_SECONDS:
            return {"error": f"duration must be 1s..24h, got {delay:.0f}s"}

        with self._lock:
            self._counter += 1
            tid = f"t{self._counter}"
            t = threading.Timer(delay, self._fire, args=(tid,))
            t.daemon = True
            self._timers[tid] = {
                "label": (label or "").strip(),
                "fires_at": time.time() + delay,
                "timer": t,
            }
        t.start()
        self._push_hud()
        fires = datetime.now() + timedelta(seconds=delay)
        logger.info(f"Timer {tid} set: {delay:.0f}s ('{label}')")
        return {"id": tid, "label": label, "fires_in_seconds": int(delay),
                "fires_at": fires.strftime("%I:%M %p")}

    def _fire(self, tid: str):
        with self._lock:
            info = self._timers.pop(tid, None)
        if info is None:
            return
        self._push_hud()
        cfg = self._config
        if cfg is None:
            return
        try:
            msg = (f"Sir, a reminder: {info['label']}."
                   if info["label"] else "Sir, your timer is up.")
            logger.info(f"Timer {tid} fired: {msg}")
            chime = cfg.sfx.alarm_wav or cfg.sfx.success_wav
            if chime:
                play_wav_file(chime, "TIMER_CHIME", cfg)
            speak_tts(msg, cfg)
            hud.set_state("idle")
        except Exception as e:
            logger.error(f"Timer announcement failed: {e}")

    def cancel(self, which: str = "") -> Dict[str, Any]:
        which = (which or "").strip().lower()
        cancelled = []
        with self._lock:
            for tid, info in list(self._timers.items()):
                if which in ("", "all", tid) or which in info["label"].lower():
                    info["timer"].cancel()
                    del self._timers[tid]
                    cancelled.append(info["label"] or tid)
        self._push_hud()
        if not cancelled:
            return {"error": f"no timer matching '{which}'"}
        return {"cancelled": cancelled}

    def list(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            items = sorted(
                ({"label": t["label"] or "(unnamed)",
                  "remaining_seconds": max(0, int(t["fires_at"] - now)),
                  "fires_at": datetime.fromtimestamp(t["fires_at"]).strftime("%I:%M %p")}
                 for t in self._timers.values()),
                key=lambda x: x["remaining_seconds"],
            )
        return {"timers": items, "count": len(items)}


timers = TimerManager()


# ============================
# Wall-clock alarms
# ============================
class AlarmManager:
    """Wall-clock alarms, set by voice or from the dashboard (POST /alarm on
    the HUD server). Unlike TimerManager's countdown timers these PERSIST
    across restarts (alarms.json beside this file) and can repeat daily or on
    weekdays. Firing is a 1 s poll loop comparing wall-clock time rather than
    threading.Timer delays — the Pi has no RTC, so an absolute delay armed
    before NTP settles (or across any clock jump) would misfire. When one
    fires: chime + spoken announcement, repeated every RING_GAP seconds up to
    RING_COUNT times unless dismissed. cancel() silences a ringing alarm
    (a ringing repeater keeps its schedule) and deletes pending ones."""

    RING_COUNT = 3
    RING_GAP = 45.0

    def __init__(self):
        self._lock = threading.Lock()
        self._alarms: Dict[str, Dict[str, Any]] = {}
        self._config: Optional["AssistantConfig"] = None
        self._counter = 0
        self._volume = 80   # 10..100, dash-controlled, persisted
        self._loaded = False   # guards saves: a mutation arriving over HTTP
                               # before attach() loads alarms.json would
                               # otherwise persist empty state over real alarms
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "alarms.json")

    def attach(self, config: "AssistantConfig"):
        self._config = config
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._counter = int(data.get("counter", 0))
            self._volume = max(10, min(100, int(data.get("volume", 80))))
            now = time.time()
            for a in data.get("alarms", []):
                if a.get("repeat"):
                    a["next_fire"] = self._next_fire(a["time"], a["repeat"])
                elif float(a.get("next_fire", 0)) <= now:
                    logger.info(f"Dropping alarm {a.get('id')} ({a.get('time')}) "
                                "missed while the assistant was down")
                    continue
                a["ringing"] = False
                self._alarms[a["id"]] = a
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Alarm load failed: {e}")
        self._loaded = True
        threading.Thread(target=self._poll_loop, daemon=True,
                         name="alarm-poll").start()
        self._push_hud()

    # ---------- persistence (call with lock held) ----------
    def _save_locked(self):
        try:
            data = {"counter": self._counter,
                    "volume": self._volume,
                    "alarms": [{k: a[k] for k in
                                ("id", "time", "label", "repeat", "next_fire")}
                               for a in self._alarms.values()]}
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"Alarm save failed: {e}")

    # ---------- schedule math ----------
    @staticmethod
    def _parse_hhmm(at_time: str) -> Tuple[int, int]:
        hh, mm = (int(p) for p in at_time.strip().split(":")[:2])
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(at_time)
        return hh, mm

    # python weekday(): Mon=0 .. Sun=6
    _DAY_IDX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

    @classmethod
    def _repeat_days(cls, repeat: str) -> Optional[set]:
        """Allowed weekdays for a repeat, or None = every day (once/daily).
        Accepts 'daily', 'weekdays', 'weekends', or a comma/space list of day
        abbreviations ('sun,mon,tue,wed,thu'). Returns None for unrecognized
        so callers can validate."""
        repeat = (repeat or "").strip().lower()
        if repeat in ("", "daily"):
            return None
        if repeat == "weekdays":
            return {0, 1, 2, 3, 4}
        if repeat == "weekends":
            return {5, 6}
        days = {cls._DAY_IDX[t[:3]] for t in re.split(r"[,\s]+", repeat)
                if t[:3] in cls._DAY_IDX}
        return days or None

    @classmethod
    def _normalize_repeat(cls, repeat: str) -> Optional[str]:
        """Canonical repeat string, or None if invalid. A day-set canonicalizes
        to Mon..Sun-ordered abbreviations ('mon,tue,wed,thu,sun')."""
        repeat = (repeat or "").strip().lower()
        if repeat in ("", "daily", "weekdays", "weekends"):
            return repeat
        days = cls._repeat_days(repeat)
        if not days:
            return None
        inv = {v: k for k, v in cls._DAY_IDX.items()}
        return ",".join(inv[i] for i in sorted(days))

    def _next_fire(self, at_time: str, repeat: str) -> float:
        hh, mm = self._parse_hhmm(at_time)
        days = self._repeat_days(repeat)
        now = datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        while target <= now or (days is not None and target.weekday() not in days):
            target += timedelta(days=1)
        return target.timestamp()

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(
                ({"id": a["id"], "time": a["time"], "label": a["label"],
                  "repeat": a["repeat"], "next_ms": int(a["next_fire"] * 1000),
                  "ringing": bool(a.get("ringing"))}
                 for a in self._alarms.values()),
                key=lambda x: x["next_ms"],
            )

    def _push_hud(self):
        hud._broadcast({"t": "alarms", "v": self.snapshot()})

    @property
    def volume(self) -> int:
        return self._volume

    def set_volume(self, level: Any) -> Dict[str, Any]:
        if not self._loaded:
            return {"error": "alarms still loading, try again"}
        try:
            level = max(10, min(100, int(level)))
        except (TypeError, ValueError):
            return {"error": f"invalid volume '{level}'"}
        with self._lock:
            self._volume = level
            self._save_locked()
        hud._broadcast({"t": "alarmvol", "v": level})
        logger.info(f"Alarm volume set to {level}%")
        return {"volume": level}

    def set_alarm(self, at_time: str = "", label: str = "",
                  repeat: str = "") -> Dict[str, Any]:
        if not self._loaded:
            return {"error": "alarms still loading, try again"}
        canon = self._normalize_repeat(repeat)
        if canon is None:
            return {"error": f"repeat must be '', 'daily', 'weekdays', 'weekends', "
                             f"or a day list like 'sun,mon,tue,wed,thu', got '{repeat}'"}
        repeat = canon
        try:
            fire = self._next_fire(at_time, repeat)
            hh, mm = self._parse_hhmm(at_time)
        except Exception:
            return {"error": f"could not parse at_time '{at_time}' (want 24h HH:MM)"}
        canonical = f"{hh:02d}:{mm:02d}"
        label = (label or "").strip()
        with self._lock:
            # same time + same label = replace, not duplicate — this is how
            # "change my 6:20 alarm to weekdays" works (the LLM re-sets it)
            for aid_, a in list(self._alarms.items()):
                if a["time"] == canonical and a["label"].lower() == label.lower():
                    del self._alarms[aid_]
            self._counter += 1
            aid = f"a{self._counter}"
            self._alarms[aid] = {"id": aid, "time": canonical,
                                 "label": label,
                                 "repeat": repeat, "next_fire": fire,
                                 "ringing": False}
            self._save_locked()
        self._push_hud()
        when = datetime.fromtimestamp(fire)
        logger.info(f"Alarm {aid} set for {when:%A %H:%M} repeat='{repeat}' ('{label}')")
        return {"id": aid, "time": canonical, "label": label, "repeat": repeat,
                "fires_at": when.strftime("%A %I:%M %p")}

    def _poll_loop(self):
        while True:
            due = []
            now = time.time()
            with self._lock:
                for a in self._alarms.values():
                    if not a["ringing"] and a["next_fire"] <= now:
                        a["ringing"] = True
                        due.append(a["id"])
            for aid in due:
                threading.Thread(target=self._ring, args=(aid,), daemon=True,
                                 name=f"alarm-{aid}").start()
            time.sleep(1.0)

    def _ring(self, aid: str):
        with self._lock:
            a = self._alarms.get(aid)
            if a is None:
                return
            label, hhmm, repeat = a["label"], a["time"], a["repeat"]
            if repeat:  # schedule the next occurrence up front
                a["next_fire"] = self._next_fire(hhmm, repeat)
            self._save_locked()
        self._push_hud()
        hh, mm = self._parse_hhmm(hhmm)
        spoken = datetime.now().replace(hour=hh, minute=mm).strftime("%I:%M %p").lstrip("0")
        msg = (f"Sir, it is {spoken}. {label}." if label
               else f"Sir, it is {spoken}. This is your alarm.")
        cfg = self._config
        for i in range(self.RING_COUNT):
            with self._lock:
                a = self._alarms.get(aid)
                if a is None or not a["ringing"]:
                    return  # dismissed
            try:
                logger.info(f"Alarm {aid} ringing ({i + 1}/{self.RING_COUNT}): {msg}")
                chime = cfg and (cfg.sfx.alarm_wav or cfg.sfx.success_wav)
                # dash-set volume applies to chime AND announcement
                set_playback_gain(self._volume / 100.0)
                try:
                    if chime:
                        play_wav_file(chime, "ALARM_CHIME", cfg)
                    if cfg:
                        speak_tts(msg, cfg)
                        hud.set_state("idle")
                finally:
                    set_playback_gain(1.0)
            except Exception as e:
                logger.error(f"Alarm announcement failed: {e}")
            for _ in range(int(self.RING_GAP)):  # dismissable mid-wait
                with self._lock:
                    a = self._alarms.get(aid)
                    if a is None or not a["ringing"]:
                        return
                time.sleep(1.0)
        self._dismiss(aid)

    def _dismiss(self, aid: str):
        with self._lock:
            a = self._alarms.get(aid)
            if a is None:
                return
            a["ringing"] = False
            if not a["repeat"]:
                del self._alarms[aid]
            self._save_locked()
        self._push_hud()

    def cancel(self, which: str = "") -> Dict[str, Any]:
        if not self._loaded:
            return {"error": "alarms still loading, try again"}
        which = (which or "").strip().lower()
        dismissed, cancelled = [], []

        def matches(a):
            return (which in ("", "all", a["id"])
                    or (which and which in a["label"].lower())
                    or which in (a["time"], a["time"].lstrip("0")))

        with self._lock:
            ringing = [a for a in self._alarms.values() if a["ringing"] and matches(a)]
            # "stop!" while one is going off silences it and touches nothing else
            targets = ringing if ringing else \
                [a for a in self._alarms.values() if matches(a)]
            for a in targets:
                name = a["label"] or a["time"]
                if a["ringing"]:
                    a["ringing"] = False
                    if not a["repeat"]:
                        del self._alarms[a["id"]]
                    dismissed.append(name)
                else:
                    del self._alarms[a["id"]]
                    cancelled.append(name)
            if targets:
                self._save_locked()
        self._push_hud()
        if not (dismissed or cancelled):
            return {"error": f"no alarm matching '{which}'"}
        out: Dict[str, Any] = {}
        if dismissed:
            out["dismissed"] = dismissed
        if cancelled:
            out["cancelled"] = cancelled
        return out

    def list(self) -> Dict[str, Any]:
        items = [{"label": a["label"] or "(unnamed)", "time": a["time"],
                  "repeat": a["repeat"] or "once", "ringing": a["ringing"],
                  "fires_at": datetime.fromtimestamp(a["next_ms"] / 1000)
                                      .strftime("%A %I:%M %p")}
                 for a in self.snapshot()]
        return {"alarms": items, "count": len(items)}


alarms = AlarmManager()


def get_pyaudio():
    """Get or create PyAudio instance."""
    global _pyaudio_instance
    if _pyaudio_instance is None and PYAUDIO_AVAILABLE:
        _pyaudio_instance = pyaudio.PyAudio()
    return _pyaudio_instance


def get_pyaudio_stream(config: AssistantConfig) -> Optional[Any]:
    """
    PATCH #3: Get or create a persistent PyAudio output stream.
    Opens once at startup, reused for every TTS call.
    Thread-safe via _pyaudio_stream_lock.
    """
    global _pyaudio_stream
    with _pyaudio_stream_lock:
        if _pyaudio_stream is not None:
            try:
                # Verify stream is still active
                if not _pyaudio_stream.is_stopped() or _pyaudio_stream.is_active():
                    return _pyaudio_stream
            except Exception:
                pass
            # Stream is dead; close and recreate
            try:
                _pyaudio_stream.close()
            except Exception:
                pass
            _pyaudio_stream = None

        pa = get_pyaudio()
        if pa is None:
            return None

        # Prefer the configured speaker by NAME; fall back to any output
        # device, since the system 'default' may point at a missing card.
        out_index = None
        try:
            want = config.audio.playback_card_name.lower()
            candidates = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxOutputChannels", 0) <= 0:
                    continue
                name = (info.get("name") or "").lower()
                if want and want in name:
                    out_index = i
                    break
                if name.startswith(("bcm2835", "vc4", "plughw", "hw", "sysdefault")):
                    candidates.append(i)
            if out_index is None and candidates:
                out_index = candidates[0]
        except Exception:
            pass

        try:
            _pyaudio_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                output_device_index=out_index,
                frames_per_buffer=config.tts.chunk_size,
            )
            logger.debug(f"PyAudio output stream opened (device index {out_index})")
        except Exception as e:
            logger.warning(f"Failed to open PyAudio stream: {e}")
            _pyaudio_stream = None

        return _pyaudio_stream


def touch_activity():
    """Update last activity timestamp for watchdog."""
    global _last_activity_time
    with _activity_lock:
        _last_activity_time = time.time()


def get_last_activity() -> float:
    with _activity_lock:
        return _last_activity_time


def _signal_handler(sig, frame):
    logger.info("Shutdown signal received, cleaning up...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================
# Connection Pooling
# ============================
def get_http_session(config: AssistantConfig) -> requests.Session:
    """Get or create a connection-pooled HTTP session."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,
        )
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session


async def get_async_session() -> aiohttp.ClientSession:
    """Get or create an async HTTP session."""
    global _async_session
    if _async_session is None or _async_session.closed:
        timeout = aiohttp.ClientTimeout(total=30)
        _async_session = aiohttp.ClientSession(timeout=timeout)
    return _async_session


async def close_async_session():
    """Close the async session."""
    global _async_session
    if _async_session and not _async_session.closed:
        await _async_session.close()
    _async_session = None


# ============================
# Retry Helper
# ============================
def retry_api_call(func: Callable, config: AssistantConfig, *args, **kwargs) -> Any:
    """Call func with exponential backoff on failure."""
    last_exc = None
    for attempt in range(config.api_max_retries):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_exc = e
            delay = min(0.5 * (1.5 ** attempt), 3.0)
            logger.warning(f"API call failed (attempt {attempt + 1}/{config.api_max_retries}): {e}")
            if attempt < config.api_max_retries - 1:
                time.sleep(delay)
    raise last_exc


def retry_api_call_fast(func: Callable, max_retries: int = 2, *args, **kwargs) -> Any:
    """Fast retry for time-critical operations (minimal delay)."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(0.2)
    raise last_exc


# ============================
# Health Monitoring
# ============================
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    last_check: float
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthMonitor:
    """Monitor health of external services."""

    def __init__(self, config: AssistantConfig):
        self.config = config
        self._health: Dict[str, ServiceHealth] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.config.health.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_health(self, service: str) -> Optional[ServiceHealth]:
        with self._lock:
            return self._health.get(service)

    def is_service_available(self, service: str) -> bool:
        health = self.get_health(service)
        if health is None:
            return True
        return health.status != HealthStatus.UNHEALTHY

    def _monitor_loop(self):
        while self._running and not _shutdown_event.is_set():
            try:
                self._check_home_assistant()
                self._check_openai()
                self._check_anthropic()
            except Exception as e:
                logger.warning(f"Health check error: {e}")

            for _ in range(int(self.config.health.check_interval_seconds)):
                if not self._running or _shutdown_event.is_set():
                    break
                time.sleep(1)

    def _check_home_assistant(self):
        if not self.config.home_assistant.url:
            return

        start = time.time()
        try:
            session = get_http_session(self.config)
            headers = {"Authorization": f"Bearer {self.config.home_assistant.token}"}
            r = session.get(
                f"{self.config.home_assistant.url}/api/",
                headers=headers,
                timeout=self.config.health.ha_timeout_seconds,
            )
            latency = (time.time() - start) * 1000

            if r.ok:
                status = HealthStatus.HEALTHY
                error = None
            else:
                status = HealthStatus.DEGRADED
                error = f"HTTP {r.status_code}"
        except Exception as e:
            latency = (time.time() - start) * 1000
            status = HealthStatus.UNHEALTHY
            error = str(e)

        with self._lock:
            self._health["home_assistant"] = ServiceHealth(
                name="home_assistant",
                status=status,
                last_check=time.time(),
                latency_ms=latency,
                error=error,
            )

        if status != HealthStatus.HEALTHY:
            logger.warning(f"Home Assistant health: {status.value} - {error}")

    def _check_openai(self):
        # PATCH #5: use a cheap max_tokens=1 chat ping instead of models.list()
        # models.list() is a paginated call that takes ~800ms and is unnecessary here.
        start = time.time()
        try:
            sync_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            latency = (time.time() - start) * 1000
            status = HealthStatus.HEALTHY
            error = None
        except Exception as e:
            latency = (time.time() - start) * 1000
            status = HealthStatus.UNHEALTHY
            error = str(e)

        with self._lock:
            self._health["openai"] = ServiceHealth(
                name="openai",
                status=status,
                last_check=time.time(),
                latency_ms=latency,
                error=error,
            )

        if status != HealthStatus.HEALTHY:
            logger.warning(f"OpenAI health: {status.value} - {error}")

    def _check_anthropic(self):
        if anthropic_client is None:
            return
        # count_tokens is free — verifies auth + connectivity without spend
        start = time.time()
        try:
            anthropic_client.messages.count_tokens(
                model=self.config.conversation.anthropic_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            latency = (time.time() - start) * 1000
            status = HealthStatus.HEALTHY
            error = None
        except Exception as e:
            latency = (time.time() - start) * 1000
            status = HealthStatus.UNHEALTHY
            error = str(e)

        with self._lock:
            self._health["anthropic"] = ServiceHealth(
                name="anthropic",
                status=status,
                last_check=time.time(),
                latency_ms=latency,
                error=error,
            )

        if status != HealthStatus.HEALTHY:
            logger.warning(f"Anthropic health: {status.value} - {error}")


# ============================
# Watchdog Timer
# ============================
class Watchdog:
    """Watchdog timer to detect and recover from hangs."""

    def __init__(self, config: AssistantConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.config.health.watchdog_enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _watch_loop(self):
        while self._running and not _shutdown_event.is_set():
            time.sleep(10)

            last = get_last_activity()
            elapsed = time.time() - last

            if elapsed > self.config.health.watchdog_timeout_seconds:
                logger.error(
                    f"Watchdog timeout! No activity for {elapsed:.0f}s. "
                    "Consider restarting the assistant."
                )
                touch_activity()


# ============================
# Entity and State Caching
# ============================
class EntityCache:
    """Cache for Home Assistant entity states."""

    def __init__(self, config: AssistantConfig):
        self.config = config
        self._states: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._entities: List[Dict[str, Any]] = []
        self._entities_at: float = 0.0
        self._areas: List[Dict[str, Any]] = []
        self._areas_at: float = 0.0
        self._lock = threading.Lock()

    def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if entity_id in self._states:
                state, cached_at = self._states[entity_id]
                if time.time() - cached_at < self.config.home_assistant.cache_ttl_seconds:
                    return state
        return None

    def set_state(self, entity_id: str, state: Dict[str, Any]):
        with self._lock:
            self._states[entity_id] = (state, time.time())

    def invalidate(self, entity_id: str):
        with self._lock:
            self._states.pop(entity_id, None)

    def invalidate_all(self):
        with self._lock:
            self._states.clear()

    def get_areas(self) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            if self._areas and time.time() - self._areas_at < self.config.home_assistant.areas_cache_ttl_seconds:
                return self._areas
        return None

    def set_areas(self, areas: List[Dict[str, Any]]):
        with self._lock:
            self._areas = areas
            self._areas_at = time.time()

    def get_entities(self) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            if self._entities and time.time() - self._entities_at < self.config.home_assistant.cache_ttl_seconds:
                return self._entities
        return None

    def set_entities(self, entities: List[Dict[str, Any]]):
        with self._lock:
            self._entities = entities
            self._entities_at = time.time()


# ============================
# Home Assistant Integration
# ============================
class HomeAssistantError(Exception):
    pass


_ENTITY_ID_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


def _validate_entity_id(entity_id: str) -> bool:
    if entity_id == "all":
        return True
    return bool(_ENTITY_ID_RE.match(entity_id))


def _sanitize_service_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unexpected or potentially dangerous fields from service data."""
    ALLOWED_DATA_KEYS = {
        "entity_id", "area_id", "device_id",
        "brightness", "brightness_pct", "brightness_step", "brightness_step_pct",
        "color_name", "rgb_color", "rgbw_color", "rgbww_color",
        "hs_color", "xy_color", "color_temp", "color_temp_kelvin", "kelvin",
        "white_value", "effect", "flash", "transition", "profile",
        "temperature", "target_temp_high", "target_temp_low", "hvac_mode",
        "fan_mode", "swing_mode", "preset_mode", "humidity",
        "media_content_id", "media_content_type", "volume_level",
        "source", "sound_mode", "option", "value", "percentage",
        "position", "tilt_position",
        # music_assistant.play_media
        "media_id", "media_type", "radio_mode", "enqueue",
        # todo.add_item / remove_item
        "item",
        # notify.send_message (desktop / phone notifications)
        "message", "title",
    }
    return {k: v for k, v in data.items() if k in ALLOWED_DATA_KEYS}


class HomeAssistantClient:
    """Home Assistant API client with caching and async support."""

    def __init__(self, config: AssistantConfig, cache: EntityCache):
        self.config = config
        self.cache = cache

    def _headers(self) -> Dict[str, str]:
        if not self.config.home_assistant.token:
            raise HomeAssistantError("HOME_ASSISTANT_TOKEN is not set.")
        return {
            "Authorization": f"Bearer {self.config.home_assistant.token}",
            "Content-Type": "application/json",
        }

    def _ws_url(self) -> str:
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
        base = self.config.home_assistant.url.rstrip("/")
        if base.startswith("https://"):
            return base.replace("https://", "wss://") + "/api/websocket"
        if base.startswith("http://"):
            return base.replace("http://", "ws://") + "/api/websocket"
        return "ws://" + base + "/api/websocket"

    def list_areas(self) -> List[Dict[str, Any]]:
        cached = self.cache.get_areas()
        if cached is not None:
            return cached

        ws_url = self._ws_url()
        ws = websocket.create_connection(ws_url, timeout=10)
        try:
            ws.recv()
            ws.send(json.dumps({"type": "auth", "access_token": self.config.home_assistant.token}))
            auth_msg = json.loads(ws.recv())
            if auth_msg.get("type") != "auth_ok":
                raise HomeAssistantError(f"WebSocket auth failed: {auth_msg}")

            ws.send(json.dumps({"id": 1, "type": "config/area_registry/list"}))
            msg = json.loads(ws.recv())

            if not msg.get("success"):
                raise HomeAssistantError(f"WS area list failed: {msg}")

            areas = msg.get("result", [])
            self.cache.set_areas(areas)
            return areas
        finally:
            ws.close()

    def list_entities(self) -> List[Dict[str, Any]]:
        cached = self.cache.get_entities()
        if cached is not None:
            return cached

        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")

        session = get_http_session(self.config)
        r = session.get(
            f"{self.config.home_assistant.url}/api/states",
            headers=self._headers(),
            timeout=10,
        )
        if not r.ok:
            raise HomeAssistantError(f"Failed to list entities: {r.status_code}")

        entities = r.json()
        self.cache.set_entities(entities)
        return entities

    def get_state(self, entity_id: str) -> Dict[str, Any]:
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
        if not entity_id or "." not in entity_id:
            raise HomeAssistantError("Invalid entity_id.")
        if not _validate_entity_id(entity_id):
            raise HomeAssistantError(f"Malformed entity_id: {entity_id}")

        cached = self.cache.get_state(entity_id)
        if cached is not None:
            logger.debug(f"Cache hit for {entity_id}")
            return cached

        session = get_http_session(self.config)
        url = f"{self.config.home_assistant.url}/api/states/{entity_id}"
        r = session.get(url, headers=self._headers(), timeout=6)

        if r.status_code == 404:
            raise HomeAssistantError(f"Entity not found: {entity_id}")
        if not r.ok:
            raise HomeAssistantError(f"HA state error: {r.status_code} {r.text[:200]}")

        state = r.json()
        self.cache.set_state(entity_id, state)
        return state

    def call_service(self, domain: str, service: str, data: Dict[str, Any]) -> Any:
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")

        domain = (domain or "").strip()
        service = (service or "").strip()

        if domain not in self.config.home_assistant.allowed_domains:
            raise HomeAssistantError(f"Domain not allowed: {domain}")
        if service not in self.config.home_assistant.allowed_services:
            raise HomeAssistantError(f"Service not allowed: {service}")

        eid = data.get("entity_id")
        if eid and isinstance(eid, str) and eid != "all" and not _validate_entity_id(eid):
            raise HomeAssistantError(f"Malformed entity_id in data: {eid}")

        data = _sanitize_service_data(data)

        url = f"{self.config.home_assistant.url}/api/services/{domain}/{service}"
        logger.debug(f"HA CALL -> {domain}.{service} data={dict(data or {})}")

        session = get_http_session(self.config)
        r = session.post(url, headers=self._headers(), json=data or {}, timeout=8)

        if not r.ok:
            raise HomeAssistantError(f"HA service error: {r.status_code} {r.text[:200]}")

        if eid:
            if isinstance(eid, str):
                self.cache.invalidate(eid)
            elif isinstance(eid, list):
                for e in eid:
                    self.cache.invalidate(e)

        try:
            return r.json()
        except Exception:
            return {"ok": True}


# ============================
# Fuzzy Matching
# ============================
def fuzzy_match_entity(
    query: str,
    entities: List[Dict[str, Any]],
    domain_filter: Optional[str] = None,
    min_score: int = 70,
) -> Optional[str]:
    """Fuzzy match a user query to an entity."""
    if not FUZZY_AVAILABLE:
        return None

    query_norm = _normalize_text(query)
    if not query_norm:
        return None

    candidates = []
    for entity in entities:
        eid = entity.get("entity_id", "")
        if not eid:
            continue

        domain = eid.split(".")[0] if "." in eid else ""
        if domain_filter and domain != domain_filter:
            continue

        attrs = entity.get("attributes", {})
        friendly_name = attrs.get("friendly_name", "")

        targets = [
            _normalize_text(friendly_name),
            _normalize_text(eid.replace(".", " ")),
        ]

        for target in targets:
            if target:
                score = fuzz.ratio(query_norm, target)
                if score >= min_score:
                    candidates.append((score, eid))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def fuzzy_match_area(
    query: str,
    areas: List[Dict[str, Any]],
    min_score: int = 70,
) -> Optional[str]:
    """Fuzzy match a user query to an area."""
    if not FUZZY_AVAILABLE:
        return None

    query_norm = _normalize_text(query)
    if not query_norm:
        return None

    candidates = []
    for area in areas:
        area_id = area.get("area_id", "")
        name = area.get("name", "")
        if not area_id:
            continue

        name_norm = _normalize_text(name)
        if name_norm:
            score = fuzz.ratio(query_norm, name_norm)
            partial = fuzz.partial_ratio(query_norm, name_norm)
            best = max(score, partial)
            if best >= min_score:
                candidates.append((best, area_id))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def infer_area_id_from_text(user_text: str, areas: List[Dict[str, Any]]) -> Optional[str]:
    """Infer area_id from user text using both exact and fuzzy matching."""
    t = _normalize_text(user_text)
    if not t:
        return None

    candidates = []
    for a in areas:
        name = a.get("name", "")
        area_id = a.get("area_id")
        if not name or not area_id:
            continue
        nn = _normalize_text(name)
        if not nn:
            continue

        words_in_text = t.split()
        area_words = nn.split()
        if all(aw in words_in_text for aw in area_words):
            candidates.append((len(nn), area_id))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return fuzzy_match_area(t, areas)


# ============================
# Light Color Handling
# ============================
COLOR_NAME_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "teal": (0, 128, 128),
    "white": (255, 255, 255),
    "warm white": (255, 244, 229),
    "cool white": (240, 255, 255),
    "lavender": (230, 190, 255),
    "coral": (255, 127, 80),
    "gold": (255, 215, 0),
    "lime": (0, 255, 0),
    "navy": (0, 0, 128),
    "olive": (128, 128, 0),
    "salmon": (250, 128, 114),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
}


def _extract_hex_rgb(text: str) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"#([0-9a-fA-F]{6})\b", text or "")
    if not m:
        return None
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _extract_color_name(text: str) -> Optional[str]:
    t = _normalize_text(text)
    if not t:
        return None
    for name in sorted(COLOR_NAME_TO_RGB.keys(), key=lambda s: len(s), reverse=True):
        if name in t:
            return name
    return None


def _extract_color_temp_kelvin(text: str) -> Optional[int]:
    t = _normalize_text(text)
    if "warm" in t and "white" in t:
        return 2700
    if ("cool" in t and "white" in t) or ("daylight" in t):
        return 5000
    if "soft" in t and "white" in t:
        return 3000
    if t.strip() == "white" or (
        " white" in t and "warm" not in t and "cool" not in t
        and "soft" not in t and "daylight" not in t
    ):
        return 4000
    m = re.search(r"\b(\d{4})\s*k\b", t)
    if m:
        k = int(m.group(1))
        if 1500 <= k <= 9000:
            return k
    return None


_BRIGHTNESS_KEYWORDS = {"percent", "%", "brightness", "dim", "brighter", "bright", "dimmer"}


def normalize_light_service_data(data: Dict[str, Any], last_user_text: str) -> Dict[str, Any]:
    """Convert common model/user color fields into HA-compatible fields."""
    data = dict(data or {})

    if "color" in data and isinstance(data["color"], str) and "color_name" not in data:
        data["color_name"] = data.pop("color")
    if "colour" in data and isinstance(data["colour"], str) and "color_name" not in data:
        data["color_name"] = data.pop("colour")

    if "color_temp_kelvin" not in data and "color_temp" not in data:
        kelvin = _extract_color_temp_kelvin(last_user_text)
        if kelvin is not None:
            data.pop("rgb_color", None)
            data.pop("hs_color", None)
            data.pop("xy_color", None)
            if isinstance(data.get("color_name"), str) and "white" in data["color_name"].lower():
                data.pop("color_name", None)
            data["color_temp_kelvin"] = kelvin
            return data

    if "rgb_color" not in data and "hs_color" not in data and "xy_color" not in data and "color_name" not in data:
        rgb = _extract_hex_rgb(last_user_text)
        if rgb:
            data["rgb_color"] = [rgb[0], rgb[1], rgb[2]]

    if "rgb_color" not in data and "hs_color" not in data and "xy_color" not in data and "color_name" not in data:
        cname = _extract_color_name(last_user_text)
        if cname:
            rgb = COLOR_NAME_TO_RGB.get(cname)
            if rgb:
                data["rgb_color"] = [rgb[0], rgb[1], rgb[2]]
            else:
                data["color_name"] = cname

    # PATCH #6: brightness regex now uses a negative lookahead (?!\s*k\b) so that
    # kelvin values like "3000k" are not incorrectly parsed as brightness "300".
    if "brightness_pct" not in data:
        norm = _normalize_text(last_user_text)
        if any(kw in norm for kw in _BRIGHTNESS_KEYWORDS):
            m = re.search(r"\b(\d{1,3})(?!\d)(?!\s*k\b)\s*%?\b", norm)
            if m:
                pct = max(0, min(100, int(m.group(1))))
                data["brightness_pct"] = pct

    return data


# ============================
# Scene Suggestions
# ============================
def suggest_scene(
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
    user_text: str,
) -> Optional[str]:
    """Suggest a relevant scene based on context."""
    try:
        entities = ha_client.list_entities()
        scenes = [e for e in entities if e.get("entity_id", "").startswith("scene.")]

        if not scenes:
            return None

        hour = datetime.now().hour
        text_lower = user_text.lower()

        suggestions = []

        if hour >= 22 or hour < 6:
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["night", "sleep", "bedtime", "evening", "dim"]):
                    suggestions.append(scene)
        elif hour >= 6 and hour < 9:
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["morning", "wake", "sunrise", "bright"]):
                    suggestions.append(scene)
        elif hour >= 17 and hour < 22:
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["evening", "dinner", "relax", "movie", "cozy"]):
                    suggestions.append(scene)

        if any(kw in text_lower for kw in ["movie", "watch", "film", "tv", "netflix"]):
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["movie", "cinema", "theater", "dim"]):
                    suggestions.append(scene)

        if any(kw in text_lower for kw in ["dinner", "eat", "food", "meal"]):
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["dinner", "dining", "meal"]):
                    suggestions.append(scene)

        if any(kw in text_lower for kw in ["work", "focus", "study", "reading"]):
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["work", "focus", "bright", "study", "reading"]):
                    suggestions.append(scene)

        if suggestions:
            scene = suggestions[0]
            friendly = scene.get("attributes", {}).get("friendly_name", scene.get("entity_id"))
            return f"Would you like me to activate the '{friendly}' scene?"

    except Exception as e:
        logger.debug(f"Scene suggestion failed: {e}")

    return None


# ============================
# State-Aware Responses
# ============================
def check_state_and_respond(
    ha_client: HomeAssistantClient,
    domain: str,
    service: str,
    entity_id: Optional[str],
    area_id: Optional[str],
) -> Optional[str]:
    """Check current state and return a response if action is redundant."""
    if not entity_id or entity_id == "all":
        return None

    try:
        state = ha_client.get_state(entity_id)
        current_state = state.get("state", "")
        friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)

        if domain == "light":
            if service == "turn_on" and current_state == "on":
                return f"The {friendly_name} is already on."
            if service == "turn_off" and current_state == "off":
                return f"The {friendly_name} is already off."

        if domain == "switch":
            if service == "turn_on" and current_state == "on":
                return f"The {friendly_name} is already on."
            if service == "turn_off" and current_state == "off":
                return f"The {friendly_name} is already off."

        if domain == "lock":
            if service == "lock" and current_state == "locked":
                return f"The {friendly_name} is already locked."
            if service == "unlock" and current_state == "unlocked":
                return f"The {friendly_name} is already unlocked."

        if domain == "cover":
            if service == "open_cover" and current_state == "open":
                return f"The {friendly_name} is already open."
            if service == "close_cover" and current_state == "closed":
                return f"The {friendly_name} is already closed."

    except Exception as e:
        logger.debug(f"State check failed: {e}")

    return None


# ============================
# Graceful Degradation
# ============================
OFFLINE_RESPONSES = {
    "hello": "Hello! I'm currently having trouble connecting to my brain, but I'm still here.",
    "hi": "Hi there! I'm having some connectivity issues, but I'll do my best.",
    "help": "I'm having trouble connecting to external services right now. Please try again in a moment.",
    "time": lambda: f"The current time is {datetime.now().strftime('%I:%M %p')}.",
    "date": lambda: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.",
    "thank": "You're welcome!",
    "thanks": "Happy to help!",
    "good morning": "Good morning! I hope you have a great day.",
    "good night": "Good night! Sleep well.",
    "good evening": "Good evening!",
    "good afternoon": "Good afternoon!",
}


def get_offline_response(text: str) -> Optional[str]:
    """Get a response when offline."""
    text_lower = text.lower().strip()

    for key, response in OFFLINE_RESPONSES.items():
        if key in text_lower:
            if callable(response):
                return response()
            return response

    return "I'm having trouble connecting to my services right now. Please try again in a moment."


# ============================
# Audio Processing
# ============================
def rms(audio_f32: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio_f32)) + 1e-12))


def save_wav(path: str, audio_i16: np.ndarray, sample_rate: int, channels: int = 1):
    audio_i16 = np.asarray(audio_i16, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def decimate_to_16k(pcm_i16: np.ndarray, mic_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Downsample with a simple low-pass anti-aliasing filter."""
    if mic_rate == target_rate:
        return pcm_i16
    ratio = mic_rate / target_rate
    step = int(round(ratio))

    n = len(pcm_i16)
    trim = n - (n % step)
    if trim == 0:
        return pcm_i16[:1]
    reshaped = pcm_i16[:trim].reshape(-1, step)
    return reshaped.astype(np.float32).mean(axis=1).astype(np.int16)


def apply_noise_reduction(
    audio_i16: np.ndarray,
    sample_rate: int,
    config: AssistantConfig,
) -> np.ndarray:
    """Apply noise reduction to audio."""
    if not NOISEREDUCE_AVAILABLE or not config.audio.noise_reduce_enabled:
        return audio_i16

    try:
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        reduced = nr.reduce_noise(
            y=audio_f32,
            sr=sample_rate,
            prop_decrease=config.audio.noise_reduce_prop_decrease,
            stationary=True,
        )
        return (reduced * 32768.0).astype(np.int16)
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}")
        return audio_i16


def normalize_volume(audio_i16: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio volume to target dB level."""
    audio_f32 = audio_i16.astype(np.float32)

    current_rms = np.sqrt(np.mean(np.square(audio_f32)) + 1e-12)
    if current_rms < 1e-6:
        return audio_i16

    current_db = 20 * np.log10(current_rms / 32768.0)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    audio_f32 = audio_f32 * gain_linear
    audio_f32 = np.tanh(audio_f32 / 32768.0) * 32768.0

    return audio_f32.astype(np.int16)


# ============================
# WebRTC VAD
# ============================
class WebRTCVAD:
    """WebRTC Voice Activity Detection wrapper."""

    def __init__(self, config: AssistantConfig):
        self.config = config
        self.vad: Optional[Any] = None

        if WEBRTC_VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(config.audio.vad_mode)
            logger.info(f"WebRTC VAD initialized (mode={config.audio.vad_mode})")
        else:
            logger.warning("WebRTC VAD not available, falling back to RMS-based detection")

    def is_speech(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        if self.vad is None:
            return True

        if sample_rate != 16000:
            return True

        frame_ms = len(pcm_bytes) / 2 / sample_rate * 1000
        if frame_ms not in (10, 20, 30):
            return True

        try:
            return self.vad.is_speech(pcm_bytes, sample_rate)
        except Exception:
            return True


# ============================
# Echo Cancellation (Basic)
# ============================
class EchoCancellation:
    """Basic echo cancellation using playback tracking."""

    def __init__(self):
        self._playing = threading.Event()
        self._last_play_end: float = 0.0
        self._suppression_tail_seconds: float = 0.3

    def start_playback(self):
        self._playing.set()

    def end_playback(self):
        self._playing.clear()
        self._last_play_end = time.time()

    def should_suppress(self) -> bool:
        if self._playing.is_set():
            return True

        if time.time() - self._last_play_end < self._suppression_tail_seconds:
            return True

        return False


echo_cancellation = EchoCancellation()


# Playback gain override (1.0 = play files as authored). aplay has no gain
# of its own, so a non-unity gain is applied by scaling into a temp WAV.
# AlarmManager sets this around alarm rings so the dashboard's ALARM VOL
# slider controls both the chime and the spoken announcement.
_playback_gain: float = 1.0


def set_playback_gain(gain: float):
    global _playback_gain
    _playback_gain = max(0.05, min(1.5, float(gain)))


def _gain_scaled_path(wav_path: str) -> Tuple[str, bool]:
    """Returns (path_to_play, is_temp_file) honoring _playback_gain."""
    gain = _playback_gain
    if abs(gain - 1.0) < 0.02:
        return wav_path, False
    try:
        with wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
        if params.sampwidth != 2:
            return wav_path, False
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) * gain
        x = np.clip(x, -32768, 32767).astype(np.int16)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
        os.close(fd)
        with wave.open(tmp, "wb") as out:
            out.setparams(params)
            out.writeframes(x.tobytes())
        return tmp, True
    except Exception:
        return wav_path, False   # volume must never break playback


def aplay_wav(wav_path: str, config: AssistantConfig):
    """Play WAV file using aplay."""
    play_path, is_temp = _gain_scaled_path(wav_path)
    try:
        with PLAYBACK_LOCK:
            echo_cancellation.start_playback()
            hud.feed_wav_async(play_path)
            try:
                p = subprocess.run(
                    ["aplay", "-D", config.audio.alsa_device, play_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                echo_cancellation.end_playback()
    finally:
        if is_temp:
            try:
                os.unlink(play_path)
            except OSError:
                pass

    if p.returncode != 0:
        raise RuntimeError(
            f"aplay failed (rc={p.returncode}) on device '{config.audio.alsa_device}'.\n"
            f"File: {wav_path}\nSTDERR:\n{p.stderr.strip()}"
        )


def aplay_wav_interruptible(
    wav_path: str,
    config: AssistantConfig,
    check_interrupt: Callable[[], bool],
) -> bool:
    """Play WAV file with support for interruption. Returns True if completed."""
    play_path, is_temp = _gain_scaled_path(wav_path)
    try:
        with PLAYBACK_LOCK:
            echo_cancellation.start_playback()
            hud.feed_wav_async(play_path)
            try:
                p = subprocess.Popen(
                    ["aplay", "-D", config.audio.alsa_device, play_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                while p.poll() is None:
                    if check_interrupt():
                        p.terminate()
                        p.wait()
                        return False
                    time.sleep(0.1)

                return p.returncode == 0
            finally:
                echo_cancellation.end_playback()
    finally:
        if is_temp:
            try:
                os.unlink(play_path)
            except OSError:
                pass


def play_wav_file(path: str, label: str, config: AssistantConfig):
    """Play a WAV file with error handling."""
    if not path:
        return
    try:
        aplay_wav(path, config)
    except Exception as e:
        logger.warning(f"{label} playback failed: {e}")


def play_success_sound(config: AssistantConfig):
    """Play success confirmation sound."""
    if config.sfx.success_wav:
        play_wav_file(config.sfx.success_wav, "SFX_SUCCESS_WAV", config)


def play_failure_sound(config: AssistantConfig):
    """Play failure sound."""
    if config.sfx.failure_wav:
        play_wav_file(config.sfx.failure_wav, "SFX_FAILURE_WAV", config)


def play_wav_file_async(path: str, label: str, config: AssistantConfig) -> Optional[Future]:
    """Play a WAV file asynchronously in background."""
    if not path:
        return None

    pool = get_thread_pool()
    return pool.submit(play_wav_file, path, label, config)


# ============================
# Streaming TTS (Real-time)
# ============================
def speak_tts_realtime(text: str, config: AssistantConfig) -> bool:
    """
    True real-time streaming TTS - plays audio chunks as they arrive.
    PATCH #3: Reuses the persistent PyAudio stream opened at startup instead of
    opening a new one per call (~50ms savings per TTS invocation).
    """
    if not PYAUDIO_AVAILABLE:
        return speak_tts_streaming(text, config)

    stream = get_pyaudio_stream(config)
    if stream is None:
        return speak_tts_streaming(text, config)

    # Chunks become audible one output-buffer-depth after stream.write();
    # delay the HUD levels by that much so the bars track the ear.
    try:
        hud_delay = min(0.6, max(0.0, float(stream.get_output_latency()))) + 0.05
    except Exception:
        hud_delay = 0.25

    echo_cancellation.start_playback()
    completed = True

    try:
        with sync_client.audio.speech.with_streaming_response.create(
            model=config.tts.model,
            voice=config.tts.voice,
            input=text,
            response_format="pcm",
        ) as response:
            buffer = bytearray()

            for chunk in response.iter_bytes(chunk_size=config.tts.chunk_size):
                if _playback_interrupt.is_set():
                    completed = False
                    break

                buffer.extend(chunk)

                while len(buffer) >= config.tts.chunk_size:
                    audio_chunk = bytes(buffer[:config.tts.chunk_size])
                    buffer = buffer[config.tts.chunk_size:]

                    if config.tts.volume_normalization and len(audio_chunk) > 0:
                        pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
                        pcm_data = normalize_volume_fast(pcm_data, config.tts.target_db)
                        audio_chunk = pcm_data.tobytes()

                    hud.feed_pcm(audio_chunk, 24000, delay=hud_delay)
                    stream.write(audio_chunk)

            # Flush remaining
            if buffer and not _playback_interrupt.is_set():
                remaining = bytes(buffer)
                if config.tts.volume_normalization and len(remaining) >= 2:
                    pcm_data = np.frombuffer(remaining, dtype=np.int16)
                    pcm_data = normalize_volume_fast(pcm_data, config.tts.target_db)
                    remaining = pcm_data.tobytes()
                hud.feed_pcm(remaining, 24000, delay=hud_delay)
                stream.write(remaining)

    except Exception as e:
        logger.warning(f"Real-time TTS failed, falling back to standard: {e}")
        echo_cancellation.end_playback()
        return speak_tts_streaming(text, config)
    finally:
        echo_cancellation.end_playback()

    return completed


def normalize_volume_fast(audio_i16: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Fast volume normalization for streaming chunks."""
    if len(audio_i16) == 0:
        return audio_i16

    audio_f32 = audio_i16.astype(np.float32)
    max_val = np.abs(audio_f32).max()

    if max_val < 100:
        return audio_i16

    target_max = 32768 * (10 ** (target_db / 20))
    gain = min(target_max / max_val, 2.0)

    audio_f32 = audio_f32 * gain
    return np.clip(audio_f32, -32768, 32767).astype(np.int16)


def speak_tts_streaming(text: str, config: AssistantConfig) -> bool:
    """Synthesize speech with streaming playback. Returns True if completed."""
    try:
        with sync_client.audio.speech.with_streaming_response.create(
            model=config.tts.model,
            voice=config.tts.voice,
            input=text,
            response_format="pcm",
        ) as response:
            pcm_fd, pcm_path = tempfile.mkstemp(suffix=".pcm", dir=BASE_DIR)
            wav_path = pcm_path.replace(".pcm", ".wav")

            try:
                with os.fdopen(pcm_fd, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if _playback_interrupt.is_set():
                            return False
                        f.write(chunk)

                pcm_data = np.fromfile(pcm_path, dtype=np.int16)

                if config.tts.volume_normalization:
                    pcm_data = normalize_volume(pcm_data, config.tts.target_db)

                save_wav(wav_path, pcm_data, 24000, channels=1)

                def check_interrupt():
                    return _playback_interrupt.is_set()

                return aplay_wav_interruptible(wav_path, config, check_interrupt)

            finally:
                for p in (pcm_path, wav_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    except Exception as e:
        logger.error(f"Streaming TTS failed: {e}")
        return speak_tts_standard(text, config)


def speak_tts_standard(text: str, config: AssistantConfig) -> bool:
    """Standard (non-streaming) TTS."""
    try:
        audio = sync_client.audio.speech.create(
            model=config.tts.model,
            voice=config.tts.voice,
            input=text,
        )
        audio_bytes = audio.read() if hasattr(audio, "read") else audio.content

        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3", dir=BASE_DIR)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)

        try:
            os.close(wav_fd)
            with os.fdopen(mp3_fd, "wb") as f:
                f.write(audio_bytes)

            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-y",
                 "-i", mp3_path, "-ac", "2", "-ar", "48000", wav_path],
                check=False,
            )

            if config.tts.volume_normalization:
                with wave.open(wav_path, "rb") as wf:
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()

                audio_data = normalize_volume(audio_data, config.tts.target_db)
                save_wav(wav_path, audio_data, sample_rate, channels)

            def check_interrupt():
                return _playback_interrupt.is_set()

            return aplay_wav_interruptible(wav_path, config, check_interrupt)

        finally:
            for p in (mp3_path, wav_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return False


def speak_tts(text: str, config: AssistantConfig) -> bool:
    """Speak text using TTS (real-time streaming for lowest latency)."""
    _playback_interrupt.clear()
    hud.set_state("speaking")

    if config.tts.streaming_enabled:
        if config.tts.direct_streaming and PYAUDIO_AVAILABLE:
            return speak_tts_realtime(text, config)
        return speak_tts_streaming(text, config)
    else:
        return speak_tts_standard(text, config)


# ============================
# Transcription (Optimized)
# ============================
def transcribe_audio_fast(wav_path: str, config: AssistantConfig) -> str:
    """Fast transcription without retry overhead."""
    try:
        with open(wav_path, "rb") as f:
            tx = sync_client.audio.transcriptions.create(
                model=config.conversation.transcribe_model,
                file=f,
            )
        return getattr(tx, "text", "").strip()
    except Exception as e:
        logger.warning(f"Fast transcription failed, will retry: {e}")
        return transcribe_audio(wav_path, config)


def transcribe_audio(wav_path: str, config: AssistantConfig) -> str:
    """Transcribe audio using Whisper with retry."""
    def _do_transcribe():
        with open(wav_path, "rb") as f:
            tx = sync_client.audio.transcriptions.create(
                model=config.conversation.transcribe_model,
                file=f,
            )
        return getattr(tx, "text", "").strip()

    return retry_api_call(_do_transcribe, config)


# ============================
# Tool Definitions
# ============================
@lru_cache(maxsize=1)
def _tools_schema():
    """Return cached tools schema for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "ha_get_state",
                "description": "Get the current state and attributes for a Home Assistant entity_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                    "required": ["entity_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_call_service",
                "description": (
                    "Call a Home Assistant service to control devices.\n"
                    "IMPORTANT: data MUST include a target: entity_id OR area_id OR device_id.\n"
                    "For light colors use light.turn_on with color_name/rgb_color/hs_color/xy_color/color_temp_kelvin."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "service": {"type": "string"},
                        "data": {"type": "object", "additionalProperties": True},
                        "entity_id": {"type": "string"},
                        "area_id": {"type": "string"},
                        "device_id": {"type": "string"},
                    },
                    "required": ["domain", "service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_list_entities",
                "description": "List all Home Assistant entities with their current states.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Optional domain filter (e.g., 'light', 'switch')",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_list_areas",
                "description": "List all Home Assistant areas/rooms.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": (
                    "Set a timer or reminder. Jarvis chimes and speaks when it fires. "
                    "Use duration_seconds for relative ('10 minutes' -> 600) or "
                    "at_time 'HH:MM' (24h) for absolute. label is spoken back (the reminder text)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_seconds": {"type": "integer"},
                        "at_time": {"type": "string", "description": "24h HH:MM"},
                        "label": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": "Cancel a timer/reminder by its label, or 'all'.",
                "parameters": {
                    "type": "object",
                    "properties": {"which": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_timers",
                "description": "List running timers/reminders with time remaining.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the live web for current information: news, sports, "
                    "weather elsewhere, prices, release dates, or any fact that "
                    "changes over time or postdates your knowledge. Returns a "
                    "concise spoken-ready answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_command",
                "description": (
                    "Control Geo's Windows desktop PC. action: lock, sleep, "
                    "shutdown, restart, playpause, next, previous, volume_up, "
                    "volume_down, mute, or launch. For launch, set target to an "
                    "app name (spotify, steam, chrome, edge, explorer, notepad, "
                    "task manager, calculator) or a website URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string",
                                   "description": "app name or URL, for launch"},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_screen",
                "description": (
                    "Take a screenshot of Geo's PC (dual monitors) and attach it "
                    "to this conversation so you can SEE it. Use when asked "
                    "what's on screen, to look at / read / explain something on "
                    "the screen, help with an error or game, or give opinions on "
                    "what Geo is looking at. Takes a few seconds; the image "
                    "arrives as the next user message."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "screen": {"type": "string", "description":
                                   "'all' (default) = both monitors in one wide "
                                   "image; '1' = left monitor only, '2' = right "
                                   "— use a single monitor when Geo says which "
                                   "one or fine text must be readable"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_artifact",
                "description": (
                    "Generate a visual HTML artifact and display it on Geo's PC screen: "
                    "charts, graphs, dashboards, TABLES, ranked LISTS, comparisons, "
                    "documents, or mini web tools. Use whenever the user wants "
                    "something SHOWN or put on screen — 'give me a table of...', "
                    "'make a list of...', 'show/draw/visualize...'. The builder "
                    "searches the web itself when the topic needs current data, so "
                    "call this directly (no separate web_search needed). Takes "
                    "several seconds to build."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec": {"type": "string", "description":
                                 "detailed description of what to build, incl. any data"},
                        "title": {"type": "string", "description": "short name"},
                        "needs_research": {"type": "boolean", "description":
                                           "true if it needs current/live data from the "
                                           "web (news, releases, prices, 'upcoming'/"
                                           "'latest'); false for timeless or purely "
                                           "visual content — false builds faster"},
                    },
                    "required": ["spec"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "daily_briefing",
                "description": (
                    "Give Geo a spoken briefing of his day: date, weather, "
                    "today's World Cup and Formula 1, and open tasks. Use for "
                    "'good morning', 'brief me', 'what's my day look like', "
                    "'what's going on today'. Read the returned 'spoken' text "
                    "aloud naturally."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "log_weight",
                "description": (
                    "Record Geo's morning weigh-in for his Summer Reset plan. "
                    "Use when he says 'log my weight ...', 'I weigh ...', "
                    "'weigh-in one seventy-nine ...'. Returns the fresh 7-day "
                    "average and week trend to read back."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lbs": {"type": "number", "description": "weight in pounds"},
                    },
                    "required": ["lbs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "workout",
                "description": (
                    "Geo's training for today (or a named weekday) from his "
                    "Summer Reset split. Use for 'what's my workout', 'what am I "
                    "training today', 'show me the workout'. Set show=true to put "
                    "the full routine on his PC screen; otherwise it returns a "
                    "spoken summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string", "description":
                                "weekday like 'monday' or 'thu'; omit for today"},
                        "show": {"type": "boolean", "description":
                                 "true = display the full routine on screen"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fitness_progress",
                "description": (
                    "Show Geo's weight-loss progress dashboard on his PC screen: "
                    "weight trend, day of the plan, projected goal date, and macro "
                    "targets. Use for 'how's my progress', 'show my weight "
                    "progress', 'am I on track'."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save a durable fact about Geo or his preferences to "
                    "long-term memory that persists across ALL future "
                    "conversations. Use when he says 'remember...', states a "
                    "lasting preference ('I take my coffee black', 'call me "
                    "boss'), or shares info clearly worth keeping (birthdays, "
                    "his teams, routines). Do NOT use for one-off task details "
                    "or timers. Confirm briefly after saving."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description":
                                 "the fact, phrased to stand alone later, e.g. "
                                 "'Geo supports Argentina in the World Cup'"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": (
                    "Remove something from long-term memory when Geo says "
                    "'forget...' or corrects a fact you stored. Matches by "
                    "description."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description":
                                  "words identifying the memory, e.g. 'coffee'"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_artifact",
                "description": (
                    "Re-open a PREVIOUSLY created artifact in its themed window "
                    "on Geo's PC — use for 'pull up / show me again / reopen "
                    "the ...'. Do NOT rebuild existing artifacts with "
                    "create_artifact. Omit query (or say 'gallery') to open the "
                    "browsable gallery of all past artifacts. If nothing "
                    "matches, the result lists recent artifact titles to offer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description":
                                  "a few words from the artifact's title, e.g. "
                                  "'smiley face', 'movies table', 'raid guide'"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "description": (
                    "Set a wake-up style alarm for a wall-clock time (persists across "
                    "restarts; announces repeatedly until dismissed). at_time is 24h "
                    "'HH:MM'. repeat may be 'daily', 'weekdays' (Mon-Fri), "
                    "'weekends', or a specific day list like 'sun,mon,tue,wed,thu'. "
                    "Setting an alarm "
                    "with the same time and label REPLACES it (use that to edit). "
                    "Use set_timer for countdowns instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "at_time": {"type": "string", "description": "24h HH:MM"},
                        "label": {"type": "string"},
                        "repeat": {"type": "string", "description":
                                   "'', 'daily', 'weekdays', 'weekends', or a "
                                   "day list e.g. 'sun,mon,tue,wed,thu'"},
                    },
                    "required": ["at_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_alarm",
                "description": (
                    "Cancel an alarm by label or time ('7:00'), or 'all'. Also "
                    "silences an alarm that is currently going off (a ringing "
                    "repeating alarm keeps its schedule)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"which": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_alarms",
                "description": "List set alarms with their next fire time.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _execute_tool_call(
    tc,
    last_user_text: str,
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
) -> Tuple[Any, bool]:
    """Execute a single OpenAI-format tool call. Returns (result_dict, success_bool)."""
    fn = tc.function.name
    try:
        parsed = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
    except Exception:
        parsed = {}
    return _execute_tool(fn, parsed, last_user_text, config, ha_client)


def _pc_ssh(config: AssistantConfig, powershell: str) -> Tuple[bool, str]:
    """Run one PowerShell snippet on Geo's PC over SSH. The snippet is built
    HERE from fixed templates — never assembled from raw LLM text — so the
    tool surface is a closed set of actions, not arbitrary remote shell.

    IMPORTANT: pass a SINGLE-level-double-quoted command; nested double
    quotes break Windows sshd ('exec request failed on channel 0')."""
    pc = config.pc_control
    try:
        p = subprocess.run(
            ["ssh", "-i", pc.key_path, "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"ConnectTimeout={int(pc.ssh_timeout)}",
             f"{pc.user}@{pc.host}", powershell],
            capture_output=True, text=True, timeout=pc.ssh_timeout + 5,
        )
        out = (p.stdout or "").strip() or (p.stderr or "").strip()
        return p.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "PC did not respond (asleep or offline?)"
    except Exception as e:
        return False, str(e)


def _pc_run_interactive(config: AssistantConfig, script: str) -> Tuple[bool, str]:
    """Run a PowerShell script IN GEO'S DESKTOP SESSION (not the invisible
    SSH session 0). SSH lands in session 0, so GUI launches/lock/media keys
    there are silently useless. We stage the script to a file and trigger it
    via a scheduled task whose principal is the interactive user — the task
    runs in the logged-in session, so windows actually appear. The task is
    registered on first use (self-heals if deleted). base64 avoids any quote
    juggling across ssh -> powershell -> file."""
    import base64
    pc = config.pc_control
    b64 = base64.b64encode(script.encode("utf-8")).decode()
    user = pc.user.split("\\")[-1]
    cmd = (
        '$d="$env:USERPROFILE\\.jarvis"; New-Item -ItemType Directory -Force -Path $d | Out-Null; '
        f'[IO.File]::WriteAllText("$d\\run.ps1",[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(\'{b64}\'))); '
        'if(-not(Get-ScheduledTask -TaskName JarvisExec -ErrorAction SilentlyContinue)){'
        '$a=New-ScheduledTaskAction -Execute \'powershell.exe\' '
        '-Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $d\\run.ps1"; '
        f'$p=New-ScheduledTaskPrincipal -UserId \'{user}\' -LogonType Interactive; '
        'Register-ScheduledTask -TaskName JarvisExec -Action $a -Principal $p -Force | Out-Null}; '
        'Start-ScheduledTask -TaskName JarvisExec'
    )
    return _pc_ssh(config, cmd)


# Captures the PRIMARY monitor in Geo's desktop session (CopyFromScreen is
# useless in SSH session 0), DPI-aware so scaled displays capture fully,
# downscaled to <=1600px wide JPEG to keep the base64 payload realtime-sized.
# It is a compiled C# exe, NOT PowerShell: Defender's AMSI signature-matches
# screenshot code in PS scripts as spyware and blocks it at parse time even
# from excluded folders — but real-time file scanning of an exe DOES honor
# the .jarvis folder exclusion. Compiled once on the PC with the stock
# framework csc.exe; bump _SCREENCAP_VER when changing the source.
_SCREENCAP_VER = "2"
_SCREENCAP_CS = r'''
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
class ScreenCap {
  [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
  // arg: "all" (default) = every monitor as one wide image;
  //      "1"/"2"/... = that monitor only, ordered left to right
  static void Main(string[] args) {
    string dir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile) + "\\.jarvis";
    try {
      SetProcessDPIAware();
      Screen[] scr = Screen.AllScreens;
      Array.Sort(scr, delegate(Screen a, Screen b) {
        return a.Bounds.X.CompareTo(b.Bounds.X); });
      Rectangle r;
      int maxW = 2560;
      if (args.Length > 0 && args[0] != "all") {
        int i = int.Parse(args[0]) - 1;
        if (i < 0) i = 0;
        if (i >= scr.Length) i = scr.Length - 1;
        r = scr[i].Bounds; maxW = 1600;
      } else if (scr.Length == 1) {
        r = scr[0].Bounds; maxW = 1600;
      } else {
        r = SystemInformation.VirtualScreen;
      }
      using (Bitmap bmp = new Bitmap(r.Width, r.Height)) {
        using (Graphics g = Graphics.FromImage(bmp))
          g.CopyFromScreen(r.X, r.Y, 0, 0, r.Size);
        Bitmap outBmp = bmp;
        if (bmp.Width > maxW) {
          int h = (int)((long)bmp.Height * maxW / bmp.Width);
          outBmp = new Bitmap(maxW, h);
          using (Graphics g2 = Graphics.FromImage(outBmp)) {
            g2.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g2.DrawImage(bmp, 0, 0, maxW, h);
          }
        }
        ImageCodecInfo enc = null;
        foreach (ImageCodecInfo c in ImageCodecInfo.GetImageEncoders())
          if (c.MimeType == "image/jpeg") enc = c;
        EncoderParameters ep = new EncoderParameters(1);
        ep.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 72L);
        outBmp.Save(dir + "\\screen.jpg", enc, ep);
        if (!object.ReferenceEquals(outBmp, bmp)) outBmp.Dispose();
      }
    } catch (Exception e) {
      System.IO.File.WriteAllText(dir + "\\screen.err", e.ToString());
    }
  }
}
'''


def _pc_stage_file(config: AssistantConfig, text: str, remote_name: str) -> bool:
    """scp a script into ~\\.jarvis on the PC. Content must travel as a FILE:
    embedding it base64 in an ssh command gets AMSI-scanned, and capture code
    (CopyFromScreen etc.) is signature-matched as spyware and blocked. The
    .jarvis folder carries a Defender exclusion (added 2026-07-11) so staged
    scripts run; the exclusion covers ONLY that folder."""
    pc = config.pc_control
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as f:
            f.write(text)
            tmp = f.name
        p = subprocess.run(
            ["scp", "-i", pc.key_path, "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"ConnectTimeout={int(pc.ssh_timeout)}",
             tmp, f"{pc.user}@{pc.host}:.jarvis/{remote_name}"],
            capture_output=True, text=True, timeout=pc.ssh_timeout + 10)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def pc_view_screen(config: AssistantConfig, screen: str = "all") -> Dict[str, Any]:
    """Screenshot the PC ('all' monitors as one wide image, or one monitor by
    number, left to right) and return it as base64 JPEG under
    '__screen_b64__' — the realtime layer converts that into an input_image
    conversation item so the voice model actually SEES the screen."""
    pc = config.pc_control
    if not pc.enabled:
        return {"error": "PC control is not configured"}
    screen = str(screen or "all").strip().lower()
    if not re.fullmatch(r"all|[1-9]", screen):
        screen = "all"
    exe = f"screencap{_SCREENCAP_VER}.exe"
    ok, out = _pc_ssh(config,
        'New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\\.jarvis" | Out-Null; '
        'Remove-Item "$env:USERPROFILE\\.jarvis\\screen.jpg","$env:USERPROFILE\\.jarvis\\screen.err" -ErrorAction SilentlyContinue; '
        f'"exe=$(Test-Path $env:USERPROFILE\\.jarvis\\{exe})"')
    if not ok:
        return {"error": f"PC unreachable ({out[:100] or 'ssh failed'}) — asleep or offline?"}
    if "exe=True" not in out:
        # one-time build with the stock .NET Framework compiler
        if not _pc_stage_file(config, _SCREENCAP_CS, "screencap.cs"):
            return {"error": "could not stage the capture source on the PC"}
        ok, out = _pc_ssh(config,
            'Remove-Item "$env:USERPROFILE\\.jarvis\\screencap*.exe" -ErrorAction SilentlyContinue; '
            '& "$env:WINDIR\\Microsoft.NET\\Framework64\\v4.0.30319\\csc.exe" /nologo /target:winexe '
            f'/out:"$env:USERPROFILE\\.jarvis\\{exe}" '
            '/r:System.Drawing.dll /r:System.Windows.Forms.dll '
            '"$env:USERPROFILE\\.jarvis\\screencap.cs" 2>&1 | Out-String')
        if not ok or "error" in (out or "").lower():
            return {"error": f"capture tool build failed: {(out or '')[:160]}"}
    ok, out = _pc_run_interactive(
        config, f'& "$env:USERPROFILE\\.jarvis\\{exe}" {screen}')
    if not ok:
        return {"error": f"capture trigger failed ({out[:100]})"}
    # single ssh call that waits server-side for the file, then streams the b64
    ok, out = _pc_ssh(config,
        '$f="$env:USERPROFILE\\.jarvis\\screen.jpg"; $e="$env:USERPROFILE\\.jarvis\\screen.err"; '
        '$d=(Get-Date).AddSeconds(10); '
        'while((Get-Date) -lt $d -and -not (Test-Path $f) -and -not (Test-Path $e)){Start-Sleep -Milliseconds 250}; '
        'if(Test-Path $f){[Convert]::ToBase64String([IO.File]::ReadAllBytes($f))}'
        'elseif(Test-Path $e){"ERR: "+(Get-Content $e -Raw)}else{"TIMEOUT"}')
    out = (out or "").strip()
    if not ok or out == "TIMEOUT" or out.startswith("ERR") or len(out) < 2000:
        detail = out[:160] if out and not out.startswith("/9j/") else "no image produced"
        return {"error": f"screen capture failed: {detail} (screen locked?)"}
    return {"status": "captured",
            "note": "screenshot attached as an image in the next user message",
            "__screen_b64__": out}


# System-wide actions: work fine from the SSH session (no desktop needed).
_PC_SYSTEM_ACTIONS = {
    "sleep":    "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
    "shutdown": "shutdown /s /t 5",
    "restart":  "shutdown /r /t 5",
}
# Desktop actions: MUST run in the interactive session (see _pc_run_interactive).
_PC_INTERACTIVE_ACTIONS = {
    "lock":        "rundll32.exe user32.dll,LockWorkStation",
    "playpause":   "(New-Object -ComObject WScript.Shell).SendKeys([char]179)",
    "next":        "(New-Object -ComObject WScript.Shell).SendKeys([char]176)",
    "previous":    "(New-Object -ComObject WScript.Shell).SendKeys([char]177)",
    "volume_up":   "1..5|%{(New-Object -ComObject WScript.Shell).SendKeys([char]175)}",
    "volume_down": "1..5|%{(New-Object -ComObject WScript.Shell).SendKeys([char]174)}",
    "mute":        "(New-Object -ComObject WScript.Shell).SendKeys([char]173)",
}


def pc_control_action(config: AssistantConfig, action: str,
                      target: str = "") -> Dict[str, Any]:
    """Execute one whitelisted PC action. action in the system/interactive
    tables, or 'launch' (target = app name from config.pc_control.apps, or
    a URL)."""
    pc = config.pc_control
    if not pc.enabled or not pc.host:
        return {"error": "PC control is not configured"}
    action = (action or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"pause": "playpause", "play": "playpause", "skip": "next",
               "lock_pc": "lock", "power_off": "shutdown", "reboot": "restart",
               "vol_up": "volume_up", "vol_down": "volume_down"}
    action = aliases.get(action, action)

    if action == "launch":
        tgt = (target or "").strip()
        if not tgt:
            return {"error": "launch needs a target (app name or URL)"}
        low = tgt.lower()
        if low in pc.apps:
            uri = pc.apps[low]
        elif re.match(r"^https?://", low) or re.match(r"^[\w.-]+\.(com|net|org|io|tv|gov|edu)\b", low):
            uri = tgt if re.match(r"^https?://", low) else "https://" + tgt
        else:
            return {"error": f"unknown app '{tgt}'. Known: {', '.join(sorted(pc.apps))}, or a website URL"}
        safe = "'" + uri.replace("'", "''") + "'"   # PowerShell single-quote escape
        ok, out = _pc_run_interactive(config, f"Start-Process {safe}")
        return ({"status": "launched", "target": tgt} if ok
                else {"error": f"launch failed: {out}"})

    if action in _PC_INTERACTIVE_ACTIONS:
        ok, out = _pc_run_interactive(config, _PC_INTERACTIVE_ACTIONS[action])
    elif action in _PC_SYSTEM_ACTIONS:
        ok, out = _pc_ssh(config, _PC_SYSTEM_ACTIONS[action])
    else:
        valid = sorted(set(_PC_SYSTEM_ACTIONS) | set(_PC_INTERACTIVE_ACTIONS))
        return {"error": f"unknown action '{action}'. Valid: {', '.join(valid)}, launch"}
    if not ok:
        return {"error": f"{action} failed: {out}"}
    return {"status": "done", "action": action}


def _artifact_system() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return (
        "You are an expert front-end engineer. Generate a SINGLE, self-contained "
        "HTML artifact to be displayed full-screen in a desktop browser.\n"
        f"Today's date is {today}.\n"
        "HARD REQUIREMENTS:\n"
        "- Output ONE complete HTML document starting with <!DOCTYPE html>.\n"
        "- ALL CSS and JavaScript INLINE. NO external resources whatsoever: no "
        "CDNs, no external scripts/stylesheets/fonts/images. Use system fonts, "
        "inline SVG, canvas, or data: URIs only.\n"
        "- Polished and responsive on a large monitor; sensible margins; readable.\n"
        "- Prefer a sleek dark theme unless the request implies otherwise. The "
        "page is shown inside a dark-navy sci-fi HUD frame (background "
        "#04070b, cyan #7fdcff / amber #ffb84d accents): give the page a very "
        "dark navy background (#05080d-ish, never pure black or white) and "
        "let the content FILL the viewport width — no narrow centered column "
        "leaving big empty side bars.\n"
        "- Keep the document LEAN: compact CSS with no unused rules or "
        "redundant declarations; no filler commentary in the markup. Elegance "
        "over volume.\n"
        "- If the request includes a CURRENT FACTS section, it was researched "
        "on the web today — build the artifact from those facts verbatim and "
        "do not swap in your own (possibly stale) training knowledge.\n"
        "- Without CURRENT FACTS: never present training knowledge as current "
        "for time-sensitive topics; for timeless/hypothetical data use "
        "reasonable illustrative values and label them as sample data.\n"
        "- Your FINAL message must be ONLY the HTML. No markdown code fences, "
        "no explanation."
    )


def _artifact_research(spec: str) -> Optional[str]:
    """Cheap pre-pass: gpt-5-mini (pennies, OpenAI credits) decides whether the
    artifact needs current facts and web-researches them. Returns a concise
    fact sheet, or None for timeless/visual requests — and None on any failure,
    so the artifact still builds from model knowledge rather than erroring."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    today = datetime.now().strftime("%A, %B %d, %Y")
    try:
        from openai import OpenAI
        r = OpenAI().with_options(timeout=75.0, max_retries=0).responses.create(
            model=os.getenv("WEB_SEARCH_MODEL", "gpt-5-mini"),
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
            instructions=(
                f"Today is {today}. You prepare source data for a visual "
                "artifact someone else will build. If the request involves "
                "current or time-sensitive information (news, releases, "
                "schedules, prices, standings, 'upcoming'/'latest' anything), "
                "search the web (at most 2 searches) and return a concise "
                "fact sheet: the concrete items, names, dates and figures the "
                "artifact should display, as plain-text bullets. Include "
                "dates. No commentary, no markup. If the request is timeless "
                "or purely visual (calculators, drawings, data already "
                "supplied by the user), reply with exactly NO_RESEARCH_NEEDED."),
            input=spec,
        )
        facts = (r.output_text or "").strip()
        if not facts or "NO_RESEARCH_NEEDED" in facts[:60]:
            return None
        return facts[:6000]
    except Exception as e:
        logger.warning(f"Artifact research skipped: {str(e)[:120]}")
        return None


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "artifact").lower()).strip("-")
    return (s or "artifact")[:40]


def _generate_artifact_html(config: AssistantConfig, spec: str,
                            needs_research: Optional[bool] = None) -> str:
    """Generate artifact HTML. Tries Claude with backoff on transient 429/5xx/
    529-overloaded errors, then falls back to OpenAI so a busy Anthropic never
    means 'no artifact'. Raises RuntimeError if every path fails."""
    art = config.artifacts
    last_err = ""

    # needs_research=False (from the voice model) skips the ~8s triage pass;
    # True/None run it — the researcher still declines timeless requests itself
    facts = None if needs_research is False else _artifact_research(spec)
    if facts:
        logger.info(f"Artifact research: {len(facts)} chars of current facts")
        spec = (f"{spec}\n\nCURRENT FACTS (web-researched today; use these as "
                f"the data source):\n{facts}")

    def _retryable(e) -> bool:
        code = getattr(e, "status_code", None)
        return code in (429, 500, 502, 503, 529) or "overload" in str(e).lower()

    if anthropic_client is not None:
        for attempt in range(3):
            try:
                # max_retries=0: a timed-out request is still billed server-side,
                # so the SDK's silent retry would double the cost of a slow run.
                # thinking disabled: transcribing facts into styled HTML doesn't
                # need reasoning tokens, and they generate at output speed
                msg = anthropic_client.with_options(max_retries=0).messages.create(
                    model=art.model, max_tokens=art.max_tokens,
                    system=_artifact_system(),
                    messages=[{"role": "user", "content": spec}],
                    thinking={"type": "disabled"},
                    timeout=90.0,
                )
                text = "".join(b.text for b in msg.content
                               if getattr(b, "type", "") == "text").strip()
                # drop any pre-HTML narration
                i = text.lower().find("<!doctype")
                return text[i:] if i > 0 else text
            except Exception as e:
                last_err = str(e)
                if _retryable(e) and attempt < 2:
                    wait = 2.0 * (attempt + 1)
                    logger.warning(f"Artifact: Claude busy ({str(e)[:80]}), "
                                   f"retry in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                break   # non-retryable, or out of retries -> try OpenAI

    # fallback: OpenAI (uses OPENAI_API_KEY). Claude overloaded or unavailable.
    try:
        from openai import OpenAI
        logger.info("Artifact: falling back to OpenAI generation")
        r = OpenAI().with_options(timeout=120.0, max_retries=0).responses.create(
            model=os.getenv("ARTIFACT_FALLBACK_MODEL", "gpt-5"),
            instructions=_artifact_system(),
            tools=[{"type": "web_search"}],
            input=spec,
        )
        html = (r.output_text or "").strip()
        i = html.lower().find("<!doctype")
        if i > 0:
            html = html[i:]
        if html:
            return html
        last_err = "OpenAI returned empty output"
    except Exception as e:
        last_err = f"{last_err}; OpenAI fallback: {e}"
    raise RuntimeError(last_err or "generation failed")


# De-borders the viewer window. KEY INSIGHT (learned the hard way): Edge/
# Chrome app windows draw their OWN title bar inside the window — stripping
# WS_CAPTION via SetWindowLong flips the style bit but the painted bar stays.
# The working fix is SetWindowRgn: clip the top CUT px (Chromium's painted
# bar, ~33px @100% DPI) out of the window's visible shape. Style bits are
# still stripped too (kills the resize frame; WS_SYSMENU kept for Alt+F4).
# Matching: title starts J.A.R.V.I.S + class Chrome_WidgetWin_1 (app windows
# only — a "J.A.R.V.I.S..." tab in Firefox once matched by title) + title
# must not contain Edge/Chrome (excludes normal tabbed windows).
# Watchdog loops the FULL 25s re-applying: Chromium re-asserts style and can
# clear regions at load events (first paint, title change, iframe load).
# Staged as ~\.jarvis\strip<ver>.ps1 via scp ONCE (never inline over ssh:
# no cmdline size limits, and AMSI scans inline base64) and invoked with
# -File <path> <W> <H>. Bump _STRIP_VER when editing so PCs restage.
# bump when viewer.html changes — Edge caches it per full URL, so reopened
# artifacts would render the stale layout without a version param
_VIEWER_VER = "2"

_STRIP_VER = "2"
_ARTIFACT_STRIP_PS = r'''param($W=1500,$H=940)
try{
Add-Type @'
using System;using System.Text;using System.Runtime.InteropServices;
public class JVW{
public delegate bool EP(IntPtr h,IntPtr l);
[DllImport("user32.dll")]public static extern bool EnumWindows(EP c,IntPtr l);
[DllImport("user32.dll")]static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
[DllImport("user32.dll")]static extern int GetClassName(IntPtr h,StringBuilder s,int n);
[DllImport("user32.dll")]static extern bool IsWindowVisible(IntPtr h);
[DllImport("user32.dll")]static extern int GetWindowLong(IntPtr h,int i);
[DllImport("user32.dll")]static extern int SetWindowLong(IntPtr h,int i,int v);
[DllImport("user32.dll")]static extern bool SetWindowPos(IntPtr h,IntPtr a,int x,int y,int w,int c,uint f);
[DllImport("user32.dll")]static extern int GetWindowRgnBox(IntPtr h,out RECT r);
[DllImport("user32.dll")]static extern int SetWindowRgn(IntPtr h,IntPtr r,bool rd);
[DllImport("gdi32.dll")]static extern IntPtr CreateRectRgn(int a,int b,int c,int d);
public struct RECT{public int L,T,R,B;}
public const int CUT=36;
public static int N;
public static void Strip(string p,int x,int y,int w,int h){N=0;
EnumWindows(delegate(IntPtr q,IntPtr l){
if(!IsWindowVisible(q))return true;
StringBuilder b=new StringBuilder(256);GetWindowText(q,b,256);string t=b.ToString();
if(!t.StartsWith(p)||t.Contains("Edge")||t.Contains("Chrome"))return true;
StringBuilder k=new StringBuilder(64);GetClassName(q,k,64);
if(k.ToString()!="Chrome_WidgetWin_1")return true;
bool did=false;
int s=GetWindowLong(q,-16);int n=s&~0x00C70000;
if(n!=s){SetWindowLong(q,-16,n);SetWindowPos(q,IntPtr.Zero,x,y,w,h,0x24);did=true;}
RECT rb;int rt=GetWindowRgnBox(q,out rb);
if(rt==0||rb.T!=CUT){SetWindowRgn(q,CreateRectRgn(0,CUT,w,h),true);did=true;}
if(did)N++;
return true;},IntPtr.Zero);}}
'@
Add-Type -AssemblyName System.Windows.Forms
$a=[System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$x=$a.X+[Math]::Max(0,[int](($a.Width-$W)/2))
$y=$a.Y+[Math]::Max(0,[int](($a.Height-$H)/2))
$d=(Get-Date).AddSeconds(25)
while((Get-Date) -lt $d){
[JVW]::Strip('J.A.R.V.I.S',$x,$y,$W,$H)
Start-Sleep -Milliseconds 150
}
}catch{}'''

# host -> staged strip version, so we scp at most once per process per PC
_strip_staged: Dict[str, str] = {}


def _ensure_strip_staged(config: AssistantConfig) -> bool:
    pc = config.pc_control
    name = f"strip{_STRIP_VER}.ps1"
    if _strip_staged.get(pc.host) == _STRIP_VER:
        return True
    ok, out = _pc_ssh(config,
        'New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\\.jarvis" | Out-Null; '
        f'"have=$(Test-Path $env:USERPROFILE\\.jarvis\\{name})"')
    if not ok:
        return False
    if "have=True" not in out and not _pc_stage_file(config, _ARTIFACT_STRIP_PS, name):
        return False
    _strip_staged[pc.host] = _STRIP_VER
    return True


def create_artifact(config: AssistantConfig, spec: str, title: str = "",
                    needs_research: Optional[bool] = None,
                    ha_client: Optional[HomeAssistantClient] = None) -> Dict[str, Any]:
    """Claude (or OpenAI fallback) writes a self-contained HTML page from
    `spec`; the Pi serves it and (if PC control is on) Jarvis opens it on
    Geo's monitor. Returns the URL either way."""
    art = config.artifacts
    if not art.enabled:
        return {"error": "artifact generation is disabled"}
    if anthropic_client is None and not os.getenv("OPENAI_API_KEY"):
        return {"error": "no generation model configured"}
    spec = (spec or "").strip()
    if not spec:
        return {"error": "no artifact description given"}
    try:
        html = _generate_artifact_html(config, spec, needs_research)
    except Exception as e:
        logger.error(f"Artifact generation failed: {e}")
        if "overload" in str(e).lower():
            return {"error": "both Claude and the fallback are busy right now; try again in a moment"}
        return {"error": f"generation failed: {str(e)[:160]}"}

    # strip accidental ```html fences if the model added them
    if html.startswith("```"):
        html = re.sub(r"^```[a-zA-Z]*\n?", "", html)
        html = re.sub(r"\n?```\s*$", "", html).strip()
    if "<" not in html:
        return {"error": "model did not return HTML"}

    try:
        view_url, opened = _save_and_open_artifact(config, html, title or spec)
    except Exception as e:
        logger.error(f"Artifact save failed: {e}")
        return {"error": f"could not save artifact: {e}"}
    return {"status": "created", "title": title or "artifact",
            "url": view_url, "opened_on_pc": opened,
            "gallery": f"{config.artifacts.url_base}/index.html"}


def _save_and_open_artifact(config: AssistantConfig, html: str,
                            title: str) -> Tuple[str, bool]:
    """Persist an artifact HTML doc, refresh the gallery, and open it in the
    themed borderless window. Shared by create_artifact (LLM-written) and the
    deterministic builders (workout, fitness progress). Returns (view_url,
    opened_on_pc)."""
    from urllib.parse import quote
    art = config.artifacts
    os.makedirs(art.dir, exist_ok=True)
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    fname = f"{_slugify(title)}-{stamp}.html"
    with open(os.path.join(art.dir, fname), "w", encoding="utf-8") as f:
        f.write(html)
    _write_artifact_gallery(art.dir)
    view_url = (f"{art.url_base}/viewer.html?v={_VIEWER_VER}&src={fname}"
                f"&title={quote((title or _artifact_label(fname))[:80])}")
    logger.info(f"Artifact saved: {fname} ({len(html)} bytes)")
    opened = art.open_on_pc and _pc_open_viewer(config, view_url)
    return view_url, opened


def _pc_open_viewer(config: AssistantConfig, view_url: str) -> bool:
    """Open a viewer/gallery URL on the PC as a themed borderless app window
    (Edge app mode + the staged strip watchdog)."""
    if not config.pc_control.enabled:
        return False
    size = os.getenv("ARTIFACT_WINDOW_SIZE", "1500,940")
    if not re.fullmatch(r"\d{3,4},\d{3,4}", size):
        size = "1500,940"
    w, h = size.split(",")
    ps = f"$u='{view_url}'\n"
    # de-borderer launches FIRST so its watchdog is hot (powershell boot
    # + C# compile ~2s) before the window exists
    if os.getenv("ARTIFACT_BORDERLESS", "1") != "0" and _ensure_strip_staged(config):
        ps += ("Start-Process -WindowStyle Hidden powershell -ArgumentList "
               "'-NoProfile','-WindowStyle','Hidden','-ExecutionPolicy','Bypass',"
               f"'-File',\"$env:USERPROFILE\\.jarvis\\strip{_STRIP_VER}.ps1\",'{w}','{h}'\n")
    ps += (f"try {{ Start-Process msedge -ArgumentList ('--app='+$u),'--window-size={size}' }} "
           f"catch {{ try {{ Start-Process chrome -ArgumentList ('--app='+$u) }} "
           f"catch {{ Start-Process $u }} }}\n")
    ok, _ = _pc_run_interactive(config, ps)
    return ok


def open_artifact(config: AssistantConfig, query: str = "") -> Dict[str, Any]:
    """Re-open a previously generated artifact (or the whole gallery) on the
    PC. Matches `query` against artifact titles; empty/'gallery' opens the
    index. Never regenerates anything."""
    art = config.artifacts
    try:
        files = sorted(
            (f for f in os.listdir(art.dir)
             if f.endswith(".html") and f not in ("index.html", "viewer.html")),
            key=lambda f: os.path.getmtime(os.path.join(art.dir, f)),
            reverse=True)
    except OSError:
        files = []
    q = re.sub(r"[^a-z0-9 ]+", " ", (query or "").lower()).strip()
    recent = [_artifact_label(f) for f in files[:8]]
    if q in ("", "gallery", "all", "list", "artifacts", "everything", "the gallery"):
        url = f"{art.url_base}/index.html"
        opened = _pc_open_viewer(config, url)
        return {"status": "opened the artifact gallery", "url": url,
                "opened_on_pc": opened, "recent_artifacts": recent}
    if not files:
        return {"error": "no saved artifacts yet"}
    import difflib
    best, best_score = None, 0.0
    qwords = set(q.split())
    for i, f in enumerate(files):
        label = re.sub(r"[^a-z0-9 ]+", " ", _artifact_label(f).lower()).strip()
        hit = len(qwords & set(label.split())) / max(1, len(qwords))
        fuzz = difflib.SequenceMatcher(None, q, label).ratio()
        score = hit * 2 + fuzz - i * 0.01     # slight recency bias on ties
        if score > best_score:
            best, best_score = f, score
    if not best or best_score < 0.45:
        return {"error": f"no saved artifact matches '{query}'",
                "recent_artifacts": recent}
    from urllib.parse import quote
    title = _artifact_label(best)
    view_url = (f"{art.url_base}/viewer.html?v={_VIEWER_VER}&src={best}"
                f"&title={quote(title[:80])}")
    opened = _pc_open_viewer(config, view_url)
    logger.info(f"Artifact reopened: {best} (query={query!r})")
    return {"status": "opened", "title": title,
            "url": view_url, "opened_on_pc": opened}


# ============================
# Daily briefing
# ============================
def _espn_json(url: str, timeout: float = 6.0) -> Optional[dict]:
    """Fetch an ESPN public JSON endpoint server-side (the dash does this in
    the browser). Returns None on any failure — the briefing degrades."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "jarvis/1.0"})
        return r.json() if r.ok else None
    except Exception:
        return None


def _f1_race_label(name: str) -> str:
    """ESPN F1 event names carry a sponsor prefix ('Moët & Chandon Belgian
    Grand Prix'). Keep the nationality + 'Grand Prix' — the nationality is the
    word right before 'Grand Prix'/'GP' (rare multi-word countries lose the
    leading word but stay intelligible when spoken)."""
    m = re.search(r"([A-Za-z']+)\s+(?:Grand Prix|GP)\b", name or "", re.I)
    return f"{m.group(1)} Grand Prix" if m else (name or "the race")


def _briefing_sports() -> List[str]:
    """Today's World Cup fixtures + the next F1 GP if it's today/tomorrow.
    Each returned string is one spoken line."""
    lines: List[str] = []
    today = datetime.now().strftime("%Y%m%d")
    wc = _espn_json("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                    f"fifa.world/scoreboard?dates={today}")
    for ev in (wc or {}).get("events", [])[:4]:
        try:
            comp = ev["competitions"][0]
            teams = comp["competitors"]
            home = next(t for t in teams if t["homeAway"] == "home")
            away = next(t for t in teams if t["homeAway"] == "away")
            hn = home["team"].get("shortDisplayName") or home["team"]["displayName"]
            an = away["team"].get("shortDisplayName") or away["team"]["displayName"]
            st = ev["status"]["type"]["state"]
            if st == "pre":
                t = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone()
                lines.append(f"World Cup: {hn} versus {an} at "
                             f"{t.strftime('%-I:%M %p').lstrip('0')}.")
            elif st == "in":
                lines.append(f"World Cup LIVE now: {hn} {home.get('score','')}, "
                             f"{an} {away.get('score','')}.")
            else:
                lines.append(f"World Cup final: {hn} {home.get('score','')}, "
                             f"{an} {away.get('score','')}.")
        except (KeyError, StopIteration, ValueError):
            continue
    f1 = _espn_json("https://site.api.espn.com/apis/site/v2/sports/racing/"
                    "f1/scoreboard")
    for ev in (f1 or {}).get("events", [])[:1]:
        try:
            t = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone()
            days = (t.date() - datetime.now().date()).days
            when = {0: "today", 1: "tomorrow"}.get(days)
            if when:
                name = _f1_race_label(ev.get("name", ""))
                lines.append(f"Formula 1: the {name} is {when} at "
                             f"{t.strftime('%-I:%M %p').lstrip('0')}.")
        except (KeyError, ValueError):
            continue
    return lines


_WEATHER_WORDS = {
    "partlycloudy": "partly cloudy", "clear-night": "clear", "snowy-rainy": "sleet",
    "lightning-rainy": "thunderstorms", "windy-variant": "windy",
    "pouring": "pouring rain", "exceptional": "severe weather",
}


def _pretty_condition(state: Optional[str]) -> str:
    s = (state or "").strip().lower()
    return _WEATHER_WORDS.get(s, s.replace("-", " "))


def _ha_today_forecast(config: AssistantConfig, ha_client) -> Optional[dict]:
    """Today's forecast row via weather.get_forecasts (return_response).
    Internal — bypasses the LLM service allowlist deliberately."""
    try:
        url = (f"{config.home_assistant.url}/api/services/weather/get_forecasts"
               "?return_response=true")
        session = get_http_session(config)
        r = session.post(url, headers=ha_client._headers(),
                         json={"type": "daily",
                               "entity_id": "weather.forecast_home"}, timeout=8)
        if not r.ok:
            return None
        sr = r.json().get("service_response", {})
        fc = sr.get("weather.forecast_home", {}).get("forecast", [])
        return fc[0] if fc else None
    except Exception:
        return None


def daily_briefing(config: AssistantConfig, ha_client) -> Dict[str, Any]:
    """Assemble a short spoken morning briefing: greeting, weather, today's
    sport, and open task count. Every section degrades independently so a
    dead data source never sinks the whole briefing."""
    now = datetime.now()
    hour = now.hour
    greeting = ("Good morning" if hour < 12 else
                "Good afternoon" if hour < 18 else "Good evening")
    parts: List[str] = [f"{greeting}, sir. It's {now.strftime('%A, %B %-d')}."]

    # weather (current from entity state + today's high/low from forecast)
    try:
        w = ha_client.get_state("weather.forecast_home")
        cond = _pretty_condition(w.get("state"))
        temp = w.get("attributes", {}).get("temperature")
        unit = w.get("attributes", {}).get("temperature_unit", "°")
        wline = f"Right now it's {round(temp)}{unit} and {cond}" if temp is not None \
            else f"Currently {cond}"
        fc = _ha_today_forecast(config, ha_client)
        if fc:
            hi, lo = fc.get("temperature"), fc.get("templow")
            fcond = _pretty_condition(fc.get("condition"))
            if hi is not None and lo is not None:
                wline += (f", with a high of {round(hi)} and a low of {round(lo)}"
                          f"{'; ' + fcond + ' expected' if fcond and fcond != cond else ''}")
        parts.append(wline + ".")
    except Exception as e:
        logger.debug(f"Briefing weather failed: {e}")

    # fitness plan: only once the reset has started
    try:
        parts.extend(_briefing_fitness())
    except Exception as e:
        logger.debug(f"Briefing fitness failed: {e}")

    # sport
    try:
        sport = _briefing_sports()
        parts.extend(sport if sport else ["No World Cup or Formula 1 on the calendar today."])
    except Exception as e:
        logger.debug(f"Briefing sports failed: {e}")

    # tasks
    try:
        todo = ha_client.get_state("todo.shopping_list")
        n = int(todo.get("state") or 0)
        if n:
            parts.append(f"You have {n} item{'s' if n != 1 else ''} on your shopping list.")
    except Exception:
        pass

    text = " ".join(parts)
    logger.info(f"Briefing assembled ({len(parts)} parts)")
    return {"status": "briefing", "spoken": text,
            "note": "Read this aloud naturally as a single briefing."}


# ============================
# Fitness: Summer Reset plan features
# ============================
WEIGHT_PATH = os.path.join(BASE_DIR, "weight_log.json")
_weight_lock = threading.Lock()


def _load_weights() -> List[Dict[str, Any]]:
    try:
        with open(WEIGHT_PATH) as f:
            data = json.load(f)
        items = data.get("entries", []) if isinstance(data, dict) else data
        return [e for e in items if isinstance(e, dict) and "lbs" in e and "date" in e]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_weights(items: List[Dict[str, Any]]) -> None:
    tmp = WEIGHT_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": items}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, WEIGHT_PATH)
    except OSError as e:
        logger.warning(f"Weight save failed: {e}")


def weight_stats() -> Dict[str, Any]:
    """Latest weight, 7-day rolling average (the number the plan says to judge
    by), change vs. the previous 7-day window, and progress toward goal."""
    items = sorted(_load_weights(), key=lambda e: e["date"])
    if not items:
        return {"count": 0}
    latest = items[-1]
    last7 = [e["lbs"] for e in items[-7:]]
    avg7 = round(sum(last7) / len(last7), 1)
    prev7 = [e["lbs"] for e in items[-14:-7]]
    trend = None
    if prev7:
        trend = round(avg7 - sum(prev7) / len(prev7), 1)   # neg = losing
    to_goal = round(latest["lbs"] - fp.GOAL_WEIGHT, 1)
    lost = round(fp.START_WEIGHT - latest["lbs"], 1)
    return {"count": len(items), "latest": latest["lbs"], "latest_date": latest["date"],
            "avg7": avg7, "week_change": trend, "to_goal": to_goal, "lost": lost}


def log_weight(lbs: Any) -> Dict[str, Any]:
    """Record a morning weigh-in. One entry per day (a second logging replaces
    the day's value). Returns the fresh 7-day average and trend to speak back."""
    try:
        val = round(float(lbs), 1)
    except (TypeError, ValueError):
        return {"error": f"'{lbs}' isn't a number I can log"}
    if not (50 <= val <= 600):
        return {"error": f"{val} lbs is outside a believable range"}
    today = datetime.now().strftime("%Y-%m-%d")
    with _weight_lock:
        items = [e for e in _load_weights() if e.get("date") != today]
        items.append({"date": today, "lbs": val})
        items.sort(key=lambda e: e["date"])
        _save_weights(items)
    st = weight_stats()
    logger.info(f"Weight logged: {val} lbs (7-day avg {st.get('avg7')})")
    try:
        hud._broadcast({"t": "weight", "v": st})   # live-update the dash tile
    except Exception:
        pass
    spoken = f"Logged {val} pounds."
    if st.get("week_change") is not None:
        direction = ("down" if st["week_change"] < 0 else
                     "up" if st["week_change"] > 0 else "flat")
        spoken += (f" Your 7-day average is {st['avg7']}, "
                   f"{direction} {abs(st['week_change'])} from last week.")
    elif st.get("count", 0) >= 2:
        spoken += f" Your average so far is {st['avg7']} pounds."
    else:
        spoken += " That's your first entry — I'll start tracking the trend."
    if st.get("to_goal") is not None and st["to_goal"] <= 0:
        spoken += " You've hit your goal weight — outstanding, sir."
    return {"status": "logged", "spoken": spoken, **st}


def _briefing_fitness() -> List[str]:
    """The training/plan lines for the morning briefing (empty before reset)."""
    today = datetime.now().date()
    day_n = fp.plan_day_number(today)
    if day_n < 1:
        days_to = (fp.RESET_DATE - today).days
        if days_to == 1:
            return ["Your Summer Reset starts tomorrow — first session is Back day."]
        return []
    lines: List[str] = []
    focus = fp.todays_focus(today)
    phase = fp.phase_for(today)
    tag = f"Day {day_n} of the reset" + (f", {phase['name'].split('—')[0].strip()}" if phase else "")
    if focus["kind"] == "rest":
        lines.append(f"{tag}. Today is full rest — no training, protect the recovery.")
    elif focus["kind"] == "cardio":
        lines.append(f"{tag}. Today is cardio: {focus['focus'].split('—')[-1].strip()}.")
    else:
        lines.append(f"{tag}. Today's lift is {focus['focus']}.")
    # weight trend
    st = weight_stats()
    if st.get("count", 0) >= 2 and st.get("week_change") is not None:
        d = ("down" if st["week_change"] < 0 else "up" if st["week_change"] > 0 else "flat")
        lines.append(f"Your 7-day weight average is {st['avg7']}, {d} "
                     f"{abs(st['week_change'])} from last week.")
    # refeed heads-up
    if fp.is_refeed_day(today):
        lines.append(f"Today is a refeed day — bump to {fp.REFEED_CALORIES} calories, "
                     "extra carbs at lunch and dinner.")
    elif fp.is_refeed_day(today + timedelta(days=1)):
        lines.append("Heads up: tomorrow is a refeed day.")
    return lines


def todays_workout(day: str = "") -> Dict[str, Any]:
    """Spoken workout for today (or a named weekday). Returns a concise summary
    plus the exercise list; the voice model reads it, or calls show_workout to
    put it on screen."""
    wd_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    d = (day or "").strip().lower()[:3]
    if d in wd_map:
        focus = fp.workout_for_weekday(wd_map[d])
        when = day.strip().title()
    else:
        focus = fp.todays_focus()
        when = "Today"
    if focus["kind"] == "rest":
        return {"status": "rest", "focus": focus["focus"],
                "spoken": f"{when} is a full rest day — no workout. Recovery is the work."}
    if focus["kind"] == "cardio":
        return {"status": "cardio", "focus": focus["focus"],
                "spoken": f"{when} is cardio: {focus['exercises'][0]}."}
    n = len([e for e in focus["exercises"] if not e.startswith("Abs")])
    return {"status": "lift", "focus": focus["focus"], "exercises": focus["exercises"],
            "spoken": f"{when} is {focus['focus']} — {n} exercises. "
                      "Say 'show me the workout' to put the full list on screen.",
            "note": "Offer to call show_workout to display the full routine."}


def _fitness_css() -> str:
    return (
        "body{margin:0;background:#05080d;color:#e8eef5;"
        "font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}"
        ".wrap{max-width:1100px;margin:0 auto;padding:4vh 5vw}"
        "h1{font-size:2.4em;margin:0 0 .1em;font-weight:800;letter-spacing:.5px;"
        "background:linear-gradient(90deg,#fff,#7fdcff);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent}"
        ".sub{color:#7fa8c9;margin:0 0 2em;font-size:1.05em}"
        ".chips{display:flex;flex-wrap:wrap;gap:1em;margin:1.5em 0}"
        ".chip{background:#0d1520;border:1px solid #22405c;border-radius:12px;"
        "padding:1em 1.4em;min-width:120px}"
        ".chip .k{color:#7fa8c9;font-size:.8em;text-transform:uppercase;letter-spacing:.1em}"
        ".chip .v{font-size:1.8em;font-weight:700;color:#7fdcff}"
        ".chip .v.amber{color:#ffb84d}"
        "ol{line-height:1.9;font-size:1.15em;padding-left:1.2em}"
        "ol li{margin-bottom:.3em}"
        ".abs{color:#ffb84d;font-weight:600}"
        "h2{color:#7fdcff;border-bottom:1px solid #22405c;padding-bottom:.3em;margin-top:1.5em}"
        ".bar{height:14px;background:#0d1520;border-radius:8px;overflow:hidden;margin:.6em 0}"
        ".bar>i{display:block;height:100%;background:linear-gradient(90deg,#7fdcff,#4fa8d8)}"
    )


def show_workout(config: AssistantConfig, day: str = "") -> Dict[str, Any]:
    """Build a themed artifact of the day's routine (deterministic, no LLM) and
    open it in the borderless window on Geo's PC."""
    wd_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    d = (day or "").strip().lower()[:3]
    wd = wd_map.get(d, datetime.now().weekday())
    focus = fp.workout_for_weekday(wd)
    dayname = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
               "Saturday", "Sunday"][wd]
    from html import escape
    items = "".join(
        f'<li class="{"abs" if e.startswith("Abs") else ""}">{escape(e)}</li>'
        for e in focus["exercises"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(dayname)} — {escape(focus['focus'])}</title>"
            f"<style>{_fitness_css()}</style></head><body><div class='wrap'>"
            f"<h1>{escape(focus['focus'])}</h1>"
            f"<p class='sub'>{dayname} · Summer Reset training</p>"
            f"<ol>{items}</ol></div></body></html>")
    try:
        view_url, opened = _save_and_open_artifact(config, html,
                                                   f"{dayname} Workout")
    except Exception as e:
        return {"error": f"could not build workout: {e}"}
    return {"status": "shown", "focus": focus["focus"], "url": view_url,
            "opened_on_pc": opened,
            "spoken": f"Put {dayname}'s {focus['focus']} workout on your screen, sir."}


def fitness_progress(config: AssistantConfig) -> Dict[str, Any]:
    """Build a themed progress dashboard artifact: weight trend sparkline,
    day-of-plan, projected goal date at current rate, and macro targets."""
    today = datetime.now().date()
    st = weight_stats()
    day_n = fp.plan_day_number(today)
    phase = fp.phase_for(today)
    items = sorted(_load_weights(), key=lambda e: e["date"])

    # projected goal date from the 7-day trend (lbs/week)
    proj = "log a couple weeks to project"
    if st.get("count", 0) >= 8 and st.get("week_change") and st["week_change"] < 0:
        weeks = st["to_goal"] / abs(st["week_change"])
        if weeks > 0:
            eta = today + timedelta(weeks=weeks)
            proj = f"~{eta:%b %-d, %Y} at this rate"

    # simple inline SVG sparkline of the last 30 entries
    spark = ""
    pts = [e["lbs"] for e in items[-30:]]
    if len(pts) >= 2:
        lo, hi = min(pts), max(pts)
        rng = (hi - lo) or 1
        w, h = 640, 120
        coords = " ".join(
            f"{i / (len(pts) - 1) * w:.1f},{h - (v - lo) / rng * h:.1f}"
            for i, v in enumerate(pts))
        spark = (f"<h2>Weight trend (last {len(pts)} weigh-ins)</h2>"
                 f"<svg viewBox='0 0 {w} {h}' style='width:100%;height:auto;"
                 f"background:#0d1520;border:1px solid #22405c;border-radius:12px'>"
                 f"<polyline fill='none' stroke='#7fdcff' stroke-width='2.5' "
                 f"points='{coords}'/></svg>"
                 f"<p class='sub'>High {hi} · Low {lo}</p>")

    pct = 0
    if st.get("latest") is not None:
        span = fp.START_WEIGHT - fp.GOAL_WEIGHT
        pct = max(0, min(100, round((fp.START_WEIGHT - st["latest"]) / span * 100)))

    latest = st.get("latest", "—")
    avg7 = st.get("avg7", "—")
    m = fp.MACROS
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Summer Reset — Progress</title><style>{_fitness_css()}</style>"
            f"</head><body><div class='wrap'>"
            f"<h1>Summer Reset — Progress</h1>"
            f"<p class='sub'>Day {max(day_n,0)} · {phase['name'] if phase else 'Pre-reset'} · "
            f"{fp.START_WEIGHT:.0f} → {fp.GOAL_WEIGHT:.0f} lbs</p>"
            f"<div class='bar'><i style='width:{pct}%'></i></div>"
            f"<div class='chips'>"
            f"<div class='chip'><div class='k'>Latest</div><div class='v'>{latest}</div></div>"
            f"<div class='chip'><div class='k'>7-day avg</div><div class='v'>{avg7}</div></div>"
            f"<div class='chip'><div class='k'>Lost</div><div class='v amber'>{st.get('lost','—')}</div></div>"
            f"<div class='chip'><div class='k'>To goal</div><div class='v'>{st.get('to_goal','—')}</div></div>"
            f"</div>"
            f"<p class='sub'>Projected goal: {proj}</p>"
            f"{spark}"
            f"<h2>Daily targets</h2><div class='chips'>"
            f"<div class='chip'><div class='k'>Calories</div><div class='v'>{m['calories']}</div></div>"
            f"<div class='chip'><div class='k'>Protein</div><div class='v'>{m['protein']}g</div></div>"
            f"<div class='chip'><div class='k'>Carbs</div><div class='v'>{m['carbs']}g</div></div>"
            f"<div class='chip'><div class='k'>Fat</div><div class='v'>{m['fat']}g</div></div>"
            f"</div></div></body></html>")
    try:
        view_url, opened = _save_and_open_artifact(config, html, "Summer Reset Progress")
    except Exception as e:
        return {"error": f"could not build progress: {e}"}
    spoken = "Here's your progress, sir."
    if st.get("count", 0) == 0:
        spoken = ("Progress board is up, but you haven't logged a weigh-in yet — "
                  "log one each morning and the trend will fill in.")
    elif st.get("week_change") is not None:
        d = "down" if st["week_change"] < 0 else "up" if st["week_change"] > 0 else "flat"
        spoken = (f"You're at {latest}, 7-day average {avg7}, {d} "
                  f"{abs(st['week_change'])} from last week. {st.get('lost',0)} pounds down so far.")
    return {"status": "shown", "url": view_url, "opened_on_pc": opened, "spoken": spoken}


# ============================
# Proactive sports announcer
# ============================
class SportsAnnouncer:
    """Background thread that watches live World Cup matches and the next F1
    session and speaks up when something happens — a goal, kickoff, or a race
    about to start. Speaks ONLY when Jarvis is idle (no active conversation,
    nothing else playing); a chat always wins. Opt out with SPORTS_ANNOUNCE=0."""

    def __init__(self, config: AssistantConfig):
        self.config = config
        self.enabled = os.getenv("SPORTS_ANNOUNCE", "1") != "0"
        self._thread: Optional[threading.Thread] = None
        self._scores: Dict[str, tuple] = {}     # match id -> (home, away) score
        self._live_seen: set = set()            # matches we've announced kickoff for
        self._f1_warned: set = set()            # event ids we've warned about
        self._primed = False                    # first poll seeds state silently

    def start(self):
        if not self.enabled:
            logger.info("Sports announcer disabled (SPORTS_ANNOUNCE=0)")
            return
        self._thread = threading.Thread(target=self._loop, name="sports-announce",
                                        daemon=True)
        self._thread.start()
        logger.info("Sports announcer started")

    def _say(self, text: str) -> bool:
        """Speak only if truly idle. Returns False if it had to defer."""
        if _conversation_active.is_set() or _shutdown_event.is_set():
            return False
        if echo_cancellation.should_suppress():
            return False
        # non-blocking: never hold up polling on a stuck speaker
        if not PLAYBACK_LOCK.acquire(blocking=False):
            return False
        try:
            hud.set_state("speaking")
            logger.info(f"Announce: {text}")
            speak_tts_realtime(text, self.config)
            return True
        except Exception as e:
            logger.warning(f"Announce failed: {e}")
            return False
        finally:
            PLAYBACK_LOCK.release()
            hud.set_state("idle")

    def _poll_once(self):
        today = datetime.now().strftime("%Y%m%d")
        wc = _espn_json("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                        f"fifa.world/scoreboard?dates={today}")
        events = (wc or {}).get("events", [])
        announcements: List[str] = []
        for ev in events:
            try:
                eid = ev["id"]
                comp = ev["competitions"][0]
                teams = comp["competitors"]
                home = next(t for t in teams if t["homeAway"] == "home")
                away = next(t for t in teams if t["homeAway"] == "away")
                hn = home["team"].get("shortDisplayName") or home["team"]["displayName"]
                an = away["team"].get("shortDisplayName") or away["team"]["displayName"]
                st = ev["status"]["type"]["state"]
                hs, as_ = int(home.get("score", 0)), int(away.get("score", 0))
            except (KeyError, StopIteration, ValueError):
                continue
            if st == "in":
                if eid not in self._live_seen:
                    self._live_seen.add(eid)
                    if self._primed:
                        announcements.append(f"Sir, {hn} versus {an} has kicked off.")
                prev = self._scores.get(eid)
                if prev is not None and (hs, as_) != prev and self._primed:
                    scorer = hn if hs > prev[0] else an
                    announcements.append(f"Goal! {scorer} scores. {hn} {hs}, {an} {as_}.")
                self._scores[eid] = (hs, as_)
            elif st == "post" and eid in self._scores and self._primed:
                announcements.append(f"Full time: {hn} {hs}, {an} {as_}.")
                self._scores.pop(eid, None)
                self._live_seen.discard(eid)

        # F1: warn ~15 min before a session that starts today
        f1 = _espn_json("https://site.api.espn.com/apis/site/v2/sports/racing/"
                        "f1/scoreboard")
        for ev in (f1 or {}).get("events", [])[:1]:
            try:
                eid = ev["id"]
                t = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone()
                mins = (t - datetime.now(t.tzinfo)).total_seconds() / 60
                if 0 < mins <= 15 and eid not in self._f1_warned and self._primed:
                    self._f1_warned.add(eid)
                    name = _f1_race_label(ev.get("name", ""))
                    announcements.append(f"Sir, the {name} starts in about "
                                         f"{int(round(mins))} minutes.")
            except (KeyError, ValueError):
                continue

        self._primed = True
        for a in announcements:
            if not self._say(a):
                break            # busy — drop the rest, they'll re-derive next poll
            time.sleep(0.5)

    def _loop(self):
        # small stagger so startup logs settle first
        _shutdown_event.wait(20)
        while not _shutdown_event.is_set():
            try:
                self._poll_once()
            except Exception as e:
                logger.debug(f"Sports announcer poll error: {e}")
            _shutdown_event.wait(self._interval())

    def _interval(self) -> float:
        # poll fast (60s) when a match is live, lazily (5 min) otherwise
        return 60.0 if self._scores else 300.0


def _write_artifact_gallery(art_dir: str):
    """Regenerate a simple index of all artifacts (newest first)."""
    try:
        from urllib.parse import quote
        files = sorted(
            (f for f in os.listdir(art_dir)
             if f.endswith(".html") and f not in ("index.html", "viewer.html")),
            key=lambda f: os.path.getmtime(os.path.join(art_dir, f)),
            reverse=True,
        )
        cards = "\n".join(
            f'<a class="c" href="./viewer.html?v={_VIEWER_VER}&src={f}&title={quote(_artifact_label(f))}">'
            f'<span class="t">{_artifact_label(f)}</span>'
            f'<span class="f">{f}</span></a>' for f in files
        ) or '<p class="empty">No artifacts yet.</p>'
        html = _ARTIFACT_GALLERY_TMPL.replace("__CARDS__", cards)
        with open(os.path.join(art_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        logger.warning(f"Gallery write failed: {e}")


def _artifact_label(fname: str) -> str:
    base = re.sub(r"-\d{4}-\d{6}\.html$", "", fname)
    return base.replace("-", " ").title() or fname


_ARTIFACT_GALLERY_TMPL = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>J.A.R.V.I.S Artifacts</title><style>
*{box-sizing:border-box}body{margin:0;background:#04070b;color:#7fdcff;
font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;padding:5vmin}
h1{font-weight:600;letter-spacing:.3em;color:#dffaff;font-size:clamp(18px,3vmin,30px)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:2vmin;margin-top:3vmin}
.c{display:flex;flex-direction:column;gap:.8vmin;padding:2.4vmin;text-decoration:none;
border:1px solid rgba(127,220,255,.25);border-radius:1vmin;background:rgba(127,220,255,.05);
color:#7fdcff;transition:none}
.c:hover{border-color:#7fdcff;box-shadow:0 0 14px rgba(127,220,255,.3)}
.t{color:#dffaff;font-size:1.15em;letter-spacing:.05em}
.f{color:rgba(127,220,255,.5);font-size:.8em;word-break:break-all}
.empty{color:rgba(127,220,255,.5)}
</style></head><body><h1>J.A.R.V.I.S · ARTIFACTS</h1>
<div class="grid">__CARDS__</div></body></html>"""


def _ma_resolve_genre_playlist(genre: str) -> Optional[Dict[str, str]]:
    """Resolve a genre/vibe word to a curated playlist via the Music
    Assistant API. A bare genre as free-text play_media matches an arbitrary
    TRACK (e.g. 'rock' -> a cover band) and radio-mode wanders from there;
    Spotify's own genre playlists ('Rock Mix', 'Jazz Classics') are what a
    'play some rock' request actually means. Returns {'uri', 'name'} or None."""
    url = os.getenv("MUSIC_ASSISTANT_URL", "ws://127.0.0.1:8095/ws")
    token = os.getenv("MUSIC_ASSISTANT_TOKEN", "")
    if not token:
        return None
    try:
        from websocket import create_connection
        ws = create_connection(url, timeout=6)
        ws.recv()  # server info
        def call(mid, cmd, args):
            ws.send(json.dumps({"message_id": mid, "command": cmd, "args": args}))
            deadline = time.time() + 8
            while time.time() < deadline:
                m = json.loads(ws.recv())
                if m.get("message_id") == mid:
                    return m
            return {}
        call("a", "auth", {"token": token})
        r = call("s", "music/search",
                 {"search_query": genre, "media_types": ["playlist"], "limit": 12})
        playlists = (r.get("result") or {}).get("playlists") or []
        ws.close()
        g = genre.strip().lower()
        def score(p):
            name = (p.get("name") or "").lower()
            owner = (p.get("owner") or "").lower()
            s = 0
            if g in name:
                s += 2
            if owner == "spotify":
                s += 3  # curated genre mixes beat user playlists
            if "mix" in name or "classics" in name or "hits" in name:
                s += 1
            return s
        playlists.sort(key=score, reverse=True)
        best = playlists[0] if playlists and score(playlists[0]) >= 3 else None
        if best and best.get("uri"):
            return {"uri": best["uri"], "name": best.get("name", genre)}
    except Exception as e:
        logger.debug(f"Genre playlist resolution failed for '{genre}': {e}")
    return None


def _execute_tool(
    fn: str,
    parsed: Dict[str, Any],
    last_user_text: str,
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
) -> Tuple[Any, bool]:
    """Execute a tool by name with parsed args. Returns (result_dict, success_bool)."""
    logger.debug(f"TOOL CALL -> {fn} args={parsed}")

    try:
        if fn == "ha_get_state":
            return ha_client.get_state(entity_id=parsed.get("entity_id", "")), True

        elif fn == "ha_call_service":
            data = parsed.get("data", {})
            if not isinstance(data, dict):
                data = {}

            for k, v in (parsed or {}).items():
                if k in ("domain", "service", "data"):
                    continue
                if k not in data:
                    data[k] = v

            domain = parsed.get("domain", "")
            service = parsed.get("service", "")

            if not any(key in data for key in ("entity_id", "area_id", "device_id")):
                areas = ha_client.list_areas()
                inferred_area = infer_area_id_from_text(last_user_text, areas)
                if inferred_area:
                    data["area_id"] = inferred_area

            if domain == "music_assistant":
                # LLM payloads are sloppy here; HA 400/500s on them. Make valid.
                if service == "play_media":
                    rm = data.get("radio_mode")
                    if isinstance(rm, str):
                        data["radio_mode"] = rm.strip().lower() in ("true", "1", "yes", "on")
                    mt = str(data.get("media_type", "")).lower()
                    if mt == "song":
                        data["media_type"] = "track"
                    elif mt in ("genre", "mood", "vibe", "music", "radio"):
                        # genre request: resolve to a curated playlist instead of
                        # letting free-text match a random track
                        hit = _ma_resolve_genre_playlist(str(data.get("media_id", "")))
                        if hit:
                            logger.info(f"Genre '{data.get('media_id')}' -> playlist {hit['name']!r}")
                            data["media_id"] = hit["uri"]
                            data["media_type"] = "playlist"
                            data["radio_mode"] = False
                        else:
                            data.pop("media_type", None)
                    elif mt and mt not in ("artist", "album", "track", "playlist",
                                           "audiobook", "podcast"):
                        data.pop("media_type", None)  # other invented values
                if not any(k in data for k in ("entity_id", "area_id", "device_id")):
                    data["entity_id"] = "media_player.jarvis_speaker"

            t = _normalize_text(last_user_text)
            if domain == "light" and service in ("turn_on", "turn_off", "toggle"):
                if any(phrase in t for phrase in ["all lights", "whole house", "everywhere", "all the lights"]):
                    data["entity_id"] = "all"

            if domain == "light" and service == "turn_on":
                data = normalize_light_service_data(data, last_user_text)

            if not any(key in data for key in ("entity_id", "area_id", "device_id")):
                raise HomeAssistantError(
                    "No target (entity_id/area_id/device_id). "
                    "Try saying a room name (e.g., 'living room') or 'all lights'."
                )

            state_response = check_state_and_respond(
                ha_client, domain, service,
                data.get("entity_id"), data.get("area_id")
            )
            if state_response:
                return {"info": state_response, "action_skipped": True}, True

            return ha_client.call_service(domain=domain, service=service, data=data), True

        elif fn == "ha_list_entities":
            entities = ha_client.list_entities()
            domain_filter = parsed.get("domain")
            if domain_filter:
                entities = [e for e in entities if e.get("entity_id", "").startswith(f"{domain_filter}.")]
            simplified = [
                {
                    "entity_id": e.get("entity_id"),
                    "state": e.get("state"),
                    "friendly_name": e.get("attributes", {}).get("friendly_name"),
                }
                for e in entities[:50]
            ]
            return {"entities": simplified, "total": len(entities)}, True

        elif fn == "ha_list_areas":
            areas = ha_client.list_areas()
            return {"areas": areas}, True

        elif fn == "get_current_time":
            now = datetime.now()
            return {
                "time": now.strftime("%I:%M %p"),
                "date": now.strftime("%A, %B %d, %Y"),
                "iso": now.isoformat(),
            }, True

        elif fn == "computer_command":
            result = pc_control_action(config, str(parsed.get("action", "")),
                                       str(parsed.get("target", "")))
            return result, "error" not in result

        elif fn == "view_screen":
            result = pc_view_screen(config, str(parsed.get("screen", "all")))
            return result, "error" not in result

        elif fn == "daily_briefing":
            result = daily_briefing(config, ha_client)
            return result, "error" not in result

        elif fn == "log_weight":
            result = log_weight(parsed.get("lbs"))
            return result, "error" not in result

        elif fn == "workout":
            if parsed.get("show"):
                result = show_workout(config, str(parsed.get("day", "")))
            else:
                result = todays_workout(str(parsed.get("day", "")))
            return result, "error" not in result

        elif fn == "fitness_progress":
            result = fitness_progress(config)
            return result, "error" not in result

        elif fn == "remember":
            result = add_memory(str(parsed.get("text", "")))
            return result, "error" not in result

        elif fn == "forget":
            result = forget_memory(str(parsed.get("query", "")))
            return result, "error" not in result

        elif fn == "open_artifact":
            result = open_artifact(config, str(parsed.get("query", "")))
            return result, "error" not in result

        elif fn == "create_artifact":
            result = create_artifact(config, str(parsed.get("spec", "")),
                                     title=str(parsed.get("title", "")),
                                     needs_research=parsed.get("needs_research"),
                                     ha_client=ha_client)
            return result, "error" not in result

        elif fn == "web_search":
            query = str(parsed.get("query", "")).strip()
            if not query:
                return {"error": "empty query"}, False
            # server-side browsing via the Responses API — one call in, one
            # concise sourced answer out; the voice model reads it aloud
            from openai import OpenAI
            # max_retries=0: the SDK default retries 2x, turning one slow
            # search into a ~77s hang; fail fast instead (tools now run off
            # the realtime receiver, but a bounded call is still better UX)
            r = OpenAI().with_options(timeout=30.0, max_retries=0).responses.create(
                model=os.getenv("WEB_SEARCH_MODEL", "gpt-5-mini"),
                tools=[{"type": "web_search"}],
                input=("Search the web and answer for a VOICE assistant to read "
                       "aloud: 2-4 plain sentences, no URLs, no markdown, no "
                       f"lists. Question: {query}"),
            )
            answer = r.output_text or ""
            answer = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", answer)  # strip md links
            answer = re.sub(r"https?://\S+", "", answer).strip()
            logger.info(f"web_search('{query[:80]}') -> {answer[:120]}")
            return {"answer": answer[:1500]}, True

        elif fn == "set_timer":
            result = timers.set_timer(
                duration_seconds=parsed.get("duration_seconds"),
                at_time=parsed.get("at_time", ""),
                label=parsed.get("label", ""),
            )
            return result, "error" not in result

        elif fn == "cancel_timer":
            result = timers.cancel(parsed.get("which", ""))
            return result, "error" not in result

        elif fn == "list_timers":
            return timers.list(), True

        elif fn == "set_alarm":
            result = alarms.set_alarm(
                parsed.get("at_time", ""),
                label=parsed.get("label", ""),
                repeat=parsed.get("repeat", ""),
            )
            return result, "error" not in result

        elif fn == "cancel_alarm":
            result = alarms.cancel(parsed.get("which", ""))
            return result, "error" not in result

        elif fn == "list_alarms":
            return alarms.list(), True

        else:
            return {"error": f"Unknown tool: {fn}"}, False

    except Exception as e:
        logger.warning(f"TOOL ERROR -> {fn}: {e}")
        return {"error": str(e)}, False


def ask_chat_with_tools(
    messages: List[Dict[str, Any]],
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
    health_monitor: HealthMonitor,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Run chat completion with tool calls.
    Returns (reply_text, updated_messages, success).
    """
    messages = list(messages)

    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m.get("content", "") or ""
            break

    if not health_monitor.is_service_available("openai"):
        offline_response = get_offline_response(last_user_text)
        return offline_response, messages, False

    all_tools_succeeded = True

    for _round in range(config.conversation.max_tool_rounds):
        def _do_completion():
            return sync_client.chat.completions.create(
                model=config.conversation.chat_model,
                messages=messages,
                tools=_tools_schema(),
                tool_choice="auto",
            )

        resp = retry_api_call(_do_completion, config)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return (msg.content or "").strip(), messages, all_tools_succeeded

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            result, success = _execute_tool_call(tc, last_user_text, config, ha_client)
            if not success:
                all_tools_succeeded = False
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(result)[:8000],
            })

    logger.warning(f"Exhausted {config.conversation.max_tool_rounds} tool rounds")

    def _final_completion():
        return sync_client.chat.completions.create(
            model=config.conversation.chat_model,
            messages=messages,
        )

    resp_final = retry_api_call(_final_completion, config)
    return (resp_final.choices[0].message.content or "").strip(), messages, all_tools_succeeded


# ============================
# Claude (Anthropic) brain
# ============================
@lru_cache(maxsize=1)
def _anthropic_tools_schema():
    """Same tools as _tools_schema(), in Anthropic format."""
    return [
        {
            "name": "ha_get_state",
            "description": "Get the current state and attributes for a Home Assistant entity_id.",
            "input_schema": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
        {
            "name": "ha_call_service",
            "description": (
                "Call a Home Assistant service to control devices.\n"
                "IMPORTANT: data MUST include a target: entity_id OR area_id OR device_id.\n"
                "For light colors use light.turn_on with color_name/rgb_color/hs_color/xy_color/color_temp_kelvin."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "service": {"type": "string"},
                    "data": {"type": "object", "additionalProperties": True},
                    "entity_id": {"type": "string"},
                    "area_id": {"type": "string"},
                    "device_id": {"type": "string"},
                },
                "required": ["domain", "service"],
            },
        },
        {
            "name": "ha_list_entities",
            "description": "List all Home Assistant entities with their current states.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter (e.g., 'light', 'switch')",
                    },
                },
            },
        },
        {
            "name": "ha_list_areas",
            "description": "List all Home Assistant areas/rooms.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_timer",
            "description": (
                "Set a timer or reminder. Jarvis chimes and speaks when it fires. "
                "Use duration_seconds for relative ('10 minutes' -> 600) or "
                "at_time 'HH:MM' (24h) for absolute. label is spoken back (the reminder text)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer"},
                    "at_time": {"type": "string", "description": "24h HH:MM"},
                    "label": {"type": "string"},
                },
            },
        },
        {
            "name": "cancel_timer",
            "description": "Cancel a timer/reminder by its label, or 'all'.",
            "input_schema": {
                "type": "object",
                "properties": {"which": {"type": "string"}},
            },
        },
        {
            "name": "list_timers",
            "description": "List running timers/reminders with time remaining.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "web_search",
            "description": (
                "Search the live web for current information: news, sports, "
                "weather elsewhere, prices, release dates, or any fact that "
                "changes over time or postdates your knowledge. Returns a "
                "concise spoken-ready answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "computer_command",
            "description": (
                "Control Geo's Windows desktop PC. action: lock, sleep, "
                "shutdown, restart, playpause, next, previous, volume_up, "
                "volume_down, mute, or launch. For launch, set target to an "
                "app name (spotify, steam, chrome, edge, explorer, notepad, "
                "task manager, calculator) or a website URL."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string",
                               "description": "app name or URL, for launch"},
                },
                "required": ["action"],
            },
        },
        {
            "name": "view_screen",
            "description": (
                "Take a screenshot of Geo's PC (dual monitors) and attach it "
                "to this conversation so you can SEE it. Use when asked "
                "what's on screen, to look at / read / explain something on "
                "the screen, help with an error or game, or give opinions on "
                "what Geo is looking at. Takes a few seconds; the image "
                "arrives as the next user message."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "screen": {"type": "string", "description":
                               "'all' (default) = both monitors in one wide "
                               "image; '1' = left monitor only, '2' = right "
                               "— use a single monitor when Geo says which "
                               "one or fine text must be readable"},
                },
            },
        },
        {
            "name": "create_artifact",
            "description": (
                "Generate a visual HTML artifact and display it on Geo's PC screen: "
                "charts, graphs, dashboards, TABLES, ranked LISTS, comparisons, "
                "documents, or mini web tools. Use whenever the user wants "
                "something SHOWN or put on screen — 'give me a table of...', "
                "'make a list of...', 'show/draw/visualize...'. The builder "
                "searches the web itself when the topic needs current data, so "
                "call this directly (no separate web_search needed). Takes "
                "several seconds to build."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "spec": {"type": "string", "description":
                             "detailed description of what to build, incl. any data"},
                    "title": {"type": "string", "description": "short name"},
                    "needs_research": {"type": "boolean", "description":
                                       "true if it needs current/live data from the "
                                       "web (news, releases, prices, 'upcoming'/"
                                       "'latest'); false for timeless or purely "
                                       "visual content — false builds faster"},
                },
                "required": ["spec"],
            },
        },
        {
            "name": "daily_briefing",
            "description": (
                "Give Geo a spoken briefing of his day: date, weather, "
                "today's World Cup and Formula 1, and open tasks. Use for "
                "'good morning', 'brief me', 'what's my day look like', "
                "'what's going on today'. Read the returned 'spoken' text "
                "aloud naturally."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "log_weight",
            "description": (
                "Record Geo's morning weigh-in for his Summer Reset plan. "
                "Use when he says 'log my weight ...', 'I weigh ...', "
                "'weigh-in one seventy-nine ...'. Returns the fresh 7-day "
                "average and week trend to read back."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lbs": {"type": "number", "description": "weight in pounds"},
                },
                "required": ["lbs"],
            },
        },
        {
            "name": "workout",
            "description": (
                "Geo's training for today (or a named weekday) from his "
                "Summer Reset split. Use for 'what's my workout', 'what am I "
                "training today', 'show me the workout'. Set show=true to put "
                "the full routine on his PC screen; otherwise it returns a "
                "spoken summary."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description":
                            "weekday like 'monday' or 'thu'; omit for today"},
                    "show": {"type": "boolean", "description":
                             "true = display the full routine on screen"},
                },
            },
        },
        {
            "name": "fitness_progress",
            "description": (
                "Show Geo's weight-loss progress dashboard on his PC screen: "
                "weight trend, day of the plan, projected goal date, and macro "
                "targets. Use for 'how's my progress', 'show my weight "
                "progress', 'am I on track'."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "remember",
            "description": (
                "Save a durable fact about Geo or his preferences to "
                "long-term memory that persists across ALL future "
                "conversations. Use when he says 'remember...', states a "
                "lasting preference ('I take my coffee black', 'call me "
                "boss'), or shares info clearly worth keeping (birthdays, "
                "his teams, routines). Do NOT use for one-off task details "
                "or timers. Confirm briefly after saving."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description":
                             "the fact, phrased to stand alone later, e.g. "
                             "'Geo supports Argentina in the World Cup'"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "forget",
            "description": (
                "Remove something from long-term memory when Geo says "
                "'forget...' or corrects a fact you stored. Matches by "
                "description."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description":
                              "words identifying the memory, e.g. 'coffee'"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "open_artifact",
            "description": (
                "Re-open a PREVIOUSLY created artifact in its themed window "
                "on Geo's PC — use for 'pull up / show me again / reopen "
                "the ...'. Do NOT rebuild existing artifacts with "
                "create_artifact. Omit query (or say 'gallery') to open the "
                "browsable gallery of all past artifacts. If nothing "
                "matches, the result lists recent artifact titles to offer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description":
                              "a few words from the artifact's title, e.g. "
                              "'smiley face', 'movies table', 'raid guide'"},
                },
            },
        },
        {
            "name": "set_alarm",
            "description": (
                "Set a wake-up style alarm for a wall-clock time (persists across "
                "restarts; announces repeatedly until dismissed). at_time is 24h "
                "'HH:MM'. repeat may be 'daily' or 'weekdays'. Setting an alarm "
                "with the same time and label REPLACES it (use that to edit). "
                "Use set_timer for countdowns instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "at_time": {"type": "string", "description": "24h HH:MM"},
                    "label": {"type": "string"},
                    "repeat": {"type": "string", "enum": ["", "daily", "weekdays"]},
                },
                "required": ["at_time"],
            },
        },
        {
            "name": "cancel_alarm",
            "description": (
                "Cancel an alarm by label or time ('7:00'), or 'all'. Also "
                "silences an alarm that is currently going off (a ringing "
                "repeating alarm keeps its schedule)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"which": {"type": "string"}},
            },
        },
        {
            "name": "list_alarms",
            "description": "List set alarms with their next fire time.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)|[.!?]$|\n")


class SentenceSpeaker:
    """Speaks streamed LLM text sentence-by-sentence so audio starts before
    the full reply exists. Sentences are synthesized and played sequentially
    by a worker thread while the next tokens are still arriving."""

    MIN_CHARS = 24  # avoid choppy TTS on abbreviations like "Dr." or "e.g."

    def __init__(self, config: AssistantConfig):
        self.config = config
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._buffer = ""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            sentence = self._queue.get()
            if sentence is None:
                return
            try:
                speak_tts(sentence, self.config)
            except Exception as e:
                logger.error(f"Streaming TTS failed for sentence: {e}")

    def feed(self, text: str):
        self._buffer += text
        while True:
            cut = None
            for m in _SENTENCE_END_RE.finditer(self._buffer):
                # Only cut once enough text has accumulated and more follows
                if m.end() >= self.MIN_CHARS and m.end() < len(self._buffer):
                    cut = m.end()
                    break
            if cut is None:
                return
            sentence = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if sentence:
                self._queue.put(sentence)

    def finish(self):
        """Flush the remaining buffer and wait for playback to complete."""
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._queue.put(tail)
        self._queue.put(None)
        self._thread.join(timeout=120)


def ask_claude_streaming(
    messages: List[Dict[str, Any]],
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
) -> Tuple[str, bool]:
    """Run the conversation turn on Claude with tool calling, streaming the
    reply into TTS sentence-by-sentence. Returns (reply_text, success).

    Tool-use exchanges stay local to this call; only plain text turns are
    kept in the shared history (which both providers can consume).
    """
    system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""

    last_user_text = ""
    api_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content:
            continue
        api_messages.append({"role": role, "content": content})
        if role == "user":
            last_user_text = content

    all_tools_succeeded = True
    spoken_parts: List[str] = []
    speaker = SentenceSpeaker(config)
    _playback_interrupt.clear()

    try:
        for _round in range(config.conversation.max_tool_rounds + 1):
            with anthropic_client.messages.stream(
                model=config.conversation.anthropic_model,
                max_tokens=1024,
                # Voice assistant: skip thinking for lowest latency
                thinking={"type": "disabled"},
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=_anthropic_tools_schema(),
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    speaker.feed(text)
                    spoken_parts.append(text)
                response = stream.get_final_message()

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result, success = _execute_tool(
                    block.name, dict(block.input or {}), last_user_text, config, ha_client
                )
                if not success:
                    all_tools_succeeded = False
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)[:8000],
                    "is_error": not success,
                })

            api_messages.append({"role": "assistant", "content": response.content})
            api_messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning(f"Exhausted {config.conversation.max_tool_rounds} tool rounds (claude)")
    finally:
        speaker.finish()

    return "".join(spoken_parts).strip(), all_tools_succeeded


def ask_and_speak(
    messages: List[Dict[str, Any]],
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
    health_monitor: HealthMonitor,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """Run one conversation turn on the configured provider AND speak the
    reply (streaming on Claude, after completion on OpenAI).
    Returns (reply_text, updated_messages, success)."""
    provider = config.conversation.provider
    if provider == "auto":
        provider = "anthropic" if anthropic_client is not None else "openai"

    if provider == "anthropic" and anthropic_client is not None:
        if not health_monitor.is_service_available("anthropic"):
            last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            offline = get_offline_response(last_user)
            speak_tts(offline, config)
            return offline, messages, False
        reply, success = ask_claude_streaming(messages, config, ha_client)
        return reply, messages, success

    reply, messages, success = ask_chat_with_tools(messages, config, ha_client, health_monitor)
    if reply:
        speak_tts(reply, config)
    return reply, messages, success


def trim_conversation(messages: List[Dict[str, Any]], config: AssistantConfig) -> List[Dict[str, Any]]:
    """
    Keep conversation within limits.
    PATCH #7: Uses index arithmetic instead of list.index() to avoid ValueError
    when message dicts have been mutated (e.g. via model_dump() on tool_calls).
    """
    if len(messages) <= config.conversation.max_history_messages + 1:
        return messages

    system = [messages[0]] if messages and messages[0].get("role") == "system" else []
    history = messages[len(system):]

    # Trim to the most recent N messages
    trim_start = max(0, len(history) - config.conversation.max_history_messages)
    trimmed = history[trim_start:]

    # Walk backwards by index to find the assistant message that precedes any
    # leading tool message — no list.index() call, no equality scan.
    while trimmed and trimmed[0].get("role") == "tool":
        abs_idx = trim_start  # position of trimmed[0] in history
        if abs_idx > 0:
            trimmed = [history[abs_idx - 1]] + trimmed
            trim_start -= 1
        else:
            break

    return system + trimmed


# ============================
# Recording with VAD
# ============================
# ============================
# Realtime speech-to-speech mode (OpenAI Realtime API, GA protocol)
# ============================
# Wake word stays local; on wake we open a WebSocket session and stream mic
# audio up / play reply audio down. Server VAD owns turn-taking (replaces the
# local record/follow-up machinery for the whole conversation). HALF-DUPLEX:
# the mic is gated while reply audio is audible — the Pi has no echo
# cancellation, so barge-in is off (interrupt_response: false). The legacy
# pipeline is untouched and is the automatic fallback when the session can't
# be established. Protocol verified live against gpt-realtime-2 2026-07-10
# (see memory: session.audio.{input,output}.format audio/pcm@24k,
# response.output_audio.delta, response.function_call_arguments.done).

def _realtime_tools_schema():
    """The Realtime API takes FLAT function entries, not chat-completions'
    nested {'type':'function','function':{...}} shape."""
    return [{"type": "function",
             "name": t["function"]["name"],
             "description": t["function"]["description"],
             "parameters": t["function"]["parameters"]}
            for t in _tools_schema()]


def decimate_to_24k(pcm_i16: np.ndarray, rate: int) -> np.ndarray:
    """48k -> 24k by averaging sample pairs (cheap anti-aliased 2:1)."""
    if rate == 24000:
        return pcm_i16
    if rate == 48000:
        n = len(pcm_i16) & ~1
        x = pcm_i16[:n].astype(np.int32)
        return ((x[0::2] + x[1::2]) // 2).astype(np.int16)
    step = max(1, int(round(rate / 24000)))
    return pcm_i16[::step]


REALTIME_INSTRUCTIONS_ADDENDUM = (
    "\n\n=== ACCENT (HARD REQUIREMENT) ===\n"
    "You speak with a REFINED BRITISH ENGLISH (Received Pronunciation) accent "
    "at ALL times — the crisp, cut-glass diction of an English butler. This is "
    "non-negotiable and applies to EVERY sentence: greetings, answers, "
    "confirmations after tool calls, error messages, and your sign-off. Never "
    "slip into an American or neutral accent, not even for a word. If you ever "
    "notice yourself drifting, immediately return to British RP. Think Michael "
    "Caine or the J.A.R.V.I.S. of the Iron Man films: dry, precise, unflappable.\n"
    "=== END ACCENT ===\n"
    "\nYou are in a LIVE VOICE conversation — everything you output is spoken "
    "aloud. Keep replies to one to three short sentences unless asked for more. "
    "Always speak English. Composed, warm but efficient; address the user as "
    "'sir' occasionally; never theatrical, never rushed. Confirm actions "
    "briefly after tools succeed; if a tool fails, say so plainly."
    "\n\nEnding: when the user signals they are done — 'thank you', 'that "
    "will be all', 'that's everything', 'goodnight', or similar — give a "
    "brief sign-off ('Very good, sir.') and THEN call end_conversation. "
    "Do not call it if they are merely pausing mid-request."
)


def run_realtime_conversation(stream, mic_frame_length: int,
                              config: AssistantConfig,
                              ha_client: HomeAssistantClient) -> bool:
    """One wake-to-idle conversation over the Realtime API.

    Returns True if the session ran (even if the user said nothing);
    False if it could not be established — caller falls back to the
    legacy pipeline for this interaction."""
    import base64

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return False

    # Mic pre-buffer: connecting takes ~1.8s (up to ~7s cold) and speech in
    # that window used to be thrown away — the user had to WAIT for the
    # session before talking ("listening takes so long to start"). Capture
    # frames from the instant we're called and flush them into the session
    # before live streaming. Frames overlapping SFX playback ("Yes sir",
    # now async) are dropped by the same echo gate as the live mic.
    prebuf: List[bytes] = []
    prebuf_stop = threading.Event()

    def prebuf_worker():
        while not prebuf_stop.is_set() and len(prebuf) < 300:   # ~10s cap
            try:
                pcm = stream.read(mic_frame_length)[0]
            except Exception:
                break
            if echo_cancellation.should_suppress():
                continue
            prebuf.append(pcm)

    prebuf_t = threading.Thread(target=prebuf_worker, daemon=True, name="rt-prebuf")
    prebuf_t.start()

    def _stop_prebuf():
        # MUST run on every exit path before another reader touches the
        # shared mic stream, or two threads interleave-steal frames
        prebuf_stop.set()
        prebuf_t.join(timeout=1.5)

    try:
        ws = websocket.create_connection(
            f"wss://api.openai.com/v1/realtime?model={config.conversation.realtime_model}",
            header=[f"Authorization: Bearer {key}"],
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Realtime connect failed ({e}); falling back to legacy pipeline")
        _stop_prebuf()
        return False

    try:
        first = json.loads(ws.recv())
        if first.get("type") != "session.created":
            logger.warning(f"Realtime: unexpected first event {first.get('type')}")
            ws.close()
            _stop_prebuf()
            return False
    except Exception as e:
        logger.warning(f"Realtime handshake failed ({e}); falling back")
        try:
            ws.close()
        except Exception:
            pass
        _stop_prebuf()
        return False

    ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": (config.conversation.system_prompt
                             + REALTIME_INSTRUCTIONS_ADDENDUM
                             + _memory_prompt_block()),
            # end_conversation is realtime-only: handled by the receiver loop
            # below, never routed to _execute_tool
            "tools": _realtime_tools_schema() + [{
                "type": "function",
                "name": "end_conversation",
                "description": ("Close the voice session. Call ONLY after the user "
                                "clearly signals they are finished ('thank you', "
                                "'that will be all', 'goodnight'), and only after "
                                "speaking a brief sign-off."),
                "parameters": {"type": "object", "properties": {}},
            }],
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                        "create_response": True,
                        "interrupt_response": False,  # half-duplex: mic is gated anyway
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": config.conversation.realtime_voice,
                },
            },
        },
    }))
    logger.info(f"Realtime session open ({config.conversation.realtime_model}, "
                f"voice={config.conversation.realtime_voice})")

    stop = threading.Event()
    # All ws.send() must be serialized: the mic thread, the receiver loop, and
    # tool worker threads all write to the socket. (recv stays on the main
    # thread — one reader, many locked writers.)
    send_lock = threading.Lock()

    def wsend(obj):
        with send_lock:
            ws.send(json.dumps(obj))

    # Tools run in worker threads so a slow one (web_search, create_artifact)
    # does NOT block the receiver — a 77s blocking web_search previously killed
    # the whole session. While a tool is pending we hold "thinking" and never
    # idle-out.
    pending = set()
    pending_lock = threading.Lock()
    activity = [time.monotonic()]   # last user/assistant activity (cross-thread)

    # ---- playback: dedicated thread pipes PCM into one long-lived aplay ----
    play_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=256)
    play_head = [0.0]   # monotonic time until which queued audio is audible

    def speaking_now() -> bool:
        return time.monotonic() < play_head[0] + 0.25   # small drain tail

    def playback_worker():
        proc = None
        try:
            proc = subprocess.Popen(
                ["aplay", "-D", config.audio.alsa_device, "-t", "raw",
                 "-f", "S16_LE", "-r", "24000", "-c", "1", "-q", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while not stop.is_set():
                try:
                    chunk = play_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                proc.stdin.write(chunk)   # blocking = natural pacing
        except Exception as e:
            logger.error(f"Realtime playback died: {e}")
        finally:
            if proc:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

    def enqueue_audio(chunk: bytes):
        now = time.monotonic()
        start = max(now, play_head[0])
        dur = len(chunk) / 48000.0          # 24k * 2 bytes
        play_head[0] = start + dur
        # HUD equalizer: feed 40ms slices with per-slice delays, matching the
        # wav-tap frame rate — one eq frame per multi-hundred-ms delta made
        # the bars mushy and out of sync with syllables (user noticed)
        step = 1920                          # 40ms of pcm16 @ 24k
        off = max(0.0, start - now) + 0.40   # 0.40 ≈ aplay open + dmix prefill
        for i in range(0, len(chunk), step):
            sub = chunk[i:i + step]
            hud.feed_pcm(sub, 24000, delay=off)
            off += len(sub) / 48000.0
        try:
            play_q.put_nowait(chunk)
        except queue.Full:
            pass   # extreme backlog: drop rather than deadlock

    # ---- mic: always read (keeps ring buffer live); append only when open ----
    def mic_worker():
        # flush speech captured while the session was connecting, then go live
        for pcm_bytes in prebuf:
            pcm24 = decimate_to_24k(np.frombuffer(pcm_bytes, dtype=np.int16),
                                    config.audio.mic_rate)
            try:
                wsend({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24.tobytes()).decode(),
                })
            except Exception:
                return
        prebuf.clear()
        while not stop.is_set():
            try:
                pcm_bytes = stream.read(mic_frame_length)[0]
            except Exception:
                break
            if speaking_now() or echo_cancellation.should_suppress():
                continue                     # half-duplex gate: frame dropped
            pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            # visual feed only while LISTENING — during thinking/generating/
            # responding the mic kept driving the equalizer with room noise,
            # overriding the HUD's synthetic loading animations (user noticed).
            # The API append below stays unconditional: server VAD must still
            # hear the user for multi-turn.
            if hud._state == "listening":
                hud.feed_pcm(pcm_bytes, config.audio.mic_rate)
            pcm24 = decimate_to_24k(pcm_i16, config.audio.mic_rate)
            try:
                wsend({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24.tobytes()).decode(),
                })
            except Exception:
                break

    _stop_prebuf()   # hand the mic stream over to mic_worker
    play_t = threading.Thread(target=playback_worker, daemon=True, name="rt-play")
    mic_t = threading.Thread(target=mic_worker, daemon=True, name="rt-mic")
    play_t.start()
    mic_t.start()

    # ---- receiver / conversation state machine ----
    ws.settimeout(0.5)
    t_start = time.monotonic()
    got_any_speech = False
    end_requested = False            # set by the end_conversation tool
    last_user_text = ""
    fn_names: Dict[str, str] = {}    # call_id -> tool name
    hud.set_state("listening")
    end_reason = "closed"

    def run_tool_async(call_id, fn, parsed, user_text):
        """Execute a tool off the receiver thread, then feed the result back."""
        try:
            result, _ok = _execute_tool(fn, parsed, user_text, config, ha_client)
        except Exception as e:
            result = {"error": str(e)}
        try:
            # screenshots ride along as an input_image item so the model
            # literally sees the screen (and can answer follow-ups about it)
            img_b64 = result.pop("__screen_b64__", None) \
                if isinstance(result, dict) else None
            if img_b64:
                wsend({"type": "conversation.item.create",
                       "item": {"type": "message", "role": "user",
                                "content": [{"type": "input_image",
                                             "image_url": f"data:image/jpeg;base64,{img_b64}"}]}})
            wsend({"type": "conversation.item.create",
                   "item": {"type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(result, default=str)[:8000]}})
            wsend({"type": "response.create"})
        except Exception:
            pass
        with pending_lock:
            pending.discard(call_id)
            still_busy = bool(pending)
        activity[0] = time.monotonic()
        if not still_busy:
            hud.set_state("thinking")   # model is composing its reply now

    try:
        while not _shutdown_event.is_set():
            now = time.monotonic()
            with pending_lock:
                busy = bool(pending)
            if now - t_start > config.conversation.realtime_max_seconds:
                end_reason = "session cap"
                break
            idle_for = now - activity[0]
            if not busy and not got_any_speech and idle_for > 12.0:
                end_reason = "no speech"
                break
            if not busy and got_any_speech \
                    and idle_for > config.conversation.realtime_idle_seconds \
                    and not speaking_now():
                end_reason = "idle"
                break
            touch_activity()
            try:
                m = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                end_reason = f"socket: {e}"
                break
            t = m.get("type", "")

            if t == "input_audio_buffer.speech_started":
                got_any_speech = True
                activity[0] = time.monotonic()
                hud.set_state("listening")
            elif t == "input_audio_buffer.speech_stopped":
                hud.set_state("thinking")
            elif t == "conversation.item.input_audio_transcription.completed":
                last_user_text = m.get("transcript", "") or last_user_text
                if last_user_text:
                    logger.info(f"You (realtime): {last_user_text.strip()}")
            elif t == "response.output_audio.delta":
                enqueue_audio(base64.b64decode(m.get("delta", "")))
                hud.set_state("speaking")
                activity[0] = time.monotonic()
            elif t == "response.output_audio_transcript.done":
                txt = (m.get("transcript") or "").strip()
                if txt:
                    logger.info(f"Assistant (realtime): {txt}")
            elif t == "response.output_item.added":
                item = m.get("item", {})
                if item.get("type") == "function_call":
                    fn_names[item.get("call_id", "")] = item.get("name", "")
            elif t == "response.function_call_arguments.done":
                call_id = m.get("call_id", "")
                fn = m.get("name") or fn_names.get(call_id, "")
                if fn == "end_conversation":
                    logger.info("Realtime: user dismissed the conversation")
                    end_requested = True
                    wsend({"type": "conversation.item.create",
                           "item": {"type": "function_call_output", "call_id": call_id,
                                    "output": '{"status": "ending"}'}})
                    continue   # no response.create — the sign-off already played
                try:
                    parsed = json.loads(m.get("arguments") or "{}")
                except Exception:
                    parsed = {}
                logger.info(f"Realtime tool: {fn}({json.dumps(parsed)[:200]})")
                # run OFF the receiver thread so slow tools keep the session alive
                with pending_lock:
                    pending.add(call_id)
                activity[0] = time.monotonic()
                # slow tools get their own HUD state (scanning-sweep animation)
                hud.set_state("generating" if fn in ("create_artifact", "web_search",
                                                     "view_screen")
                              else "thinking")
                threading.Thread(target=run_tool_async, name=f"rt-tool-{fn}",
                                 args=(call_id, fn, parsed, last_user_text),
                                 daemon=True).start()
            elif t == "response.done":
                if end_requested:
                    end_reason = "dismissed"
                    break        # finally-block drains the sign-off audio
                activity[0] = time.monotonic()
                with pending_lock:
                    busy = bool(pending)
                if not speaking_now() and not busy:
                    hud.set_state("listening")
            elif t == "error":
                logger.warning(f"Realtime error event: {json.dumps(m.get('error'))[:300]}")
    finally:
        stop.set()
        try:
            ws.close()
        except Exception:
            pass
        mic_t.join(timeout=2)
        # let queued reply audio finish before yielding the speaker
        deadline = time.monotonic() + 8
        while speaking_now() and time.monotonic() < deadline:
            time.sleep(0.1)
        play_q.put(None)
        play_t.join(timeout=4)
        hud.set_state("idle")
        logger.info(f"Realtime session ended ({end_reason}, "
                    f"{time.monotonic() - t_start:.0f}s)")
    return True


def record_utterance_after_wake(
    stream,
    mic_frame_length: int,
    mic_rate: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
) -> Optional[str]:
    """
    Record audio with VAD-based speech detection.

    PATCH #1: The frame is always read from the stream first, even when
    echo-suppression is active.  Previously the loop did `continue` before
    reading, causing the sounddevice ring-buffer to fill up and the assistant
    to fall behind real-time after every TTS playback.

    PATCH #2: The VAD frame passed to webrtcvad is sized dynamically from the
    actual frame duration rather than being hard-coded to 480 samples.
    webrtcvad requires exactly 10 / 20 / 30 ms at 16 kHz:
      - 10 ms = 160 samples
      - 20 ms = 320 samples
      - 30 ms = 480 samples
    We compute frame_ms from the mic frame length and pick the closest valid
    VAD frame size, so the code still works correctly if MIC_RATE changes
    (e.g. 44100 Hz gives 441 mic samples → 147 at 16 kHz, nearest valid = 160).
    """
    chunks: List[np.ndarray] = []
    max_frames = int((config.audio.max_utterance_seconds * mic_rate) / mic_frame_length)
    frames_per_second = mic_rate / mic_frame_length

    # PATCH #2: compute the VAD frame size once, based on actual frame duration
    frame_duration_ms = (mic_frame_length / mic_rate) * 1000  # ms per mic frame
    decimated_samples_per_frame = int(round(mic_frame_length * 16000 / mic_rate))
    # Pick the largest valid webrtcvad frame that fits within our decimated frame
    for valid_ms in (30, 20, 10):
        valid_samples = int(16000 * valid_ms / 1000)  # 480, 320, or 160
        if decimated_samples_per_frame >= valid_samples:
            vad_frame_samples = valid_samples
            break
    else:
        vad_frame_samples = 160  # 10 ms fallback

    noise_rms_values: List[float] = []
    logger.info("Listening for your question...")

    # Drain frames that buffered while the wake chime played (blocking) so
    # ambient calibration below measures the room NOW — stale chime-echo
    # frames inflate the noise floor and deafen the adaptive threshold.
    try:
        while stream.read_available >= mic_frame_length:
            stream.read(mic_frame_length)
    except Exception:
        pass

    for i in range(config.audio.noise_calibration_frames):
        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunks.append(pcm_i16)
        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        noise_rms_values.append(rms(audio_f32))

    ambient_rms = float(np.median(noise_rms_values)) if noise_rms_values else 0.005
    adaptive_threshold = max(
        config.audio.silence_rms_threshold,
        ambient_rms * config.audio.adaptive_threshold_mult
    )
    logger.debug(f"Ambient RMS: {ambient_rms:.5f} -> adaptive threshold: {adaptive_threshold:.5f}")

    speech_detected = False
    speech_frame_count = 0
    silence_count = 0
    leading_silence_frames = int(config.audio.leading_silence_timeout * frames_per_second)
    min_speech_frames = int(config.audio.min_speech_seconds * frames_per_second)
    silence_frames_needed = int(config.audio.silence_seconds_to_stop * frames_per_second)

    vad_win_size = getattr(config.audio, 'vad_window_size', 2)
    energy_window: deque = deque(maxlen=3)
    vad_window: deque = deque(maxlen=3)

    peak_speech_rms: float = 0.0
    energy_drop_ratio = getattr(config.audio, 'energy_drop_ratio', 0.25)
    speech_resume_count = 0

    remaining_frames = max_frames - config.audio.noise_calibration_frames
    leading_silence_count = 0

    for frame_idx in range(remaining_frames):
        if _shutdown_event.is_set():
            return None

        # PATCH #1: ALWAYS read the frame first to drain the ring-buffer.
        # Only skip processing (not reading) when echo-suppressing.
        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        if echo_cancellation.should_suppress():
            # Frame consumed but discarded — buffer stays in sync
            continue

        hud.feed_pcm(pcm_bytes, mic_rate)
        chunks.append(pcm_i16)

        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        frame_rms = rms(audio_f32)
        energy_window.append(frame_rms)
        smoothed_rms = float(np.mean(energy_window))

        pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
        # PATCH #2: use dynamically-computed vad_frame_samples
        vad_frame = pcm_16k[:vad_frame_samples] if len(pcm_16k) >= vad_frame_samples else pcm_16k
        is_speech_vad = vad.is_speech(vad_frame.tobytes(), 16000)
        vad_window.append(is_speech_vad)

        vad_votes = sum(vad_window)
        energy_above = smoothed_rms >= adaptive_threshold
        vad_says_speech = vad_votes >= 2
        is_speech = energy_above or vad_says_speech

        if not speech_detected:
            if is_speech:
                speech_detected = True
                speech_frame_count = 1
                silence_count = 0
                peak_speech_rms = smoothed_rms
                logger.info(f"Speech detected at frame {frame_idx} (RMS: {smoothed_rms:.5f}, VAD: {vad_votes}/3)")
            else:
                leading_silence_count += 1
                if leading_silence_count >= leading_silence_frames:
                    logger.info("No speech detected within timeout, aborting.")
                    return None
        else:
            if is_speech:
                speech_frame_count += 1
                if smoothed_rms > peak_speech_rms:
                    peak_speech_rms = smoothed_rms
                if silence_count == 0:
                    speech_resume_count = 0
                else:
                    speech_resume_count += 1
                    if speech_resume_count >= 3:
                        silence_count = 0
                        speech_resume_count = 0
            else:
                speech_resume_count = 0
                silence_count += 1

                # The energy-drop test is nearly always true in real silence,
                # so this floor IS the effective stop latency for a finished
                # command — keep it a substantial fraction of the full stop
                # (at //2 with the old 0.35s config it fired after ~0.24s,
                # which cut people off mid-thought).
                quick_stop = (
                    speech_frame_count >= min_speech_frames and
                    peak_speech_rms > 0 and
                    smoothed_rms < peak_speech_rms * energy_drop_ratio and
                    silence_count >= max(4, int(silence_frames_needed * 0.7))
                )

                if quick_stop or (speech_frame_count >= min_speech_frames and silence_count >= silence_frames_needed):
                    logger.info(
                        f"End of speech: {speech_frame_count} frames, "
                        f"{silence_count} silence, quick={quick_stop}, "
                        f"RMS={smoothed_rms:.5f}, peak={peak_speech_rms:.5f}"
                    )
                    break

    if not speech_detected:
        logger.info("No speech detected in recording.")
        return None

    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)

    trailing_silence_samples = int(silence_count * mic_frame_length)
    if trailing_silence_samples > 0 and trailing_silence_samples < len(audio_i16):
        audio_i16 = audio_i16[:-trailing_silence_samples]

    # Downsample to 16 kHz before upload: STT models are 16 kHz-native and the
    # upload shrinks 3x, which cuts transcription round-trip time
    audio_16k = decimate_to_16k(audio_i16, mic_rate)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
    os.close(out_fd)
    save_wav(out_path, audio_16k, 16000, channels=1)
    return out_path


def record_followup(
    stream,
    mic_frame_length: int,
    mic_rate: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
    wake_detector: Any,
    ambient_threshold: float,
) -> Tuple[Optional[str], bool]:
    """
    Listen for follow-up speech or wake word.
    Returns: Tuple of (wav_path or None, was_wake_word_detected)

    followup_window_seconds bounds the wait for speech to START; once speech
    begins, recording continues until followup_silence_timeout of quiet (or
    max_utterance_seconds), like the main recorder. Previously the window
    capped the WHOLE recording, so any answer that started late got truncated
    at the window edge.

    Other fixes vs the original version:
    - The mic ring-buffer is drained before the window opens. Nothing reads
      the stream while the assistant thinks/speaks, so the first reads used
      to return STALE audio — often the assistant's own voice — and the
      window could expire before the user ever spoke.
    - A short pre-roll is prepended so detection latency doesn't clip the
      first syllable off the transcription.
    - The VAD frame is sized to a valid webrtcvad length (the full 80 ms
      frame made is_speech() return True unconditionally, reducing detection
      to energy-only at a doubled threshold).
    - min-speech uses min_speech_seconds directly (the old 1.5x multiplier
      discarded short answers like "yes").
    - Honors echo suppression while a playback tail is still audible.
    """
    frames_per_second = mic_rate / mic_frame_length
    window_frames = int(config.conversation.followup_window_seconds * frames_per_second)
    silence_timeout_frames = max(2, int(config.conversation.followup_silence_timeout * frames_per_second))
    max_speech_frames = int(config.audio.max_utterance_seconds * frames_per_second)
    min_speech_frames = max(2, int(config.audio.min_speech_seconds * frames_per_second))

    # ambient threshold from the caller keeps follow-up detection consistent
    # with the main utterance recorder's calibrated noise floor
    followup_threshold = max(ambient_threshold, config.audio.silence_rms_threshold * 2.0)

    # largest valid webrtcvad frame (10/20/30 ms @ 16 kHz) within our frame
    decimated_per_frame = int(round(mic_frame_length * 16000 / mic_rate))
    vad_frame_samples = 160
    for valid_ms in (30, 20, 10):
        if decimated_per_frame >= int(16000 * valid_ms / 1000):
            vad_frame_samples = int(16000 * valid_ms / 1000)
            break

    # drain audio that buffered while the assistant was thinking/speaking
    drained = 0
    try:
        while stream.read_available >= mic_frame_length:
            stream.read(mic_frame_length)
            drained += 1
    except Exception:
        pass
    if drained:
        logger.debug(f"Follow-up: drained {drained} stale buffered frames")

    pre_roll: deque = deque(maxlen=4)   # ~0.3s of audio from before detection
    chunks: List[np.ndarray] = []
    speech_detected = False
    speech_frame_count = 0
    silence_count = 0

    energy_window: deque = deque(maxlen=3)
    vad_window: deque = deque(maxlen=3)

    logger.debug("Listening for follow-up...")

    frame_idx = 0
    while True:
        if _shutdown_event.is_set():
            return None, False
        if not speech_detected and frame_idx >= window_frames:
            break
        if speech_detected and len(chunks) >= max_speech_frames:
            break
        frame_idx += 1

        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        if echo_cancellation.should_suppress():
            continue  # frame consumed but discarded — buffer stays in sync

        hud.feed_pcm(pcm_bytes, mic_rate)

        pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
        if len(pcm_16k) >= wake_detector.frame_length:
            if wake_detector.process(pcm_16k[:wake_detector.frame_length]):
                logger.info("Wake word detected during follow-up window")
                return None, True

        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        frame_rms = rms(audio_f32)
        energy_window.append(frame_rms)
        smoothed_rms = float(np.mean(energy_window))

        vad_frame = pcm_16k[:vad_frame_samples] if len(pcm_16k) >= vad_frame_samples else pcm_16k
        is_speech_vad = vad.is_speech(vad_frame.tobytes(), 16000)
        vad_window.append(is_speech_vad)
        vad_votes = sum(vad_window)

        is_speech = (smoothed_rms >= followup_threshold) and (vad_votes >= 2)

        if is_speech:
            if not speech_detected:
                speech_detected = True
                chunks.extend(pre_roll)  # keep the onset detection would clip
                logger.debug(f"Follow-up speech detected (RMS: {smoothed_rms:.4f}, VAD: {vad_votes}/3)")
            speech_frame_count += 1
            silence_count = 0
            chunks.append(pcm_i16)
        elif speech_detected:
            silence_count += 1
            chunks.append(pcm_i16)
            if speech_frame_count >= min_speech_frames and silence_count >= silence_timeout_frames:
                logger.debug(f"Follow-up end: {speech_frame_count} speech frames")
                break
        else:
            pre_roll.append(pcm_i16)

    if not speech_detected or speech_frame_count < min_speech_frames:
        logger.debug(f"No valid follow-up speech (detected={speech_detected}, frames={speech_frame_count}, min={min_speech_frames})")
        return None, False

    logger.info(f"Follow-up captured {len(chunks) / frames_per_second:.1f}s of speech")

    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    audio_i16 = apply_noise_reduction(audio_i16, mic_rate, config)
    audio_16k = decimate_to_16k(audio_i16, mic_rate)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
    os.close(out_fd)
    save_wav(out_path, audio_16k, 16000, channels=1)
    return out_path, False


# ============================
# Audio device resolution
# ============================
def resolve_mic_device(config: AssistantConfig) -> int:
    """Find the microphone by name — ALSA/sounddevice indices shuffle across
    reboots whenever USB devices change, so a fixed index is unreliable."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        logger.warning(f"Could not enumerate audio devices: {e}")
        return config.audio.mic_device_index

    want = config.audio.mic_device_name.lower()
    if want:
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and want in d["name"].lower():
                logger.info(f"Mic resolved by name '{config.audio.mic_device_name}' -> index {i} ({d['name']})")
                return i
        logger.warning(f"No input device matching '{config.audio.mic_device_name}'; falling back")

    idx = config.audio.mic_device_index
    if 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0:
        return idx
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            logger.warning(f"Configured mic index {idx} has no input; using index {i} ({d['name']})")
            return i
    return idx


def resolve_playback_device(config: AssistantConfig) -> str:
    """Find the ALSA playback device by card name, falling back to the
    configured ALSA_DEVICE if it's still a valid playback card, else 'default'."""
    want = config.audio.playback_card_name.lower()
    cards_text = ""
    try:
        with open("/proc/asound/cards") as f:
            cards_text = f.read()
    except Exception:
        pass

    if want and cards_text:
        for m in re.finditer(r"^\s*(\d+)\s+\[(\S+)\s*\]:\s*\S+\s+-\s+(.*)$", cards_text, re.M):
            num, ident, desc = m.group(1), m.group(2), m.group(3)
            if want in ident.lower() or want in desc.lower():
                # The named speaker is present: play through 'default', which
                # /etc/asound.conf routes to it VIA DMIX — so TTS/chimes mix
                # with Music Assistant instead of colliding on exclusive hw.
                logger.info(f"Playback resolved by name '{config.audio.playback_card_name}' "
                            f"-> default (dmix on card {num}, {desc.strip()})")
                return "default"
        logger.warning(f"No sound card matching '{config.audio.playback_card_name}' (speaker unplugged?)")

    # Validate the configured device: its card must exist AND have a playback PCM
    dev = config.audio.alsa_device
    m = re.match(r"(?:plug)?hw:(\d+)", dev)
    if m and os.path.exists(f"/proc/asound/card{m.group(1)}/pcm0p"):
        return dev
    if not m and dev != "default":
        return dev

    # Last resort: first card that can actually play audio (e.g. headphone
    # jack). 'default' itself may be pinned to a missing card in asound.conf.
    try:
        import glob as _glob
        for pcm in sorted(_glob.glob("/proc/asound/card[0-9]*/pcm0p")):
            card = re.search(r"card(\d+)", pcm).group(1)
            fallback = f"plughw:{card},0"
            logger.warning(f"Playback falling back to first available card: {fallback}")
            return fallback
    except Exception:
        pass
    return "default"


# ============================
# Wake word engines
# ============================
class WyomingWakeWord:
    """Wake word detection via a wyoming-openwakeword server (fully local).

    Speaks the Wyoming protocol over TCP: each event is a JSON header line
    ending in \\n with data_length/payload_length, followed by that many bytes
    of JSON data and raw payload. Audio is streamed as 16 kHz 16-bit mono
    audio-chunk events; the server replies with detection events.
    """

    frame_length = 512  # samples @ 16 kHz per process() call

    def __init__(self, host: str, port: int, model: str, refractory_seconds: float = 2.0):
        self.host = host
        self.port = port
        self.model = model
        self.refractory_seconds = refractory_seconds
        self._sock: Optional[socket.socket] = None
        self._recv_buffer = bytearray()
        self._timestamp_ms = 0
        self._last_detection_time = 0.0
        self._last_connect_attempt = 0.0
        self._connect()

    def _connect(self):
        self._close_socket()
        self._last_connect_attempt = time.time()
        sock = socket.create_connection((self.host, self.port), timeout=5)
        sock.settimeout(5)
        self._sock = sock
        self._recv_buffer.clear()
        self._timestamp_ms = 0
        self._write_event("detect", {"names": [self.model]})
        self._write_event("audio-start", {"rate": 16000, "width": 2, "channels": 1, "timestamp": 0})

    def _close_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _write_event(self, etype: str, data: Optional[Dict[str, Any]] = None, payload: bytes = b""):
        data_bytes = json.dumps(data or {}).encode("utf-8")
        header = {
            "type": etype,
            "version": "1.0.0",
            "data_length": len(data_bytes),
            "payload_length": len(payload),
        }
        self._sock.sendall(json.dumps(header).encode("utf-8") + b"\n" + data_bytes + payload)

    def _drain_events(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Non-blocking read of any complete events the server has sent."""
        import select as _select
        events = []
        while True:
            readable, _, _ = _select.select([self._sock], [], [], 0)
            if not readable:
                break
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("wyoming server closed connection")
            self._recv_buffer.extend(chunk)

        while True:
            newline = self._recv_buffer.find(b"\n")
            if newline < 0:
                break
            try:
                header = json.loads(self._recv_buffer[:newline])
            except json.JSONDecodeError:
                del self._recv_buffer[:newline + 1]
                continue
            data_len = header.get("data_length") or 0
            payload_len = header.get("payload_length") or 0
            total = newline + 1 + data_len + payload_len
            if len(self._recv_buffer) < total:
                break  # incomplete event; wait for more bytes
            data = header.get("data") or {}
            if data_len:
                try:
                    data = json.loads(self._recv_buffer[newline + 1:newline + 1 + data_len])
                except json.JSONDecodeError:
                    data = {}
            del self._recv_buffer[:total]
            events.append((header.get("type", ""), data))
        return events

    def process(self, pcm_16k: np.ndarray) -> bool:
        """Stream one frame of 16 kHz int16 audio; True when the wake word fired."""
        if self._sock is None:
            # Reconnect at most once per second so a down container doesn't spin
            if time.time() - self._last_connect_attempt < 1.0:
                return False
            try:
                self._connect()
                logger.info("Reconnected to wyoming-openwakeword")
            except Exception as e:
                logger.warning(f"openwakeword reconnect failed: {e}")
                return False

        payload = np.asarray(pcm_16k, dtype=np.int16).tobytes()
        try:
            self._write_event(
                "audio-chunk",
                {"rate": 16000, "width": 2, "channels": 1, "timestamp": self._timestamp_ms},
                payload,
            )
            self._timestamp_ms += (len(payload) // 2) * 1000 // 16000
            events = self._drain_events()
        except Exception as e:
            logger.warning(f"openwakeword connection error: {e}")
            self._close_socket()
            return False

        for etype, data in events:
            if etype == "detection":
                now = time.time()
                if now - self._last_detection_time >= self.refractory_seconds:
                    self._last_detection_time = now
                    logger.debug(f"openwakeword detection: {data}")
                    return True
        return False

    def reset(self):
        """Reconnect so stale buffered audio/detections can't fire a false wake."""
        try:
            self._connect()
        except Exception as e:
            logger.warning(f"openwakeword reset failed (will retry in loop): {e}")
            self._close_socket()

    def delete(self):
        self._close_socket()


class PorcupineDetector:
    """Porcupine wrapped in the same interface as WyomingWakeWord."""

    def __init__(self, config: AssistantConfig):
        if not PORCUPINE_AVAILABLE:
            raise RuntimeError("pvporcupine is not installed")
        access_key = os.environ["PICOVOICE_ACCESS_KEY"]
        sensitivity = max(0.0, min(1.0, config.wake_word.sensitivity))
        if config.wake_word.keyword_path:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[config.wake_word.keyword_path],
                sensitivities=[sensitivity],
            )
            self.label = f"custom ppn: {config.wake_word.keyword_path}"
        else:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[config.wake_word.wake_word],
                sensitivities=[sensitivity],
            )
            self.label = f"built-in: {config.wake_word.wake_word}"
        self.frame_length = self._porcupine.frame_length

    def process(self, pcm_16k: np.ndarray) -> bool:
        if len(pcm_16k) < self.frame_length:
            return False
        return self._porcupine.process(pcm_16k[:self.frame_length].tolist()) >= 0

    def reset(self):
        pass

    def delete(self):
        self._porcupine.delete()


def create_wake_detector(config: AssistantConfig) -> Tuple[Any, str]:
    """Create the configured wake word detector."""
    if config.wake_word.engine == "porcupine":
        detector = PorcupineDetector(config)
        return detector, f"porcupine ({detector.label})"

    detector = WyomingWakeWord(
        host=config.wake_word.oww_host,
        port=config.wake_word.oww_port,
        model=config.wake_word.oww_model,
        refractory_seconds=config.wake_word.refractory_seconds,
    )
    return detector, f"openwakeword '{config.wake_word.oww_model}' @ {config.wake_word.oww_host}:{config.wake_word.oww_port}"


# ============================
# Debug / Startup Info
# ============================
def _print_startup_info(config: AssistantConfig, mic_device: int):
    """Print startup diagnostics."""
    env_path = os.path.join(BASE_DIR, ".env")
    logger.info(f"Loaded .env from: {env_path}")
    logger.info(f"ALSA_DEVICE (playback): {config.audio.alsa_device}")
    logger.info(f"TTS voice: {config.tts.voice} | TTS model: {config.tts.model}")
    logger.info(f"Streaming TTS: {config.tts.streaming_enabled} | Direct streaming: {config.tts.direct_streaming}")
    logger.info(f"SFX: enabled={config.sfx.enabled} style={config.sfx.style} volume={config.sfx.volume}")

    for label, path in [
        ("SFX_STARTUP_WAV", config.sfx.startup_wav),
        ("SFX_AFTER_WAKE_WAV", config.sfx.after_wake_wav),
        ("SFX_AFTER_QUESTION_WAV", config.sfx.after_question_wav),
        ("SFX_PROCESSING_WAV", config.sfx.processing_wav),
        ("SFX_LISTENING_WAV", config.sfx.listening_wav),
        ("SFX_SUCCESS_WAV", config.sfx.success_wav),
        ("SFX_FAILURE_WAV", config.sfx.failure_wav),
    ]:
        if path:
            logger.info(f"{label}: {path}")

    logger.info(f"Wake sensitivity: {config.wake_word.sensitivity}")
    logger.info(
        f"Max history: {config.conversation.max_history_messages} | "
        f"Max tool rounds: {config.conversation.max_tool_rounds}"
    )
    logger.info(
        f"Silence detection: threshold={config.audio.silence_rms_threshold}, "
        f"stop_after={config.audio.silence_seconds_to_stop}s, "
        f"min_speech={config.audio.min_speech_seconds}s, "
        f"leading_timeout={config.audio.leading_silence_timeout}s, "
        f"adaptive_mult={config.audio.adaptive_threshold_mult}x"
    )
    logger.info(
        f"Health: monitor={config.health.enabled}, "
        f"watchdog={config.health.watchdog_enabled}"
    )
    logger.info(f"debug_tools={config.debug_tools} | debug_ha={config.debug_ha}")

    if config.home_assistant.url:
        logger.info(f"HA URL: {config.home_assistant.url}")
    if config.home_assistant.token:
        logger.info(f"HA TOKEN: (set) length={len(config.home_assistant.token)}")

    try:
        logger.info(f"Mic device: {mic_device} | {sd.query_devices(mic_device)['name']}")
    except Exception:
        pass


# ============================
# Initialization & Prewarming
# ============================
def initialize_openai_clients():
    """Initialize OpenAI clients with connection pooling."""
    global sync_client, async_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    sync_client = OpenAI(
        api_key=api_key,
        max_retries=0,
        timeout=30.0,
    )
    async_client = AsyncOpenAI(
        api_key=api_key,
        max_retries=0,
        timeout=30.0,
    )
    logger.info("OpenAI clients initialized")


def initialize_anthropic_client(config: AssistantConfig):
    """Initialize the Claude client when an API key is configured."""
    global anthropic_client

    if not ANTHROPIC_AVAILABLE:
        logger.info("anthropic package not installed; brain stays on OpenAI")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY not set; brain stays on OpenAI (add it to .env to use Claude)")
        return

    anthropic_client = anthropic.Anthropic(max_retries=1, timeout=30.0)
    logger.info(f"Anthropic client initialized (model: {config.conversation.anthropic_model})")


def prewarm_connections(config: AssistantConfig, ha_client: HomeAssistantClient):
    """
    Prewarm connections to reduce latency on first request.
    Runs in background thread during startup.
    """
    pool = get_thread_pool()

    def _prewarm_openai():
        # PATCH #4: use a cheap max_tokens=1 completion instead of models.list().
        # models.list() is a paginated API call that takes ~800ms — far too slow
        # for what is essentially a TCP connection warm-up.
        try:
            sync_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logger.debug("OpenAI connection prewarmed")
        except Exception as e:
            logger.debug(f"OpenAI prewarm failed (non-critical): {e}")

    def _prewarm_ha():
        try:
            if config.home_assistant.url:
                ha_client.list_entities()
                ha_client.list_areas()
                logger.debug("Home Assistant connection prewarmed, entities cached")
        except Exception as e:
            logger.debug(f"HA prewarm failed (non-critical): {e}")

    def _prewarm_pyaudio():
        # PATCH #3: open the persistent PyAudio stream at startup
        try:
            if PYAUDIO_AVAILABLE:
                get_pyaudio_stream(config)
                logger.debug("PyAudio persistent stream opened")
        except Exception as e:
            logger.debug(f"PyAudio prewarm failed: {e}")

    def _prewarm_anthropic():
        if anthropic_client is None:
            return
        try:
            anthropic_client.messages.count_tokens(
                model=config.conversation.anthropic_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            logger.debug("Anthropic connection prewarmed")
        except Exception as e:
            logger.debug(f"Anthropic prewarm failed (non-critical): {e}")

    futures = [
        pool.submit(_prewarm_openai),
        pool.submit(_prewarm_ha),
        pool.submit(_prewarm_pyaudio),
        pool.submit(_prewarm_anthropic),
    ]

    for f in futures:
        try:
            f.result(timeout=5.0)
        except Exception:
            pass

    logger.info("Connection prewarming complete")


def cleanup():
    """Cleanup resources on shutdown."""
    global _thread_pool, _pyaudio_stream, _pyaudio_instance, _http_session

    if _thread_pool:
        _thread_pool.shutdown(wait=False)
        _thread_pool = None

    # PATCH #3: close the persistent PyAudio stream on shutdown
    with _pyaudio_stream_lock:
        if _pyaudio_stream:
            try:
                _pyaudio_stream.stop_stream()
                _pyaudio_stream.close()
            except Exception:
                pass
            _pyaudio_stream = None

    if _pyaudio_instance and PYAUDIO_AVAILABLE:
        try:
            _pyaudio_instance.terminate()
        except Exception:
            pass
        _pyaudio_instance = None

    if _http_session:
        try:
            _http_session.close()
        except Exception:
            pass
        _http_session = None


# ============================
# Main Loop
# ============================

# Module-level storage for the last-computed ambient threshold so follow-up
# mode can reuse it without re-calibrating.  Reset to None after each
# wake-word cycle.
_last_ambient_threshold: float = 0.0


def _ma_provider_watchdog():
    """Music Assistant's Spotify provider fails to load when the container
    boots before the network is ready — and MA never retries on its own
    (observed twice; symptom: 'No playable items found' on every play).
    Check every 90s and reload the provider if it's down."""
    url = os.getenv("MUSIC_ASSISTANT_URL", "ws://127.0.0.1:8095/ws")
    token = os.getenv("MUSIC_ASSISTANT_TOKEN", "")
    if not token:
        return
    from websocket import create_connection
    while not _shutdown_event.is_set():
        time.sleep(90)
        try:
            ws = create_connection(url, timeout=6)
            ws.recv()
            def call(mid, cmd, args=None):
                ws.send(json.dumps({"message_id": mid, "command": cmd,
                                    "args": args or {}}))
                deadline = time.time() + 10
                while time.time() < deadline:
                    m = json.loads(ws.recv())
                    if m.get("message_id") == mid:
                        return m
                return {}
            call("a", "auth", {"token": token})
            provs = call("p", "providers").get("result") or []
            spotify = [p for p in provs if p.get("domain") == "spotify"]
            if spotify and not any(p.get("available") for p in spotify):
                cfgs = call("c", "config/providers").get("result") or []
                inst = next((c["instance_id"] for c in cfgs
                             if c.get("domain") == "spotify"), None)
                if inst:
                    call("r", "config/providers/reload", {"instance_id": inst})
                    logger.warning("Music Assistant Spotify provider was down - reloaded it")
            ws.close()
        except Exception:
            pass  # MA down/restarting; try again next cycle


def _telemetry_loop(config: AssistantConfig, ha_client: HomeAssistantClient):
    """Feed real system + home stats to the HUD's SYSTEMS panel (replacing
    its synthetic random-walk). Skips all work unless a HUD is connected.
    Rows are [key, bar_percent, display_text]."""
    prev_idle = prev_total = 0
    home_rows: List[List[Any]] = []
    last_home = 0.0
    while not _shutdown_event.is_set():
        time.sleep(3.0)
        if not hud._clients:
            continue
        try:
            rows: List[List[Any]] = []
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
            idle, total = vals[3] + vals[4], sum(vals)
            if prev_total and total > prev_total:
                cpu = 100.0 * (1 - (idle - prev_idle) / (total - prev_total))
                rows.append(["CPU", round(cpu), f"{cpu:.0f}%"])
            prev_idle, prev_total = idle, total

            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    tc = int(f.read().strip()) / 1000.0
                rows.append(["TEMP", round(min(100, max(0, (tc - 30) / 55 * 100))),
                             f"{tc:.0f}°C"])
            except OSError:
                pass

            mi: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    mi[k] = int(v.strip().split()[0])
            if mi.get("MemTotal"):
                mp = 100.0 * (1 - mi.get("MemAvailable", 0) / mi["MemTotal"])
                rows.append(["MEMORY", round(mp), f"{mp:.0f}%"])

            st = os.statvfs("/")
            dp = 100.0 * (1 - st.f_bavail / st.f_blocks)
            rows.append(["DISK", round(dp), f"{dp:.0f}%"])

            if time.time() - last_home > 10:
                last_home = time.time()
                home_rows = []
                try:
                    ents = ha_client.list_entities()
                    lights = [e for e in ents
                              if e.get("entity_id", "").startswith("light.")]
                    on = sum(1 for e in lights if e.get("state") == "on")
                    if lights:
                        home_rows.append(["LIGHTS", round(100 * on / len(lights)),
                                          f"{on}/{len(lights)}"])
                    playing = [e for e in ents
                               if e.get("entity_id", "").startswith("media_player.")
                               and e.get("state") == "playing"]
                    home_rows.append(["MEDIA", 70 if playing else 4,
                                      "LIVE" if playing else "IDLE"])
                    # now-playing line + album art for the HUD. The entity
                    # cache (60s TTL) makes track info stale, so fetch the
                    # desk speaker fresh; the cached list only finds OTHER
                    # players. entity_picture is a same-origin signed URL.
                    playing.sort(key=lambda e: e.get("entity_id") != "media_player.jarvis_speaker")
                    try:
                        session = get_http_session(config)
                        r = session.get(
                            f"{config.home_assistant.url}/api/states/media_player.jarvis_speaker",
                            headers={"Authorization": f"Bearer {config.home_assistant.token}"},
                            timeout=4)
                        fresh = r.json() if r.ok else {}
                        if fresh.get("state") == "playing":
                            playing.insert(0, fresh)
                        else:
                            playing = [p for p in playing
                                       if p.get("entity_id") != "media_player.jarvis_speaker"]
                    except Exception:
                        pass
                    self_np, self_npi = "", ""
                    if playing:
                        a = playing[0].get("attributes", {})
                        artist, title = a.get("media_artist", ""), a.get("media_title", "")
                        self_np = " — ".join(p for p in (artist, title) if p)
                        self_npi = a.get("entity_picture") or ""
                    _telemetry_loop.now_playing = self_np
                    _telemetry_loop.now_playing_img = self_npi
                except Exception:
                    pass  # HA hiccup; keep last known home rows

            hud._broadcast({"t": "tele", "v": rows + home_rows,
                            "np": getattr(_telemetry_loop, "now_playing", ""),
                            "npi": getattr(_telemetry_loop, "now_playing_img", "")})
        except Exception:
            pass  # telemetry must never take the assistant down


def handle_interaction(
    stream,
    mic_frame_length: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
    wake_detector: Any,
    messages: List[Dict[str, Any]],
    ha_client: HomeAssistantClient,
    health_monitor: HealthMonitor,
) -> Tuple[List[Dict[str, Any]], bool, float]:
    """
    Handle a single interaction cycle.
    Returns: (updated_messages, should_continue_followup, ambient_threshold)

    PATCH #8: now returns the ambient_threshold so record_followup can use it.
    PATCH #10: plays listening_wav at the correct moment (when recording starts).
    """
    global _last_ambient_threshold
    touch_activity()
    hud.set_state("listening")

    # PATCH #10: play listening sound just before we start recording
    play_wav_file_async(config.sfx.listening_wav, "SFX_LISTENING_WAV", config)

    utter_wav = record_utterance_after_wake(
        stream, mic_frame_length, config.audio.mic_rate, config, vad
    )

    if utter_wav is None:
        logger.info("No utterance captured, returning to wake word listening.")
        return messages, False, _last_ambient_threshold

    # Immediate audio feedback: play processing sound as soon as recording stops
    play_wav_file_async(config.sfx.processing_wav, "SFX_PROCESSING", config)
    hud.set_state("thinking")

    try:
        text = transcribe_audio_fast(utter_wav, config)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        speak_tts("Sorry, I couldn't understand that. Try again.", config)
        return messages, False, _last_ambient_threshold
    finally:
        try:
            os.unlink(utter_wav)
        except OSError:
            pass

    if not text:
        return messages, False, _last_ambient_threshold

    logger.info(f"You: {text}")
    messages.append({"role": "user", "content": text})

    play_wav_file_async(config.sfx.after_question_wav, "SFX_AFTER_QUESTION_WAV", config)

    try:
        # ask_and_speak handles TTS itself (streamed sentence-by-sentence on Claude)
        reply, messages, success = ask_and_speak(messages, config, ha_client, health_monitor)
        messages.append({"role": "assistant", "content": reply})

        if success:
            play_wav_file(config.sfx.success_wav, "SFX_SUCCESS_WAV", config)
        else:
            play_wav_file(config.sfx.failure_wav, "SFX_FAILURE_WAV", config)

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        reply = "Sorry, something went wrong processing that request."
        messages.append({"role": "assistant", "content": reply})
        play_wav_file(config.sfx.failure_wav, "SFX_FAILURE_WAV", config)
        try:
            speak_tts(reply, config)
        except Exception as tts_e:
            logger.error(f"TTS playback failed: {tts_e}")

    messages = trim_conversation(messages, config)
    logger.info(f"Assistant: {reply}")

    return messages, config.conversation.followup_enabled, _last_ambient_threshold


def main():
    """Main entry point."""
    global _wake_detector_ref, _last_ambient_threshold

    config = load_config()

    # Resolve audio devices by name (card numbers move around across reboots)
    mic_device = resolve_mic_device(config)
    config.audio.mic_device_index = mic_device
    config.audio.alsa_device = resolve_playback_device(config)

    initialize_openai_clients()
    initialize_anthropic_client(config)

    # load persisted alarms BEFORE the HTTP server can accept /alarm
    # mutations — a request in the gap used to persist empty state over
    # the real alarm list (bit us 2026-07-10)
    alarms.attach(config)

    if config.hud_events_enabled:
        hud.start(config.hud_events_port)

    _print_startup_info(config, mic_device)

    wake_detector, wake_label = create_wake_detector(config)
    _wake_detector_ref = wake_detector

    vad = WebRTCVAD(config)
    cache = EntityCache(config)
    ha_client = HomeAssistantClient(config, cache)
    health_monitor = HealthMonitor(config)
    watchdog = Watchdog(config)

    prewarm_connections(config, ha_client)

    # Desk-assistant extras: voice timers, HA-triggered announcements,
    # and real telemetry for the HUD (alarms attach earlier, pre-HTTP)
    timers.attach(config)

    def _announce(text: str):
        try:
            logger.info(f"Announcement: {text}")
            speak_tts(text, config)
        except Exception as e:
            logger.error(f"Announcement failed: {e}")
        finally:
            hud.set_state("idle")

    hud.say_cb = lambda text: get_thread_pool().submit(_announce, text)
    threading.Thread(target=_telemetry_loop, args=(config, ha_client),
                     daemon=True, name="hud-tele").start()
    threading.Thread(target=_ma_provider_watchdog, daemon=True,
                     name="ma-watchdog").start()

    logger.info(f"Assistant running. Wake word mode: {wake_label}")

    play_wav_file(config.sfx.startup_wav, "SFX_STARTUP", config)

    messages: List[Dict[str, Any]] = [{"role": "system",
        "content": config.conversation.system_prompt + _memory_prompt_block()}]

    wake_frame_length = wake_detector.frame_length
    mic_frame_length = int(wake_frame_length * config.audio.mic_rate / config.audio.porcupine_rate)

    logger.info(f"Mic rate: {config.audio.mic_rate} Hz | Wake rate: {config.audio.porcupine_rate} Hz")
    logger.info(f"Mic frame: {mic_frame_length} | Wake frame: {wake_frame_length}")

    health_monitor.start()
    watchdog.start()
    sports_announcer = SportsAnnouncer(config)
    sports_announcer.start()

    try:
        with sd.RawInputStream(
            device=mic_device,
            samplerate=config.audio.mic_rate,
            blocksize=mic_frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while not _shutdown_event.is_set():
                touch_activity()

                pcm_bytes = stream.read(mic_frame_length)[0]
                pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_16k = decimate_to_16k(pcm_i16, config.audio.mic_rate)

                if len(pcm_16k) < wake_frame_length:
                    continue

                if wake_detector.process(pcm_16k[:wake_frame_length]):
                    logger.info("Wake word detected!")
                    hud.set_state("listening")  # pop the HUD equalizer immediately
                    _conversation_active.set()  # mute the sports announcer
                    # async: "Yes sir" overlaps the ~2s realtime connect instead
                    # of preceding it — the echo gate keeps it out of the mic
                    play_wav_file_async(config.sfx.after_wake_wav,
                                        "SFX_AFTER_WAKE_WAV", config)

                    if config.conversation.realtime_enabled and \
                            run_realtime_conversation(stream, mic_frame_length,
                                                      config, ha_client):
                        _conversation_active.clear()
                        wake_detector.reset()
                        hud.set_state("idle")
                        continue
                    # realtime disabled or unreachable — legacy pipeline

                    messages, enable_followup, ambient_threshold = handle_interaction(
                        stream, mic_frame_length, config, vad, wake_detector,
                        messages, ha_client, health_monitor
                    )

                    # PATCH #8: pass calibrated ambient_threshold into follow-up loop
                    while enable_followup and not _shutdown_event.is_set():
                        touch_activity()
                        hud.set_state("listening")
                        followup_wav, was_wake_word = record_followup(
                            stream, mic_frame_length, config.audio.mic_rate, config, vad,
                            wake_detector, ambient_threshold
                        )

                        if was_wake_word:
                            logger.info("Wake word detected during follow-up - starting fresh conversation")
                            messages = [{"role": "system",
                                "content": config.conversation.system_prompt + _memory_prompt_block()}]
                            break

                        elif followup_wav is None:
                            logger.debug("Follow-up timeout, returning to wake word detection")
                            break

                        else:
                            play_wav_file_async(config.sfx.processing_wav, "SFX_PROCESSING", config)

                            try:
                                followup_text = transcribe_audio_fast(followup_wav, config)
                            except Exception as e:
                                logger.error(f"Follow-up transcription failed: {e}")
                                break
                            finally:
                                try:
                                    os.unlink(followup_wav)
                                except OSError:
                                    pass

                            if not followup_text:
                                break

                            logger.info(f"You (follow-up): {followup_text}")
                            messages.append({"role": "user", "content": followup_text})

                            try:
                                reply, messages, success = ask_and_speak(messages, config, ha_client, health_monitor)
                                messages.append({"role": "assistant", "content": reply})

                                if success:
                                    play_wav_file(config.sfx.success_wav, "SFX_SUCCESS_WAV", config)
                                else:
                                    play_wav_file(config.sfx.failure_wav, "SFX_FAILURE_WAV", config)

                            except Exception as e:
                                logger.error(f"Follow-up chat completion failed: {e}")
                                break

                            messages = trim_conversation(messages, config)
                            logger.info(f"Assistant: {reply}")

                            enable_followup = config.conversation.followup_enabled

                    # Flush stale buffered audio/detections before resuming
                    # wake listening (matters for the wyoming stream)
                    _conversation_active.clear()
                    wake_detector.reset()
                    hud.set_state("idle")

    finally:
        watchdog.stop()
        health_monitor.stop()
        cleanup()
        wake_detector.delete()
        _wake_detector_ref = None
        logger.info("Assistant shut down.")


if __name__ == "__main__":
    main()