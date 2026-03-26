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
from datetime import datetime
from enum import Enum
from functools import lru_cache

import numpy as np
import sounddevice as sd
import pvporcupine
import aiohttp
import requests
import websocket
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import queue

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
_porcupine_ref: Optional[pvporcupine.Porcupine] = None
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

        try:
            _pyaudio_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                frames_per_buffer=config.tts.chunk_size,
            )
            logger.debug("PyAudio output stream opened")
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


def aplay_wav(wav_path: str, config: AssistantConfig):
    """Play WAV file using aplay."""
    with PLAYBACK_LOCK:
        echo_cancellation.start_playback()
        try:
            p = subprocess.run(
                ["aplay", "-D", config.audio.alsa_device, wav_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            echo_cancellation.end_playback()

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
    with PLAYBACK_LOCK:
        echo_cancellation.start_playback()
        try:
            p = subprocess.Popen(
                ["aplay", "-D", config.audio.alsa_device, wav_path],
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

                    stream.write(audio_chunk)

            # Flush remaining
            if buffer and not _playback_interrupt.is_set():
                remaining = bytes(buffer)
                if config.tts.volume_normalization and len(remaining) >= 2:
                    pcm_data = np.frombuffer(remaining, dtype=np.int16)
                    pcm_data = normalize_volume_fast(pcm_data, config.tts.target_db)
                    remaining = pcm_data.tobytes()
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
    ]


def _execute_tool_call(
    tc,
    last_user_text: str,
    config: AssistantConfig,
    ha_client: HomeAssistantClient,
) -> Tuple[Any, bool]:
    """Execute a single tool call. Returns (result_dict, success_bool)."""
    fn = tc.function.name
    try:
        parsed = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
    except Exception:
        parsed = {}

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

    # PATCH #10: play the listening sound now that we are actively recording
    # (moved here from handle_interaction so it fires at the right moment)

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

                quick_stop = (
                    speech_frame_count >= min_speech_frames and
                    peak_speech_rms > 0 and
                    smoothed_rms < peak_speech_rms * energy_drop_ratio and
                    silence_count >= max(3, silence_frames_needed // 2)
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

    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
    os.close(out_fd)
    save_wav(out_path, audio_i16, mic_rate, channels=1)
    return out_path


def record_followup(
    stream,
    mic_frame_length: int,
    mic_rate: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
    porcupine: pvporcupine.Porcupine,
    ambient_threshold: float,
) -> Tuple[Optional[str], bool]:
    """
    Listen for follow-up speech or wake word.
    Returns: Tuple of (wav_path or None, was_wake_word_detected)

    PATCH #8: accepts ambient_threshold from the caller (computed during the
    preceding utterance recording) instead of using a hardcoded *3.0 multiplier.
    Falls back to silence_rms_threshold * 2 when no ambient value is provided.
    """
    frames_per_second = mic_rate / mic_frame_length
    timeout_frames = int(config.conversation.followup_window_seconds * frames_per_second)
    silence_timeout_frames = int(config.conversation.followup_silence_timeout * frames_per_second)

    # PATCH #8: use caller-supplied ambient threshold so follow-up detection is
    # consistent with the main utterance recorder's calibrated noise floor.
    followup_threshold = max(ambient_threshold, config.audio.silence_rms_threshold * 2.0)

    chunks: List[np.ndarray] = []
    speech_detected = False
    speech_frame_count = 0
    silence_count = 0
    min_speech_frames = int(max(0.4, config.audio.min_speech_seconds * 1.5) * frames_per_second)

    energy_window: deque = deque(maxlen=3)
    vad_window: deque = deque(maxlen=3)

    logger.debug("Listening for follow-up...")

    for frame_idx in range(timeout_frames):
        if _shutdown_event.is_set():
            return None, False

        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
        if len(pcm_16k) >= porcupine.frame_length:
            if porcupine.process(pcm_16k[:porcupine.frame_length].tolist()) >= 0:
                logger.info("Wake word detected during follow-up window")
                return None, True

        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        frame_rms = rms(audio_f32)
        energy_window.append(frame_rms)
        smoothed_rms = float(np.mean(energy_window))

        is_speech_vad = vad.is_speech(pcm_16k.tobytes(), 16000)
        vad_window.append(is_speech_vad)
        vad_votes = sum(vad_window)

        # PATCH #8: use ambient-calibrated threshold instead of hardcoded *3.0
        is_speech = (smoothed_rms >= followup_threshold) and (vad_votes >= 3)

        if is_speech:
            if not speech_detected:
                speech_detected = True
                logger.debug(f"Follow-up speech detected (RMS: {smoothed_rms:.4f}, VAD: {vad_votes}/3)")
            speech_frame_count += 1
            silence_count = 0
            chunks.append(pcm_i16)
        else:
            if speech_detected:
                silence_count += 1
                chunks.append(pcm_i16)

                if speech_frame_count >= min_speech_frames and silence_count >= silence_timeout_frames:
                    logger.debug(f"Follow-up end: {speech_frame_count} speech frames")
                    break

    if not speech_detected or speech_frame_count < min_speech_frames:
        logger.debug(f"No valid follow-up speech (detected={speech_detected}, frames={speech_frame_count}, min={min_speech_frames})")
        return None, False

    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    audio_i16 = apply_noise_reduction(audio_i16, mic_rate, config)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
    os.close(out_fd)
    save_wav(out_path, audio_i16, mic_rate, channels=1)
    return out_path, False


# ============================
# Porcupine init
# ============================
def create_porcupine(config: AssistantConfig) -> Tuple[pvporcupine.Porcupine, str]:
    """Create Porcupine wake word engine."""
    access_key = os.environ["PICOVOICE_ACCESS_KEY"]

    if config.wake_word.keyword_path:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[config.wake_word.keyword_path],
            sensitivities=[max(0.0, min(1.0, config.wake_word.sensitivity))],
        )
        return porcupine, f"custom ppn: {config.wake_word.keyword_path}"

    porcupine = pvporcupine.create(
        access_key=access_key,
        keywords=[config.wake_word.wake_word],
        sensitivities=[max(0.0, min(1.0, config.wake_word.sensitivity))],
    )
    return porcupine, f"built-in: {[config.wake_word.wake_word]}"


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

    futures = [
        pool.submit(_prewarm_openai),
        pool.submit(_prewarm_ha),
        pool.submit(_prewarm_pyaudio),
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


def handle_interaction(
    stream,
    mic_frame_length: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
    porcupine: pvporcupine.Porcupine,
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
        reply, messages, success = ask_chat_with_tools(messages, config, ha_client, health_monitor)
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

    messages = trim_conversation(messages, config)

    logger.info(f"Assistant: {reply}")

    if reply:
        try:
            speak_tts(reply, config)
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")

    return messages, config.conversation.followup_enabled, _last_ambient_threshold


def main():
    """Main entry point."""
    global _porcupine_ref, _last_ambient_threshold

    config = load_config()
    mic_device = config.audio.mic_device_index

    initialize_openai_clients()

    _print_startup_info(config, mic_device)

    porcupine, wake_label = create_porcupine(config)
    _porcupine_ref = porcupine

    vad = WebRTCVAD(config)
    cache = EntityCache(config)
    ha_client = HomeAssistantClient(config, cache)
    health_monitor = HealthMonitor(config)
    watchdog = Watchdog(config)

    prewarm_connections(config, ha_client)

    logger.info(f"Assistant running. Wake word mode: {wake_label}")

    play_wav_file(config.sfx.startup_wav, "SFX_STARTUP", config)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": config.conversation.system_prompt}]

    porc_frame_length = porcupine.frame_length
    mic_frame_length = int(porc_frame_length * config.audio.mic_rate / config.audio.porcupine_rate)

    logger.info(f"Mic rate: {config.audio.mic_rate} Hz | Porcupine rate: {config.audio.porcupine_rate} Hz")
    logger.info(f"Mic frame: {mic_frame_length} | Porcupine frame: {porc_frame_length}")

    health_monitor.start()
    watchdog.start()

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

                if len(pcm_16k) < porc_frame_length:
                    continue

                if porcupine.process(pcm_16k[:porc_frame_length].tolist()) >= 0:
                    logger.info("Wake word detected!")
                    play_wav_file(config.sfx.after_wake_wav, "SFX_AFTER_WAKE_WAV", config)

                    messages, enable_followup, ambient_threshold = handle_interaction(
                        stream, mic_frame_length, config, vad, porcupine,
                        messages, ha_client, health_monitor
                    )

                    # PATCH #8: pass calibrated ambient_threshold into follow-up loop
                    while enable_followup and not _shutdown_event.is_set():
                        touch_activity()
                        followup_wav, was_wake_word = record_followup(
                            stream, mic_frame_length, config.audio.mic_rate, config, vad,
                            porcupine, ambient_threshold
                        )

                        if was_wake_word:
                            logger.info("Wake word detected during follow-up - starting fresh conversation")
                            messages = [{"role": "system", "content": config.conversation.system_prompt}]
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
                                reply, messages, success = ask_chat_with_tools(messages, config, ha_client, health_monitor)
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

                            if reply:
                                try:
                                    speak_tts(reply, config)
                                except Exception as e:
                                    logger.error(f"Follow-up TTS playback failed: {e}")

                            enable_followup = config.conversation.followup_enabled

    finally:
        watchdog.stop()
        health_monitor.stop()
        cleanup()
        porcupine.delete()
        _porcupine_ref = None
        logger.info("Assistant shut down.")


if __name__ == "__main__":
    main()