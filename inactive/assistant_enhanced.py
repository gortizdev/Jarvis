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
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
from functools import lru_cache
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
import sounddevice as sd
import pvporcupine
import aiohttp
import requests
import websocket
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

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
# Dataclass Configuration
# ============================
@dataclass
class AudioConfig:
    """Audio device and processing configuration."""
    mic_device_index: int = 2
    alsa_device: str = "default"
    mic_rate: int = 48000
    porcupine_rate: int = 16000
    
    # WebRTC VAD settings
    vad_mode: int = 3  # 0-3, 3 is most aggressive
    vad_frame_ms: int = 30  # Must be 10, 20, or 30ms
    
    # Recording parameters
    max_utterance_seconds: float = 15.0
    silence_rms_threshold: float = 0.012
    silence_seconds_to_stop: float = 0.75
    min_speech_seconds: float = 0.4
    leading_silence_timeout: float = 4.0
    adaptive_threshold_mult: float = 2.5
    noise_calibration_frames: int = 10
    
    # Noise suppression
    noise_reduce_enabled: bool = True
    noise_reduce_prop_decrease: float = 0.6


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    wake_word: str = "picovoice"
    keyword_path: str = ""
    sensitivity: float = 0.65


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    model: str = "gpt-4o-mini-tts"
    voice: str = "fable"
    streaming_enabled: bool = True
    volume_normalization: bool = True
    target_db: float = -20.0  # Target loudness in dB


@dataclass
class SFXConfig:
    """Sound effects configuration."""
    enabled: bool = True
    style: str = "jarvis"
    volume: float = 0.55
    after_wake_wav: str = ""
    after_question_wav: str = ""
    success_wav: str = ""
    failure_wav: str = ""
    listening_wav: str = ""


@dataclass
class HomeAssistantConfig:
    """Home Assistant connection configuration."""
    url: str = ""
    token: str = ""
    
    # Entity/state caching
    cache_ttl_seconds: float = 60.0
    areas_cache_ttl_seconds: float = 300.0
    
    # Allowed domains and services
    allowed_domains: set = field(default_factory=lambda: {
        "light", "switch", "fan", "cover", "climate", "media_player",
        "scene", "script", "automation", "input_boolean", "button",
        "select", "number", "lock",
    })
    
    allowed_services: set = field(default_factory=lambda: {
        "turn_on", "turn_off", "toggle", "open_cover", "close_cover",
        "stop_cover", "set_temperature", "set_hvac_mode", "media_play",
        "media_pause", "media_stop", "play_media", "volume_set",
        "select_source", "press", "select_option", "set_value",
        "set_percentage", "set_preset_mode", "set_fan_mode", "lock", "unlock",
    })


@dataclass
class ConversationConfig:
    """Conversation and model configuration."""
    chat_model: str = "gpt-4o"
    transcribe_model: str = "gpt-4o-mini-transcribe"
    max_history_messages: int = 30
    max_tool_rounds: int = 5
    
    # Follow-up mode
    followup_enabled: bool = True
    followup_window_seconds: float = 5.0
    followup_silence_timeout: float = 2.0
    
    # Barge-in
    bargein_enabled: bool = True
    
    system_prompt: str = ""


@dataclass
class HealthConfig:
    """Health monitoring configuration."""
    enabled: bool = True
    check_interval_seconds: float = 60.0
    ha_timeout_seconds: float = 5.0
    openai_timeout_seconds: float = 10.0
    
    # Watchdog
    watchdog_enabled: bool = True
    watchdog_timeout_seconds: float = 120.0


@dataclass  
class AssistantConfig:
    """Root configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    sfx: SFXConfig = field(default_factory=SFXConfig)
    home_assistant: HomeAssistantConfig = field(default_factory=HomeAssistantConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    
    # Retry settings
    api_max_retries: int = 3
    api_retry_base_delay: float = 1.0
    
    # Debug flags
    debug_tools: bool = True
    debug_ha: bool = True
    log_level: str = "INFO"


# ============================
# Configuration Loader
# ============================
def _clean_env(v: Optional[str]) -> str:
    if v is None:
        return ""
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v


def _env_float(name: str, default: float) -> float:
    raw = _clean_env(os.getenv(name))
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _clean_env(os.getenv(name))
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _clean_env(os.getenv(name)).lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    path = path.strip()
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def load_config() -> AssistantConfig:
    """Load configuration from environment variables."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    
    default_system_prompt = (
        "You are a highly capable, calm, and precise home voice assistant.\n"
        "Be a little witty and confident, but not verbose.\n"
        "Keep responses concise and conversational.\n"
        "If you take an action, confirm it succinctly.\n"
        "\n"
        "IMPORTANT TOOL RULES:\n"
        "- When calling ha_call_service you MUST include a target in data (entity_id or area_id or device_id).\n"
        "- If the user mentions a room/area, prefer area_id.\n"
        "- If the user says 'all lights' / 'whole house' / 'everywhere', use entity_id='all'.\n"
        "- For light color requests, call light.turn_on with one of:\n"
        "  color_name (string), rgb_color ([r,g,b]), hs_color ([h,s]), xy_color ([x,y]),\n"
        "  color_temp_kelvin (int), and optionally brightness_pct (0-100).\n"
        "- ALWAYS check current state before actions when relevant - inform user if already in desired state.\n"
        "- Suggest relevant scenes when appropriate based on time of day or context.\n"
    )
    
    config = AssistantConfig(
        audio=AudioConfig(
            mic_device_index=_env_int("MIC_DEVICE_INDEX", 2),
            alsa_device=_clean_env(os.getenv("ALSA_DEVICE")) or "default",
            mic_rate=_env_int("MIC_RATE", 48000),
            vad_mode=_env_int("VAD_MODE", 3),
            max_utterance_seconds=_env_float("MAX_UTTERANCE_SECONDS", 15.0),
            silence_rms_threshold=_env_float("SILENCE_RMS_THRESHOLD", 0.012),
            silence_seconds_to_stop=_env_float("SILENCE_SECONDS_TO_STOP", 0.75),
            min_speech_seconds=_env_float("MIN_SPEECH_SECONDS", 0.4),
            leading_silence_timeout=_env_float("LEADING_SILENCE_TIMEOUT", 4.0),
            adaptive_threshold_mult=_env_float("ADAPTIVE_THRESHOLD_MULT", 2.5),
            noise_calibration_frames=_env_int("NOISE_CALIBRATION_FRAMES", 10),
            noise_reduce_enabled=_env_bool("NOISE_REDUCE_ENABLED", True),
        ),
        wake_word=WakeWordConfig(
            wake_word=_clean_env(os.getenv("WAKE_WORD")) or "picovoice",
            keyword_path=_clean_env(os.getenv("PORCUPINE_KEYWORD_PATH")),
            sensitivity=_env_float("WAKE_SENSITIVITY", 0.65),
        ),
        tts=TTSConfig(
            model=_clean_env(os.getenv("TTS_MODEL")) or "gpt-4o-mini-tts",
            voice=_clean_env(os.getenv("TTS_VOICE")) or "fable",
            streaming_enabled=_env_bool("TTS_STREAMING", True),
            volume_normalization=_env_bool("TTS_VOLUME_NORM", True),
            target_db=_env_float("TTS_TARGET_DB", -20.0),
        ),
        sfx=SFXConfig(
            enabled=_env_bool("SFX_ENABLED", True),
            style=(_clean_env(os.getenv("SFX_STYLE")) or "jarvis").lower(),
            volume=_env_float("SFX_VOLUME", 0.55),
            after_wake_wav=_resolve_path(_clean_env(os.getenv("SFX_AFTER_WAKE_WAV")), base_dir),
            after_question_wav=_resolve_path(_clean_env(os.getenv("SFX_AFTER_QUESTION_WAV")), base_dir),
            success_wav=_resolve_path(_clean_env(os.getenv("SFX_SUCCESS_WAV")), base_dir),
            failure_wav=_resolve_path(_clean_env(os.getenv("SFX_FAILURE_WAV")), base_dir),
            listening_wav=_resolve_path(_clean_env(os.getenv("SFX_LISTENING_WAV")), base_dir),
        ),
        home_assistant=HomeAssistantConfig(
            url=_clean_env(os.getenv("HOME_ASSISTANT_URL")).rstrip("/"),
            token=_clean_env(os.getenv("HOME_ASSISTANT_TOKEN")),
            cache_ttl_seconds=_env_float("HA_CACHE_TTL", 60.0),
            areas_cache_ttl_seconds=_env_float("HA_AREAS_CACHE_TTL", 300.0),
        ),
        conversation=ConversationConfig(
            chat_model=_clean_env(os.getenv("CHAT_MODEL")) or "gpt-4o",
            transcribe_model=_clean_env(os.getenv("TRANSCRIBE_MODEL")) or "gpt-4o-mini-transcribe",
            max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 30),
            max_tool_rounds=_env_int("MAX_TOOL_ROUNDS", 5),
            followup_enabled=_env_bool("FOLLOWUP_ENABLED", True),
            followup_window_seconds=_env_float("FOLLOWUP_WINDOW_SECONDS", 5.0),
            followup_silence_timeout=_env_float("FOLLOWUP_SILENCE_TIMEOUT", 2.0),
            bargein_enabled=_env_bool("BARGEIN_ENABLED", True),
            system_prompt=_clean_env(os.getenv("SYSTEM_PROMPT")) or default_system_prompt,
        ),
        health=HealthConfig(
            enabled=_env_bool("HEALTH_CHECK_ENABLED", True),
            check_interval_seconds=_env_float("HEALTH_CHECK_INTERVAL", 60.0),
            watchdog_enabled=_env_bool("WATCHDOG_ENABLED", True),
            watchdog_timeout_seconds=_env_float("WATCHDOG_TIMEOUT", 120.0),
        ),
        api_max_retries=_env_int("API_MAX_RETRIES", 3),
        api_retry_base_delay=_env_float("API_RETRY_BASE_DELAY", 1.0),
        debug_tools=_env_bool("DEBUG_TOOLS", True),
        debug_ha=_env_bool("DEBUG_HA", True),
        log_level=(_clean_env(os.getenv("LOG_LEVEL")) or "INFO").upper(),
    )
    
    logger.setLevel(getattr(logging, config.log_level, logging.INFO))
    return config


# ============================
# Global State
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYBACK_LOCK = threading.Lock()
_shutdown_event = threading.Event()
_porcupine_ref: Optional[pvporcupine.Porcupine] = None
_playback_interrupt = threading.Event()  # For barge-in support
_last_activity_time = time.time()  # For watchdog
_activity_lock = threading.Lock()

# Connection pool session
_http_session: Optional[requests.Session] = None
_async_session: Optional[aiohttp.ClientSession] = None

# OpenAI clients
sync_client: Optional[OpenAI] = None
async_client: Optional[AsyncOpenAI] = None


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
        # Connection pooling adapter
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,  # We handle retries ourselves
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
        """Start background health monitoring."""
        if not self.config.health.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def get_health(self, service: str) -> Optional[ServiceHealth]:
        """Get health status of a service."""
        with self._lock:
            return self._health.get(service)
    
    def is_service_available(self, service: str) -> bool:
        """Check if a service is available."""
        health = self.get_health(service)
        if health is None:
            return True  # Assume available if not checked
        return health.status != HealthStatus.UNHEALTHY
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running and not _shutdown_event.is_set():
            try:
                self._check_home_assistant()
                self._check_openai()
            except Exception as e:
                logger.warning(f"Health check error: {e}")
            
            # Sleep in small intervals to allow quick shutdown
            for _ in range(int(self.config.health.check_interval_seconds)):
                if not self._running or _shutdown_event.is_set():
                    break
                time.sleep(1)
    
    def _check_home_assistant(self):
        """Check Home Assistant connectivity."""
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
        """Check OpenAI API connectivity."""
        start = time.time()
        try:
            # Simple models list call to verify connectivity
            models = sync_client.models.list()
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
        """Start watchdog monitoring."""
        if not self.config.health.watchdog_enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Watchdog started")
    
    def stop(self):
        """Stop watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _watch_loop(self):
        """Monitor for activity timeouts."""
        while self._running and not _shutdown_event.is_set():
            time.sleep(10)
            
            last = get_last_activity()
            elapsed = time.time() - last
            
            if elapsed > self.config.health.watchdog_timeout_seconds:
                logger.error(
                    f"Watchdog timeout! No activity for {elapsed:.0f}s. "
                    "Consider restarting the assistant."
                )
                # Could trigger auto-restart here if desired
                # For now, just log and reset timer
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
        """Get cached state if still valid."""
        with self._lock:
            if entity_id in self._states:
                state, cached_at = self._states[entity_id]
                if time.time() - cached_at < self.config.home_assistant.cache_ttl_seconds:
                    return state
        return None
    
    def set_state(self, entity_id: str, state: Dict[str, Any]):
        """Cache entity state."""
        with self._lock:
            self._states[entity_id] = (state, time.time())
    
    def invalidate(self, entity_id: str):
        """Invalidate cache for an entity."""
        with self._lock:
            self._states.pop(entity_id, None)
    
    def invalidate_all(self):
        """Invalidate all cached states."""
        with self._lock:
            self._states.clear()
    
    def get_areas(self) -> Optional[List[Dict[str, Any]]]:
        """Get cached areas if still valid."""
        with self._lock:
            if self._areas and time.time() - self._areas_at < self.config.home_assistant.areas_cache_ttl_seconds:
                return self._areas
        return None
    
    def set_areas(self, areas: List[Dict[str, Any]]):
        """Cache areas list."""
        with self._lock:
            self._areas = areas
            self._areas_at = time.time()
    
    def get_entities(self) -> Optional[List[Dict[str, Any]]]:
        """Get cached entities if still valid."""
        with self._lock:
            if self._entities and time.time() - self._entities_at < self.config.home_assistant.cache_ttl_seconds:
                return self._entities
        return None
    
    def set_entities(self, entities: List[Dict[str, Any]]):
        """Cache entities list."""
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


class HomeAssistantClient:
    """Home Assistant API client with caching and async support."""
    
    def __init__(self, config: AssistantConfig, cache: EntityCache):
        self.config = config
        self.cache = cache
        self._ws: Optional[websocket.WebSocket] = None
    
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
        """List all areas via WebSocket."""
        cached = self.cache.get_areas()
        if cached is not None:
            return cached
        
        ws_url = self._ws_url()
        ws = websocket.create_connection(ws_url, timeout=10)
        try:
            ws.recv()  # auth_required
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
        """List all entity states."""
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
        """Get entity state with caching."""
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
        if not entity_id or "." not in entity_id:
            raise HomeAssistantError("Invalid entity_id.")
        if not _validate_entity_id(entity_id):
            raise HomeAssistantError(f"Malformed entity_id: {entity_id}")
        
        # Check cache first
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
    
    async def async_get_state(self, entity_id: str) -> Dict[str, Any]:
        """Async version of get_state."""
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
        if not entity_id or "." not in entity_id:
            raise HomeAssistantError("Invalid entity_id.")
        if not _validate_entity_id(entity_id):
            raise HomeAssistantError(f"Malformed entity_id: {entity_id}")
        
        cached = self.cache.get_state(entity_id)
        if cached is not None:
            return cached
        
        session = await get_async_session()
        url = f"{self.config.home_assistant.url}/api/states/{entity_id}"
        headers = self._headers()
        
        async with session.get(url, headers=headers) as r:
            if r.status == 404:
                raise HomeAssistantError(f"Entity not found: {entity_id}")
            if not r.ok:
                text = await r.text()
                raise HomeAssistantError(f"HA state error: {r.status} {text[:200]}")
            
            state = await r.json()
            self.cache.set_state(entity_id, state)
            return state
    
    def call_service(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any],
    ) -> Any:
        """Call a Home Assistant service."""
        if not self.config.home_assistant.url:
            raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
        
        domain = (domain or "").strip()
        service = (service or "").strip()
        
        if domain not in self.config.home_assistant.allowed_domains:
            raise HomeAssistantError(f"Domain not allowed: {domain}")
        if service not in self.config.home_assistant.allowed_services:
            raise HomeAssistantError(f"Service not allowed: {service}")
        
        # Validate entity_id if present
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
        
        # Invalidate cache for affected entities
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
    
    async def async_call_service(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any],
    ) -> Any:
        """Async version of call_service."""
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
        logger.debug(f"HA CALL (async) -> {domain}.{service} data={dict(data or {})}")
        
        session = await get_async_session()
        headers = self._headers()
        
        async with session.post(url, headers=headers, json=data or {}) as r:
            if not r.ok:
                text = await r.text()
                raise HomeAssistantError(f"HA service error: {r.status} {text[:200]}")
            
            if eid:
                if isinstance(eid, str):
                    self.cache.invalidate(eid)
                elif isinstance(eid, list):
                    for e in eid:
                        self.cache.invalidate(e)
            
            try:
                return await r.json()
            except Exception:
                return {"ok": True}


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


# ============================
# Fuzzy Matching
# ============================
def _normalize_text(s: str) -> str:
    s = (s or "").replace("_", " ").strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == "#").strip()


def fuzzy_match_entity(
    query: str,
    entities: List[Dict[str, Any]],
    domain_filter: Optional[str] = None,
    min_score: int = 70,
) -> Optional[str]:
    """
    Fuzzy match a user query to an entity.
    Returns the best matching entity_id or None.
    """
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
        
        # Build searchable text from entity attributes
        attrs = entity.get("attributes", {})
        friendly_name = attrs.get("friendly_name", "")
        
        # Create multiple match targets
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
    """
    Fuzzy match a user query to an area.
    Returns the best matching area_id or None.
    """
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
            # Also check partial ratio for substring matches
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
    
    # First try exact word boundary matching
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
    
    # Fall back to fuzzy matching
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
    
    if "brightness_pct" not in data:
        norm = _normalize_text(last_user_text)
        if any(kw in norm for kw in _BRIGHTNESS_KEYWORDS):
            m = re.search(r"\b(\d{1,3})\s*%?\b", norm)
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
    """
    Suggest a relevant scene based on context.
    Returns a suggestion string or None.
    """
    try:
        entities = ha_client.list_entities()
        scenes = [e for e in entities if e.get("entity_id", "").startswith("scene.")]
        
        if not scenes:
            return None
        
        hour = datetime.now().hour
        text_lower = user_text.lower()
        
        # Context-based suggestions
        suggestions = []
        
        # Time-based
        if hour >= 22 or hour < 6:
            # Night time
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["night", "sleep", "bedtime", "evening", "dim"]):
                    suggestions.append(scene)
        elif hour >= 6 and hour < 9:
            # Morning
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["morning", "wake", "sunrise", "bright"]):
                    suggestions.append(scene)
        elif hour >= 17 and hour < 22:
            # Evening
            for scene in scenes:
                name = scene.get("attributes", {}).get("friendly_name", "").lower()
                if any(kw in name for kw in ["evening", "dinner", "relax", "movie", "cozy"]):
                    suggestions.append(scene)
        
        # Activity-based from user text
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
            # Return the most relevant scene name
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
    """
    Check current state and return a response if action is redundant.
    Returns a response string or None if action should proceed.
    """
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
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(np.square(audio_f32)) + 1e-12)
    if current_rms < 1e-6:
        return audio_i16
    
    # Convert to dB
    current_db = 20 * np.log10(current_rms / 32768.0)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain with soft clipping
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
        """Check if audio frame contains speech."""
        if self.vad is None:
            return True  # Fall back to always true, let RMS handle it
        
        # WebRTC VAD requires 16kHz audio
        if sample_rate != 16000:
            return True
        
        # Frame must be 10, 20, or 30ms
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
    """
    Basic echo cancellation using playback tracking.
    
    This is a simple approach that tracks when audio is being played
    and suppresses mic input during that time. A proper AEC would
    use adaptive filtering (like WebRTC AEC), but that requires
    more complex setup with playback reference signals.
    """
    
    def __init__(self):
        self._playing = threading.Event()
        self._last_play_end: float = 0.0
        self._suppression_tail_seconds: float = 0.3
    
    def start_playback(self):
        """Mark that playback is starting."""
        self._playing.set()
    
    def end_playback(self):
        """Mark that playback has ended."""
        self._playing.clear()
        self._last_play_end = time.time()
    
    def should_suppress(self) -> bool:
        """Check if mic input should be suppressed due to echo."""
        if self._playing.is_set():
            return True
        
        # Also suppress briefly after playback ends
        if time.time() - self._last_play_end < self._suppression_tail_seconds:
            return True
        
        return False


# ============================
# Audio Playback
# ============================
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
    chunk_seconds: float = 0.5,
) -> bool:
    """
    Play WAV file with support for interruption.
    Returns True if completed, False if interrupted.
    """
    # For barge-in support, we need to play in chunks and check for interrupt
    # This is a simplified approach - ideally we'd use a proper audio library
    
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
        play_wav_file(config.sfx.success_wav, "success", config)


def play_failure_sound(config: AssistantConfig):
    """Play failure sound."""
    if config.sfx.failure_wav:
        play_wav_file(config.sfx.failure_wav, "failure", config)


# ============================
# Streaming TTS
# ============================
def speak_tts_streaming(text: str, config: AssistantConfig) -> bool:
    """
    Synthesize speech with streaming playback.
    Returns True if completed, False if interrupted.
    """
    try:
        # Stream the audio
        with sync_client.audio.speech.with_streaming_response.create(
            model=config.tts.model,
            voice=config.tts.voice,
            input=text,
            response_format="pcm",
        ) as response:
            # Create a temp file for PCM data
            pcm_fd, pcm_path = tempfile.mkstemp(suffix=".pcm", dir=BASE_DIR)
            wav_path = pcm_path.replace(".pcm", ".wav")
            
            try:
                # Stream to file
                with os.fdopen(pcm_fd, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if _playback_interrupt.is_set():
                            return False
                        f.write(chunk)
                
                # Convert PCM to WAV (24kHz, 16-bit, mono)
                pcm_data = np.fromfile(pcm_path, dtype=np.int16)
                
                # Apply volume normalization if enabled
                if config.tts.volume_normalization:
                    pcm_data = normalize_volume(pcm_data, config.tts.target_db)
                
                save_wav(wav_path, pcm_data, 24000, channels=1)
                
                # Play with interrupt checking
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
        # Fall back to non-streaming
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
            
            # Apply volume normalization
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
    """Speak text using TTS (streaming if enabled)."""
    _playback_interrupt.clear()
    
    if config.tts.streaming_enabled:
        return speak_tts_streaming(text, config)
    else:
        return speak_tts_standard(text, config)


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
# Recording with VAD
# ============================
def record_utterance_after_wake(
    stream,
    mic_frame_length: int,
    mic_rate: int,
    config: AssistantConfig,
    vad: WebRTCVAD,
) -> Optional[str]:
    """Record audio with VAD-based speech detection."""
    chunks: List[np.ndarray] = []
    max_frames = int((config.audio.max_utterance_seconds * mic_rate) / mic_frame_length)
    frames_per_second = mic_rate / mic_frame_length
    
    # Phase 1: calibrate ambient noise
    noise_rms_values: List[float] = []
    logger.info("Listening for your question...")
    
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
    
    # Phase 2: record with VAD + RMS fallback
    speech_detected = False
    speech_frame_count = 0
    silence_count = 0
    leading_silence_frames = int(config.audio.leading_silence_timeout * frames_per_second)
    min_speech_frames = int(config.audio.min_speech_seconds * frames_per_second)
    silence_frames_needed = int(config.audio.silence_seconds_to_stop * frames_per_second)
    
    energy_window: deque = deque(maxlen=5)
    vad_window: deque = deque(maxlen=5)
    
    remaining_frames = max_frames - config.audio.noise_calibration_frames
    leading_silence_count = 0
    
    for frame_idx in range(remaining_frames):
        if _shutdown_event.is_set():
            return None
        
        # Check echo cancellation
        if echo_cancellation.should_suppress():
            continue
        
        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunks.append(pcm_i16)
        
        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        frame_rms = rms(audio_f32)
        energy_window.append(frame_rms)
        smoothed_rms = float(np.mean(energy_window))
        
        # Use WebRTC VAD if available
        pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
        is_speech_vad = vad.is_speech(pcm_16k.tobytes(), 16000)
        vad_window.append(is_speech_vad)
        
        # Combine VAD and RMS for robust detection
        vad_votes = sum(vad_window)
        is_speech = (smoothed_rms >= adaptive_threshold) or (vad_votes >= 2)
        
        if not speech_detected:
            if is_speech:
                speech_detected = True
                speech_frame_count = 1
                silence_count = 0
                logger.debug(f"Speech detected at frame {frame_idx} (RMS: {smoothed_rms:.5f}, VAD: {vad_votes}/5)")
            else:
                leading_silence_count += 1
                if leading_silence_count >= leading_silence_frames:
                    logger.info("No speech detected within timeout, aborting.")
                    return None
        else:
            if is_speech:
                speech_frame_count += 1
                silence_count = 0
            else:
                silence_count += 1
                if speech_frame_count >= min_speech_frames and silence_count >= silence_frames_needed:
                    logger.debug(
                        f"End of speech: {speech_frame_count} speech frames, "
                        f"{silence_count} silence frames"
                    )
                    break
    
    if not speech_detected:
        logger.info("No speech detected in recording.")
        return None
    
    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    
    # Trim trailing silence
    trailing_silence_samples = int(silence_count * mic_frame_length)
    if trailing_silence_samples > 0 and trailing_silence_samples < len(audio_i16):
        audio_i16 = audio_i16[:-trailing_silence_samples]
    
    # Apply noise reduction
    audio_i16 = apply_noise_reduction(audio_i16, mic_rate, config)
    
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
) -> Tuple[Optional[str], bool]:
    """
    Listen for follow-up speech or wake word.
    
    Returns:
        Tuple of (wav_path or None, was_wake_word_detected)
    """
    frames_per_second = mic_rate / mic_frame_length
    timeout_frames = int(config.conversation.followup_window_seconds * frames_per_second)
    silence_timeout_frames = int(config.conversation.followup_silence_timeout * frames_per_second)
    
    chunks: List[np.ndarray] = []
    speech_detected = False
    speech_frame_count = 0
    silence_count = 0
    min_speech_frames = int(config.audio.min_speech_seconds * frames_per_second)
    
    energy_window: deque = deque(maxlen=5)
    
    logger.debug("Listening for follow-up...")
    
    for frame_idx in range(timeout_frames):
        if _shutdown_event.is_set():
            return None, False
        
        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Check for wake word (barge-in)
        pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
        if len(pcm_16k) >= porcupine.frame_length:
            if porcupine.process(pcm_16k[:porcupine.frame_length].tolist()) >= 0:
                logger.info("Wake word detected during follow-up window")
                return None, True
        
        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        frame_rms = rms(audio_f32)
        energy_window.append(frame_rms)
        smoothed_rms = float(np.mean(energy_window))
        
        is_speech = smoothed_rms >= config.audio.silence_rms_threshold * 2
        
        if is_speech:
            if not speech_detected:
                speech_detected = True
                logger.debug("Follow-up speech detected")
            speech_frame_count += 1
            silence_count = 0
            chunks.append(pcm_i16)
        else:
            if speech_detected:
                silence_count += 1
                chunks.append(pcm_i16)
                
                if speech_frame_count >= min_speech_frames and silence_count >= silence_timeout_frames:
                    break
    
    if not speech_detected or speech_frame_count < min_speech_frames:
        return None, False
    
    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    audio_i16 = apply_noise_reduction(audio_i16, mic_rate, config)
    
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=BASE_DIR)
    os.close(out_fd)
    save_wav(out_path, audio_i16, mic_rate, channels=1)
    return out_path, False


# ============================
# Transcription
# ============================
def transcribe_audio(wav_path: str, config: AssistantConfig) -> str:
    """Transcribe audio using Whisper."""
    def _do_transcribe():
        with open(wav_path, "rb") as f:
            tx = sync_client.audio.transcriptions.create(
                model=config.conversation.transcribe_model,
                file=f,
            )
        return getattr(tx, "text", "").strip()
    
    return retry_api_call(_do_transcribe, config)


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
            delay = config.api_retry_base_delay * (2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt + 1}/{config.api_max_retries}): {e}")
            if attempt < config.api_max_retries - 1:
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
    raise last_exc


# ============================
# Tool Definitions and Execution
# ============================
def _tools_schema():
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
    """
    Execute a single tool call.
    Returns (result_dict, success_bool).
    """
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
            
            # Check for "all" patterns
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
            
            # State-aware check
            state_response = check_state_and_respond(
                ha_client, domain, service,
                data.get("entity_id"), data.get("area_id")
            )
            if state_response:
                return {"info": state_response, "action_skipped": True}, True
            
            result = ha_client.call_service(domain=domain, service=service, data=data)
            return result, True
        
        elif fn == "ha_list_entities":
            entities = ha_client.list_entities()
            domain_filter = parsed.get("domain")
            if domain_filter:
                entities = [e for e in entities if e.get("entity_id", "").startswith(f"{domain_filter}.")]
            # Return simplified view
            simplified = [
                {
                    "entity_id": e.get("entity_id"),
                    "state": e.get("state"),
                    "friendly_name": e.get("attributes", {}).get("friendly_name"),
                }
                for e in entities[:50]  # Limit to prevent token overflow
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


# ============================
# Chat Completion with Tools
# ============================
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
    
    # Check if we're offline
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
# Conversation Management
# ============================
def trim_conversation(messages: List[Dict[str, Any]], config: AssistantConfig) -> List[Dict[str, Any]]:
    """Keep conversation within limits."""
    if len(messages) <= config.conversation.max_history_messages + 1:
        return messages
    
    system = [messages[0]] if messages and messages[0].get("role") == "system" else []
    history = messages[len(system):]
    
    trimmed = history[-(config.conversation.max_history_messages):]
    
    while trimmed and trimmed[0].get("role") == "tool":
        idx = history.index(trimmed[0])
        if idx > 0:
            trimmed.insert(0, history[idx - 1])
        else:
            break
    
    return system + trimmed


# ============================
# Porcupine Setup
# ============================
def create_porcupine(config: AssistantConfig) -> Tuple[pvporcupine.Porcupine, str]:
    """Create Porcupine wake word detector."""
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
# Startup Info
# ============================
def print_startup_info(config: AssistantConfig):
    """Print configuration info on startup."""
    logger.info("=" * 50)
    logger.info("Enhanced Voice Assistant Starting")
    logger.info("=" * 50)
    logger.info(f"Audio device: {config.audio.alsa_device}")
    logger.info(f"Mic device index: {config.audio.mic_device_index}")
    logger.info(f"TTS: {config.tts.model} / {config.tts.voice}")
    logger.info(f"Chat model: {config.conversation.chat_model}")
    logger.info(f"Wake sensitivity: {config.wake_word.sensitivity}")
    
    if WEBRTC_VAD_AVAILABLE:
        logger.info(f"WebRTC VAD: enabled (mode={config.audio.vad_mode})")
    else:
        logger.info("WebRTC VAD: not available (using RMS fallback)")
    
    if NOISEREDUCE_AVAILABLE and config.audio.noise_reduce_enabled:
        logger.info("Noise reduction: enabled")
    else:
        logger.info("Noise reduction: disabled")
    
    if FUZZY_AVAILABLE:
        logger.info("Fuzzy matching: enabled")
    else:
        logger.info("Fuzzy matching: not available")
    
    logger.info(f"Follow-up mode: {'enabled' if config.conversation.followup_enabled else 'disabled'}")
    logger.info(f"Barge-in: {'enabled' if config.conversation.bargein_enabled else 'disabled'}")
    logger.info(f"TTS streaming: {'enabled' if config.tts.streaming_enabled else 'disabled'}")
    logger.info(f"Volume normalization: {'enabled' if config.tts.volume_normalization else 'disabled'}")
    logger.info(f"Health monitoring: {'enabled' if config.health.enabled else 'disabled'}")
    logger.info(f"Watchdog: {'enabled' if config.health.watchdog_enabled else 'disabled'}")
    
    if config.home_assistant.url:
        logger.info(f"Home Assistant: {config.home_assistant.url}")
    
    logger.info("=" * 50)
    
    try:
        device_info = sd.query_devices(config.audio.mic_device_index)
        logger.info(f"Mic: {device_info['name']}")
    except Exception:
        pass


# ============================
# Main Loop
# ============================
def main():
    global _porcupine_ref, sync_client, async_client
    
    # Load configuration
    config = load_config()
    
    # Initialize OpenAI clients
    sync_client = OpenAI()
    async_client = AsyncOpenAI()
    
    # Print startup info
    print_startup_info(config)
    
    # Initialize components
    cache = EntityCache(config)
    ha_client = HomeAssistantClient(config, cache)
    health_monitor = HealthMonitor(config)
    watchdog = Watchdog(config)
    vad = WebRTCVAD(config)
    
    # Start background services
    health_monitor.start()
    watchdog.start()
    
    # Create Porcupine
    porcupine, wake_label = create_porcupine(config)
    _porcupine_ref = porcupine
    logger.info(f"Wake word: {wake_label}")
    
    # Initialize conversation
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": config.conversation.system_prompt}
    ]
    
    # Calculate frame sizes
    porc_frame_length = porcupine.frame_length
    mic_rate = config.audio.mic_rate
    mic_frame_length = int(porc_frame_length * mic_rate / config.audio.porcupine_rate)
    
    logger.info(f"Mic rate: {mic_rate} Hz | Frame: {mic_frame_length}")
    logger.info("Ready! Listening for wake word...")
    
    try:
        with sd.RawInputStream(
            device=config.audio.mic_device_index,
            samplerate=mic_rate,
            blocksize=mic_frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while not _shutdown_event.is_set():
                touch_activity()
                
                # Read audio frame
                pcm_bytes = stream.read(mic_frame_length)[0]
                pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_16k = decimate_to_16k(pcm_i16, mic_rate)
                
                if len(pcm_16k) < porc_frame_length:
                    continue
                
                # Check for wake word
                if porcupine.process(pcm_16k[:porc_frame_length].tolist()) >= 0:
                    request_id = new_request_id()
                    RequestContextFilter.set_request_id(request_id)
                    
                    logger.info("Wake word detected!")
                    touch_activity()
                    
                    # Play wake sound
                    play_wav_file(config.sfx.after_wake_wav, "wake", config)
                    
                    # Record utterance
                    utter_wav = record_utterance_after_wake(
                        stream, mic_frame_length, mic_rate, config, vad
                    )
                    
                    if utter_wav is None:
                        logger.info("No utterance captured, resuming wake word detection")
                        RequestContextFilter.clear_request_id()
                        continue
                    
                    # Play "processing" sound
                    play_wav_file(config.sfx.after_question_wav, "processing", config)
                    
                    try:
                        text = transcribe_audio(utter_wav, config)
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                        speak_tts("Sorry, I couldn't understand that.", config)
                        RequestContextFilter.clear_request_id()
                        continue
                    finally:
                        try:
                            os.unlink(utter_wav)
                        except OSError:
                            pass
                    
                    if not text:
                        RequestContextFilter.clear_request_id()
                        continue
                    
                    # Process conversation loop (including follow-ups)
                    while True:
                        touch_activity()
                        logger.info(f"You: {text}")
                        messages.append({"role": "user", "content": text})
                        
                        try:
                            reply, messages, tools_succeeded = ask_chat_with_tools(
                                messages, config, ha_client, health_monitor
                            )
                            messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            logger.error(f"Chat completion failed: {e}")
                            reply = "Sorry, something went wrong."
                            messages.append({"role": "assistant", "content": reply})
                            tools_succeeded = False
                        
                        messages = trim_conversation(messages, config)
                        logger.info(f"Assistant: {reply}")
                        
                        # Play success/failure sound
                        if tools_succeeded:
                            play_success_sound(config)
                        else:
                            play_failure_sound(config)
                        
                        # Speak response
                        if reply:
                            completed = speak_tts(reply, config)
                            if not completed and config.conversation.bargein_enabled:
                                logger.info("TTS interrupted by barge-in")
                        
                        # Check for scene suggestion
                        suggestion = suggest_scene(config, ha_client, text)
                        if suggestion:
                            logger.debug(f"Scene suggestion: {suggestion}")
                        
                        # Follow-up mode
                        if not config.conversation.followup_enabled:
                            break
                        
                        followup_wav, was_wake = record_followup(
                            stream, mic_frame_length, mic_rate, config, vad, porcupine
                        )
                        
                        if was_wake:
                            # Wake word detected, start fresh recording
                            play_wav_file(config.sfx.after_wake_wav, "wake", config)
                            utter_wav = record_utterance_after_wake(
                                stream, mic_frame_length, mic_rate, config, vad
                            )
                            if utter_wav:
                                play_wav_file(config.sfx.after_question_wav, "processing", config)
                                try:
                                    text = transcribe_audio(utter_wav, config)
                                    continue  # Process new utterance
                                finally:
                                    try:
                                        os.unlink(utter_wav)
                                    except OSError:
                                        pass
                            break
                        
                        if followup_wav:
                            play_wav_file(config.sfx.after_question_wav, "processing", config)
                            try:
                                text = transcribe_audio(followup_wav, config)
                                if text:
                                    logger.info("Follow-up detected")
                                    continue  # Process follow-up
                            except Exception as e:
                                logger.error(f"Follow-up transcription failed: {e}")
                            finally:
                                try:
                                    os.unlink(followup_wav)
                                except OSError:
                                    pass
                        
                        # No follow-up, exit loop
                        break
                    
                    RequestContextFilter.clear_request_id()
                    logger.info("Ready! Listening for wake word...")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Cleanup
        porcupine.delete()
        _porcupine_ref = None
        health_monitor.stop()
        watchdog.stop()
        
        # Close async session
        loop = asyncio.new_event_loop()
        loop.run_until_complete(close_async_session())
        loop.close()
        
        # Close sync session
        if _http_session:
            _http_session.close()
        
        logger.info("Assistant shut down.")


if __name__ == "__main__":
    main()
