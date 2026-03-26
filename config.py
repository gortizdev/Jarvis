"""Configuration for voice assistant."""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


# ============================
# Structured Logging
# ============================
logger = logging.getLogger("voice_assistant")


def _setup_logger(log_level: str = "INFO"):
    """Setup logger if not already configured."""
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))


# ============================
# Configuration Helpers
# ============================
def _clean_env(v: Optional[str]) -> str:
    """Clean environment variable value."""
    if v is None:
        return ""
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v


def _env_float(name: str, default: float) -> float:
    """Get float environment variable."""
    raw = _clean_env(os.getenv(name))
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    """Get int environment variable."""
    raw = _clean_env(os.getenv(name))
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Get bool environment variable."""
    raw = _clean_env(os.getenv(name)).lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve path relative to base directory."""
    if not path:
        return ""
    path = path.strip()
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


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
    vad_mode: int = 3
    vad_frame_ms: int = 30
    max_utterance_seconds: float = 15.0
    silence_rms_threshold: float = 0.015
    silence_seconds_to_stop: float = 0.35
    min_speech_seconds: float = 0.3
    leading_silence_timeout: float = 2.0
    adaptive_threshold_mult: float = 3.0
    noise_calibration_frames: int = 3
    noise_reduce_enabled: bool = False
    noise_reduce_prop_decrease: float = 0.5
    vad_window_size: int = 2
    energy_drop_ratio: float = 0.15


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
    target_db: float = -20.0
    direct_streaming: bool = True
    # PATCH #3: chunk_size kept; stream is now reused across calls (see assistant.py)
    chunk_size: int = 2048
    buffer_chunks: int = 2


@dataclass
class SFXConfig:
    """Sound effects configuration."""
    enabled: bool = True
    style: str = "jarvis"
    volume: float = 0.55
    startup_wav: str = ""
    after_wake_wav: str = ""
    after_question_wav: str = ""
    processing_wav: str = ""
    success_wav: str = ""
    failure_wav: str = ""
    # PATCH #10: listening_wav is now actually played in assistant.py
    listening_wav: str = ""


@dataclass
class HomeAssistantConfig:
    """Home Assistant connection configuration."""
    url: str = ""
    token: str = ""
    cache_ttl_seconds: float = 60.0
    areas_cache_ttl_seconds: float = 300.0
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
    bargein_enabled: bool = True
    followup_enabled: bool = True
    followup_window_seconds: float = 3.0
    followup_silence_timeout: float = 0.5
    system_prompt: str = ""


@dataclass
class HealthConfig:
    """Health monitoring configuration."""
    enabled: bool = True
    check_interval_seconds: float = 60.0
    ha_timeout_seconds: float = 5.0
    openai_timeout_seconds: float = 10.0
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
    api_max_retries: int = 3
    api_retry_base_delay: float = 1.0
    # PATCH #9: debug flags default to False; enable via .env for development
    debug_tools: bool = False
    debug_ha: bool = False
    log_level: str = "INFO"


# ============================
# Configuration Loader
# ============================
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
            silence_rms_threshold=_env_float("SILENCE_RMS_THRESHOLD", 0.015),
            silence_seconds_to_stop=_env_float("SILENCE_SECONDS_TO_STOP", 0.35),
            min_speech_seconds=_env_float("MIN_SPEECH_SECONDS", 0.3),
            leading_silence_timeout=_env_float("LEADING_SILENCE_TIMEOUT", 2.0),
            adaptive_threshold_mult=_env_float("ADAPTIVE_THRESHOLD_MULT", 3.0),
            noise_calibration_frames=_env_int("NOISE_CALIBRATION_FRAMES", 3),
            noise_reduce_enabled=_env_bool("NOISE_REDUCE_ENABLED", False),
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
            startup_wav=_resolve_path(_clean_env(os.getenv("SFX_STARTUP_WAV")), base_dir),
            after_wake_wav=_resolve_path(_clean_env(os.getenv("SFX_AFTER_WAKE_WAV")), base_dir),
            after_question_wav=_resolve_path(_clean_env(os.getenv("SFX_AFTER_QUESTION_WAV")), base_dir),
            processing_wav=_resolve_path(_clean_env(os.getenv("SFX_PROCESSING_WAV")), base_dir),
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
            bargein_enabled=_env_bool("BARGEIN_ENABLED", True),
            followup_enabled=_env_bool("FOLLOWUP_ENABLED", True),
            followup_window_seconds=_env_float("FOLLOWUP_WINDOW_SECONDS", 3.0),
            followup_silence_timeout=_env_float("FOLLOWUP_SILENCE_TIMEOUT", 0.5),
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
        # PATCH #9: default False; set DEBUG_TOOLS=true / DEBUG_HA=true in .env for dev
        debug_tools=_env_bool("DEBUG_TOOLS", False),
        debug_ha=_env_bool("DEBUG_HA", False),
        log_level=(_clean_env(os.getenv("LOG_LEVEL")) or "INFO").upper(),
    )

    logger.setLevel(getattr(logging, config.log_level, logging.INFO))
    return config