import os
import wave
import time
import threading
import subprocess
import json
import websocket
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import sounddevice as sd
import pvporcupine
import requests
from openai import OpenAI
from dotenv import load_dotenv


# ============================
# Helpers
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


def _resolve_path(path: str) -> str:
    """
    Allow either absolute paths or paths relative to the assistant.py directory.
    """
    if not path:
        return ""
    path = path.strip()
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


# ============================
# Home Assistant WS + Area helpers
# ============================
class HomeAssistantError(Exception):
    pass


def _ha_ws_url() -> str:
    if not HA_URL:
        raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
    base = HA_URL.rstrip("/")
    if base.startswith("https://"):
        return base.replace("https://", "wss://") + "/api/websocket"
    if base.startswith("http://"):
        return base.replace("http://", "ws://") + "/api/websocket"
    return "ws://" + base + "/api/websocket"


def ha_list_areas_ws() -> List[Dict[str, Any]]:
    ws_url = _ha_ws_url()
    ws = websocket.create_connection(ws_url, timeout=10)

    # auth_required
    ws.recv()
    ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    auth_msg = json.loads(ws.recv())
    if auth_msg.get("type") != "auth_ok":
        ws.close()
        raise HomeAssistantError(f"WebSocket auth failed: {auth_msg}")

    ws.send(json.dumps({"id": 1, "type": "config/area_registry/list"}))
    msg = json.loads(ws.recv())
    ws.close()

    if not msg.get("success"):
        raise HomeAssistantError(f"WS area list failed: {msg}")
    return msg.get("result", [])


def _normalize_area_name(s: str) -> str:
    s = (s or "").replace("_", " ").strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()


def resolve_area_id(area_hint: str) -> Optional[str]:
    hint = _normalize_area_name(area_hint)
    if not hint:
        return None
    try:
        areas = ha_list_areas_ws()
    except Exception:
        return None

    # exact match
    for a in areas:
        if _normalize_area_name(a.get("name", "")) == hint:
            return a.get("area_id")

    # contains match
    for a in areas:
        name = _normalize_area_name(a.get("name", ""))
        if hint in name or name in hint:
            return a.get("area_id")

    return None


# ============================
# Load environment variables
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ============================
# Audio / Devices
# ============================
MIC_DEVICE_INDEX = _env_int("MIC_DEVICE_INDEX", 2)

# PortAudio output device is informational only (playback uses aplay/ALSA_DEVICE)
OUTPUT_DEVICE_INDEX = _env_int("OUTPUT_DEVICE_INDEX", 1)
OUTPUT_DEVICE_MATCH = _clean_env(os.getenv("OUTPUT_DEVICE_MATCH")).lower()

# Playback uses ALSA via aplay. For HDMI where hw is "busy", use PulseAudio:
#   ALSA_DEVICE=default
ALSA_DEVICE = _clean_env(os.getenv("ALSA_DEVICE")) or "default"

# ============================
# Home Assistant REST config
# ============================
HA_URL = _clean_env(os.getenv("HOME_ASSISTANT_URL")).rstrip("/")
HA_TOKEN = _clean_env(os.getenv("HOME_ASSISTANT_TOKEN"))

# Debug logging: prints tool calls and HA errors to console/journalctl
DEBUG_TOOLS = _env_bool("DEBUG_TOOLS", True)
DEBUG_HA = _env_bool("DEBUG_HA", True)

# Expand allowlists so "other devices" don't get blocked unnecessarily
HA_ALLOWED_DOMAINS = {
    "light",
    "switch",
    "fan",
    "cover",
    "climate",
    "media_player",
    "scene",
    "script",
    "automation",
    "input_boolean",
    "button",
    "select",
    "number",
    "lock",
}

HA_ALLOWED_SERVICES = {
    "turn_on",
    "turn_off",
    "toggle",
    "open_cover",
    "close_cover",
    "stop_cover",
    "set_temperature",
    "set_hvac_mode",
    "media_play",
    "media_pause",
    "media_stop",
    "play_media",
    "volume_set",
    "select_source",
    "press",
    "select_option",
    "set_value",
    "set_percentage",
    "set_preset_mode",
    "set_fan_mode",
    "lock",
    "unlock",
}

# ============================
# Recording
# ============================
MAX_UTTERANCE_SECONDS = _env_float("MAX_UTTERANCE_SECONDS", 12.0)
SILENCE_RMS_THRESHOLD = _env_float("SILENCE_RMS_THRESHOLD", 0.012)
SILENCE_SECONDS_TO_STOP = _env_float("SILENCE_SECONDS_TO_STOP", 1.0)

# ============================
# OpenAI config
# ============================
CHAT_MODEL = _clean_env(os.getenv("CHAT_MODEL")) or "gpt-5.2"
TRANSCRIBE_MODEL = _clean_env(os.getenv("TRANSCRIBE_MODEL")) or "gpt-4o-mini-transcribe"
TTS_MODEL = _clean_env(os.getenv("TTS_MODEL")) or "gpt-4o-mini-tts"
TTS_VOICE = _clean_env(os.getenv("TTS_VOICE")) or "fable"

SYSTEM_PROMPT = _clean_env(os.getenv("SYSTEM_PROMPT")) or (
    "You are a highly capable, calm, and precise home voice assistant.\n"
    "Be a little witty and confident, but not verbose.\n"
    "Keep responses concise and conversational.\n"
    "If you take an action, confirm it succinctly.\n"
    "\n"
    "CRITICAL HOME ASSISTANT CONTROL RULES:\n"
    "1) Do NOT ask the user for an area name if they already said one.\n"
    "2) Prefer calling Home Assistant services using area targeting when the user mentions an area.\n"
    "   Use ha_call_service with data including 'area_id' derived from the spoken area.\n"
    "3) If the user says 'all lights', 'everywhere', or 'whole house', call light.turn_off/turn_on using\n"
    "   data {'entity_id': 'all'} without asking follow-ups.\n"
    "4) Never invent specific entity_ids.\n"
    "5) If Home Assistant returns an error like area not found, ask ONE follow-up.\n"
)

client = OpenAI()

# ============================
# SFX / Earcons + custom WAV hooks
# ============================
SFX_ENABLED = _env_bool("SFX_ENABLED", True)
SFX_STYLE = (_clean_env(os.getenv("SFX_STYLE")) or "jarvis").lower()  # jarvis | classic
SFX_VOLUME = _env_float("SFX_VOLUME", 0.55)
SFX_PROCESS_INTERVAL = _env_float("SFX_PROCESS_INTERVAL", 0.9)

SFX_WAKE_WAV = _resolve_path(_clean_env(os.getenv("SFX_WAKE_WAV")))
SFX_READY_WAV = _resolve_path(_clean_env(os.getenv("SFX_READY_WAV")))
SFX_PROCESS_WAV = _resolve_path(_clean_env(os.getenv("SFX_PROCESS_WAV")))

SFX_AFTER_WAKE_WAV = _resolve_path(_clean_env(os.getenv("SFX_AFTER_WAKE_WAV")))
SFX_AFTER_QUESTION_WAV = _resolve_path(_clean_env(os.getenv("SFX_AFTER_QUESTION_WAV")))

# ============================
# Wake Word (Porcupine)
# ============================
WAKE_KEYWORDS = [_clean_env(os.getenv("WAKE_WORD")) or "picovoice"]  # built-in fallback
PORCUPINE_KEYWORD_PATH = _clean_env(os.getenv("PORCUPINE_KEYWORD_PATH"))  # custom .ppn path
WAKE_SENSITIVITY = _env_float("WAKE_SENSITIVITY", 0.65)

PORCUPINE_RATE = 16000
MIC_RATE = 48000


# ============================
# Audio helpers
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
    if mic_rate == target_rate:
        return pcm_i16
    step = int(round(mic_rate / target_rate))
    return pcm_i16[::step]


def mono_to_stereo_interleaved(pcm_mono_i16: np.ndarray) -> np.ndarray:
    pcm_mono_i16 = np.asarray(pcm_mono_i16, dtype=np.int16)
    return np.repeat(pcm_mono_i16, 2)


def aplay_wav(wav_path: str):
    p = subprocess.run(
        ["aplay", "-D", ALSA_DEVICE, wav_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"aplay failed (rc={p.returncode}) on device '{ALSA_DEVICE}'.\n"
            f"File: {wav_path}\n"
            f"STDERR:\n{p.stderr.strip()}"
        )


def play_wav_file(path: str, label: str):
    if not path:
        return
    try:
        aplay_wav(path)
    except Exception as e:
        print(f"⚠️ {label} playback failed: {e}")


# ============================
# Device pickers (mic uses sounddevice; playback uses ALSA aplay)
# ============================
def pick_input_device(preferred_index: Optional[int] = None) -> int:
    devices = sd.query_devices()

    if preferred_index is not None:
        try:
            if devices[preferred_index].get("max_input_channels", 0) > 0:
                return preferred_index
        except Exception:
            pass

    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return i

    raise RuntimeError("No input-capable audio device found.")


def pick_output_device(preferred_index: Optional[int] = None, name_match: str = "") -> int:
    devices = sd.query_devices()

    if name_match:
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and name_match in (d.get("name", "").lower()):
                return i

    if preferred_index is not None:
        try:
            if devices[preferred_index].get("max_output_channels", 0) > 0:
                return preferred_index
        except Exception:
            pass

    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            return i

    raise RuntimeError("No output-capable audio device found.")


# ============================
# SFX generators
# ============================
def _gen_sine(freq: float, dur_s: float, sr: int, vol: float) -> np.ndarray:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    x = np.sin(2 * np.pi * freq * t) * vol
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _gen_click(dur_s: float, sr: int, vol: float) -> np.ndarray:
    n = int(sr * dur_s)
    x = (np.random.randn(n).astype(np.float32) * vol) * 0.35
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _gen_chirp(f0: float, f1: float, dur_s: float, sr: int, vol: float) -> np.ndarray:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    k = (f1 - f0) / max(dur_s, 1e-6)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    x = np.sin(phase) * vol
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    aa = np.pad(a, (0, n - len(a)), mode="constant")
    bb = np.pad(b, (0, n - len(b)), mode="constant")
    x = aa.astype(np.int32) + bb.astype(np.int32)
    return np.clip(x, -32768, 32767).astype(np.int16)


def _concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([p for p in parts if p is not None and len(p) > 0])


def _write_and_play_pcm_stereo(pcm_mono: np.ndarray, sr: int, filename: str):
    wav_path = os.path.join(BASE_DIR, filename)
    pcm_stereo = mono_to_stereo_interleaved(pcm_mono)
    save_wav(wav_path, pcm_stereo, sr, channels=2)
    aplay_wav(wav_path)


def sfx_wake():
    if not SFX_ENABLED:
        return
    if SFX_WAKE_WAV:
        aplay_wav(SFX_WAKE_WAV)
        return

    sr = 48000
    v = float(SFX_VOLUME)

    if SFX_STYLE == "classic":
        pcm = _gen_sine(880, 0.08, sr, v * 0.9)
        _write_and_play_pcm_stereo(pcm, sr, "sfx_wake.wav")
        return

    click = _gen_click(0.010, sr, v * 0.75)
    chirp = _gen_chirp(980, 1450, 0.090, sr, v * 0.75)
    pcm = _mix(click, chirp)
    _write_and_play_pcm_stereo(pcm, sr, "sfx_wake.wav")


def sfx_ready():
    if not SFX_ENABLED:
        return
    if SFX_READY_WAV:
        aplay_wav(SFX_READY_WAV)
        return

    sr = 48000
    v = float(SFX_VOLUME)

    if SFX_STYLE == "classic":
        pcm = _gen_sine(659, 0.08, sr, v * 0.85)
        _write_and_play_pcm_stereo(pcm, sr, "sfx_ready.wav")
        return

    click = _gen_click(0.008, sr, v * 0.7)
    chirp = _gen_chirp(1500, 1050, 0.090, sr, v * 0.75)
    ping = _gen_sine(1200, 0.045, sr, v * 0.45)
    pcm = _concat(_mix(click, chirp), np.zeros(int(sr * 0.012), dtype=np.int16), ping)
    _write_and_play_pcm_stereo(pcm, sr, "sfx_ready.wav")


def sfx_process_tick():
    if not SFX_ENABLED:
        return
    if SFX_PROCESS_WAV:
        aplay_wav(SFX_PROCESS_WAV)
        return

    sr = 48000
    v = float(SFX_VOLUME)

    if SFX_STYLE == "classic":
        pcm = _gen_sine(523, 0.08, sr, v * 0.85)
        _write_and_play_pcm_stereo(pcm, sr, "sfx_process.wav")
        return

    blip = _gen_chirp(1200, 1450, 0.050, sr, v * 0.35)
    click = _gen_click(0.006, sr, v * 0.35)
    pcm = _mix(blip, click)
    _write_and_play_pcm_stereo(pcm, sr, "sfx_process.wav")


def start_processing_loop(stop_event: threading.Event, interval: float = 0.9):
    def loop():
        while not stop_event.is_set():
            try:
                sfx_process_tick()
            except Exception as e:
                print(f"⚠️ SFX process tick failed: {e}")
            for _ in range(max(1, int(interval / 0.1))):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

    threading.Thread(target=loop, daemon=True).start()


# ============================
# Home Assistant REST helpers
# ============================
def _ha_headers() -> Dict[str, str]:
    if not HA_TOKEN:
        raise HomeAssistantError("HOME_ASSISTANT_TOKEN is not set.")
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def ha_get_state(entity_id: str) -> Dict[str, Any]:
    if not HA_URL:
        raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
    if not entity_id or "." not in entity_id:
        raise HomeAssistantError("Invalid entity_id.")
    url = f"{HA_URL}/api/states/{entity_id}"
    r = requests.get(url, headers=_ha_headers(), timeout=6)
    if r.status_code == 404:
        raise HomeAssistantError(f"Entity not found: {entity_id}")
    if not r.ok:
        raise HomeAssistantError(f"HA state error: {r.status_code} {r.text[:200]}")
    return r.json()


def ha_call_service(domain: str, service: str, data: Dict[str, Any]) -> Any:
    if not HA_URL:
        raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
    domain = (domain or "").strip()
    service = (service or "").strip()

    if domain not in HA_ALLOWED_DOMAINS:
        raise HomeAssistantError(f"Domain not allowed: {domain}")
    if service not in HA_ALLOWED_SERVICES:
        raise HomeAssistantError(f"Service not allowed: {service}")

    # Translate spoken area name/slug into real HA area_id (via WebSocket)
    if isinstance(data, dict) and "area_id" in data and isinstance(data["area_id"], str):
        resolved = resolve_area_id(data["area_id"])
        if resolved:
            data["area_id"] = resolved

    url = f"{HA_URL}/api/services/{domain}/{service}"

    if DEBUG_HA:
        safe_data = dict(data or {})
        print(f"🧰 HA CALL -> {domain}.{service} data={safe_data}")

    r = requests.post(url, headers=_ha_headers(), data=json.dumps(data or {}), timeout=8)
    if not r.ok:
        raise HomeAssistantError(f"HA service error: {r.status_code} {r.text[:200]}")
    try:
        return r.json()
    except Exception:
        return {"ok": True}


# ============================
# OpenAI: Transcribe + Tool loop
# ============================
def transcribe_audio(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        tx = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f)
    return getattr(tx, "text", "").strip()


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
                    "IMPORTANT: Put targets in the 'data' object (e.g., data.area_id or data.entity_id)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "service": {"type": "string"},
                        "data": {
                            "type": "object",
                            "description": (
                                "Service data/target. Use keys like: entity_id, area_id, device_id, "
                                "and any service-specific fields."
                            ),
                            "additionalProperties": True,
                        },
                    },
                    "required": ["domain", "service", "data"],
                },
            },
        },
    ]


def ask_chat_with_tools(messages: List[Dict[str, Any]]) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=_tools_schema(),
        tool_choice="auto",
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return (msg.content or "").strip()

    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
        }
    )

    for tc in tool_calls:
        fn = tc.function.name
        try:
            parsed = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
        except Exception:
            parsed = {}

        if DEBUG_TOOLS:
            print(f"🔧 TOOL CALL -> {fn} args={parsed}")

        try:
            if fn == "ha_get_state":
                result = ha_get_state(entity_id=parsed.get("entity_id", ""))
            elif fn == "ha_call_service":
                # ✅ ROBUSTNESS FIX:
                # If the model mistakenly sends target fields (area_id/entity_id/etc.)
                # at the top-level instead of inside "data", merge them into data.
                data = parsed.get("data", {})
                if not isinstance(data, dict):
                    data = {}

                # Merge any extra keys into data (except domain/service/data)
                for k, v in (parsed or {}).items():
                    if k in ("domain", "service", "data"):
                        continue
                    # don't overwrite if already provided in data
                    if k not in data:
                        data[k] = v

                result = ha_call_service(
                    domain=parsed.get("domain", ""),
                    service=parsed.get("service", ""),
                    data=data,
                )
            else:
                result = {"error": f"Unknown tool: {fn}"}
        except Exception as e:
            result = {"error": str(e)}
            if DEBUG_TOOLS:
                print(f"❌ TOOL ERROR -> {fn}: {e}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": json.dumps(result)[:8000],
            }
        )

    resp2 = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return (resp2.choices[0].message.content or "").strip()


def speak_tts(text: str):
    audio = client.audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=text)
    audio_bytes = audio.read() if hasattr(audio, "read") else audio.content

    mp3_path = os.path.join(BASE_DIR, "tts.mp3")
    wav_path = os.path.join(BASE_DIR, "tts.wav")

    with open(mp3_path, "wb") as f:
        f.write(audio_bytes)

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-y", "-i", mp3_path, "-ac", "2", "-ar", "48000", wav_path],
        check=False,
    )

    aplay_wav(wav_path)


# ============================
# Recording
# ============================
def record_utterance_after_wake(stream, mic_frame_length: int, mic_rate: int) -> str:
    chunks: List[np.ndarray] = []
    silence_frames_needed = int((SILENCE_SECONDS_TO_STOP * mic_rate) / mic_frame_length)
    silence_count = 0
    max_frames = int((MAX_UTTERANCE_SECONDS * mic_rate) / mic_frame_length)

    print("🎙️ Listening for your question...")

    for _ in range(max_frames):
        pcm_bytes = stream.read(mic_frame_length)[0]
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunks.append(pcm_i16)

        audio_f32 = pcm_i16.astype(np.float32) / 32768.0
        if rms(audio_f32) < SILENCE_RMS_THRESHOLD:
            silence_count += 1
            if silence_count >= silence_frames_needed and len(chunks) > 5:
                break
        else:
            silence_count = 0

    audio_i16 = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    out_path = os.path.join(BASE_DIR, "utterance.wav")
    save_wav(out_path, audio_i16, mic_rate, channels=1)
    return out_path


# ============================
# Porcupine init
# ============================
def create_porcupine() -> Tuple[pvporcupine.Porcupine, str]:
    access_key = os.environ["PICOVOICE_ACCESS_KEY"]

    if PORCUPINE_KEYWORD_PATH:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[PORCUPINE_KEYWORD_PATH],
            sensitivities=[max(0.0, min(1.0, WAKE_SENSITIVITY))],
        )
        return porcupine, f"custom ppn: {PORCUPINE_KEYWORD_PATH}"

    porcupine = pvporcupine.create(
        access_key=access_key,
        keywords=WAKE_KEYWORDS,
        sensitivities=[max(0.0, min(1.0, WAKE_SENSITIVITY))],
    )
    return porcupine, f"built-in: {WAKE_KEYWORDS}"


# ============================
# Debug
# ============================
def _print_debug(out_info_device: int):
    print(f"Loaded .env from: {ENV_PATH}")
    print(f"ALSA_DEVICE (playback): {ALSA_DEVICE}")
    print(f"TTS voice: {TTS_VOICE} | TTS model: {TTS_MODEL}")
    print(f"SFX: enabled={SFX_ENABLED} style={SFX_STYLE} volume={SFX_VOLUME}")
    if SFX_AFTER_WAKE_WAV:
        print(f"SFX_AFTER_WAKE_WAV: {SFX_AFTER_WAKE_WAV}")
    if SFX_AFTER_QUESTION_WAV:
        print(f"SFX_AFTER_QUESTION_WAV: {SFX_AFTER_QUESTION_WAV}")
    print(f"Wake sensitivity: {WAKE_SENSITIVITY}")

    print(f"DEBUG_TOOLS={DEBUG_TOOLS} DEBUG_HA={DEBUG_HA}")

    if HA_URL:
        print(f"HA URL: {HA_URL}")
    else:
        print("⚠️ HOME_ASSISTANT_URL not set.")
    if HA_TOKEN:
        print(f"HA TOKEN: (set) length={len(HA_TOKEN)}")
    else:
        print("⚠️ HOME_ASSISTANT_TOKEN not set.")

    try:
        print("PortAudio output device (informational):", out_info_device, "|", sd.query_devices(out_info_device)["name"])
    except Exception:
        pass


# ============================
# Main
# ============================
def main():
    mic_device = pick_input_device(preferred_index=MIC_DEVICE_INDEX)
    out_info_device = pick_output_device(preferred_index=OUTPUT_DEVICE_INDEX, name_match=OUTPUT_DEVICE_MATCH)

    print("Using mic device:", mic_device, "|", sd.query_devices(mic_device)["name"])
    _print_debug(out_info_device)

    # Basic audio test (earcons only)
    try:
        if SFX_ENABLED:
            print("🔊 Audio test (wake + ready)...")
            sfx_wake()
            sfx_ready()
            print("✅ Audio test done.")
    except Exception as e:
        print(f"❌ Audio test failed: {e}")

    porcupine, wake_label = create_porcupine()
    print(f"✅ Assistant running. Wake word mode: {wake_label}")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    porc_frame_length = porcupine.frame_length
    mic_frame_length = int(porc_frame_length * MIC_RATE / PORCUPINE_RATE)

    print(f"   Mic rate: {MIC_RATE} Hz | Porcupine rate: {PORCUPINE_RATE} Hz")
    print(f"   Mic frame length: {mic_frame_length} | Porcupine frame length: {porc_frame_length}")

    try:
        with sd.RawInputStream(
            device=mic_device,
            samplerate=MIC_RATE,
            blocksize=mic_frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                pcm_bytes = stream.read(mic_frame_length)[0]
                pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_16k = decimate_to_16k(pcm_i16, MIC_RATE)

                if len(pcm_16k) < porc_frame_length:
                    continue

                if porcupine.process(pcm_16k[:porc_frame_length].tolist()) >= 0:
                    # Wake detected
                    try:
                        sfx_wake()
                    except Exception as e:
                        print(f"⚠️ Wake SFX failed: {e}")

                    play_wav_file(SFX_AFTER_WAKE_WAV, "SFX_AFTER_WAKE_WAV")

                    utter_wav = record_utterance_after_wake(stream, mic_frame_length, MIC_RATE)

                    play_wav_file(SFX_AFTER_QUESTION_WAV, "SFX_AFTER_QUESTION_WAV")

                    stop_processing = threading.Event()
                    start_processing_loop(stop_processing, interval=SFX_PROCESS_INTERVAL)

                    reply = ""
                    try:
                        text = transcribe_audio(utter_wav)
                        if not text:
                            continue

                        print(f"👤 You: {text}")
                        messages.append({"role": "user", "content": text})

                        reply = ask_chat_with_tools(messages)
                        messages.append({"role": "assistant", "content": reply})

                        print(f"🤖 Assistant: {reply}")

                    finally:
                        stop_processing.set()

                    try:
                        sfx_ready()
                    except Exception as e:
                        print(f"⚠️ Ready SFX failed: {e}")

                    if reply:
                        try:
                            speak_tts(reply)
                        except Exception as e:
                            print(f"⚠️ TTS playback failed: {e}")

    finally:
        porcupine.delete()


if __name__ == "__main__":
    main()

