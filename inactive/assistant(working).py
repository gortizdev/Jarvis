import os
import wave
import time
import threading
import subprocess
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import pvporcupine
import requests
import websocket
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
    if not path:
        return ""
    path = path.strip()
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _normalize_text(s: str) -> str:
    s = (s or "").replace("_", " ").strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()


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

# Playback uses ALSA via aplay (USB speaker = plughw:X,0)
ALSA_DEVICE = _clean_env(os.getenv("ALSA_DEVICE")) or "default"

# Serialize playback (prevents overlapping aplay calls)
PLAYBACK_LOCK = threading.Lock()

# ============================
# Home Assistant REST/WS config
# ============================
HA_URL = _clean_env(os.getenv("HOME_ASSISTANT_URL")).rstrip("/")
HA_TOKEN = _clean_env(os.getenv("HOME_ASSISTANT_TOKEN"))

DEBUG_TOOLS = _env_bool("DEBUG_TOOLS", True)
DEBUG_HA = _env_bool("DEBUG_HA", True)

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
    "IMPORTANT TOOL RULES:\n"
    "- When calling ha_call_service you MUST include a target in data (entity_id or area_id or device_id).\n"
    "- If the user mentions a room/area, prefer area_id.\n"
    "- If the user says 'all lights' / 'whole house' / 'everywhere', use entity_id='all'.\n"
)

client = OpenAI()

# ============================
# SFX / WAV hooks
# ============================
SFX_ENABLED = _env_bool("SFX_ENABLED", True)
SFX_STYLE = (_clean_env(os.getenv("SFX_STYLE")) or "jarvis").lower()
SFX_VOLUME = _env_float("SFX_VOLUME", 0.55)
SFX_PROCESS_INTERVAL = _env_float("SFX_PROCESS_INTERVAL", 0.9)

SFX_AFTER_WAKE_WAV = _resolve_path(_clean_env(os.getenv("SFX_AFTER_WAKE_WAV")))
SFX_AFTER_QUESTION_WAV = _resolve_path(_clean_env(os.getenv("SFX_AFTER_QUESTION_WAV")))

# ============================
# Wake Word (Porcupine)
# ============================
WAKE_WORD = _clean_env(os.getenv("WAKE_WORD")) or "picovoice"
PORCUPINE_KEYWORD_PATH = _clean_env(os.getenv("PORCUPINE_KEYWORD_PATH"))
WAKE_SENSITIVITY = _env_float("WAKE_SENSITIVITY", 0.65)

PORCUPINE_RATE = 16000
MIC_RATE = 48000


# ============================
# Home Assistant helpers
# ============================
class HomeAssistantError(Exception):
    pass


def _ha_headers() -> Dict[str, str]:
    if not HA_TOKEN:
        raise HomeAssistantError("HOME_ASSISTANT_TOKEN is not set.")
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def _ha_ws_url() -> str:
    if not HA_URL:
        raise HomeAssistantError("HOME_ASSISTANT_URL is not set.")
    base = HA_URL.rstrip("/")
    if base.startswith("https://"):
        return base.replace("https://", "wss://") + "/api/websocket"
    if base.startswith("http://"):
        return base.replace("http://", "ws://") + "/api/websocket"
    return "ws://" + base + "/api/websocket"


# Cache area registry so we can infer room names without asking follow-ups.
_AREAS_CACHE: List[Dict[str, Any]] = []
_AREAS_CACHE_AT: float = 0.0
_AREAS_CACHE_TTL_S = 300.0  # 5 minutes


def ha_list_areas_ws() -> List[Dict[str, Any]]:
    ws_url = _ha_ws_url()
    ws = websocket.create_connection(ws_url, timeout=10)

    ws.recv()  # auth_required
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


def get_cached_areas() -> List[Dict[str, Any]]:
    global _AREAS_CACHE, _AREAS_CACHE_AT
    now = time.time()
    if _AREAS_CACHE and (now - _AREAS_CACHE_AT) < _AREAS_CACHE_TTL_S:
        return _AREAS_CACHE
    try:
        _AREAS_CACHE = ha_list_areas_ws()
        _AREAS_CACHE_AT = now
    except Exception:
        # keep old cache if available
        pass
    return _AREAS_CACHE


def infer_area_id_from_text(user_text: str) -> Optional[str]:
    """
    Look for a known HA area name inside the user's utterance.
    Returns area_id (UUID-like string) or None.
    """
    t = _normalize_text(user_text)
    if not t:
        return None
    areas = get_cached_areas()
    # Prefer longest names first (e.g., "living room" before "room")
    candidates = []
    for a in areas:
        name = a.get("name", "")
        area_id = a.get("area_id")
        if not name or not area_id:
            continue
        nn = _normalize_text(name)
        if nn and nn in t:
            candidates.append((len(nn), area_id))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


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

    url = f"{HA_URL}/api/services/{domain}/{service}"

    if DEBUG_HA:
        print(f"🧰 HA CALL -> {domain}.{service} data={dict(data or {})}")

    r = requests.post(url, headers=_ha_headers(), data=json.dumps(data or {}), timeout=8)
    if not r.ok:
        raise HomeAssistantError(f"HA service error: {r.status_code} {r.text[:200]}")
    try:
        return r.json()
    except Exception:
        return {"ok": True}


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


def aplay_wav(wav_path: str):
    # serialize playback
    with PLAYBACK_LOCK:
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
                    "IMPORTANT: data MUST include a target: entity_id OR area_id OR device_id."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "service": {"type": "string"},
                        "data": {"type": "object", "additionalProperties": True},
                        # allow accidental top-level targets; we'll merge them anyway
                        "entity_id": {"type": "string"},
                        "area_id": {"type": "string"},
                        "device_id": {"type": "string"},
                    },
                    "required": ["domain", "service"],
                },
            },
        },
    ]


def _infer_targets_if_missing(domain: str, service: str, data: Dict[str, Any], last_user_text: str) -> Dict[str, Any]:
    """
    If tool call omitted targets, infer from the user's last utterance.
    """
    data = dict(data or {})

    # If user asked for all-lights / whole house.
    t = _normalize_text(last_user_text)
    if domain == "light" and service in ("turn_on", "turn_off", "toggle"):
        if any(phrase in t for phrase in ["all lights", "whole house", "everywhere", "all the lights"]):
            data.setdefault("entity_id", "all")
            return data

    # If the user mentioned a known HA area name, use area_id.
    if "area_id" not in data and "entity_id" not in data and "device_id" not in data:
        inferred_area_id = infer_area_id_from_text(last_user_text)
        if inferred_area_id:
            data["area_id"] = inferred_area_id

    return data


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

    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m.get("content", "") or ""
            break

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
                # Merge accidental top-level targets into data
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

                # If STILL missing targets, infer from last user utterance
                if not any(key in data for key in ("entity_id", "area_id", "device_id")):
                    data = _infer_targets_if_missing(domain, service, data, last_user_text)

                # If STILL missing, fail fast with a useful error instead of 400 spam
                if not any(key in data for key in ("entity_id", "area_id", "device_id")):
                    raise HomeAssistantError(
                        "No target (entity_id/area_id/device_id). "
                        "Try saying a room name (e.g., 'living room') or 'all lights'."
                    )

                result = ha_call_service(domain=domain, service=service, data=data)

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
        keywords=[WAKE_WORD],
        sensitivities=[max(0.0, min(1.0, WAKE_SENSITIVITY))],
    )
    return porcupine, f"built-in: {[WAKE_WORD]}"


# ============================
# Debug
# ============================
def _print_debug(mic_device: int):
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
    if HA_TOKEN:
        print(f"HA TOKEN: (set) length={len(HA_TOKEN)}")

    try:
        print("Mic device:", mic_device, "|", sd.query_devices(mic_device)["name"])
    except Exception:
        pass


# ============================
# Main
# ============================
def main():
    mic_device = MIC_DEVICE_INDEX
    _print_debug(mic_device)

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
                    play_wav_file(SFX_AFTER_WAKE_WAV, "SFX_AFTER_WAKE_WAV")

                    utter_wav = record_utterance_after_wake(stream, mic_frame_length, MIC_RATE)

                    play_wav_file(SFX_AFTER_QUESTION_WAV, "SFX_AFTER_QUESTION_WAV")

                    reply = ""
                    text = transcribe_audio(utter_wav)
                    if not text:
                        continue

                    print(f"👤 You: {text}")
                    messages.append({"role": "user", "content": text})

                    reply = ask_chat_with_tools(messages)
                    messages.append({"role": "assistant", "content": reply})

                    print(f"🤖 Assistant: {reply}")

                    if reply:
                        speak_tts(reply)

    finally:
        porcupine.delete()


if __name__ == "__main__":
    main()
