"""Async helpers for xAI TTS and STT HTTP APIs."""

from __future__ import annotations

import logging
import struct
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData

from .voice_const import (
    XAI_STT_URL,
    XAI_TTS_URL,
    XAI_VOICES,
    XAI_VOICES_URL,
    normalize_stt_language,
    normalize_tts_language,
)

_LOGGER = logging.getLogger(__name__)

_TIMEOUT_SHORT = ClientTimeout(total=20)
_TIMEOUT_TTS = ClientTimeout(total=60)
_TIMEOUT_STT = ClientTimeout(total=120)


class XAIVoiceError(Exception):
    """Raised when an xAI voice API call fails."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM fmt chunk size
        1,  # audio format PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm


async def async_validate_voice_access(
    session: ClientSession, api_key: str
) -> tuple[bool, str]:
    """
    Check whether the API key can use xAI voice endpoints.

    Returns (ok, detail_message).
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    # Prefer lightweight voices list if available
    try:
        async with session.get(
            XAI_VOICES_URL, headers=headers, timeout=_TIMEOUT_SHORT
        ) as resp:
            if resp.status == 200:
                return True, "Voice API accessible (voices list OK)"
            if resp.status in (401, 403):
                text = await resp.text()
                return False, f"API key rejected for voice ({resp.status}): {text[:200]}"
            # 404 = endpoint path differs; fall through to TTS probe
            _LOGGER.debug("Voices list returned %s, probing TTS", resp.status)
    except ClientError as err:
        _LOGGER.debug("Voices list failed: %s", err)

    # Tiny TTS probe
    try:
        async with session.post(
            XAI_TTS_URL,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "text": "Hi",
                "voice_id": "eve",
                "language": "en",
                "output_format": {
                    "codec": "mp3",
                    "sample_rate": 24000,
                    "bit_rate": 64000,
                },
            },
            timeout=_TIMEOUT_TTS,
        ) as resp:
            if resp.status == 200:
                return True, "Voice API accessible (TTS probe OK)"
            text = await resp.text()
            if resp.status in (401, 403):
                return (
                    False,
                    "This API key cannot access xAI Voice (TTS/STT). "
                    "Enable Voice in the xAI console or create a key with voice permissions. "
                    f"({resp.status})",
                )
            return False, f"Voice probe failed ({resp.status}): {text[:240]}"
    except ClientError as err:
        return False, f"Could not reach xAI Voice API: {err}"


async def async_list_voices(
    session: ClientSession, api_key: str
) -> dict[str, str]:
    """Return voice_id → name mapping (falls back to built-in catalog)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(
            XAI_VOICES_URL, headers=headers, timeout=_TIMEOUT_SHORT
        ) as resp:
            if resp.status != 200:
                return dict(XAI_VOICES)
            data = await resp.json(content_type=None)
    except (ClientError, ValueError):
        return dict(XAI_VOICES)

    voices: dict[str, str] = {}
    items = data if isinstance(data, list) else data.get("voices") or data.get("data") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        vid = item.get("voice_id") or item.get("id") or item.get("name")
        name = item.get("name") or item.get("display_name") or vid
        if vid:
            voices[str(vid)] = str(name)
    return voices or dict(XAI_VOICES)


async def async_tts(
    session: ClientSession,
    api_key: str,
    *,
    text: str,
    voice_id: str = "eve",
    language: str = "en",
    speed: float = 1.0,
) -> bytes:
    """Synthesize speech; returns raw MP3 bytes."""
    if not text or not text.strip():
        raise XAIVoiceError("Empty TTS text")
    # xAI max 15_000 chars per request
    message = text.strip()
    if len(message) > 15000:
        message = message[:15000]

    lang = normalize_tts_language(language)
    payload: dict[str, Any] = {
        "text": message,
        "voice_id": voice_id or "eve",
        "language": lang,
        "output_format": {
            "codec": "mp3",
            "sample_rate": 24000,
            "bit_rate": 128000,
        },
    }
    if speed and abs(speed - 1.0) > 0.01:
        payload["speed"] = max(0.7, min(1.5, float(speed)))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            XAI_TTS_URL, headers=headers, json=payload, timeout=_TIMEOUT_TTS
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                detail = body.decode("utf-8", errors="replace")[:300]
                raise XAIVoiceError(
                    f"xAI TTS failed ({resp.status}): {detail}",
                    status=resp.status,
                )
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type:
                # Timestamped envelope
                import base64
                import json

                data = json.loads(body)
                audio_b64 = data.get("audio")
                if not audio_b64:
                    raise XAIVoiceError("TTS JSON response missing audio")
                return base64.b64decode(audio_b64)
            return body
    except ClientError as err:
        raise XAIVoiceError(f"xAI TTS network error: {err}") from err


async def async_stt(
    session: ClientSession,
    api_key: str,
    *,
    audio: bytes,
    language: str | None = "en",
    sample_rate: int = 16000,
    is_raw_pcm: bool = False,
    channels: int = 1,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
) -> str:
    """Transcribe audio bytes; returns transcript text."""
    if not audio:
        raise XAIVoiceError("Empty audio for STT")

    if is_raw_pcm:
        # Prefer WAV wrapper for reliability
        audio = _pcm16_to_wav(audio, sample_rate=sample_rate, channels=channels)
        filename = "audio.wav"
        content_type = "audio/wav"

    lang = normalize_stt_language(language)
    headers = {"Authorization": f"Bearer {api_key}"}

    form = FormData()
    # Fields before file
    if lang:
        form.add_field("language", lang)
        form.add_field("format", "true")
    form.add_field(
        "file",
        audio,
        filename=filename,
        content_type=content_type,
    )

    try:
        async with session.post(
            XAI_STT_URL, headers=headers, data=form, timeout=_TIMEOUT_STT
        ) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise XAIVoiceError(
                    f"xAI STT failed ({resp.status}): {detail}",
                    status=resp.status,
                )
            data = await resp.json(content_type=None)
    except ClientError as err:
        raise XAIVoiceError(f"xAI STT network error: {err}") from err

    text = data.get("text") if isinstance(data, dict) else None
    if text is None:
        raise XAIVoiceError(f"Unexpected STT response: {str(data)[:200]}")
    return str(text).strip()
