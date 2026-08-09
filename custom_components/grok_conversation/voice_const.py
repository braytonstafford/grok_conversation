"""xAI Voice constants (TTS / STT)."""

from __future__ import annotations

from homeassistant.components.tts import ATTR_VOICE

# Config option keys
CONF_TTS_VOICE = "tts_voice"
CONF_TTS_LANGUAGE = "tts_language"
CONF_TTS_SPEED = "tts_speed"
CONF_STT_LANGUAGE = "stt_language"
CONF_STT_FORMAT = "stt_format"
CONF_ENABLE_TTS = "enable_tts"
CONF_ENABLE_STT = "enable_stt"
CONF_VOICE_API_OK = "voice_api_ok"

RECOMMENDED_TTS_VOICE = "eve"
RECOMMENDED_TTS_LANGUAGE = "en"
RECOMMENDED_TTS_SPEED = 1.0
RECOMMENDED_STT_LANGUAGE = "en"
RECOMMENDED_ENABLE_TTS = True
RECOMMENDED_ENABLE_STT = True

XAI_API_BASE = "https://api.x.ai/v1"
XAI_TTS_URL = f"{XAI_API_BASE}/tts"
XAI_STT_URL = f"{XAI_API_BASE}/stt"
XAI_VOICES_URL = f"{XAI_API_BASE}/tts/voices"

# Built-in voices from xAI docs (voice_id → display label)
XAI_VOICES: dict[str, str] = {
    "eve": "Eve — energetic and upbeat",
    "ara": "Ara — warm and friendly",
    "rex": "Rex — confident and clear",
    "sal": "Sal — smooth and balanced",
    "leo": "Leo — authoritative and strong",
    "carina": "Carina — soft, empathetic",
    "luna": "Luna — gentle, patient",
    "iris": "Iris — friendly, upbeat",
    "helios": "Helios — upbeat, versatile assistant",
    "celeste": "Celeste — compassionate, reassuring",
    "ursa": "Ursa — friendly, warm",
    "rigel": "Rigel — precise, professional",
    "cosmo": "Cosmo — bright, curious",
    "lux": "Lux — grounded, calm",
    "atlas": "Atlas — confident, commanding",
    "castor": "Castor — charismatic, easygoing",
    "naksh": "Naksh — warm, thoughtful",
    "lumen": "Lumen — warm, articulate",
    "sirius": "Sirius — quick-witted, playful",
    "orion": "Orion — rich, cinematic",
    "altair": "Altair — elegant, premium",
    "perseus": "Perseus — strong, trustworthy",
    "zenith": "Zenith — sharp, focused",
    "helix": "Helix — bold, dynamic",
    "kepler": "Kepler — inventive, charismatic",
    "zagan": "Zagan — powerful, dramatic",
}

# TTS languages (BCP-47). HA Assist language dropdown uses these codes.
# Include both short and locale forms commonly used by Assist.
TTS_LANGUAGES: list[str] = [
    "en",
    "en-US",
    "en-GB",
    "ar-EG",
    "ar-SA",
    "ar-AE",
    "bn",
    "zh",
    "zh-CN",
    "fr",
    "fr-FR",
    "de",
    "de-DE",
    "hi",
    "id",
    "it",
    "it-IT",
    "ja",
    "ja-JP",
    "ko",
    "ko-KR",
    "pt-BR",
    "pt-PT",
    "ru",
    "ru-RU",
    "es",
    "es-ES",
    "es-MX",
    "tr",
    "vi",
    "auto",
]

# STT languages from xAI docs (25+)
STT_LANGUAGES: list[str] = [
    "en",
    "en-US",
    "en-GB",
    "ar",
    "cs",
    "da",
    "nl",
    "fil",
    "fr",
    "fr-FR",
    "de",
    "de-DE",
    "hi",
    "id",
    "it",
    "it-IT",
    "ja",
    "ja-JP",
    "ko",
    "ko-KR",
    "mk",
    "ms",
    "fa",
    "pl",
    "pt",
    "pt-BR",
    "ro",
    "ru",
    "ru-RU",
    "es",
    "es-ES",
    "es-MX",
    "sv",
    "th",
    "tr",
    "vi",
]

# Map HA Assist locale → xAI language param
def normalize_tts_language(language: str | None) -> str:
    """Normalize HA language code to xAI TTS language."""
    if not language:
        return RECOMMENDED_TTS_LANGUAGE
    lang = language.replace("_", "-").strip()
    if lang.lower() == "auto":
        return "auto"
    # Exact match
    for supported in TTS_LANGUAGES:
        if supported.lower() == lang.lower():
            return supported if supported != "auto" else "auto"
    # Primary subtag
    primary = lang.split("-")[0].lower()
    for supported in TTS_LANGUAGES:
        if supported.lower() == primary:
            return supported
    # English display name from some STT UIs
    if lang.lower() in ("english", "en_us", "en_gb"):
        return "en"
    return primary or RECOMMENDED_TTS_LANGUAGE


def normalize_stt_language(language: str | None) -> str | None:
    """Normalize HA language for STT formatting (primary subtag)."""
    if not language:
        return RECOMMENDED_STT_LANGUAGE
    lang = language.replace("_", "-").strip()
    lower = lang.lower()
    if lower in ("auto",):
        return None
    if lower in ("english",):
        return "en"
    primary = lang.split("-")[0].lower()
    # Valid STT primary codes from docs
    known = {
        "ar", "cs", "da", "nl", "en", "fil", "fr", "de", "hi", "id", "it",
        "ja", "ko", "mk", "ms", "fa", "pl", "pt", "ro", "ru", "es", "sv",
        "th", "tr", "vi",
    }
    if primary in known:
        return primary
    return primary or RECOMMENDED_STT_LANGUAGE


# Re-export for options
TTS_VOICE_OPTION = ATTR_VOICE  # "voice" — required by Assist UI
