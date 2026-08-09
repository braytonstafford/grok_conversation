"""xAI Grok Text-to-Speech entity for Home Assistant Assist."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .voice_api import XAIVoiceError, async_list_voices, async_tts
from .voice_const import (
    CONF_TTS_LANGUAGE,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_LANGUAGE,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
    TTS_LANGUAGES,
    XAI_VOICES,
    normalize_tts_language,
)

_LOGGER = logging.getLogger(__name__)
PARALLEL_UPDATES = 4


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up xAI TTS entity from a config entry."""
    if not config_entry.options.get("enable_tts", True):
        _LOGGER.debug("xAI TTS disabled in options")
        return

    session = async_get_clientsession(hass)
    api_key = config_entry.data.get("api_key", "")
    voices = dict(XAI_VOICES)
    try:
        voices = await async_list_voices(session, api_key)
    except Exception as err:  # noqa: BLE001
        _LOGGER.debug("Using built-in voice catalog: %s", err)

    # Cache on hass data for options flow
    hass.data.setdefault(DOMAIN, {}).setdefault(config_entry.entry_id, {})[
        "tts_voices"
    ] = voices

    async_add_entities([XAIGrokTTSEntity(config_entry, voices)])


class XAIGrokTTSEntity(TextToSpeechEntity):
    """xAI TTS engine (appears in Assist → Text-to-speech)."""

    # Standalone name — avoid "Grok xAI Grok" from device + entity name concat
    _attr_has_entity_name = False
    _attr_name = "xAI TTS"
    _attr_supported_options = [ATTR_VOICE]

    def __init__(
        self, config_entry: ConfigEntry, voices: dict[str, str]
    ) -> None:
        self._entry = config_entry
        self._voices_map = voices
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title or "xAI Grok",
            manufacturer="xAI",
            model="xAI Voice TTS",
            entry_type=DeviceEntryType.SERVICE,
        )
        # Assist language dropdown
        self._attr_supported_languages = list(TTS_LANGUAGES)
        default_lang = config_entry.options.get(
            CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE
        )
        if default_lang not in self._attr_supported_languages:
            default_lang = RECOMMENDED_TTS_LANGUAGE
        self._attr_default_language = default_lang

    @property
    def default_options(self) -> dict[str, Any]:
        """Default voice for Assist UI / Try voice."""
        voice = self._entry.options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        if voice not in self._voices_map:
            voice = next(iter(self._voices_map), RECOMMENDED_TTS_VOICE)
        return {ATTR_VOICE: voice}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Voices for the Assist voice dropdown (all xAI voices are multilingual)."""
        default = self._entry.options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        items = [
            Voice(vid, label) for vid, label in sorted(self._voices_map.items(), key=lambda x: x[1].lower())
        ]
        # Put configured default first
        items.sort(key=lambda v: (0 if v.voice_id == default else 1, v.name.lower()))
        return items

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Generate MP3 audio via xAI /v1/tts."""
        voice_id = options.get(
            ATTR_VOICE,
            self._entry.options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE),
        )
        speed = float(
            self._entry.options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)
        )
        # Prefer Assist-selected language; fall back to integration default
        lang = normalize_tts_language(
            language
            or self._entry.options.get(CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE)
        )
        api_key = self._entry.data.get("api_key", "")
        session = async_get_clientsession(self.hass)

        _LOGGER.debug(
            "xAI TTS: voice=%s lang=%s chars=%s", voice_id, lang, len(message or "")
        )
        try:
            audio = await async_tts(
                session,
                api_key,
                text=message,
                voice_id=str(voice_id),
                language=lang,
                speed=speed,
            )
        except XAIVoiceError as err:
            _LOGGER.error("xAI TTS error: %s", err)
            raise HomeAssistantError(str(err)) from err

        return "mp3", audio
