"""The Grok Conversation integration."""

from __future__ import annotations

import base64
from mimetypes import guess_file_type
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.images_response import ImagesResponse
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .api_helpers import (
    async_chat_completion,
    async_responses_completion,
    extract_usage,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SHOW_CITATIONS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    IMAGE_QUALITIES,
    IMAGE_SIZES,
    IMAGE_STYLES,
    LIVE_SEARCH_OFF,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
    SERVICE_ASK,
    SERVICE_CLEAR_MEMORY,
    SERVICE_GENERATE_CONTENT,
    SERVICE_GENERATE_IMAGE,
    SERVICE_HOME_BRIEFING,
    SERVICE_PHOTO_ANALYSIS,
    SERVICE_QUERY_IMAGE,
    SERVICE_RESET_STATS,
)
from .usage import UsageTracker
from .voice_api import async_validate_voice_access

PLATFORMS = (
    Platform.CONVERSATION,
    Platform.SENSOR,
    Platform.TTS,
    Platform.STT,
)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

OpenAIConfigEntry = ConfigEntry  # runtime_data: openai.AsyncClient



def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    try:
        mime_type, _ = guess_file_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(file_path, "rb") as image_file:
            return (mime_type, base64.b64encode(image_file.read()).decode("utf-8"))
    except (OSError, IOError) as err:
        raise HomeAssistantError(f"Error reading file {file_path}: {err}") from err


def _validate_config_entry(hass: HomeAssistant, entry_id: str) -> OpenAIConfigEntry:
    """Validate and return config entry."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="invalid_config_entry",
            translation_placeholders={"config_entry": entry_id},
        )
    return entry  # type: ignore[return-value]


def _entry_client(entry: OpenAIConfigEntry) -> openai.AsyncClient:
    return entry.runtime_data


def _usage_tracker(hass: HomeAssistant, entry_id: str) -> UsageTracker | None:
    data = hass.data.get(DOMAIN, {}).get(entry_id)
    if not data:
        return None
    return data.get("usage")


async def _record_usage(
    hass: HomeAssistant,
    entry_id: str,
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    service: str,
) -> None:
    tracker = _usage_tracker(hass, entry_id)
    if tracker:
        await tracker.async_record(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            service=service,
        )


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Grok Conversation services (once)."""
    hass.data.setdefault(DOMAIN, {})

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with grok."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client = _entry_client(entry)
        model = call.data.get("model") or entry.options.get(
            "image_model", RECOMMENDED_IMAGE_GENERATION_MODEL
        )

        try:
            response: ImagesResponse = await client.images.generate(
                model=model,
                prompt=call.data[CONF_PROMPT],
                size=call.data.get("size", "1024x1024"),
                quality=call.data.get("quality", "standard"),
                style=call.data.get("style", "vivid"),
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        image_data = response.data[0]
        result: dict[str, Any] = {"url": image_data.url, "model": model}
        if getattr(image_data, "revised_prompt", None):
            result["revised_prompt"] = image_data.revised_prompt
        await _record_usage(
            hass,
            entry.entry_id,
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            service="generate_image",
        )
        return result

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to Grok and return the response (supports images)."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client = _entry_client(entry)

        content: list[dict[str, Any]] = [
            {"type": "text", "text": call.data[CONF_PROMPT]}
        ]

        has_images = False

        def append_files_to_content() -> None:
            nonlocal has_images
            for filename in call.data.get(CONF_FILENAMES, []):
                if not hass.config.is_allowed_path(filename):
                    raise HomeAssistantError(
                        f"Cannot read `{filename}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`"
                    )
                if not Path(filename).exists():
                    raise HomeAssistantError(f"`{filename}` does not exist")
                mime_type, base64_file = encode_file(filename)
                if "image/" not in mime_type:
                    raise HomeAssistantError(
                        "Only images are supported by the xAI API, "
                        f"`{filename}` is not an image file"
                    )
                has_images = True
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_file}",
                            "detail": "auto",
                        },
                    }
                )

        if call.data.get(CONF_FILENAMES):
            await hass.async_add_executor_job(append_files_to_content)

        if has_images:
            model = call.data.get("model") or entry.options.get(
                "vision_model", RECOMMENDED_VISION_MODEL
            )
        else:
            model = call.data.get("model") or entry.options.get(
                CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
            )

        live_search = call.data.get(
            CONF_LIVE_SEARCH, entry.options.get(CONF_LIVE_SEARCH, LIVE_SEARCH_OFF)
        )
        show_citations = call.data.get(
            CONF_SHOW_CITATIONS,
            entry.options.get(CONF_SHOW_CITATIONS, True),
        )
        max_tokens = call.data.get(
            CONF_MAX_TOKENS, entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        )
        temperature = call.data.get(
            CONF_TEMPERATURE,
            entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
        )
        top_p = call.data.get(
            CONF_TOP_P, entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
        )
        reasoning_effort = call.data.get(
            CONF_REASONING_EFFORT, entry.options.get(CONF_REASONING_EFFORT)
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": content}  # type: ignore[typeddict-item]
        ]

        try:
            if live_search and live_search != LIVE_SEARCH_OFF and not has_images:
                text, p_tok, c_tok = await async_responses_completion(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": call.data[CONF_PROMPT]}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    live_search=live_search,
                    show_citations=bool(show_citations),
                    reasoning_effort=reasoning_effort,
                )
            else:
                response = await async_chat_completion(
                    client,
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    reasoning_effort=reasoning_effort,
                    user=call.context.user_id,
                )
                text = response.choices[0].message.content or ""
                p_tok, c_tok = extract_usage(response)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        await _record_usage(
            hass,
            entry.entry_id,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            service="generate_content",
        )
        return {"text": text, "model": model, "prompt_tokens": p_tok, "completion_tokens": c_tok}

    async def ask_service(call: ServiceCall) -> ServiceResponse:
        """Stateless one-shot ask with optional live search overrides."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client = _entry_client(entry)

        instructions = (call.data.get("instructions") or "").strip()
        input_data = (call.data.get("input_data") or "").strip()
        if not instructions or not input_data:
            raise ServiceValidationError("instructions and input_data are required")

        model = call.data.get("model") or entry.options.get(
            CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
        )
        max_tokens = call.data.get(
            CONF_MAX_TOKENS, entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        )
        temperature = call.data.get(CONF_TEMPERATURE)
        top_p = call.data.get(CONF_TOP_P)
        reasoning_effort = call.data.get(CONF_REASONING_EFFORT)
        live_search = call.data.get(CONF_LIVE_SEARCH, LIVE_SEARCH_OFF)
        show_citations = call.data.get(CONF_SHOW_CITATIONS, True)
        location = call.data.get(CONF_LOCATION_CONTEXT) or entry.options.get(
            CONF_LOCATION_CONTEXT, ""
        )
        if location:
            instructions = f"{instructions}\n\nUser home location context: {location}"

        try:
            if live_search and live_search != LIVE_SEARCH_OFF:
                text, p_tok, c_tok = await async_responses_completion(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": input_data}],
                    system_prompt=instructions,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    live_search=live_search,
                    show_citations=bool(show_citations),
                    reasoning_effort=reasoning_effort,
                )
            else:
                response = await async_chat_completion(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": input_data},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    reasoning_effort=reasoning_effort,
                )
                text = response.choices[0].message.content or ""
                p_tok, c_tok = extract_usage(response)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error in ask service: {err}") from err

        await _record_usage(
            hass,
            entry.entry_id,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            service="ask",
        )
        return {
            "status": "ok",
            "response_text": text,
            "model": model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        }

    async def photo_analysis(call: ServiceCall) -> ServiceResponse:
        """Analyze one or more images with Grok vision."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client = _entry_client(entry)
        model = call.data.get("model") or entry.options.get(
            "vision_model", RECOMMENDED_VISION_MODEL
        )
        prompt = call.data["prompt"]
        images = call.data.get("images") or []
        if isinstance(images, str):
            images = [line.strip() for line in images.splitlines() if line.strip()]

        if not images:
            raise ServiceValidationError("At least one image path or URL is required")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        def build_images() -> None:
            for image in images:
                url = image
                if isinstance(image, dict):
                    url = image.get("url") or image.get("path") or ""
                if not url:
                    continue
                parsed = urlparse(str(url))
                if parsed.scheme in ("http", "https"):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": str(url), "detail": "auto"},
                        }
                    )
                    continue
                path = str(url)
                if not hass.config.is_allowed_path(path):
                    raise HomeAssistantError(
                        f"Cannot read `{path}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted"
                    )
                if not Path(path).exists():
                    raise HomeAssistantError(f"`{path}` does not exist")
                mime_type, b64 = encode_file(path)
                if "image/" not in mime_type:
                    raise HomeAssistantError(f"`{path}` is not an image")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}",
                            "detail": "auto",
                        },
                    }
                )

        await hass.async_add_executor_job(build_images)
        if len(content) < 2:
            raise ServiceValidationError("No valid images provided")

        try:
            response = await async_chat_completion(
                client,
                model=model,
                messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                max_tokens=call.data.get("max_tokens", 800),
                temperature=call.data.get(CONF_TEMPERATURE),
                top_p=call.data.get(CONF_TOP_P),
            )
            text = response.choices[0].message.content or ""
            p_tok, c_tok = extract_usage(response)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error analyzing photo: {err}") from err

        await _record_usage(
            hass,
            entry.entry_id,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            service="photo_analysis",
        )
        return {
            "status": "ok",
            "response_text": text,
            "model": model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        }

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Legacy query_image service (compat with photo_analysis)."""
        images_in = call.data.get("images", [])
        normalized: list[str] = []
        for img in images_in:
            if isinstance(img, dict):
                url = img.get("url") or img.get("path") or ""
                if url:
                    normalized.append(str(url))
            elif img:
                normalized.append(str(img))
        # Build a minimal namespace-like object for photo_analysis fields
        class _Data(dict):
            def get(self, key, default=None):  # noqa: ANN001
                return super().get(key, default)

            def __getitem__(self, key):  # noqa: ANN001
                return super().__getitem__(key)

        class _Call:
            def __init__(self, data):
                self.data = data
                self.context = call.context

        return await photo_analysis(
            _Call(  # type: ignore[arg-type]
                _Data(
                    {
                        "config_entry": call.data["config_entry"],
                        "prompt": call.data["prompt"],
                        "model": call.data.get("model"),
                        "max_tokens": call.data.get("max_tokens", 300),
                        "images": normalized,
                    }
                )
            )
        )

    async def clear_memory(call: ServiceCall) -> ServiceResponse:
        """Best-effort clear of conversation agent memory / chat logs."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        cleared = {"conversation_agent": False, "notes": []}
        try:
            # HA stores conversation history via conversation component; reload agent.
            await hass.config_entries.async_reload(entry.entry_id)
            cleared["conversation_agent"] = True
            cleared["notes"].append(
                "Reloaded integration to reset in-memory agent state. "
                "Home Assistant chat log history is managed by the conversation integration."
            )
        except Exception as err:  # noqa: BLE001
            raise HomeAssistantError(f"Failed to clear memory: {err}") from err
        return {"status": "ok", **cleared}

    async def reset_stats(call: ServiceCall) -> ServiceResponse:
        """Reset token usage counters."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        tracker = _usage_tracker(hass, entry.entry_id)
        if not tracker:
            raise HomeAssistantError("Usage tracker not available")
        await tracker.async_reset()
        return {"status": "ok", "message": "Usage statistics reset"}

    async def home_briefing(call: ServiceCall) -> ServiceResponse:
        """Competitive advantage: summarize current home state via Grok."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client = _entry_client(entry)
        model = call.data.get("model") or entry.options.get(
            CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
        )

        domains = call.data.get("domains") or [
            "light",
            "climate",
            "lock",
            "cover",
            "alarm_control_panel",
            "binary_sensor",
            "sensor",
            "weather",
            "person",
        ]
        include_unavailable = bool(call.data.get("include_unavailable", False))
        max_entities = int(call.data.get("max_entities", 80))

        lines: list[str] = []
        count = 0
        for state in hass.states.async_all():
            domain = state.domain
            if domain not in domains:
                continue
            if not include_unavailable and state.state in (
                "unavailable",
                "unknown",
            ):
                continue
            # Prefer exposed-looking entities; skip noisy internals
            if state.entity_id.startswith(("sensor.date", "sensor.time")):
                continue
            friendly = state.attributes.get("friendly_name") or state.entity_id
            unit = state.attributes.get("unit_of_measurement")
            value = f"{state.state}{(' ' + unit) if unit else ''}"
            lines.append(f"- {friendly} ({state.entity_id}): {value}")
            count += 1
            if count >= max_entities:
                break

        location = entry.options.get(CONF_LOCATION_CONTEXT) or ""
        tz = str(hass.config.time_zone or "")
        focus = call.data.get("focus") or "Give a concise spoken home status briefing."
        system = (
            "You are Grok preparing a Home Assistant briefing. "
            "Be accurate, prioritize security (locks, doors, alarm), climate, "
            "and anything unusual. Keep it under 120 words unless asked otherwise."
        )
        if location:
            system += f" Home location: {location}."
        if tz:
            system += f" Timezone: {tz}."

        user_payload = f"{focus}\n\nCurrent entity snapshot:\n" + (
            "\n".join(lines) if lines else "(no matching entities)"
        )

        live_search = call.data.get(CONF_LIVE_SEARCH, LIVE_SEARCH_OFF)
        try:
            if live_search and live_search != LIVE_SEARCH_OFF:
                text, p_tok, c_tok = await async_responses_completion(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": user_payload}],
                    system_prompt=system,
                    max_tokens=call.data.get(CONF_MAX_TOKENS, 500),
                    live_search=live_search,
                    show_citations=bool(call.data.get(CONF_SHOW_CITATIONS, False)),
                )
            else:
                response = await async_chat_completion(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_payload},
                    ],
                    max_tokens=call.data.get(CONF_MAX_TOKENS, 500),
                    temperature=0.7,
                )
                text = response.choices[0].message.content or ""
                p_tok, c_tok = extract_usage(response)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating home briefing: {err}") from err

        await _record_usage(
            hass,
            entry.entry_id,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            service="home_briefing",
        )
        return {
            "status": "ok",
            "response_text": text,
            "entities_included": count,
            "model": model,
        }

    def _cfg_entry_selector() -> dict:
        return selector.ConfigEntrySelector({"integration": DOMAIN})

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional("model"): cv.string,
                vol.Optional(CONF_MAX_TOKENS): cv.positive_int,
                vol.Optional(CONF_TEMPERATURE): vol.Coerce(float),
                vol.Optional(CONF_TOP_P): vol.Coerce(float),
                vol.Optional(CONF_REASONING_EFFORT): cv.string,
                vol.Optional(CONF_LIVE_SEARCH): cv.string,
                vol.Optional(CONF_SHOW_CITATIONS): cv.boolean,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional("model"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
                vol.Optional("quality", default="standard"): vol.In(IMAGE_QUALITIES),
                vol.Optional("style", default="vivid"): vol.In(IMAGE_STYLES),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_ASK,
        ask_service,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Required("instructions"): cv.string,
                vol.Required("input_data"): cv.string,
                vol.Optional("model"): cv.string,
                vol.Optional(CONF_MAX_TOKENS): cv.positive_int,
                vol.Optional(CONF_TEMPERATURE): vol.Coerce(float),
                vol.Optional(CONF_TOP_P): vol.Coerce(float),
                vol.Optional(CONF_REASONING_EFFORT): cv.string,
                vol.Optional(CONF_LIVE_SEARCH, default=LIVE_SEARCH_OFF): cv.string,
                vol.Optional(CONF_SHOW_CITATIONS, default=True): cv.boolean,
                vol.Optional(CONF_LOCATION_CONTEXT): cv.string,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_PHOTO_ANALYSIS,
        photo_analysis,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Required("prompt"): cv.string,
                vol.Required("images"): vol.Any(cv.string, [cv.string], [dict]),
                vol.Optional("model"): cv.string,
                vol.Optional("max_tokens", default=800): cv.positive_int,
                vol.Optional(CONF_TEMPERATURE): vol.Coerce(float),
                vol.Optional(CONF_TOP_P): vol.Coerce(float),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Required("prompt"): cv.string,
                vol.Required("images"): vol.All(cv.ensure_list, [vol.Any(cv.string, dict)]),
                vol.Optional("model", default=RECOMMENDED_VISION_MODEL): cv.string,
                vol.Optional("max_tokens", default=300): cv.positive_int,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_MEMORY,
        clear_memory,
        schema=vol.Schema({vol.Required("config_entry"): _cfg_entry_selector()}),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_STATS,
        reset_stats,
        schema=vol.Schema({vol.Required("config_entry"): _cfg_entry_selector()}),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_HOME_BRIEFING,
        home_briefing,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): _cfg_entry_selector(),
                vol.Optional("focus"): cv.string,
                vol.Optional("domains"): vol.All(cv.ensure_list, [cv.string]),
                vol.Optional("max_entities", default=80): cv.positive_int,
                vol.Optional("include_unavailable", default=False): cv.boolean,
                vol.Optional("model"): cv.string,
                vol.Optional(CONF_MAX_TOKENS): cv.positive_int,
                vol.Optional(CONF_LIVE_SEARCH): cv.string,
                vol.Optional(CONF_SHOW_CITATIONS): cv.boolean,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: OpenAIConfigEntry) -> bool:
    """Set up Grok Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url="https://api.x.ai/v1",
        http_client=get_async_client(hass),
    )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    tracker = UsageTracker(hass, entry.entry_id)
    await tracker.async_load()

    # Probe Voice API (TTS/STT) — conversation still works if voice is denied
    session = async_get_clientsession(hass)
    voice_ok, voice_detail = await async_validate_voice_access(
        session, entry.data[CONF_API_KEY]
    )
    if voice_ok:
        LOGGER.info("xAI Voice API OK: %s", voice_detail)
    else:
        LOGGER.warning(
            "xAI Voice API not available for this key — TTS/STT engines "
            "may fail until voice is enabled on the key. Detail: %s",
            voice_detail,
        )

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "usage": tracker,
        "voice_ok": voice_ok,
        "voice_detail": voice_detail,
    }

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Grok."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok
