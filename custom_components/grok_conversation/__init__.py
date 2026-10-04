"""The Grok Conversation integration."""

from __future__ import annotations

import base64
from mimetypes import guess_file_type
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlparse

import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.images_response import ImagesResponse
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
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
    CONF_FALLBACK_MODEL,
    CONF_FAST_MODEL,
    CONF_FILENAMES,
    CONF_IMAGE_MODEL,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SHOW_CITATIONS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VISION_MODEL,
    DEFAULT_AI_TASK_NAME,
    DOMAIN,
    IMAGE_ASPECT_RATIOS,
    IMAGE_QUALITIES,
    IMAGE_QUALITY_DOCUMENTED,
    IMAGE_RESOLUTIONS,
    IMAGE_RESPONSE_FORMATS,
    IMAGE_SIZES,
    IMAGE_STYLES,
    LIVE_SEARCH_OFF,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_FAST_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
    RETIRED_CHAT_MODELS,
    RETIRED_VISION_MODELS,
    SERVICE_ASK,
    SERVICE_CLEAR_MEMORY,
    SERVICE_GENERATE_CONTENT,
    SERVICE_GENERATE_IMAGE,
    SERVICE_HOME_BRIEFING,
    SERVICE_PHOTO_ANALYSIS,
    SERVICE_QUERY_IMAGE,
    SERVICE_RESET_STATS,
    SIZE_TO_ASPECT_RATIO,
    remap_retired_chat_model,
)
from .entity import resolve_vision_model
from .usage import UsageTracker
from .voice_api import async_validate_voice_access

PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
    Platform.TTS,
    Platform.STT,
)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

OpenAIConfigEntry = ConfigEntry  # runtime_data: openai.AsyncClient

# One deprecation WARNING per field per Home Assistant run.
_DEPRECATED_FIELD_WARNED: set[str] = set()
_RETIRED_VISION_WARNED = False
_QUALITY_UNSUPPORTED_WARNED = False


def _warn_deprecated_image_field(field: str, message: str) -> None:
    """Log a one-shot deprecation warning for a generate_image field."""
    if field in _DEPRECATED_FIELD_WARNED:
        return
    _DEPRECATED_FIELD_WARNED.add(field)
    LOGGER.warning(message)


def model_supports_image_quality(model: str | None) -> bool:
    """Return True if ``quality`` is documented for this imagine model.

    xAI only supports ``quality`` on ``grok-imagine-image-2.0`` (and dated /
    ``-latest`` aliases that start with ``grok-imagine-image-2.0-``).
    """
    mid = (model or "").strip().lower()
    if not mid:
        return False
    return mid == "grok-imagine-image-2.0" or mid.startswith(
        "grok-imagine-image-2.0-"
    )


def build_image_generate_kwargs(call_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Build kwargs for client.images.generate from service call data.

    Never sends ``size`` or ``style``. Sends documented ``quality``
    (``low``/``medium``/``auto``) via ``extra_body`` only when the model
    supports it (``grok-imagine-image-2.0``).
    """
    global _QUALITY_UNSUPPORTED_WARNED

    aspect_ratio = call_data.get("aspect_ratio")
    size = call_data.get("size")
    style = call_data.get("style")
    quality = call_data.get("quality")
    resolution = call_data.get("resolution")
    n = int(call_data.get("n", 1))
    response_format = call_data.get("response_format", "url")

    if style is not None:
        _warn_deprecated_image_field(
            "style",
            "generate_image field 'style' is deprecated and ignored by xAI; "
            "it will be removed in a future release.",
        )

    if size is not None:
        mapped = SIZE_TO_ASPECT_RATIO.get(size)
        if aspect_ratio is None and mapped:
            aspect_ratio = mapped
            _warn_deprecated_image_field(
                "size",
                "generate_image field 'size' is deprecated; mapped "
                f"'{size}' → aspect_ratio '{mapped}'. "
                "It will be removed in a future release.",
            )
        else:
            _warn_deprecated_image_field(
                "size",
                "generate_image field 'size' is deprecated and ignored "
                f"(aspect_ratio already set to '{aspect_ratio}'). "
                "It will be removed in a future release.",
            )

    if quality is not None and quality not in IMAGE_QUALITY_DOCUMENTED:
        _warn_deprecated_image_field(
            "quality",
            "generate_image field 'quality' value "
            f"'{quality}' is deprecated and ignored (use low|medium|auto for "
            "grok-imagine-image-2.0). It will be removed in a future release.",
        )
        quality = None

    if quality is not None and not model_supports_image_quality(model):
        if not _QUALITY_UNSUPPORTED_WARNED:
            _QUALITY_UNSUPPORTED_WARNED = True
            LOGGER.warning(
                "generate_image field 'quality' is only supported for "
                "grok-imagine-image-2.0; ignoring for model '%s'.",
                model,
            )
        quality = None

    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": call_data[CONF_PROMPT],
        "n": n,
        "response_format": response_format,
    }
    extra_body: dict[str, Any] = {}
    if aspect_ratio is not None:
        extra_body["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        extra_body["resolution"] = resolution
    if quality is not None:
        extra_body["quality"] = quality
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def format_images_response(
    response: ImagesResponse,
    *,
    model: str,
    response_format: str,
) -> dict[str, Any]:
    """Format an ImagesResponse into a backward-compatible service result."""
    if not response.data:
        raise HomeAssistantError("Image generation returned empty data")

    images: list[dict[str, Any]] = []
    for item in response.data:
        entry: dict[str, Any] = {}
        url = getattr(item, "url", None)
        b64 = getattr(item, "b64_json", None)
        mime = getattr(item, "mime_type", None)
        if url:
            entry["url"] = url
        if b64:
            entry["b64_json"] = b64
        if mime:
            entry["mime_type"] = mime
        if "url" not in entry and "b64_json" not in entry:
            raise HomeAssistantError(
                "Image generation response missing url and b64_json"
            )
        images.append(entry)

    result: dict[str, Any] = {"model": model, "images": images}
    if response_format == "url" and images[0].get("url"):
        result["url"] = images[0]["url"]
    revised = getattr(response.data[0], "revised_prompt", None)
    if revised:
        result["revised_prompt"] = revised
    return result


def resolve_service_vision_model(
    call_model: str | None,
    entry_options: MappingProxyType[str, Any] | dict[str, Any],
) -> str:
    """Resolve vision model for image services with retired-id guard.

    Prefer the entry chat model when it supports images (shared #36
    resolution). Fall back to ``vision_model`` / recommended. Per-call
    overrides that use retired ``grok-2-vision-*`` ids are remapped.
    """
    global _RETIRED_VISION_WARNED

    if call_model:
        if call_model in RETIRED_VISION_MODELS:
            if not _RETIRED_VISION_WARNED:
                _RETIRED_VISION_WARNED = True
                LOGGER.warning(
                    "Vision model '%s' is retired; using '%s' instead. "
                    "Update your automation or integration options.",
                    call_model,
                    RECOMMENDED_VISION_MODEL,
                )
            return RECOMMENDED_VISION_MODEL
        return remap_retired_chat_model(call_model, RECOMMENDED_VISION_MODEL)

    chat = remap_retired_chat_model(
        entry_options.get(CONF_CHAT_MODEL),
        RECOMMENDED_CHAT_MODEL,
    )
    return resolve_vision_model(dict(entry_options), chat)


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
            CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_GENERATION_MODEL
        )
        response_format = call.data.get("response_format", "url")
        kwargs = build_image_generate_kwargs(dict(call.data), model)

        try:
            response: ImagesResponse = await client.images.generate(**kwargs)
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        result = format_images_response(
            response, model=model, response_format=response_format
        )
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
            model = resolve_service_vision_model(
                call.data.get("model"), entry.options
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
        model = resolve_service_vision_model(call.data.get("model"), entry.options)
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
                vol.Optional("aspect_ratio"): vol.In(IMAGE_ASPECT_RATIOS),
                vol.Optional("resolution"): vol.In(IMAGE_RESOLUTIONS),
                vol.Optional("quality"): vol.In(IMAGE_QUALITIES),
                vol.Optional("n", default=1): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=10)
                ),
                vol.Optional("response_format", default="url"): vol.In(
                    IMAGE_RESPONSE_FORMATS
                ),
                # Deprecated — still validated so existing automations work
                vol.Optional("size"): vol.In(IMAGE_SIZES),
                vol.Optional("style"): vol.In(IMAGE_STYLES),
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
                vol.Optional("model"): cv.string,
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
        # Legacy HTTPX clients are supported at runtime only (openai 3.x / HTTPX2).
        http_client=cast(Any, get_async_client(hass)),
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


def _add_ai_task_subentry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Add default AI Task subentry when missing."""
    if any(
        sub.subentry_type == "ai_task_data" for sub in entry.subentries.values()
    ):
        return
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(dict(RECOMMENDED_AI_TASK_OPTIONS)),
            subentry_type="ai_task_data",
            title=DEFAULT_AI_TASK_NAME,
            unique_id=None,
        ),
    )


def _migrate_retired_model_keys(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Rewrite retired model ids in an options/subentry dict.

    Returns list of (key, old, new) replacements applied.
    """
    replacements: list[tuple[str, str, str]] = []
    key_map = (
        (CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL, RETIRED_VISION_MODELS),
        (CONF_FAST_MODEL, RECOMMENDED_FAST_MODEL, RETIRED_CHAT_MODELS),
        (CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL, RETIRED_CHAT_MODELS),
        (CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL, RETIRED_CHAT_MODELS),
    )
    for key, recommended, retired in key_map:
        current = data.get(key)
        if current in retired:
            data[key] = recommended
            replacements.append((key, str(current), recommended))
    return replacements


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry (AI Task subentry + retired vision/fast models)."""
    LOGGER.debug(
        "Migrating %s from version %s.%s",
        entry.entry_id,
        entry.version,
        entry.minor_version,
    )

    if entry.version > 1:
        return False

    if entry.version == 1 and entry.minor_version < 2:
        _add_ai_task_subentry(hass, entry)
        hass.config_entries.async_update_entry(entry, minor_version=2)

    if entry.version == 1 and entry.minor_version < 3:
        options = dict(entry.options)
        update_kwargs: dict[str, Any] = {"minor_version": 3}
        for key, old, new in _migrate_retired_model_keys(options):
            update_kwargs["options"] = options
            LOGGER.info(
                "Migrated retired %s '%s' → '%s' for entry %s",
                key,
                old,
                new,
                entry.entry_id,
            )

        # Rewrite retired ids on conversation + ai_task_data subentries.
        for subentry in entry.subentries.values():
            if subentry.subentry_type not in ("conversation", "ai_task_data"):
                continue
            data = dict(subentry.data)
            changes = _migrate_retired_model_keys(data)
            if not changes:
                continue
            hass.config_entries.async_update_subentry(entry, subentry, data=data)
            for key, old, new in changes:
                LOGGER.info(
                    "Migrated retired %s '%s' → '%s' for subentry %s (%s)",
                    key,
                    old,
                    new,
                    subentry.subentry_id,
                    subentry.subentry_type,
                )

        hass.config_entries.async_update_entry(entry, **update_kwargs)

    LOGGER.debug(
        "Migration to version %s.%s successful",
        entry.version,
        entry.minor_version,
    )
    return True
