"""The Grok Conversation integration."""

from __future__ import annotations

import base64
from mimetypes import guess_file_type
from pathlib import Path

import openai
from openai.types.images_response import ImagesResponse
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
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
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PAYLOAD_TEMPLATE,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    IMAGE_QUALITIES,
    IMAGE_SIZES,
    IMAGE_STYLES,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
)

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_GENERATE_CONTENT = "generate_content"

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type OpenAIConfigEntry = ConfigEntry[openai.AsyncClient]


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
    return entry


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Grok Conversation."""

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with grok."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client: openai.AsyncClient = entry.runtime_data

        try:
            response: ImagesResponse = await client.images.generate(
                model=RECOMMENDED_IMAGE_GENERATION_MODEL,
                prompt=call.data[CONF_PROMPT],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        # Return format matching OpenAI integration: url and revised_prompt
        image_data = response.data[0]
        result = {"url": image_data.url}
        # Add revised_prompt if available in response
        if hasattr(image_data, "revised_prompt") and image_data.revised_prompt:
            result["revised_prompt"] = image_data.revised_prompt
        return result

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to Grok and return the response."""
        entry = _validate_config_entry(hass, call.data["config_entry"])
        client: openai.AsyncClient = entry.runtime_data

        content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=call.data[CONF_PROMPT])
        ]

        has_images = False
        def append_files_to_content() -> None:
            nonlocal has_images
            for filename in call.data[CONF_FILENAMES]:
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
                        "Only images are supported by the xAI API,"
                        f"`{filename}` is not an image file"
                    )
                has_images = True
                content.append(
                    ResponseInputImageParam(
                        type="input_image",
                        file_id=filename,
                        image_url=f"data:{mime_type};base64,{base64_file}",
                        detail="auto",
                    )
                )

        if CONF_FILENAMES in call.data:
            await hass.async_add_executor_job(append_files_to_content)

        # Use vision model when images are provided, otherwise use chat model
        if has_images:
            model: str = RECOMMENDED_VISION_MODEL
        else:
            model: str = entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        messages: ResponseInputParam = [
            EasyInputMessageParam(type="message", role="user", content=content)
        ]

        try:
            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": entry.options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": entry.options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                "user": call.context.user_id,
            }

            # Add reasoning_effort if the model supports it (Grok reasoning models)
            # Only add reasoning_effort for models that support it (models with "reasoning" in the name)
            reasoning_effort = entry.options.get(CONF_REASONING_EFFORT)
            if reasoning_effort and reasoning_effort != "none" and "reasoning" in model.lower():
                model_args["reasoning_effort"] = reasoning_effort

            response: Response = await client.chat.completions.create(**model_args)

        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err
        except FileNotFoundError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        return {"text": response.output_text}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
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
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
                vol.Optional("quality", default="standard"): vol.In(IMAGE_QUALITIES),
                vol.Optional("style", default="vivid"): vol.In(IMAGE_STYLES),
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

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Grok."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)