"""AI Task platform for xAI Grok."""

from __future__ import annotations

import base64
from json import JSONDecodeError
from typing import TYPE_CHECKING

import openai
from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_PROMPT,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    remap_retired_chat_model,
)
from .entity import GrokBaseLLMEntity, model_supports_vision

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import OpenAIConfigEntry

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [GrokAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GrokAITaskEntity(ai_task.AITaskEntity, GrokBaseLLMEntity):
    """Grok AI Task entity (generate_data + generate_image)."""

    def __init__(
        self, entry: OpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        features = ai_task.AITaskEntityFeature.GENERATE_DATA

        # HA 2026.7.0 already defines these; getattr keeps older cores safe.
        support_attachments = getattr(
            ai_task.AITaskEntityFeature, "SUPPORT_ATTACHMENTS", None
        )
        generate_image = getattr(
            ai_task.AITaskEntityFeature, "GENERATE_IMAGE", None
        )

        chat_model = remap_retired_chat_model(
            subentry.data.get(CONF_CHAT_MODEL),
            RECOMMENDED_CHAT_MODEL,
        )
        # Attachments for generate_data (vision) and generate_image (edits).
        if support_attachments is not None and (
            model_supports_vision(str(chat_model)) or generate_image is not None
        ):
            features |= support_attachments
        if generate_image is not None:
            features |= generate_image

        self._attr_supported_features = features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        try:
            await self._async_handle_chat_log(
                chat_log,
                structure=task.structure,
                structure_name=task.name,
                service="ai_task",
                fallback_model="",  # no conversation fallback for AI Task
            )
        except HomeAssistantError:
            raise
        except Exception as err:
            LOGGER.error("Error in AI Task generate_data: %s", err, exc_info=True)
            raise HomeAssistantError(f"Error talking to xAI: {err}") from err

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError(
                "Error with Grok structured response"
            ) from err

        if not isinstance(data, dict):
            LOGGER.error(
                "Structured response was not a dict: %s. Response: %s",
                type(data).__name__,
                text,
            )
            raise HomeAssistantError("Error with Grok structured response")

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task (optional reference attachments → edits)."""
        from . import (
            async_generate_or_edit_image,
            async_reference_images_from_attachments,
            format_images_response,
        )

        model = self.subentry.data.get(
            CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_GENERATION_MODEL
        )
        # Always request b64 so GenImageTaskResult can return raw bytes.
        call_data: dict = {
            CONF_PROMPT: task.instructions,
            "n": 1,
            "response_format": "b64_json",
        }

        attachments = getattr(task, "attachments", None) or None
        reference_images = None
        if attachments:
            reference_images = await async_reference_images_from_attachments(
                self.hass, list(attachments)
            )

        client = self.entry.runtime_data
        try:
            response = await async_generate_or_edit_image(
                client,
                model=model,
                call_data=call_data,
                reference_images=reference_images,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        formatted = format_images_response(
            response, model=model, response_format="b64_json"
        )
        b64 = formatted["images"][0].get("b64_json")
        if not b64:
            raise HomeAssistantError("Image generation response missing b64_json")

        mime_type = formatted["images"][0].get("mime_type") or "image/jpeg"
        image_data = base64.b64decode(b64)

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=formatted.get("revised_prompt") or task.instructions,
            )
        )

        return ai_task.GenImageTaskResult(
            image_data=image_data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            model=model,
            revised_prompt=formatted.get("revised_prompt"),
        )
