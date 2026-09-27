"""AI Task platform for xAI Grok."""

from __future__ import annotations

from json import JSONDecodeError
from typing import TYPE_CHECKING

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import (
    CONF_CHAT_MODEL,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
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
    """Grok AI Task entity (generate_data; generate_image deferred)."""

    def __init__(
        self, entry: OpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
        )
        chat_model = remap_retired_chat_model(
            subentry.data.get(CONF_CHAT_MODEL),
            RECOMMENDED_CHAT_MODEL,
        )
        if model_supports_vision(str(chat_model)):
            features |= ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        # GENERATE_IMAGE is deferred to a follow-up PR. The subentry may still
        # carry an optional image_model field for schema stability.
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
