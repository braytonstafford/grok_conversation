"""Conversation support for xAI Grok."""

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any, Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.chat import ChatCompletionChunk
from voluptuous_openapi import convert
from typing import AsyncIterator

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OpenAIConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict:
    """Format tool specification for xAI chat completions API."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
            "description": tool.description or "",
            "strict": False,
        }
    }


def _convert_content_to_param(
    content: conversation.Content,
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    if isinstance(content, conversation.ToolResultContent):
        return [
            FunctionCallOutput(
                type="function_call_output",
                call_id=content.tool_call_id,
                output=json.dumps(content.tool_result),
            )
        ]

    if content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "developer":
            role = "system"
        messages.append(
            EasyInputMessageParam(type="message", role=role, content=content.content)
        )

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        messages.extend(
            {
                "type": "function_call",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.tool_args),
                },
                "call_id": tool_call.id,
            }
            for tool_call in content.tool_calls
        )
    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncIterator[ChatCompletionChunk],
) -> AsyncGenerator[dict, None]:
    """
    Transform an xAI chat completions delta stream into Home Assistant format.
    Yields dictionaries with role, content, or tool calls for incremental updates.
    """
    current_tool_call = None
    tool_call_counter = 0
    async for chunk in result:
        for choice in chunk.choices:
            if choice.delta:
                # Handle role
                if choice.delta.role:
                    yield {"role": choice.delta.role}
                # Handle content
                if choice.delta.content:
                    yield {"content": choice.delta.content}
                # Handle function calls (tool calls)
                if choice.delta.function_call:
                    if current_tool_call is None:
                        # Start a new tool call
                        current_tool_call = {
                            "name": choice.delta.function_call.name,
                            "arguments": choice.delta.function_call.arguments or ""
                        }
                    else:
                        # Append to existing tool call arguments
                        if choice.delta.function_call.arguments:
                            current_tool_call["arguments"] += choice.delta.function_call.arguments
                    # Check if arguments are complete (i.e., valid JSON)
                    try:
                        parsed_args = json.loads(current_tool_call["arguments"])
                        # If parsing succeeds, yield the complete tool call
                        tool_id = str(tool_call_counter)
                        yield {
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "tool_name": current_tool_call["name"],
                                    "tool_args": parsed_args
                                }
                            ]
                        }
                        current_tool_call = None
                        tool_call_counter += 1
                    except json.JSONDecodeError:
                        # Arguments are not yet complete
                        pass
                # Log usage stats if available
                if chunk.usage:
                    chat_log.async_trace(
                        {
                            "stats": {
                                "input_tokens": chunk.usage.prompt_tokens,
                                "output_tokens": chunk.usage.completion_tokens,
                            }
                        }
                    )


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Grok conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="xAI",
            model="Grok",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[dict] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        client = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools

            if model.startswith("o"):
                model_args["reasoning"] = {
                    "effort": options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    )
                }

            try:
                result = await client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by xAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to xAI: %s", err)
                raise HomeAssistantError("Error talking to xAI") from err

            # Collect all deltas into a single string
            full_response = ""
            async for content in chat_log.async_add_delta_content_stream(
                user_input.agent_id, _transform_stream(chat_log, result)
            ):
                if isinstance(content, dict) and "content" in content:
                    full_response += content["content"]

            # Add the full response as a single AssistantContent
            if full_response or tools:  # Add even if empty to ensure AssistantContent
                chat_log.add_content(conversation.AssistantContent(content=full_response or "No response generated."))
            else:
                LOGGER.warning("No content or tools in response, adding default AssistantContent")
                chat_log.add_content(conversation.AssistantContent(content="No response generated."))

            messages.extend(_convert_content_to_param(conversation.AssistantContent(content=full_response)))

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        LOGGER.debug("chat_log.content types: %s", [type(c) for c in chat_log.content])
        if not chat_log.content or not isinstance(chat_log.content[-1], conversation.AssistantContent):
            LOGGER.warning("Last content item is not AssistantContent: %s", type(chat_log.content[-1]) if chat_log.content else "Empty")
            intent_response.async_set_speech("Sorry, I couldn't generate a response.")
        else:
            intent_response.async_set_speech(full_response or "No response generated.")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)