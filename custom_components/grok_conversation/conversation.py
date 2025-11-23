"""Conversation support for xAI Grok."""

from collections.abc import AsyncGenerator, Callable
import json
import re
from typing import Any, Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ToolParam,
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
from .exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    convert_to_template,
    get_function_executor,
    validate_authentication,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _strip_json_from_response(response: str) -> str:
    """Strip JSON objects from the end of LLM responses."""
    if not response:
        return response

    # Look for JSON objects at the end of the response
    # Find the last opening brace and check if everything after it is valid JSON
    last_brace_index = response.rfind('{')
    if last_brace_index == -1:
        return response

    # Extract potential JSON from the last brace to the end
    potential_json = response[last_brace_index:]
    try:
        # Try to parse it as JSON
        json.loads(potential_json)
        # If successful, remove the JSON part
        return response[:last_brace_index].strip()
    except json.JSONDecodeError:
        # Not valid JSON, check for nested braces
        return response


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
) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


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
            ResponseFunctionToolCallParam(
                type="function_call",
                name=tool_call.tool_name,
                arguments=json.dumps(tool_call.tool_args),
                call_id=tool_call.id,
            )
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
        LOGGER.debug("Processing chunk: %s", chunk)
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
                # Log finish reason if available
                if choice.finish_reason:
                    LOGGER.debug("Stream finished with reason: %s", choice.finish_reason)


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
        """Call the API with function calling support."""
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

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        client = self.entry.runtime_data

        # Get exposed entities if LLM HASS API is enabled
        exposed_entities = []
        if options.get(CONF_LLM_HASS_API):
            exposed_entities = llm.async_get_api(self.hass, options[CONF_LLM_HASS_API]).async_get_exposed_entities()

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Prepare tools for the API call if we have exposed entities
            tools = []
            if exposed_entities:
                tools = [_format_tool(tool, None) for tool in llm.async_get_api(self.hass, options[CONF_LLM_HASS_API]).async_get_tools()]

            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "stream": False,
            }

            # Add reasoning_effort if the model supports it (Grok reasoning models)
            # Only add reasoning_effort for models that support it (models with "reasoning" in the name)
            reasoning_effort = options.get(CONF_REASONING_EFFORT)
            if reasoning_effort and reasoning_effort != "none" and "reasoning" in model.lower():
                model_args["reasoning_effort"] = reasoning_effort

            if tools:
                model_args["tools"] = tools
                model_args["tool_choice"] = "auto"

            try:
                LOGGER.debug("Sending API request: %s", model_args)
                result = await client.chat.completions.create(**model_args)

                choice = result.choices[0]
                message = choice.message

                # Handle tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Add assistant message with tool calls
                    assistant_content = conversation.AssistantContent(
                        agent_id=user_input.agent_id,
                        content=message.content or ""
                    )

                    tool_calls = []
                    for tool_call in message.tool_calls:
                        tool_calls.append(
                            conversation.ToolCall(
                                id=tool_call.id,
                                tool_name=tool_call.function.name,
                                tool_args=json.loads(tool_call.function.arguments)
                            )
                        )

                    assistant_content.tool_calls = tool_calls
                    async for _ in chat_log.async_add_assistant_content(assistant_content):
                        pass  # Consume the async generator
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.tool_name,
                                    "arguments": json.dumps(tc.tool_args)
                                }
                            }
                            for tc in tool_calls
                        ]
                    })

                    # Execute tool calls
                    for tool_call in message.tool_calls:
                        try:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)

                            # Execute the tool using LLM API
                            tool_result = await llm.async_get_api(self.hass, options[CONF_LLM_HASS_API]).async_call_tool(
                                tool_name, tool_args
                            )

                            # Add tool result to messages
                            async for _ in chat_log.async_add_tool_result(
                                conversation.ToolResultContent(
                                    tool_call_id=tool_call.id,
                                    tool_result=tool_result
                                )
                            ):
                                pass  # Consume the async generator
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_result)
                            })

                        except Exception as err:
                            LOGGER.error("Error executing tool %s: %s", tool_call.function.name, err)
                            # Add error result
                            async for _ in chat_log.async_add_tool_result(
                                conversation.ToolResultContent(
                                    tool_call_id=tool_call.id,
                                    tool_result={"error": str(err)}
                                )
                            ):
                                pass  # Consume the async generator
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": str(err)})
                            })

                    # Continue the loop to get the final response
                    continue

                else:
                    # No tool calls, this is the final response
                    full_response = message.content or ""
                    # Strip any JSON metadata from the end of the response
                    full_response = _strip_json_from_response(full_response)
                    LOGGER.debug("API response: %s", full_response)

                    # Add the response as AssistantContent
                    if full_response:
                        async for _ in chat_log.async_add_assistant_content(
                            conversation.AssistantContent(
                                agent_id=user_input.agent_id,
                                content=full_response
                            )
                        ):
                            pass  # Consume the async generator
                        messages.append({"role": "assistant", "content": full_response})
                    else:
                        LOGGER.warning("No assistant content received from API response")

                # Log usage stats if available
                if result.usage:
                    chat_log.async_trace(
                        {
                            "stats": {
                                "input_tokens": result.usage.prompt_tokens,
                                "output_tokens": result.usage.completion_tokens,
                            }
                        }
                    )

                # Check finish reason
                if choice.finish_reason == "length":
                    raise TokenLengthExceededError(options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS))

                break  # Exit loop after successful response

            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by xAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to xAI: %s", err, exc_info=True)
                raise HomeAssistantError(f"Error talking to xAI: {err}") from err
            except Exception as err:
                LOGGER.error("Unexpected error in conversation handler: %s", err, exc_info=True)
                raise HomeAssistantError(f"Unexpected error: {err}") from err

        # Create intent response
        intent_response = intent.IntentResponse(language=user_input.language)

        # Get the last assistant content for speech
        last_assistant_content = None
        for content in reversed(chat_log.content):
            if isinstance(content, conversation.AssistantContent):
                last_assistant_content = content
                break

        if last_assistant_content and last_assistant_content.content:
            intent_response.async_set_speech(last_assistant_content.content)
        else:
            intent_response.async_set_speech("Sorry, I couldn't generate a response.")

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