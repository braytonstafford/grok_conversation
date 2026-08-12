"""Conversation support for xAI Grok."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any, AsyncIterator, Literal

import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.llm import ToolInput
from homeassistant.util import dt as dt_util

from . import OpenAIConfigEntry
from .api_helpers import (
    async_chat_completion,
    async_responses_completion,
    extract_usage,
    looks_like_search_query,
    looks_like_simple_query,
)
from .const import (
    CONF_AUTO_MODEL_ROUTING,
    CONF_BUDGET_WARN_USD,
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_FAST_MODEL,
    CONF_HOME_CONTEXT,
    CONF_INTERACTION_MODE,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_OPTIMIZED,
    DOMAIN,
    LIVE_SEARCH_OFF,
    LOGGER,
    MODE_CHAT_ONLY,
    MODE_PIPELINE,
    MODE_TOOLS,
    RECOMMENDED_AUTO_MODEL_ROUTING,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_FAST_MODEL,
    RECOMMENDED_HOME_CONTEXT,
    RECOMMENDED_INTERACTION_MODE,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VOICE_OPTIMIZED,
    VOICE_OPTIMIZED_SUFFIX,
)
from .exceptions import TokenLengthExceededError
from .usage import UsageTracker

MAX_TOOL_ITERATIONS = 10


def _strip_json_from_response(response: str) -> str:
    """Strip JSON objects from the end of LLM responses."""
    if not response:
        return response
    last_brace_index = response.rfind("{")
    if last_brace_index == -1:
        return response
    potential_json = response[last_brace_index:]
    try:
        json.loads(potential_json)
        return response[:last_brace_index].strip()
    except json.JSONDecodeError:
        return response


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _sanitize_tool_schema(schema: Any) -> dict[str, Any]:
    """Normalize tool JSON schema for xAI function calling.

    xAI rejects roots that are anyOf/oneOf unions (e.g. HA HassStartTimer).
    Merge properties from every object branch so alternate arg shapes
    (name vs hours/minutes/seconds) are all still available to the model.
    """
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    def _merge_union_branches(branches: list[Any]) -> dict[str, Any] | None:
        """Collapse anyOf/oneOf/allOf into one object schema.

        Properties from every object-like branch are merged so no timer /
        alternate-shape parameters are dropped. ``required`` is dropped
        because it is no longer valid across merged alternatives.
        """
        props: dict[str, Any] = {}
        objectish = 0
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            # Recurse first so nested unions flatten too
            branch = _clean(branch)
            if not isinstance(branch, dict):
                continue
            if branch.get("type") == "object" or "properties" in branch:
                objectish += 1
                nested = branch.get("properties")
                if isinstance(nested, dict):
                    props.update(nested)
        if objectish:
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": True,
            }
        # No object branch — keep first dict branch (cleaned)
        for branch in branches:
            if isinstance(branch, dict):
                return _clean(branch)
        return None

    def _clean(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = dict(node)

        # Collapse union keys to a single object-compatible schema
        for union_key in ("anyOf", "oneOf", "allOf"):
            if union_key not in node or not isinstance(node[union_key], list):
                continue
            branches = node[union_key]
            merged = _merge_union_branches(branches)
            rest = {
                k: v
                for k, v in node.items()
                if k not in (union_key, "required")
            }
            if isinstance(merged, dict):
                # Branch content wins for type/properties; outer keys fill gaps
                node = {**rest, **merged}
            else:
                node = rest
            # Only collapse one union key per pass; re-enter via recursion below
            break

        # Normalize type lists that include object
        t = node.get("type")
        if isinstance(t, list):
            if "object" in t:
                node["type"] = "object"
            elif "array" in t:
                node["type"] = "array"
            elif t:
                node["type"] = t[0]

        if "properties" in node and isinstance(node["properties"], dict):
            node["properties"] = {
                key: _clean(value) for key, value in node["properties"].items()
            }
        if "items" in node:
            node["items"] = _clean(node["items"])
        if "additionalProperties" in node and isinstance(
            node["additionalProperties"], dict
        ):
            node["additionalProperties"] = _clean(node["additionalProperties"])

        # Drop nested unsupported combinators left behind
        for bad in ("not",):
            node.pop(bad, None)

        return node

    cleaned = _clean(schema)

    # Root must be an object for xAI
    if not isinstance(cleaned, dict):
        cleaned = {}
    if cleaned.get("type") != "object" and "properties" not in cleaned:
        # Non-object root (string/number/etc.) — wrap
        cleaned = {
            "type": "object",
            "properties": {"value": cleaned} if cleaned else {},
            "additionalProperties": True,
        }
    cleaned.setdefault("type", "object")
    if cleaned["type"] != "object":
        cleaned = {
            "type": "object",
            "properties": {"value": cleaned},
            "additionalProperties": True,
        }
    cleaned.setdefault("properties", {})
    if not isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {}

    # Strip root-level keys xAI rejects on tool parameter schemas
    for bad in ("oneOf", "anyOf", "allOf", "not", "enum"):
        cleaned.pop(bad, None)
    # Drop required after union merge — alternatives make it invalid
    if "required" in cleaned and not cleaned.get("properties"):
        cleaned.pop("required", None)

    return cleaned


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification for xAI / OpenAI-compatible function calling."""
    # HA scripts/tools with selector: fields need llm.selector_serializer so
    # voluptuous_openapi.convert() does not try to use TextSelector as a dict key.
    serializer = custom_serializer or llm.selector_serializer
    try:
        raw_schema = convert(tool.parameters, custom_serializer=serializer)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning(
            "Failed to convert schema for tool %s (%s); using empty object schema",
            tool.name,
            err,
        )
        raw_schema = {"type": "object", "properties": {}}

    schema = _sanitize_tool_schema(raw_schema)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _convert_content_to_param(
    content: conversation.Content,
) -> list[ChatCompletionMessageParam]:
    """Convert any native chat message for this agent to the native format."""
    messages: list[ChatCompletionMessageParam] = []

    if isinstance(content, conversation.ToolResultContent):
        tool_message = {
            "role": "tool",
            "content": json.dumps(content.tool_result),
            "tool_call_id": content.tool_call_id,
        }
        messages.append(tool_message)  # type: ignore[arg-type]
        return messages

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        tool_calls_list = []
        for tool_call in content.tool_calls:
            if hasattr(tool_call, "function"):
                tool_calls_list.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            elif hasattr(tool_call, "tool_name"):
                tool_calls_list.append(
                    {
                        "id": tool_call.id
                        if hasattr(tool_call, "id")
                        else str(hash(tool_call)),
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args)
                            if hasattr(tool_call, "tool_args")
                            else "{}",
                        },
                    }
                )
            elif isinstance(tool_call, dict):
                tool_calls_list.append(
                    {
                        "id": tool_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_call.get(
                                "tool_name", tool_call.get("name", "")
                            ),
                            "arguments": json.dumps(
                                tool_call.get(
                                    "tool_args", tool_call.get("arguments", {})
                                )
                            ),
                        },
                    }
                )
            else:
                tool_calls_list.append(
                    {
                        "id": getattr(tool_call, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(
                                tool_call,
                                "tool_name",
                                getattr(tool_call, "name", ""),
                            ),
                            "arguments": json.dumps(
                                getattr(
                                    tool_call,
                                    "tool_args",
                                    getattr(tool_call, "arguments", {}),
                                )
                            ),
                        },
                    }
                )

        messages.append(
            {
                "role": "assistant",
                "content": content.content or "",
                "tool_calls": tool_calls_list,
            }
        )  # type: ignore[arg-type]
        return messages

    if hasattr(content, "content") and content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": content.content})
    return messages


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncIterator[ChatCompletionChunk],
) -> AsyncGenerator[dict, None]:
    """Transform an xAI chat completions delta stream into Home Assistant format."""
    current_tool_call = None
    tool_call_counter = 0
    async for chunk in result:
        for choice in chunk.choices:
            if not choice.delta:
                continue
            if choice.delta.role:
                yield {"role": choice.delta.role}
            if choice.delta.content:
                yield {"content": choice.delta.content}
            if choice.delta.function_call:
                if current_tool_call is None:
                    current_tool_call = {
                        "name": choice.delta.function_call.name,
                        "arguments": choice.delta.function_call.arguments or "",
                    }
                else:
                    if choice.delta.function_call.arguments:
                        current_tool_call[
                            "arguments"
                        ] += choice.delta.function_call.arguments
                try:
                    parsed_args = json.loads(current_tool_call["arguments"])
                    tool_id = str(tool_call_counter)
                    yield {
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "tool_name": current_tool_call["name"],
                                "tool_args": parsed_args,
                            }
                        ]
                    }
                    current_tool_call = None
                    tool_call_counter += 1
                except json.JSONDecodeError:
                    pass
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
        self._attr_supported_features = conversation.ConversationEntityFeature(0)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        self._update_control_feature()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    def _update_control_feature(self) -> None:
        mode = self.entry.options.get(
            CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE
        )
        llm_hass_api = self.entry.options.get(CONF_LLM_HASS_API)
        if mode == MODE_CHAT_ONLY or not llm_hass_api:
            self._attr_supported_features = conversation.ConversationEntityFeature(0)
            return
        api_ids = llm_hass_api if isinstance(llm_hass_api, list) else [llm_hass_api]
        api_ids = [api_id for api_id in api_ids if api_id != "none"]
        if not api_ids:
            self._attr_supported_features = conversation.ConversationEntityFeature(0)
            return
        try:
            llm.async_get_api(self.hass, api_ids[0])
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        except Exception:  # noqa: BLE001
            self._attr_supported_features = conversation.ConversationEntityFeature(0)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _usage_tracker(self) -> UsageTracker | None:
        data = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id)
        if not data:
            return None
        return data.get("usage")

    async def _record_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        tracker = self._usage_tracker()
        if tracker:
            await tracker.async_record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                service="conversation",
            )
            budget = float(
                self.entry.options.get(CONF_BUDGET_WARN_USD, 0) or 0
            )
            if budget > 0 and tracker.snapshot.estimated_cost_usd >= budget:
                LOGGER.warning(
                    "Grok estimated spend $%.4f exceeded budget warn $%.2f",
                    tracker.snapshot.estimated_cost_usd,
                    budget,
                )
                self.hass.bus.async_fire(
                    f"{DOMAIN}_budget_warning",
                    {
                        "entry_id": self.entry.entry_id,
                        "estimated_cost_usd": tracker.snapshot.estimated_cost_usd,
                        "budget_warn_usd": budget,
                    },
                )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API with function calling support."""
        try:
            return await self._async_handle_message_inner(user_input, chat_log)
        except Exception as err:  # noqa: BLE001
            LOGGER.error(
                "Unexpected error in conversation handler: %s", err, exc_info=True
            )
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(
                "Sorry, I encountered an unexpected error. Please try again."
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id if chat_log else "",
                continue_conversation=False,
            )

    def _is_tool_result_helpful(self, tool_name: str, tool_result: Any) -> bool:
        """Legacy helper — kept for compatibility; always prefer real payloads."""
        return True

    @staticmethod
    def _tool_result_payload(tool_result: Any) -> str:
        """Serialize tool results for the model. Always pass through real data."""
        try:
            return json.dumps(tool_result, default=str)
        except TypeError:
            return json.dumps({"result": str(tool_result)})

    @staticmethod
    def _pick_speech_content(chat_log: conversation.ChatLog) -> str | None:
        """Prefer the last assistant message that is not a tool-call turn."""
        fallback: str | None = None
        for content in reversed(chat_log.content):
            if not isinstance(content, conversation.AssistantContent):
                continue
            text = (content.content or "").strip()
            if not text:
                continue
            # Skip intermediate tool-call turns (often OOC / "calling tool…")
            if getattr(content, "tool_calls", None):
                if fallback is None:
                    fallback = text
                continue
            return text
        return fallback

    def _build_factual_context(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Location, local time, presence, weather — needed by live search.

        Always attach this to the search pass even when HA tools are present
        (#27). Persona/voice bits stay separate so they can stay gated.
        """
        options = self.entry.options
        parts: list[str] = []

        location = (options.get(CONF_LOCATION_CONTEXT) or "").strip()
        if location:
            parts.append(
                "Home location context for local queries "
                f"(use this for 'near me', local weather, open now, distance): {location}."
            )
        else:
            tz = self.hass.config.time_zone
            if tz:
                parts.append(f"Home Assistant timezone: {tz}.")

        if options.get(CONF_HOME_CONTEXT, RECOMMENDED_HOME_CONTEXT):
            now = dt_util.now()
            parts.append(
                f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M')}."
            )
            people = []
            for state in self.hass.states.async_all("person"):
                people.append(
                    f"{state.attributes.get('friendly_name', state.entity_id)}={state.state}"
                )
            if people:
                parts.append("Person presence: " + ", ".join(people[:12]) + ".")
            weather = next(
                (
                    s
                    for s in self.hass.states.async_all("weather")
                    if s.state not in ("unavailable", "unknown")
                ),
                None,
            )
            if weather:
                temp = weather.attributes.get("temperature")
                unit = weather.attributes.get("temperature_unit", "")
                parts.append(
                    f"Weather entity {weather.entity_id}: {weather.state}"
                    + (f", {temp}{unit}" if temp is not None else "")
                    + "."
                )

        return "\n".join(parts)

    def _build_persona_context(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Voice-style and user-name hints (not required for search grounding)."""
        options = self.entry.options
        parts: list[str] = []

        if options.get(CONF_VOICE_OPTIMIZED, RECOMMENDED_VOICE_OPTIMIZED):
            parts.append(VOICE_OPTIMIZED_SUFFIX.strip())

        if options.get(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME):
            name = self._resolve_user_name(user_input)
            if name:
                parts.append(
                    f"The current user is named {name}. Address them by name when natural."
                )

        return "\n".join(parts)

    def _build_extra_system_prompt(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Assemble persona + factual context for the main LLM / tool path."""
        parts = [
            p
            for p in (
                self._build_persona_context(user_input),
                self._build_factual_context(user_input),
            )
            if p
        ]
        return "\n".join(parts)

    def _resolve_user_name(
        self, user_input: conversation.ConversationInput
    ) -> str | None:
        """Best-effort user display name from person entity or HA user."""
        context = user_input.context
        user_id = getattr(context, "user_id", None) if context else None
        if user_id:
            for state in self.hass.states.async_all("person"):
                if state.attributes.get("user_id") == user_id:
                    return state.attributes.get("friendly_name") or state.name
            user = self.hass.auth.async_get_user(user_id) if self.hass.auth else None
            if user and user.name:
                return user.name
        return None

    def _select_model(self, user_text: str, options: dict) -> str:
        """Pick chat/fast model based on auto-routing."""
        primary = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        fast = options.get(CONF_FAST_MODEL, RECOMMENDED_FAST_MODEL)
        if options.get(CONF_AUTO_MODEL_ROUTING, RECOMMENDED_AUTO_MODEL_ROUTING):
            if looks_like_simple_query(user_text) and not looks_like_search_query(
                user_text
            ):
                return fast or primary
        return primary

    async def _try_pipeline(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult | None:
        """Try built-in Home Assistant conversation agent first."""
        try:
            result = await conversation.async_converse(
                self.hass,
                text=user_input.text,
                conversation_id=None,
                context=user_input.context,
                language=user_input.language,
                agent_id="conversation.home_assistant",
                device_id=user_input.device_id,
            )
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Pipeline HA agent failed: %s", err)
            return None

        speech = ""
        if result and result.response:
            speech = result.response.speech.get("plain", {}).get("speech", "")
            # Also check response response types
            if not speech and hasattr(result.response, "as_dict"):
                data = result.response.as_dict()
                speech = (
                    data.get("speech", {}).get("plain", {}).get("speech", "") or ""
                )
        if not speech:
            return None
        lowered = speech.lower()
        fallback_markers = (
            "sorry",
            "i am not aware",
            "i'm not aware",
            "don't know",
            "do not know",
            "no intent",
            "not sure how",
            "can you rephrase",
        )
        if any(m in lowered for m in fallback_markers):
            return None
        return result

    async def _async_handle_message_inner(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Inner method that handles the actual conversation logic."""
        options = self.entry.options
        mode = options.get(CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE)

        LOGGER.info(
            "Grok handling message mode=%s device_id=%s text=%s",
            mode,
            user_input.device_id,
            user_input.text,
        )

        # Intelligent pipeline: HA intent first
        if mode == MODE_PIPELINE:
            piped = await self._try_pipeline(user_input)
            if piped is not None:
                LOGGER.debug("Served via HA intent pipeline")
                return piped

        # Chat-only: disable LLM HASS API tools
        llm_api_option = None if mode == MODE_CHAT_ONLY else options.get(CONF_LLM_HASS_API)

        extra = self._build_extra_system_prompt(user_input)
        user_extra = user_input.extra_system_prompt or ""
        combined_extra = "\n".join(p for p in (extra, user_extra) if p)

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_api_option,
                options.get(CONF_PROMPT),
                combined_extra or None,
            )
        except conversation.ConverseError as err:
            LOGGER.error("ConverseError in async_provide_llm_data: %s", err)
            return err.as_conversation_result()

        model = self._select_model(user_input.text, options)
        fallback_model = options.get(CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL)
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        # Prefix username on latest user message when enabled
        if options.get(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME):
            name = self._resolve_user_name(user_input)
            if name and messages:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        content = messages[i].get("content")
                        if isinstance(content, str) and not content.startswith(
                            f"[{name}]"
                        ):
                            messages[i] = {
                                **messages[i],
                                "content": f"[{name}] {content}",
                            }
                        break

        client = self.entry.runtime_data
        live_search = options.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
        show_citations = options.get(CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS)

        # Live search via Responses API. Previously this was skipped whenever HA
        # tools were present (#26), which made Live Search a no-op for Assist users.
        use_search = bool(
            live_search
            and live_search != LIVE_SEARCH_OFF
            and (
                looks_like_search_query(user_input.text)
                or mode == MODE_CHAT_ONLY
            )
        )
        ha_tools_available = bool(chat_log.llm_api and chat_log.llm_api.tools)

        if use_search:
            search_system = [
                "You have live web/X search. Answer with current facts only.",
                "Be concise. Prefer scores, times, and concrete outcomes.",
                "If results are uncertain, say what you found and what is unknown.",
                "When the user says 'near me' / local / open now, use the home "
                "location and local time below — do not ask them for a city.",
            ]
            if options.get(CONF_PROMPT):
                search_system.append(str(options.get(CONF_PROMPT)))
            # Factual context ALWAYS reaches the search pass (#27), even with HA tools
            factual = self._build_factual_context(user_input)
            if factual:
                search_system.append(factual)
            if not ha_tools_available:
                # Persona / voice only on search-only path (avoid bloating tool path twice)
                persona = self._build_persona_context(user_input)
                if persona:
                    search_system.append(persona)
                if user_extra:
                    search_system.append(user_extra)

            search_messages = [
                m for m in messages if m.get("role") != "system"
            ]
            try:
                text, p_tok, c_tok = await async_responses_completion(
                    client,
                    model=model,
                    messages=search_messages,  # type: ignore[arg-type]
                    system_prompt="\n\n".join(search_system) or None,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    temperature=options.get(
                        CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                    ),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    live_search=live_search,
                    show_citations=bool(show_citations),
                    reasoning_effort=options.get(CONF_REASONING_EFFORT),
                )
                text = _strip_json_from_response(text)
                await self._record_usage(model, p_tok, c_tok)

                if text and not ha_tools_available:
                    # Search-only path (chat_only or no LLM HASS API)
                    async for _ in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(
                            agent_id=user_input.agent_id, content=text
                        )
                    ):
                        pass
                    intent_response = intent.IntentResponse(
                        language=user_input.language
                    )
                    intent_response.async_set_speech(text)
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                        continue_conversation=chat_log.continue_conversation,
                    )

                if text and ha_tools_available:
                    # Two-pass (#26): inject live findings into the tool loop so
                    # Assist control and live search work together.
                    brief = text.strip()
                    if len(brief) > 4000:
                        brief = brief[:4000] + "…"
                    search_note = (
                        "Live search results for this user question "
                        "(use these facts; do not claim you lack real-time data):\n"
                        f"{brief}"
                    )
                    new_messages: list[dict[str, Any]] = []
                    inserted = False
                    for msg in messages:
                        if (
                            not inserted
                            and isinstance(msg, dict)
                            and msg.get("role") == "system"
                            and isinstance(msg.get("content"), str)
                        ):
                            new_messages.append(
                                {
                                    "role": "system",
                                    "content": f"{msg['content']}\n\n{search_note}",
                                }
                            )
                            inserted = True
                        else:
                            new_messages.append(msg)
                    if not inserted:
                        new_messages.insert(
                            0, {"role": "system", "content": search_note}
                        )
                    messages = new_messages
                    LOGGER.debug(
                        "Injected live search brief (%s chars) into tool loop",
                        len(brief),
                    )
            except Exception as err:  # noqa: BLE001
                LOGGER.warning(
                    "Live search path failed (%s); continuing without search context",
                    err,
                )

        # Tool-calling chat completions loop
        models_to_try = [model]
        if fallback_model and fallback_model != model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for active_model in models_to_try:
            try:
                result = await self._tool_loop(
                    user_input=user_input,
                    chat_log=chat_log,
                    messages=list(messages),
                    model=active_model,
                    options=options,
                    client=client,
                )
                return result
            except openai.RateLimitError as err:
                last_error = err
                LOGGER.error("Rate limited by xAI on %s: %s", active_model, err)
                break
            except openai.OpenAIError as err:
                last_error = err
                LOGGER.warning(
                    "Model %s failed (%s); trying fallback if available",
                    active_model,
                    err,
                )
                continue
            except TokenLengthExceededError:
                raise
            except Exception as err:  # noqa: BLE001
                last_error = err
                LOGGER.warning("Unexpected error on %s: %s", active_model, err)
                continue

        if isinstance(last_error, openai.RateLimitError):
            raise HomeAssistantError("Rate limited or insufficient funds") from last_error
        raise HomeAssistantError(f"Error talking to xAI: {last_error}") from last_error

    async def _tool_loop(
        self,
        *,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        messages: list,
        model: str,
        options,
        client,
    ) -> conversation.ConversationResult:
        """Run chat completion tool iterations for one model."""
        for _iteration in range(MAX_TOOL_ITERATIONS):
            tools: list[dict[str, Any]] | None = None
            if chat_log.llm_api:
                tools = []
                for tool in chat_log.llm_api.tools:
                    try:
                        tools.append(
                            _format_tool(tool, chat_log.llm_api.custom_serializer)
                        )
                    except Exception as err:  # noqa: BLE001
                        LOGGER.warning(
                            "Skipping tool %s due to schema error: %s",
                            getattr(tool, "name", "?"),
                            err,
                        )
                if not tools:
                    tools = None

            try:
                result = await async_chat_completion(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    temperature=options.get(
                        CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                    ),
                    tools=tools,
                    tool_choice="auto" if tools else None,
                    reasoning_effort=options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    ),
                    user=chat_log.conversation_id,
                )
            except openai.OpenAIError:
                raise

            choice = result.choices[0]
            message = choice.message
            p_tok, c_tok = extract_usage(result)

            if getattr(message, "tool_calls", None):
                ha_tool_calls = []
                for tc in message.tool_calls:
                    tc.external = True
                    ha_tool_calls.append(tc)

                assistant_content = conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=message.content or "",
                    tool_calls=ha_tool_calls,
                )
                async for _ in chat_log.async_add_assistant_content(assistant_content):
                    pass
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                for tool_call in message.tool_calls:
                    try:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError as err:
                            tool_result = {
                                "error": f"Invalid tool arguments JSON: {err}",
                                "raw_arguments": tool_call.function.arguments,
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": self._tool_result_payload(tool_result),
                                }
                            )
                            continue
                        if not chat_log.llm_api:
                            tool_result = {
                                "error": "LLM HASS API not configured. Enable Home Assistant LLM API / Assist control in Grok Conversation options."
                            }
                        else:
                            tool = next(
                                (
                                    t
                                    for t in chat_log.llm_api.tools
                                    if t.name == tool_name
                                ),
                                None,
                            )
                            if tool is None:
                                tool_result = {
                                    "error": f"Tool {tool_name} not found"
                                }
                            else:
                                # ToolInput only accepts tool_name/tool_args;
                                # context/user/device live on LLMContext (as_llm_context).
                                tool_input = ToolInput(
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                )
                                tool_result = await tool.async_call(
                                    self.hass,
                                    tool_input,
                                    user_input.as_llm_context(DOMAIN),
                                )
                                LOGGER.debug(
                                    "Tool %s result: %s", tool_name, tool_result
                                )
                        # Always return the real tool payload. Filtering "unhelpful"
                        # results caused the model to invent successful actions (#8/#16).
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": self._tool_result_payload(tool_result),
                            }
                        )
                    except Exception as err:  # noqa: BLE001
                        LOGGER.error(
                            "Error executing tool %s: %s",
                            tool_call.function.name,
                            err,
                            exc_info=True,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": self._tool_result_payload(
                                    {
                                        "error": str(err),
                                        "tool": tool_call.function.name,
                                    }
                                ),
                            }
                        )

                await self._record_usage(model, p_tok, c_tok)
                continue

            full_response = _strip_json_from_response(message.content or "")
            if full_response:
                async for _ in chat_log.async_add_assistant_content(
                    conversation.AssistantContent(
                        agent_id=user_input.agent_id, content=full_response
                    )
                ):
                    pass
                messages.append({"role": "assistant", "content": full_response})

            if result.usage:
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": result.usage.prompt_tokens,
                            "output_tokens": result.usage.completion_tokens,
                        }
                    }
                )
            await self._record_usage(model, p_tok, c_tok)

            if choice.finish_reason == "length":
                raise TokenLengthExceededError(
                    options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
                )
            break

        intent_response = intent.IntentResponse(language=user_input.language)
        speech = self._pick_speech_content(chat_log)
        if speech:
            intent_response.async_set_speech(speech)
        else:
            intent_response.async_set_speech(
                "Sorry, I couldn't generate a response."
            )

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)
