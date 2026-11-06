"""Conversation regression tests for the shared LLM tool loop (#34)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_INTERACTION_MODE,
    DOMAIN,
    MODE_CHAT_ONLY,
    MODE_PIPELINE,
    MODE_TOOLS,
)

# HA 2026.10+; absent on 2026.7–2026.9.
ToolResult = getattr(llm, "ToolResult", None)


def _completion_response(text: str = "", *, tool_calls=None, model: str | None = None):
    """Build a minimal chat.completions-like response."""
    message = MagicMock()
    message.content = text
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 11
    usage.completion_tokens = 7
    result = MagicMock()
    result.choices = [choice]
    result.usage = usage
    result.model = model
    return result


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class _RecordingTool(llm.Tool):
    """Test tool that records calls."""

    name = "test_light"
    description = "Control a test light"
    parameters = vol.Schema({vol.Required("action"): str}, extra=vol.ALLOW_EXTRA)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> Any:
        self.calls.append(dict(tool_input.tool_args))
        data = {"ok": True, "action": tool_input.tool_args.get("action")}
        if ToolResult is not None:
            return ToolResult(data=data)
        return data


def _mock_llm_api(hass: HomeAssistant, tool: llm.Tool) -> llm.APIInstance:
    api = MagicMock(spec=llm.API)
    api.hass = hass
    api.id = "test"
    api.name = "Test"
    return llm.APIInstance(
        api=api,
        api_prompt="Test tools available.",
        llm_context=llm.LLMContext(
            platform=DOMAIN,
            context=Context(),
            language="en",
            assistant="conversation",
            device_id=None,
        ),
        tools=[tool],
    )


def _conversation_entity(hass: HomeAssistant, entry: MockConfigEntry):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def test_conversation_tool_loop_executes_tool(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Tool calls from the model execute through the shared chat-log loop."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            _completion_response("The light is on."),
        ]
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-tools")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert tool.calls == [{"action": "turn_on"}]
    assert mock_openai_client.chat.completions.create.await_count == 2
    assert any(
        isinstance(c, conversation.AssistantContent) and c.content == "The light is on."
        for c in chat_log.content
    )


async def test_conversation_fallback_model_on_primary_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Primary OpenAIError falls through to the fallback model."""
    entity = _conversation_entity(hass, mock_config_entry)

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            openai.APIError(
                message="primary down",
                request=MagicMock(),
                body=None,
            ),
            _completion_response("Fallback answered."),
        ]
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-fallback")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert mock_openai_client.chat.completions.create.await_count == 2
    first_kwargs = mock_openai_client.chat.completions.create.await_args_list[0].kwargs
    second_kwargs = mock_openai_client.chat.completions.create.await_args_list[1].kwargs
    assert first_kwargs["model"] == "grok-4.3-latest"
    assert second_kwargs["model"] == "grok-4.6"
    assert any(
        isinstance(c, conversation.AssistantContent)
        and c.content == "Fallback answered."
        for c in chat_log.content
    )


async def test_conversation_fallback_does_not_inherit_partial_tools(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Fallback request must not include the failed model's partial tool turns."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            openai.APIError(
                message="boom after tool",
                request=MagicMock(),
                body=None,
            ),
            _completion_response("Recovered without prior tools."),
        ]
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-partial")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    # Tool ran once on the primary attempt
    assert tool.calls == [{"action": "turn_on"}]
    # Third call is the fallback's first request — must be clean (no tool roles)
    fallback_kwargs = mock_openai_client.chat.completions.create.await_args_list[2].kwargs
    assert fallback_kwargs["model"] == "grok-4.6"
    roles = [m.get("role") for m in fallback_kwargs["messages"]]
    assert "tool" not in roles
    assert roles.count("assistant") == 0 or all(
        not m.get("tool_calls")
        for m in fallback_kwargs["messages"]
        if m.get("role") == "assistant"
    )
    assert roles == ["system", "user"]


async def test_invalid_tool_argument_json_returns_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Malformed tool-argument JSON is returned as a tool error, not executed."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                tool_calls=[_tool_call("test_light", "{not-json")]
            ),
            _completion_response("I could not parse that tool call."),
        ]
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-bad-json")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(conversation.UserContent(content="Toggle light"))

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Toggle light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={CONF_CHAT_MODEL: "grok-4.3-latest"},
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="",
    )

    assert tool.calls == []
    # Second request should include a tool message with the JSON error
    second_kwargs = mock_openai_client.chat.completions.create.await_args_list[1].kwargs
    tool_msgs = [m for m in second_kwargs["messages"] if m.get("role") == "tool"]
    assert tool_msgs
    assert "Invalid tool arguments JSON" in tool_msgs[0]["content"]


CONTROL = conversation.ConversationEntityFeature.CONTROL
NO_CONTROL = conversation.ConversationEntityFeature(0)


class _NamedAPI(llm.API):
    """Minimal registered LLM API for multi-API control tests."""

    def __init__(self, hass: HomeAssistant, api_id: str, name: str) -> None:
        super().__init__(hass=hass, id=api_id, name=name)

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt="",
            llm_context=llm_context,
            tools=[],
        )


def _supported_features(
    hass: HomeAssistant, entry: MockConfigEntry
) -> conversation.ConversationEntityFeature:
    entity = _conversation_entity(hass, entry)
    state = hass.states.get(entity.entity_id)
    assert state is not None
    return state.attributes["supported_features"]


async def _reload_with_options(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    **updates: object,
) -> None:
    """Update options and reload so _update_control_feature re-runs."""
    hass.config_entries.async_update_entry(
        entry, options={**dict(entry.options), **updates}
    )
    with (
        patch(
            "custom_components.grok_conversation.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
        patch(
            "custom_components.grok_conversation.get_async_client",
            return_value=None,
        ),
    ):
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
    entry.runtime_data = mock_openai_client


async def test_control_feature_with_registered_assist_api(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Tools mode + registered assist API advertises CONTROL (#49)."""
    assert llm.LLM_API_ASSIST in {api.id for api in llm.async_get_apis(hass)}
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == CONTROL


async def test_control_feature_absent_without_llm_api(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Tools mode without llm_hass_api does not advertise CONTROL."""
    assert mock_config_entry.options.get(CONF_INTERACTION_MODE) == MODE_TOOLS
    assert CONF_LLM_HASS_API not in mock_config_entry.options
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL


async def test_control_feature_invalid_api_not_advertised(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Configured but unregistered API ids do not advertise CONTROL."""
    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: ["not-a-registered-api"],
        },
    )
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL


async def test_control_feature_chat_only_not_advertised(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Chat-only mode never advertises CONTROL even with assist selected."""
    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL


async def test_control_feature_multiple_apis_any_registered(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """CONTROL when any configured API is registered (multi-select)."""
    unregister = llm.async_register_api(hass, _NamedAPI(hass, "custom_tools", "Custom"))
    try:
        await _reload_with_options(
            hass,
            mock_config_entry,
            mock_openai_client,
            **{
                CONF_INTERACTION_MODE: MODE_TOOLS,
                CONF_LLM_HASS_API: ["missing-api", "custom_tools"],
            },
        )
        assert _supported_features(hass, mock_config_entry) == CONTROL

        await _reload_with_options(
            hass,
            mock_config_entry,
            mock_openai_client,
            **{
                CONF_INTERACTION_MODE: MODE_TOOLS,
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST, "custom_tools"],
            },
        )
        assert _supported_features(hass, mock_config_entry) == CONTROL
    finally:
        unregister()


async def test_control_feature_options_update_transitions(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Options reload transitions set and clear CONTROL appropriately."""
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_PIPELINE,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: ["still-missing"],
        },
    )
    assert _supported_features(hass, mock_config_entry) == NO_CONTROL

    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == CONTROL


async def test_control_feature_does_not_break_tool_calling(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Advertising CONTROL must not change the shared chat-log tool loop."""
    await _reload_with_options(
        hass,
        mock_config_entry,
        mock_openai_client,
        **{
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )
    assert _supported_features(hass, mock_config_entry) == CONTROL

    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            _completion_response("The light is on."),
        ]
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-control-tools")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert tool.calls == [{"action": "turn_on"}]
    assert mock_openai_client.chat.completions.create.await_count == 2
