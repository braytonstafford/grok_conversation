"""Tests for reasoning-effort gating (#40 / #42 / PR #44).

Capability check is an allow-list of documented families (4.3/4.5/4.6/4.7,
``grok-build-latest``, ``*multi-agent*``). ``*non-reasoning*`` and everything
else (including ``*-reasoning`` tags without a documented effort list) omit
the parameter. Unset / ``none`` never sends.
"""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import Context, HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.api_helpers import (
    _should_send_reasoning_effort,
    _supports_reasoning_effort,
    async_chat_completion,
    async_responses_completion,
)
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_INTERACTION_MODE,
    CONF_LIVE_SEARCH,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    RECOMMENDED_REASONING_EFFORT,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Documented suffix-less families
        ("grok-4.3", True),
        ("grok-4.3-latest", True),
        ("  grok-4.3  ", True),
        ("grok-4.5", True),
        ("grok-4.5-latest", True),
        ("grok-4.6", True),
        ("grok-4.7", True),
        ("grok-build-latest", True),
        ("grok-4.20-multi-agent-0309", True),
        ("Grok-4.3-Latest", True),
        # Prefix boundary: must not match longer dotted minors
        ("grok-4.30", False),
        ("grok-4.50", False),
        ("grok-4.200", False),
        ("grok-4.3x", False),
        # Explicit non-reasoning — always excluded (including ids that would
        # otherwise match an allow-list prefix or multi-agent)
        ("grok-4.20-0309-non-reasoning", False),
        ("grok-4-1-fast-non-reasoning", False),
        ("grok-4-fast-non-reasoning", False),
        ("GROK-4.20-0309-NON-REASONING", False),
        ("grok-4.20-non-reasoning-latest", False),
        ("grok-4.20-multi-agent-non-reasoning", False),
        ("grok-4.3-non-reasoning", False),
        ("grok-4.5-non-reasoning", False),
        # Reasoning-tagged but no documented effort list → omit
        ("grok-4-fast-reasoning", False),
        ("grok-4.20-0309-reasoning", False),
        ("grok-4-1-fast-reasoning", False),
        ("Grok-4-Fast-Reasoning", False),
        ("grok-4.20-reasoning-latest", False),
        ("grok-4.20", False),
        ("grok-4.20-0309", False),
        # Ambiguous / unsupported → omit
        ("grok-4", False),
        ("grok-4-latest", False),
        ("grok-3-mini", False),
        ("grok-3-mini-fast", False),
        ("grok-3", False),
        ("grok-build-0.1", False),
        ("grok-code-fast-1", False),
        ("grok-imagine-image", False),
        ("totally-unknown-model", False),
        ("grok-5", False),
        ("", False),
        (None, False),
    ],
)
def test_supports_reasoning_effort(model: str | None, expected: bool) -> None:
    """Allow-list + multi-agent; non-reasoning / undocumented / unknowns False."""
    assert _supports_reasoning_effort(model) is expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("grok-4.3-latest", "low", True),
        ("grok-4.3-latest", "MEDIUM", True),
        ("grok-4.3-latest", "none", False),
        ("grok-4.3-latest", "NONE", False),
        ("grok-4.3-latest", " None ", False),
        ("grok-4.3-latest", None, False),
        ("grok-4.3-latest", "", False),
        ("grok-4.20-0309-reasoning", "high", False),
        ("grok-4-fast-reasoning", "low", False),
        ("grok-4.20-0309-non-reasoning", "low", False),
    ],
)
def test_should_send_reasoning_effort(
    model: str, effort: str | None, expected: bool
) -> None:
    """Send only explicit effort on documented families; none is case-insensitive."""
    assert _should_send_reasoning_effort(model, effort) is expected


def test_recommended_reasoning_effort_is_provider_default() -> None:
    """Recommended / options-flow default is stored none (not sent)."""
    assert RECOMMENDED_REASONING_EFFORT == "none"


def _mock_responses_client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.output_text = "ok"
    response.citations = None
    response.output = []
    response.usage = MagicMock(
        input_tokens=1,
        output_tokens=1,
        prompt_tokens=None,
        completion_tokens=None,
    )
    client.responses.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "grok-4.20-0309-non-reasoning",
        "grok-4-1-fast-non-reasoning",
        "GROK-4-1-FAST-NON-REASONING",
        "grok-4.20-0309-reasoning",
        "grok-4-fast-reasoning",
        "grok-3-mini",
        "grok-4",
        "grok-build-0.1",
        "totally-unknown-model",
    ],
)
async def test_chat_completion_omits_reasoning_for_unsupported(
    model: str,
) -> None:
    """Chat Completions must not send reasoning_effort for unsupported ids."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock())

    await async_chat_completion(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="low",
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert kwargs["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "grok-4.3-latest",
        "grok-4.5",
        "grok-4.6",
        "grok-4.7",
        "grok-build-latest",
        "grok-4.20-multi-agent-0309",
    ],
)
async def test_chat_completion_includes_reasoning_for_supported(
    model: str,
) -> None:
    """Chat Completions must send reasoning_effort for supported ids."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock())

    await async_chat_completion(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="medium",
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_chat_completion_normalizes_effort_case_and_whitespace() -> None:
    """Chat Completions send strip().lower() of the effort value."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock())

    await async_chat_completion(
        client,
        model="grok-4.3-latest",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=" HIGH ",
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "NONE", " None ", None])
async def test_chat_completion_skips_effort_none(effort: str | None) -> None:
    """Effort none/None is never sent, even on a supported model."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock())

    await async_chat_completion(
        client,
        model="grok-4.3-latest",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-4-fast-reasoning",
        "grok-3-mini",
        "grok-4",
        "totally-unknown-model",
    ],
)
async def test_responses_omits_reasoning_for_unsupported(model: str) -> None:
    """Responses API must not send reasoning for unsupported ids."""
    client = _mock_responses_client()

    await async_responses_completion(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="low",
    )

    kwargs = client.responses.create.await_args.kwargs
    assert "reasoning" not in kwargs
    assert kwargs["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "grok-4.3-latest",
        "grok-4.5",
        "grok-4.6",
        "grok-4.7",
        "grok-4.20-multi-agent-0309",
    ],
)
async def test_responses_includes_reasoning_for_supported(model: str) -> None:
    """Responses API must send reasoning.effort for supported ids."""
    client = _mock_responses_client()

    await async_responses_completion(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )

    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_responses_normalizes_effort_case_and_whitespace() -> None:
    """Responses API send strip().lower() of the effort value."""
    client = _mock_responses_client()

    await async_responses_completion(
        client,
        model="grok-4.7",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=" HIGH ",
    )

    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "NONE", None])
async def test_responses_skips_effort_none(effort: str | None) -> None:
    """Effort none/None is never sent on the Responses path either."""
    client = _mock_responses_client()

    await async_responses_completion(
        client,
        model="grok-4.7",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )

    kwargs = client.responses.create.await_args.kwargs
    assert "reasoning" not in kwargs


# ---------------------------------------------------------------------------
# Call-site tests: entity chat path + conversation search path (#44 review)
# ---------------------------------------------------------------------------


def _completion_response(text: str = "ok"):
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 1
    result = MagicMock()
    result.choices = [choice]
    result.usage = usage
    result.model = "grok-4.3-latest"
    return result


def _conversation_entity(hass: HomeAssistant, entry: MockConfigEntry):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


@pytest.mark.asyncio
async def test_entity_chat_omits_effort_when_unset(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Main chat path: unset option => provider default (not sent) on supported model."""
    entity = _conversation_entity(hass, mock_config_entry)
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("hi")
    )
    chat_log = ChatLog(hass=hass, conversation_id="conv-effort-unset")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))

    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
            # CONF_REASONING_EFFORT intentionally absent
        },
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_entity_chat_sends_explicit_effort_on_supported(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Main chat path: explicit medium => sent on supported model."""
    entity = _conversation_entity(hass, mock_config_entry)
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("hi")
    )
    chat_log = ChatLog(hass=hass, conversation_id="conv-effort-set")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))

    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
            CONF_REASONING_EFFORT: "medium",
        },
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_entity_chat_never_sends_effort_on_unsupported(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Main chat path: unsupported model never receives effort even if set."""
    entity = _conversation_entity(hass, mock_config_entry)
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("hi")
    )
    chat_log = ChatLog(hass=hass, conversation_id="conv-effort-unsupported")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))

    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.20-0309-reasoning",
        options={
            CONF_CHAT_MODEL: "grok-4.20-0309-reasoning",
            CONF_FALLBACK_MODEL: "grok-4.6",
            CONF_REASONING_EFFORT: "high",
        },
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs


def _set_entry_options(entry: MockConfigEntry, **updates: object) -> None:
    """Replace entry.options without triggering a config-entry reload."""
    merged = {**dict(entry.options), **updates}
    # Drop keys explicitly set to a sentinel? Callers omit CONF_REASONING_EFFORT
    # by not including it; if present in RECOMMENDED_OPTIONS it stays unless
    # popped below.
    object.__setattr__(entry, "options", MappingProxyType(merged))


def _mock_search_response(text: str = "latest score is 3-1") -> MagicMock:
    response = MagicMock()
    response.output_text = text
    response.citations = None
    response.output = []
    response.usage = MagicMock(
        input_tokens=1,
        output_tokens=1,
        prompt_tokens=None,
        completion_tokens=None,
    )
    return response


@pytest.mark.asyncio
async def test_search_pass_omits_effort_when_unset(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Live-search pass: unset option => not sent on supported model."""
    opts = {
        **dict(mock_config_entry.options),
        CONF_INTERACTION_MODE: "chat_only",
        CONF_LIVE_SEARCH: "web",
        CONF_CHAT_MODEL: "grok-4.3-latest",
        CONF_RECOMMENDED: True,
    }
    opts.pop(CONF_REASONING_EFFORT, None)
    object.__setattr__(mock_config_entry, "options", MappingProxyType(opts))

    mock_openai_client.responses.create = AsyncMock(
        return_value=_mock_search_response()
    )

    entity = _conversation_entity(hass, mock_config_entry)
    chat_log = ChatLog(hass=hass, conversation_id="conv-search-unset")
    user_input = conversation.ConversationInput(
        text="What is the latest news?",
        context=Context(),
        conversation_id="conv-search-unset",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entity_id,
    )

    with patch.object(entity, "_select_model", return_value="grok-4.3-latest"):
        result = await entity._async_handle_message_inner(  # noqa: SLF001
            user_input, chat_log
        )

    assert result is not None
    kwargs = mock_openai_client.responses.create.await_args.kwargs
    assert "reasoning" not in kwargs


@pytest.mark.asyncio
async def test_search_pass_sends_explicit_effort_on_supported(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Live-search pass: explicit low => sent on supported model."""
    _set_entry_options(
        mock_config_entry,
        **{
            CONF_INTERACTION_MODE: "chat_only",
            CONF_LIVE_SEARCH: "web",
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_REASONING_EFFORT: "low",
            CONF_RECOMMENDED: False,
        },
    )

    mock_openai_client.responses.create = AsyncMock(
        return_value=_mock_search_response()
    )

    entity = _conversation_entity(hass, mock_config_entry)
    chat_log = ChatLog(hass=hass, conversation_id="conv-search-set")
    user_input = conversation.ConversationInput(
        text="What is the latest news?",
        context=Context(),
        conversation_id="conv-search-set",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entity_id,
    )

    with patch.object(entity, "_select_model", return_value="grok-4.3-latest"):
        result = await entity._async_handle_message_inner(  # noqa: SLF001
            user_input, chat_log
        )

    assert result is not None
    kwargs = mock_openai_client.responses.create.await_args.kwargs
    assert kwargs["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_search_pass_never_sends_effort_on_unsupported(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Live-search pass: unsupported model never receives effort."""
    _set_entry_options(
        mock_config_entry,
        **{
            CONF_INTERACTION_MODE: "chat_only",
            CONF_LIVE_SEARCH: "web",
            CONF_CHAT_MODEL: "grok-4.20-0309-reasoning",
            CONF_REASONING_EFFORT: "high",
            CONF_RECOMMENDED: False,
        },
    )

    mock_openai_client.responses.create = AsyncMock(
        return_value=_mock_search_response()
    )

    entity = _conversation_entity(hass, mock_config_entry)
    chat_log = ChatLog(hass=hass, conversation_id="conv-search-unsupported")
    user_input = conversation.ConversationInput(
        text="What is the latest news?",
        context=Context(),
        conversation_id="conv-search-unsupported",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entity_id,
    )

    with patch.object(
        entity, "_select_model", return_value="grok-4.20-0309-reasoning"
    ):
        result = await entity._async_handle_message_inner(  # noqa: SLF001
            user_input, chat_log
        )

    assert result is not None
    kwargs = mock_openai_client.responses.create.await_args.kwargs
    assert "reasoning" not in kwargs
