"""Tests for reasoning-effort gating on reasoning vs non-reasoning model ids (#40)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.grok_conversation.api_helpers import (
    _supports_reasoning_effort,
    async_chat_completion,
    async_responses_completion,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("grok-4-fast-reasoning", True),
        ("grok-4.20-0309-reasoning", True),
        ("grok-4-1-fast-reasoning", True),
        ("Grok-4-Fast-Reasoning", True),
        ("grok-4.20-0309-non-reasoning", False),
        ("grok-4-1-fast-non-reasoning", False),
        ("grok-4-fast-non-reasoning", False),
        ("GROK-4.20-0309-NON-REASONING", False),
        ("grok-4.3", False),
        ("", False),
        (None, False),
    ],
)
def test_supports_reasoning_effort(model: str | None, expected: bool) -> None:
    """Helper is true only for reasoning ids, not non-reasoning or plain ids."""
    assert _supports_reasoning_effort(model) is expected  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "grok-4.20-0309-non-reasoning",
        "grok-4-1-fast-non-reasoning",
        "GROK-4-1-FAST-NON-REASONING",
    ],
)
async def test_chat_completion_omits_reasoning_for_non_reasoning(
    model: str,
) -> None:
    """Chat Completions must not send reasoning_effort for non-reasoning ids."""
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
        "grok-4-fast-reasoning",
        "grok-4.20-0309-reasoning",
        "Grok-4-Fast-Reasoning",
    ],
)
async def test_chat_completion_includes_reasoning_for_reasoning(
    model: str,
) -> None:
    """Chat Completions must send reasoning_effort for reasoning ids."""
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
@pytest.mark.parametrize("effort", ["none", None])
async def test_chat_completion_skips_effort_none(effort: str | None) -> None:
    """Effort none/None is never sent, even on a reasoning model."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock())

    await async_chat_completion(
        client,
        model="grok-4-fast-reasoning",
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
        "grok-4-1-fast-non-reasoning",
        "GROK-4.20-0309-NON-REASONING",
    ],
)
async def test_responses_omits_reasoning_for_non_reasoning(model: str) -> None:
    """Responses API must not send reasoning for non-reasoning ids."""
    client = MagicMock()
    response = MagicMock()
    response.output_text = "ok"
    response.citations = None
    response.output = []
    response.usage = MagicMock(input_tokens=1, output_tokens=1, prompt_tokens=None, completion_tokens=None)
    client.responses.create = AsyncMock(return_value=response)

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
        "grok-4-fast-reasoning",
        "grok-4.20-0309-reasoning",
        "Grok-4.20-0309-Reasoning",
    ],
)
async def test_responses_includes_reasoning_for_reasoning(model: str) -> None:
    """Responses API must send reasoning.effort for reasoning ids."""
    client = MagicMock()
    response = MagicMock()
    response.output_text = "ok"
    response.citations = None
    response.output = []
    response.usage = MagicMock(input_tokens=1, output_tokens=1, prompt_tokens=None, completion_tokens=None)
    client.responses.create = AsyncMock(return_value=response)

    await async_responses_completion(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )

    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", None])
async def test_responses_skips_effort_none(effort: str | None) -> None:
    """Effort none/None is never sent on the Responses path either."""
    client = MagicMock()
    response = MagicMock()
    response.output_text = "ok"
    response.citations = None
    response.output = []
    response.usage = MagicMock(input_tokens=1, output_tokens=1, prompt_tokens=None, completion_tokens=None)
    client.responses.create = AsyncMock(return_value=response)

    await async_responses_completion(
        client,
        model="grok-4-fast-reasoning",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )

    kwargs = client.responses.create.await_args.kwargs
    assert "reasoning" not in kwargs
