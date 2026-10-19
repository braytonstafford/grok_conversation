"""Tests for reasoning-effort gating (#40 / #42).

Capability check is an allow-list of documented families plus explicit
``*reasoning*`` ids; ``*non-reasoning*`` is always excluded. Unknown ids
default to omit (conservative).
"""

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
        # Explicit reasoning-tagged ids (incl. retired / case variants)
        ("grok-4-fast-reasoning", True),
        ("grok-4.20-0309-reasoning", True),
        ("grok-4-1-fast-reasoning", True),
        ("Grok-4-Fast-Reasoning", True),
        ("grok-4.20-reasoning-latest", True),
        # Explicit non-reasoning — always excluded
        ("grok-4.20-0309-non-reasoning", False),
        ("grok-4-1-fast-non-reasoning", False),
        ("grok-4-fast-non-reasoning", False),
        ("GROK-4.20-0309-NON-REASONING", False),
        ("grok-4.20-non-reasoning-latest", False),
        # Suffix-less families documented to accept effort (#42)
        ("grok-4.3", True),
        ("grok-4.3-latest", True),
        ("grok-4.5", True),
        ("grok-4.5-latest", True),
        ("grok-4.6", True),
        ("grok-4.7", True),
        ("grok-4.20", True),
        ("grok-4.20-0309", True),
        ("grok-4.20-multi-agent-0309", True),
        ("grok-build-latest", True),
        # Ambiguous / unsupported → omit (conservative)
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
    """Allow-list + reasoning tag; non-reasoning and unknowns are False."""
    assert _supports_reasoning_effort(model) is expected  # type: ignore[arg-type]


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
        "grok-4-fast-reasoning",
        "grok-4.20-0309-reasoning",
        "Grok-4-Fast-Reasoning",
        "grok-4.3-latest",
        "grok-4.5",
        "grok-4.6",
        "grok-4.7",
        "grok-4.20",
        "grok-build-latest",
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
@pytest.mark.parametrize("effort", ["none", None])
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
        "grok-4-1-fast-non-reasoning",
        "GROK-4.20-0309-NON-REASONING",
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
        "grok-4-fast-reasoning",
        "grok-4.20-0309-reasoning",
        "Grok-4.20-0309-Reasoning",
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
@pytest.mark.parametrize("effort", ["none", None])
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
