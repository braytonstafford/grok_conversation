"""Regression tests for conversation live-search helpers (v1.8.0 fixes)."""

from custom_components.grok_conversation.api_helpers import (
    looks_like_non_search_query,
    should_use_live_search,
)
from custom_components.grok_conversation.const import (
    CONF_VISION_MODEL,
    RECOMMENDED_VISION_MODEL,
)
from custom_components.grok_conversation.entity import (
    model_supports_vision,
    resolve_vision_model,
)


def test_pipeline_mode_deny_list() -> None:
    """Pipeline mode searches by default except deny-listed phrases (#30)."""
    assert should_use_live_search(
        "what's the score of the game",
        interaction_mode="pipeline",
        live_search="web",
    )
    assert not should_use_live_search(
        "tell me a joke",
        interaction_mode="pipeline",
        live_search="web",
    )
    assert looks_like_non_search_query("hello")
    assert not should_use_live_search(
        "hello", interaction_mode="pipeline", live_search="web"
    )


def test_tools_mode_allow_list_only() -> None:
    """Tools mode keeps the stricter allow-list."""
    assert should_use_live_search(
        "latest news", interaction_mode="tools", live_search="full"
    )
    assert not should_use_live_search(
        "turn on the lights", interaction_mode="tools", live_search="full"
    )


def test_chat_only_always_searches_when_enabled() -> None:
    """Chat-only always uses live search when enabled."""
    assert should_use_live_search(
        "anything at all", interaction_mode="chat_only", live_search="web"
    )
    assert not should_use_live_search(
        "anything", interaction_mode="chat_only", live_search="off"
    )


def test_vision_model_detection() -> None:
    """Current Grok 4 models accept images; retired grok-2-vision / imagine do not."""
    assert model_supports_vision("grok-4.3-latest")
    assert model_supports_vision("grok-4.6")
    assert model_supports_vision("grok-4-1-fast-non-reasoning")
    assert not model_supports_vision("grok-2-vision-1212")
    assert not model_supports_vision("grok-imagine-image")
    assert not model_supports_vision("grok-3-mini-fast")


def test_resolve_vision_model_prefers_chat_model() -> None:
    """Attachment routing keeps the subentry/chat model when it supports images."""
    assert (
        resolve_vision_model(
            {CONF_VISION_MODEL: "grok-4.3-latest"},
            "grok-4-1-fast-non-reasoning",
        )
        == "grok-4-1-fast-non-reasoning"
    )


def test_resolve_vision_model_skips_retired_vision() -> None:
    """Stored grok-2-vision-* is ignored; fall back to a current model."""
    assert (
        resolve_vision_model(
            {CONF_VISION_MODEL: "grok-2-vision-1212"},
            "grok-3-mini-fast",
        )
        == RECOMMENDED_VISION_MODEL
    )
    assert RECOMMENDED_VISION_MODEL.startswith("grok-4")
