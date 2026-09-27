"""Regression tests for conversation live-search helpers (v1.8.0 fixes)."""

import pytest

from custom_components.grok_conversation.api_helpers import (
    looks_like_non_search_query,
    should_use_live_search,
)
from custom_components.grok_conversation.const import (
    CONF_VISION_MODEL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_VISION_MODEL,
    _RETIRED_CHAT_WARNED,
    remap_retired_chat_model,
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
    assert not model_supports_vision("text-only-custom")


def test_resolve_vision_model_prefers_chat_model() -> None:
    """Attachment routing keeps the subentry/chat model when it supports images."""
    assert (
        resolve_vision_model(
            {CONF_VISION_MODEL: "grok-4.3"},
            "grok-4.6",
        )
        == "grok-4.6"
    )


def test_resolve_vision_model_falls_back_when_chat_text_only() -> None:
    """When chat cannot take images, use configured vision_model."""
    assert (
        resolve_vision_model(
            {CONF_VISION_MODEL: "grok-4.6"},
            "text-only-custom",
        )
        == "grok-4.6"
    )


def test_resolve_vision_model_skips_retired_vision() -> None:
    """Stored grok-2-vision-* is ignored; fall back to a current model."""
    assert (
        resolve_vision_model(
            {CONF_VISION_MODEL: "grok-2-vision-1212"},
            "text-only-custom",
        )
        == RECOMMENDED_VISION_MODEL
    )
    assert RECOMMENDED_VISION_MODEL.startswith("grok-4")


def test_resolve_vision_model_remaps_retired_fast_chat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retired fast chat model is remapped before vision preference."""
    _RETIRED_CHAT_WARNED.clear()
    with caplog.at_level("WARNING"):
        assert (
            resolve_vision_model(
                {CONF_VISION_MODEL: "grok-4.6"},
                "grok-4-1-fast-non-reasoning",
            )
            == RECOMMENDED_CHAT_MODEL
        )
    assert "retired" in caplog.text.lower()


def test_remap_retired_fast_and_fallback_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runtime remap for May 15 retired fast/fallback defaults."""
    from custom_components.grok_conversation.const import (
        RECOMMENDED_FALLBACK_MODEL,
        RECOMMENDED_FAST_MODEL,
    )

    _RETIRED_CHAT_WARNED.clear()
    with caplog.at_level("WARNING"):
        assert (
            remap_retired_chat_model(
                "grok-4-1-fast-non-reasoning", RECOMMENDED_FAST_MODEL
            )
            == RECOMMENDED_FAST_MODEL
        )
        assert (
            remap_retired_chat_model("grok-3-mini-fast", RECOMMENDED_FALLBACK_MODEL)
            == RECOMMENDED_FALLBACK_MODEL
        )
    assert RECOMMENDED_FAST_MODEL == "grok-4.3"
    assert RECOMMENDED_FALLBACK_MODEL == "grok-4.3"
    assert "grok-4-1-fast-non-reasoning" in caplog.text
    assert "grok-3-mini-fast" in caplog.text
    # Current models pass through
    assert remap_retired_chat_model("grok-4.6", RECOMMENDED_FAST_MODEL) == "grok-4.6"
