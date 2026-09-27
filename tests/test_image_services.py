"""Acceptance tests for generate_image + vision model fixes (#35)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation import (
    _DEPRECATED_FIELD_WARNED,
    build_image_generate_kwargs,
    format_images_response,
    resolve_service_vision_model,
)
from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_VISION_MODEL,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_VISION_MODEL,
    RETIRED_VISION_MODELS,
    SERVICE_GENERATE_CONTENT,
    SERVICE_GENERATE_IMAGE,
    SERVICE_PHOTO_ANALYSIS,
    SERVICE_QUERY_IMAGE,
)


@pytest.fixture(autouse=True)
def _reset_deprecation_flags() -> None:
    """Reset one-shot warning flags between tests."""
    _DEPRECATED_FIELD_WARNED.clear()
    import custom_components.grok_conversation as mod

    mod._RETIRED_VISION_WARNED = False


def _image_item(*, url=None, b64_json=None, mime_type=None, revised_prompt=None):
    return SimpleNamespace(
        url=url,
        b64_json=b64_json,
        mime_type=mime_type,
        revised_prompt=revised_prompt,
    )


def test_build_kwargs_prompt_only_sends_defaults() -> None:
    """Prompt-only call sends n=1 + response_format=url, no size/style/quality."""
    kwargs = build_image_generate_kwargs({CONF_PROMPT: "a cat"}, "grok-imagine-image")
    assert kwargs == {
        "model": "grok-imagine-image",
        "prompt": "a cat",
        "n": 1,
        "response_format": "url",
    }
    assert "extra_body" not in kwargs
    assert "size" not in kwargs
    assert "style" not in kwargs
    assert "quality" not in kwargs


def test_build_kwargs_forwards_documented_params() -> None:
    """aspect_ratio, resolution, quality=medium, n=3 go to the API."""
    kwargs = build_image_generate_kwargs(
        {
            CONF_PROMPT: "a dog",
            "aspect_ratio": "16:9",
            "resolution": "2k",
            "quality": "medium",
            "n": 3,
            "response_format": "url",
        },
        "grok-imagine-image-2.0",
    )
    assert kwargs["n"] == 3
    assert kwargs["response_format"] == "url"
    assert kwargs["extra_body"] == {
        "aspect_ratio": "16:9",
        "resolution": "2k",
        "quality": "medium",
    }


def test_build_kwargs_legacy_size_style_quality(caplog: pytest.LogCaptureFixture) -> None:
    """Legacy size/style/quality validate; only mapped aspect_ratio is sent."""
    with caplog.at_level("WARNING"):
        kwargs = build_image_generate_kwargs(
            {
                CONF_PROMPT: "sunset",
                "size": "1792x1024",
                "style": "natural",
                "quality": "hd",
            },
            "grok-imagine-image",
        )
    assert kwargs["extra_body"] == {"aspect_ratio": "16:9"}
    assert "quality" not in kwargs["extra_body"]
    assert "size" not in kwargs
    assert "style" not in kwargs
    text = caplog.text
    assert "size" in text and "deprecated" in text
    assert "style" in text and "deprecated" in text
    assert "quality" in text and "deprecated" in text


def test_format_images_url_response() -> None:
    """url format returns top-level url + images list."""
    response = SimpleNamespace(
        data=[
            _image_item(url="https://example.com/a.png", mime_type="image/png"),
            _image_item(url="https://example.com/b.png"),
        ]
    )
    result = format_images_response(
        response, model="grok-imagine-image", response_format="url"
    )
    assert result["url"] == "https://example.com/a.png"
    assert result["model"] == "grok-imagine-image"
    assert result["images"] == [
        {"url": "https://example.com/a.png", "mime_type": "image/png"},
        {"url": "https://example.com/b.png"},
    ]
    assert "revised_prompt" not in result


def test_format_images_b64_response() -> None:
    """b64_json format returns images[].b64_json and no top-level url."""
    response = SimpleNamespace(
        data=[_image_item(b64_json="abc123", mime_type="image/jpeg")]
    )
    result = format_images_response(
        response, model="grok-imagine-image", response_format="b64_json"
    )
    assert "url" not in result
    assert result["images"] == [{"b64_json": "abc123", "mime_type": "image/jpeg"}]


def test_format_images_empty_raises() -> None:
    """Empty data raises HomeAssistantError."""
    with pytest.raises(HomeAssistantError, match="empty data"):
        format_images_response(
            SimpleNamespace(data=[]),
            model="grok-imagine-image",
            response_format="url",
        )


def test_format_images_missing_payload_raises() -> None:
    """Missing url and b64_json raises HomeAssistantError."""
    with pytest.raises(HomeAssistantError, match="missing url"):
        format_images_response(
            SimpleNamespace(data=[_image_item()]),
            model="grok-imagine-image",
            response_format="url",
        )


def test_resolve_vision_model_defaults() -> None:
    """No model → entry option → recommended default."""
    assert resolve_service_vision_model(None, {}) == RECOMMENDED_VISION_MODEL
    assert RECOMMENDED_VISION_MODEL == "grok-4.3"
    assert (
        resolve_service_vision_model(None, {CONF_VISION_MODEL: "grok-4.6"})
        == "grok-4.6"
    )
    assert resolve_service_vision_model("grok-4.5", {}) == "grok-4.5"


def test_resolve_vision_model_remaps_retired(caplog: pytest.LogCaptureFixture) -> None:
    """Per-call retired id remaps to recommended with one warning."""
    with caplog.at_level("WARNING"):
        assert (
            resolve_service_vision_model("grok-2-vision-1212", {})
            == RECOMMENDED_VISION_MODEL
        )
        assert (
            resolve_service_vision_model("grok-2-vision-latest", {})
            == RECOMMENDED_VISION_MODEL
        )
    assert caplog.text.count("retired") == 1
    assert RETIRED_VISION_MODELS == {
        "grok-2-vision-1212",
        "grok-2-vision",
        "grok-2-vision-latest",
        "grok-vision-beta",
    }


async def test_generate_image_service_call(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Service call with prompt only hits images.generate correctly."""
    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=[_image_item(url="https://cdn.example/img.png", mime_type="image/png")]
        )
    )

    response = await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        {
            "config_entry": mock_config_entry.entry_id,
            CONF_PROMPT: "a lighthouse",
        },
        blocking=True,
        return_response=True,
    )

    mock_openai_client.images.generate.assert_awaited_once()
    kwargs = mock_openai_client.images.generate.await_args.kwargs
    assert kwargs["prompt"] == "a lighthouse"
    assert kwargs["n"] == 1
    assert kwargs["response_format"] == "url"
    assert "size" not in kwargs
    assert "style" not in kwargs
    assert "quality" not in kwargs
    assert "extra_body" not in kwargs
    assert response["url"] == "https://cdn.example/img.png"
    assert response["images"][0]["url"] == "https://cdn.example/img.png"


async def test_generate_image_b64_and_params(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Documented extras are forwarded via extra_body; b64 response has no url."""
    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=[_image_item(b64_json="Zm9v", mime_type="image/png")]
        )
    )

    response = await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        {
            "config_entry": mock_config_entry.entry_id,
            CONF_PROMPT: "abstract art",
            "aspect_ratio": "9:16",
            "resolution": "2k",
            "quality": "medium",
            "n": 2,
            "response_format": "b64_json",
        },
        blocking=True,
        return_response=True,
    )

    kwargs = mock_openai_client.images.generate.await_args.kwargs
    assert kwargs["n"] == 2
    assert kwargs["response_format"] == "b64_json"
    assert kwargs["extra_body"] == {
        "aspect_ratio": "9:16",
        "resolution": "2k",
        "quality": "medium",
    }
    assert "url" not in response
    assert response["images"][0]["b64_json"] == "Zm9v"


async def test_generate_image_api_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """OpenAIError maps to HomeAssistantError."""
    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock(
        side_effect=openai.OpenAIError("boom")
    )

    with pytest.raises(HomeAssistantError, match="Error generating image"):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_GENERATE_IMAGE,
            {
                "config_entry": mock_config_entry.entry_id,
                CONF_PROMPT: "fail",
            },
            blocking=True,
            return_response=True,
        )


async def test_photo_analysis_uses_vision_default(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """photo_analysis uses recommended vision model when unset."""
    with patch(
        "custom_components.grok_conversation.async_chat_completion",
        new_callable=AsyncMock,
    ) as mock_chat:
        mock_chat.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="a plant"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )
        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_PHOTO_ANALYSIS,
            {
                "config_entry": mock_config_entry.entry_id,
                "prompt": "What is this?",
                "images": ["https://example.com/plant.jpg"],
            },
            blocking=True,
            return_response=True,
        )

    assert mock_chat.await_args.kwargs["model"] == RECOMMENDED_VISION_MODEL
    assert response["model"] == RECOMMENDED_VISION_MODEL


async def test_photo_analysis_uses_option_and_remaps_retired(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Option vision_model is used; retired per-call override remaps."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options={**dict(RECOMMENDED_OPTIONS), CONF_VISION_MODEL: "grok-4.6"},
        version=1,
        minor_version=3,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": dict(RECOMMENDED_AI_TASK_OPTIONS),
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)

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
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    entry.runtime_data = mock_openai_client

    with patch(
        "custom_components.grok_conversation.async_chat_completion",
        new_callable=AsyncMock,
    ) as mock_chat:
        mock_chat.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        resp = await hass.services.async_call(
            DOMAIN,
            SERVICE_PHOTO_ANALYSIS,
            {
                "config_entry": entry.entry_id,
                "prompt": "Describe",
                "images": ["https://example.com/a.jpg"],
            },
            blocking=True,
            return_response=True,
        )
        assert resp["model"] == "grok-4.6"

        with caplog.at_level("WARNING"):
            resp2 = await hass.services.async_call(
                DOMAIN,
                SERVICE_QUERY_IMAGE,
                {
                    "config_entry": entry.entry_id,
                    "prompt": "Describe",
                    "images": [{"url": "https://example.com/a.jpg"}],
                    "model": "grok-2-vision-1212",
                },
                blocking=True,
                return_response=True,
            )
        assert resp2["model"] == RECOMMENDED_VISION_MODEL
        assert "retired" in caplog.text


async def test_generate_content_with_image_uses_vision(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    tmp_path,
) -> None:
    """generate_content with an image file uses the vision model."""
    img = tmp_path / "snap.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    with (
        patch(
            "custom_components.grok_conversation.async_chat_completion",
            new_callable=AsyncMock,
        ) as mock_chat,
        patch(
            "homeassistant.core.Config.is_allowed_path",
            return_value=True,
        ),
        patch(
            "custom_components.grok_conversation.encode_file",
            return_value=("image/jpeg", "qq"),
        ),
    ):
        mock_chat.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="seen"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        resp = await hass.services.async_call(
            DOMAIN,
            SERVICE_GENERATE_CONTENT,
            {
                "config_entry": mock_config_entry.entry_id,
                CONF_PROMPT: "What is in this image?",
                "filenames": [str(img)],
            },
            blocking=True,
            return_response=True,
        )

    assert mock_chat.await_args.kwargs["model"] == RECOMMENDED_VISION_MODEL
    assert resp["model"] == RECOMMENDED_VISION_MODEL


async def test_migrate_entry_rewrites_retired_vision(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
) -> None:
    """Migration to minor 3 rewrites retired vision_model; leaves others alone."""
    assert await async_setup_component(hass, "homeassistant", {})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options={
            CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
            CONF_VISION_MODEL: "grok-2-vision-1212",
            "live_search": "web",
        },
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

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
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.minor_version == 3
    assert entry.options[CONF_VISION_MODEL] == RECOMMENDED_VISION_MODEL
    assert entry.options["live_search"] == "web"
    assert entry.options[CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL
    # AI Task subentry added by minor-2 migration
    assert any(s.subentry_type == "ai_task_data" for s in entry.subentries.values())


async def test_migrate_entry_without_vision_model(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
) -> None:
    """v1.1 entry without vision_model migrates cleanly to minor 3."""
    assert await async_setup_component(hass, "homeassistant", {})
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options={CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL},
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

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
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.minor_version == 3
    assert CONF_VISION_MODEL not in entry.options or entry.options.get(
        CONF_VISION_MODEL
    ) not in RETIRED_VISION_MODELS


async def test_options_flow_saves_and_rejects_retired_vision(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Options flow saves vision_model and rejects retired ids."""
    with (
        patch(
            "custom_components.grok_conversation.config_flow.async_list_chat_models",
            return_value=[RECOMMENDED_CHAT_MODEL, RECOMMENDED_VISION_MODEL, "grok-4.6"],
        ),
        patch(
            "custom_components.grok_conversation.config_flow.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ),
        patch(
            "custom_components.grok_conversation.config_flow.get_async_client",
            return_value=None,
        ),
    ):
        result = await hass.config_entries.options.async_init(
            mock_config_entry.entry_id
        )
        assert result["type"] == FlowResultType.FORM

        # Reject retired
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                CONF_RECOMMENDED: True,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_VISION_MODEL: "grok-2-vision-1212",
            },
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"][CONF_VISION_MODEL] == "model_retired"

        # Save valid
        result = await hass.config_entries.options.async_init(
            mock_config_entry.entry_id
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                CONF_RECOMMENDED: True,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_VISION_MODEL: "grok-4.6",
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_VISION_MODEL] == "grok-4.6"
