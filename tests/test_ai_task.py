"""Tests for AI Task platform."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task, conversation
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, selector
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

try:
    import probatio
except ImportError:  # pragma: no cover
    probatio = None

from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_IMAGE_MODEL,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
)


def _structure_schema(fields: dict):
    """Build a structure schema compatible with the installed HA version."""
    if probatio is not None:
        mapped = {}
        for key, value in fields.items():
            if isinstance(key, vol.Required):
                mapped[probatio.Required(key.schema)] = value
            elif isinstance(key, vol.Optional):
                mapped[probatio.Optional(key.schema)] = value
            else:
                mapped[key] = value
        return probatio.Schema(mapped)
    return vol.Schema(fields)


def _completion_response(text: str, *, tool_calls=None):
    """Build a minimal chat.completions-like response."""
    message = MagicMock()
    message.content = text
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    result = MagicMock()
    result.choices = [choice]
    result.usage = usage
    return result


@pytest.fixture
def ai_task_entity_id(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> str:
    """Return the AI Task entity id created for the default subentry."""
    registry = er.async_get(hass)
    entities = [
        e
        for e in registry.entities.values()
        if e.domain == "ai_task" and e.config_entry_id == mock_config_entry.entry_id
    ]
    assert entities, "Expected an ai_task entity from the default subentry"
    return entities[0].entity_id


async def test_setup_creates_ai_task_entity(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    ai_task_entity_id: str,
) -> None:
    """Setting up the entry creates the ai_task entity from the default subentry."""
    assert ai_task_entity_id == "ai_task.grok_ai_task"

    state = hass.states.get(ai_task_entity_id)
    assert state is not None
    # has_entity_name + _attr_name=None → friendly name is the device title only
    assert state.name == "Grok AI Task"

    registry = er.async_get(hass)
    entry = registry.async_get(ai_task_entity_id)
    assert entry is not None
    assert entry.config_entry_id == mock_config_entry.entry_id
    assert entry.config_subentry_id is not None
    assert entry.unique_id == entry.config_subentry_id
    assert entry.entity_id == "ai_task.grok_ai_task"


async def test_migrate_adds_ai_task_subentry(
    hass: HomeAssistant, mock_openai_client: MagicMock
) -> None:
    """Migrating a v1.1 entry adds the ai_task_data subentry."""
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

    # Preserve a conversation entity unique_id across migration
    registry = er.async_get(hass)
    registry.async_get_or_create(
        "conversation",
        DOMAIN,
        entry.entry_id,
        config_entry=entry,
        suggested_object_id="grok",
    )
    conv_before = registry.async_get_entity_id(
        "conversation", DOMAIN, entry.entry_id
    )
    options_before = dict(entry.options)

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
    assert any(
        s.subentry_type == "ai_task_data" for s in entry.subentries.values()
    )
    assert dict(entry.options) == options_before
    assert (
        registry.async_get_entity_id("conversation", DOMAIN, entry.entry_id)
        == conv_before
    )


async def test_generate_data_free_text(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Free-text generate_data returns a string."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("The driveway is clear.")
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="Driveway",
        entity_id=ai_task_entity_id,
        instructions="Describe the driveway",
    )

    assert result.data == "The driveway is clear."
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "response_format" not in call_kwargs


async def test_generate_data_structured(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Structured generate_data sends json_schema response_format and returns dict."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response('{"cars": 2}')
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="Driveway check",
        entity_id=ai_task_entity_id,
        instructions="How many cars?",
        structure=_structure_schema(
            {
                vol.Required("cars"): selector.selector({"number": {}}),
                vol.Optional("notes"): selector.selector({"text": {}}),
            }
        ),
    )

    assert result.data == {"cars": 2}
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["strict"] is True
    assert call_kwargs["response_format"]["json_schema"]["name"] == "driveway_check"
    schema = call_kwargs["response_format"]["json_schema"]["schema"]
    # Strict mode: every property (including optional) must be required;
    # optional fields are nullable.
    assert set(schema["required"]) == {"cars", "notes"}
    assert "null" in schema["properties"]["notes"]["type"]
    assert call_kwargs["max_tokens"] >= 2500


async def test_generate_data_records_usage(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """AI Task token usage lands in UsageTracker / usage sensors."""
    from custom_components.grok_conversation.const import DOMAIN

    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("Counted.")
    )

    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    before = tracker.snapshot.request_count

    await ai_task.async_generate_data(
        hass,
        task_name="Usage",
        entity_id=ai_task_entity_id,
        instructions="Count",
    )
    await hass.async_block_till_done()

    assert tracker.snapshot.request_count == before + 1
    assert tracker.snapshot.prompt_tokens >= 10
    assert tracker.snapshot.completion_tokens >= 5
    assert "ai_task" in tracker.snapshot.by_service
    assert tracker.snapshot.by_service["ai_task"]["request_count"] >= 1

    # Sensor entity_id is derived from device + name; find via registry.
    registry = er.async_get(hass)
    prompt_entities = [
        e
        for e in registry.entities.values()
        if e.domain == "sensor"
        and e.config_entry_id == mock_config_entry.entry_id
        and e.unique_id.endswith("_prompt_tokens")
    ]
    assert prompt_entities
    prompt_state = hass.states.get(prompt_entities[0].entity_id)
    assert prompt_state is not None
    assert float(prompt_state.state) >= 10


async def test_generate_data_invalid_json(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Invalid JSON raises HomeAssistantError."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("NOT JSON")
    )

    with pytest.raises(HomeAssistantError, match="Error with Grok structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="Driveway check",
            entity_id=ai_task_entity_id,
            instructions="How many cars?",
            structure=_structure_schema(
                {vol.Required("cars"): selector.selector({"number": {}})}
            ),
        )


async def test_generate_data_image_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Image attachments become image_url content parts."""
    image_path = tmp_path / "driveway.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_completion_response("Two cars")
    )

    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=MagicMock(
                url="http://example.com/driveway.jpg",
                mime_type="image/jpeg",
                path=image_path,
            ),
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Driveway",
            entity_id=ai_task_entity_id,
            instructions="How many cars?",
            attachments=[
                {
                    "media_content_id": "media-source://media/driveway.jpg",
                    "media_content_type": "image/jpeg",
                }
            ],
        )

    assert result.data == "Two cars"
    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == RECOMMENDED_CHAT_MODEL
    messages = kwargs["messages"]
    user_msg = next(m for m in messages if m.get("role") == "user")
    assert isinstance(user_msg["content"], list)
    assert any(part.get("type") == "image_url" for part in user_msg["content"])


async def test_generate_data_non_image_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Non-image attachments raise HomeAssistantError."""
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=MagicMock(
                url="http://example.com/notes.pdf",
                mime_type="application/pdf",
                path=pdf_path,
            ),
        ),
        pytest.raises(HomeAssistantError, match="Only images are supported"),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Docs",
            entity_id=ai_task_entity_id,
            instructions="Summarize",
            attachments=[
                {
                    "media_content_id": "media-source://media/notes.pdf",
                    "media_content_type": "application/pdf",
                }
            ],
        )


async def test_generate_data_api_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """xAI API errors map to HomeAssistantError."""
    import openai

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=openai.APIError(
            message="boom",
            request=MagicMock(),
            body=None,
        )
    )

    with pytest.raises(HomeAssistantError, match="Error talking to xAI"):
        await ai_task.async_generate_data(
            hass,
            task_name="Fail",
            entity_id=ai_task_entity_id,
            instructions="Anything",
        )


def _ai_task_entity(hass: HomeAssistant, entity_id: str):
    """Return the AI Task entity instance."""
    from homeassistant.components.ai_task.const import DATA_COMPONENT

    entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
    assert entity is not None
    return entity


def _ai_task_features(hass: HomeAssistant, entity_id: str):
    """Return supported features for the AI Task entity."""
    return _ai_task_entity(hass, entity_id).supported_features


async def test_ai_task_entity_supports_generate_image(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    ai_task_entity_id: str,
) -> None:
    """AI Task entity advertises GENERATE_IMAGE + SUPPORT_ATTACHMENTS."""
    features = _ai_task_features(hass, ai_task_entity_id)
    generate_image = getattr(ai_task.AITaskEntityFeature, "GENERATE_IMAGE", None)
    support_attachments = getattr(
        ai_task.AITaskEntityFeature, "SUPPORT_ATTACHMENTS", None
    )
    assert generate_image is not None
    assert support_attachments is not None
    assert generate_image in features
    assert support_attachments in features
    assert ai_task.AITaskEntityFeature.GENERATE_DATA in features


def _fake_chat_log():
    """Minimal chat log stand-in for direct _async_generate_image calls."""
    log = MagicMock()
    log.conversation_id = "conv-test"
    log.async_add_assistant_content_without_tools = MagicMock()
    return log


async def test_generate_image_text_only(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """ai_task.generate_image without attachments uses images.generate."""
    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"img-bytes").decode(),
                    url=None,
                    mime_type="image/png",
                    revised_prompt=None,
                )
            ]
        )
    )
    mock_openai_client.post = AsyncMock()

    entity = _ai_task_entity(hass, ai_task_entity_id)
    result = await entity._async_generate_image(
        ai_task.GenImageTask(
            name="Sky",
            instructions="A blue sky",
        ),
        _fake_chat_log(),
    )

    mock_openai_client.images.generate.assert_awaited_once()
    mock_openai_client.post.assert_not_called()
    kwargs = mock_openai_client.images.generate.await_args.kwargs
    assert kwargs["prompt"] == "A blue sky"
    assert kwargs["response_format"] == "b64_json"
    assert "image" not in kwargs
    assert result.image_data == b"img-bytes"
    assert result.mime_type == "image/png"


async def test_generate_image_with_attachment_posts_edits(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Reference attachment → POST /images/edits with data URI on edit model."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=3,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": {
                    **dict(RECOMMENDED_AI_TASK_OPTIONS),
                    CONF_IMAGE_MODEL: "grok-imagine-image-2.0",
                },
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

    registry = er.async_get(hass)
    entities = [
        e
        for e in registry.entities.values()
        if e.domain == "ai_task" and e.config_entry_id == entry.entry_id
    ]
    entity_id = entities[0].entity_id
    entity = _ai_task_entity(hass, entity_id)

    image_path = tmp_path / "driveway.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock()
    mock_openai_client.post = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"edited").decode(),
                    url=None,
                    mime_type="image/jpeg",
                    revised_prompt="sketch of driveway",
                )
            ]
        )
    )

    attachment = conversation.Attachment(
        media_content_id="media-source://media/driveway.jpg",
        mime_type="image/jpeg",
        path=image_path,
    )
    result = await entity._async_generate_image(
        ai_task.GenImageTask(
            name="Sketch",
            instructions="Render as pencil sketch",
            attachments=[attachment],
        ),
        _fake_chat_log(),
    )

    mock_openai_client.images.generate.assert_not_called()
    mock_openai_client.post.assert_awaited_once()
    assert mock_openai_client.post.await_args.args[0] == "/images/edits"
    body = mock_openai_client.post.await_args.kwargs["body"]
    assert body["model"] == "grok-imagine-image-2.0"
    assert body["prompt"] == "Render as pencil sketch"
    assert body["image"]["type"] == "image_url"
    assert body["image"]["url"].startswith("data:image/jpeg;base64,")
    assert result.image_data == b"edited"
    assert result.revised_prompt == "sketch of driveway"


async def test_generate_image_attachment_unsupported_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Default image model rejects edit attachments with HomeAssistantError."""
    entity = _ai_task_entity(hass, ai_task_entity_id)
    image_path = tmp_path / "driveway.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    mock_openai_client.post = AsyncMock()

    with pytest.raises(HomeAssistantError, match="does not support image editing"):
        await entity._async_generate_image(
            ai_task.GenImageTask(
                name="Sketch",
                instructions="Edit",
                attachments=[
                    conversation.Attachment(
                        media_content_id="media-source://media/driveway.jpg",
                        mime_type="image/jpeg",
                        path=image_path,
                    )
                ],
            ),
            _fake_chat_log(),
        )

    mock_openai_client.post.assert_not_called()


async def test_generate_image_ha_attachments_getattr_safe(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """GenImageTask without attachments attr (older shape) still generates."""
    entity = _ai_task_entity(hass, ai_task_entity_id)
    mock_openai_client.images = MagicMock()
    mock_openai_client.images.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"ok").decode(),
                    url=None,
                    mime_type="image/png",
                    revised_prompt=None,
                )
            ]
        )
    )
    # Simulate a task object that may lack attachments on older HA shapes.
    task = SimpleNamespace(name="Sky", instructions="Blue sky")
    assert getattr(task, "attachments", None) is None

    result = await entity._async_generate_image(task, _fake_chat_log())
    assert result.image_data == b"ok"
    mock_openai_client.images.generate.assert_awaited_once()


async def test_generate_image_six_attachments_raises_before_read(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
    tmp_path: Path,
) -> None:
    """More than 5 AI Task attachments raises before any file read."""
    from custom_components.grok_conversation.const import IMAGE_EDIT_MAX_IMAGES

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=3,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": {
                    **dict(RECOMMENDED_AI_TASK_OPTIONS),
                    CONF_IMAGE_MODEL: "grok-imagine-image-2.0",
                },
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

    registry = er.async_get(hass)
    entity_id = next(
        e.entity_id
        for e in registry.entities.values()
        if e.domain == "ai_task" and e.config_entry_id == entry.entry_id
    )
    entity = _ai_task_entity(hass, entity_id)
    mock_openai_client.post = AsyncMock()

    attachments = []
    for i in range(IMAGE_EDIT_MAX_IMAGES + 1):
        path = tmp_path / f"a{i}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        attachments.append(
            conversation.Attachment(
                media_content_id=f"media-source://media/a{i}.jpg",
                mime_type="image/jpeg",
                path=path,
            )
        )

    with (
        patch(
            "custom_components.grok_conversation._read_path_bytes",
            side_effect=AssertionError("must not read"),
        ),
        pytest.raises(HomeAssistantError, match=str(IMAGE_EDIT_MAX_IMAGES)),
    ):
        await entity._async_generate_image(
            ai_task.GenImageTask(
                name="Too many",
                instructions="Edit",
                attachments=attachments,
            ),
            _fake_chat_log(),
        )
    mock_openai_client.post.assert_not_called()
