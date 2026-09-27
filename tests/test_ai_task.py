"""Tests for AI Task platform."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
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
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
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
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
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
