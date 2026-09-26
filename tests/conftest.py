"""Shared fixtures for Grok Conversation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
)

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for all tests."""
    return


@pytest.fixture(autouse=True)
async def setup_ha(hass: HomeAssistant) -> None:
    """Set up the homeassistant component (exposed entities, etc.)."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_openai_client():
    """Mock openai.AsyncOpenAI used by the integration."""
    client = MagicMock()
    client.platform_headers = MagicMock(return_value={})
    client.with_options.return_value.models.list = MagicMock(
        return_value=MagicMock(data=[MagicMock(id="grok-4.3-latest")])
    )
    client.models.list = AsyncMock(
        return_value=MagicMock(data=[MagicMock(id="grok-4.3-latest")])
    )
    client.chat.completions.create = AsyncMock()
    client.responses.create = AsyncMock()
    return client


@pytest.fixture
async def mock_config_entry(
    hass: HomeAssistant,
    mock_openai_client: MagicMock,
) -> MockConfigEntry:
    """Create a loaded config entry with default AI Task subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=2,
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
            "custom_components.grok_conversation.config_flow.openai.AsyncOpenAI",
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
        patch(
            "custom_components.grok_conversation.config_flow.get_async_client",
            return_value=None,
        ),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    entry.runtime_data = mock_openai_client
    return entry
