"""Tests for config flow and AI Task subentry flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import voluptuous as vol
import yaml
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import SelectSelector
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.config_flow import (
    RECOMMENDED_OPTIONS,
    openai_config_option_schema,
)
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
)


async def test_user_flow_creates_ai_task_subentry(
    hass: HomeAssistant, mock_openai_client: MagicMock
) -> None:
    """User config flow creates an entry with a default ai_task_data subentry."""
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()

    with (
        patch(
            "custom_components.grok_conversation.config_flow.validate_input",
            return_value={"voice_ok": True, "voice_detail": "ok"},
        ),
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
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-test"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = result["result"]
    assert any(
        s.subentry_type == "ai_task_data" for s in entry.subentries.values()
    )


async def test_ai_task_subentry_create_and_reconfigure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Subentry flow can create and reconfigure an ai_task_data subentry."""
    with (
        patch(
            "custom_components.grok_conversation.config_flow.async_list_chat_models",
            return_value=[RECOMMENDED_CHAT_MODEL],
        ),
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
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, "ai_task_data"),
            context={"source": SOURCE_USER},
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Custom Task",
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_GENERATION_MODEL,
                CONF_RECOMMENDED: True,
            },
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Custom Task"

    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.title == "Custom Task"
    )

    with (
        patch(
            "custom_components.grok_conversation.config_flow.async_list_chat_models",
            return_value=[RECOMMENDED_CHAT_MODEL, "grok-4.5"],
        ),
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
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, "ai_task_data"),
            context={
                "source": "reconfigure",
                "subentry_id": subentry.subentry_id,
            },
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_CHAT_MODEL: "grok-4.5",
                CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_GENERATION_MODEL,
                CONF_RECOMMENDED: True,
            },
        )
        await hass.async_block_till_done()

    assert result["type"] in (FlowResultType.ABORT, FlowResultType.CREATE_ENTRY)
    updated = mock_config_entry.subentries[subentry.subentry_id]
    assert updated.data[CONF_CHAT_MODEL] == "grok-4.5"


async def test_reasoning_effort_provider_default_in_options_schema(
    hass: HomeAssistant,
) -> None:
    """Advanced options expose none (provider default) and it round-trips (#40 review)."""
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()

    options = {**RECOMMENDED_OPTIONS, CONF_RECOMMENDED: False}
    schema_dict = openai_config_option_schema(hass, options)

    effort_key = next(
        key
        for key in schema_dict
        if getattr(key, "schema", key) == CONF_REASONING_EFFORT
    )
    selector = schema_dict[effort_key]
    assert isinstance(selector, SelectSelector)
    effort_values = [opt["value"] for opt in selector.config["options"]]
    effort_labels = [opt["label"] for opt in selector.config["options"]]
    assert "none" in effort_values
    assert "Provider default (not sent)" in effort_labels

    validated = vol.Schema(schema_dict)(
        {
            CONF_RECOMMENDED: False,
            CONF_REASONING_EFFORT: "none",
        }
    )
    assert validated[CONF_REASONING_EFFORT] == "none"

    services = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "custom_components"
            / "grok_conversation"
            / "services.yaml"
        ).read_text(encoding="utf-8")
    )
    ask_opts = services["ask"]["fields"]["reasoning_effort"]["selector"]["select"][
        "options"
    ]
    ask_values = [
        opt["value"] if isinstance(opt, dict) else opt for opt in ask_opts
    ]
    assert "none" in ask_values
