"""Config flow for Grok Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    GROK_SYSTEM_PROMPT,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    UNSUPPORTED_MODELS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_PROMPT: GROK_SYSTEM_PROMPT,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    def sync_validate():
        client = openai.AsyncOpenAI(
            api_key=data[CONF_API_KEY],
            base_url="https://api.x.ai/v1",
            http_client=get_async_client(hass),
        )
        return client.with_options(timeout=10.0).models.list()

    try:
        await hass.async_add_executor_job(sync_validate)
    except openai.APIConnectionError:
        raise
    except openai.AuthenticationError:
        raise
    except Exception as err:
        _LOGGER.exception("Unexpected exception during validation")
        raise


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Grok Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="Grok",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options
        errors: dict[str, str] = {}

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                # Handle LLM HASS API selection (can be list or single value)
                llm_hass_api = user_input.get(CONF_LLM_HASS_API)
                if llm_hass_api:
                    # Get available API IDs for validation (same as used in schema)
                    try:
                        available_apis = llm.async_get_apis(self.hass)
                        available_api_ids = {api.id for api in available_apis}
                        _LOGGER.debug(
                            "Available LLM APIs: %s",
                            {api.id: api.name for api in available_apis}
                        )
                    except Exception as err:
                        _LOGGER.error("Error getting available LLM APIs: %s", err, exc_info=True)
                        available_api_ids = set()
                    
                    # Normalize to list for easier processing
                    if isinstance(llm_hass_api, str):
                        api_list = [llm_hass_api]
                    elif isinstance(llm_hass_api, list):
                        api_list = list(llm_hass_api)
                    else:
                        api_list = []
                    
                    # Remove "none" from list if present
                    if "none" in api_list:
                        api_list.remove("none")
                    
                    # If list is empty after removing "none", omit the key
                    if not api_list:
                        user_input.pop(CONF_LLM_HASS_API, None)
                    else:
                        # Validate that all LLM APIs exist
                        invalid_apis = [api_id for api_id in api_list if api_id not in available_api_ids]
                        if invalid_apis:
                            errors[CONF_LLM_HASS_API] = "llm_api_not_found"
                            _LOGGER.warning(
                                "LLM API(s) not found: %s. Available APIs: %s. User input: %s",
                                invalid_apis,
                                available_api_ids,
                                llm_hass_api
                            )
                        else:
                            # Store as list for consistency
                            user_input[CONF_LLM_HASS_API] = api_list
                            _LOGGER.debug("Validated LLM HASS API selection: %s", api_list)
                else:
                    # No API selected, omit the key
                    user_input.pop(CONF_LLM_HASS_API, None)

                if user_input.get(CONF_CHAT_MODEL) in UNSUPPORTED_MODELS:
                    errors[CONF_CHAT_MODEL] = "model_not_supported"
                elif not errors:
                    return self.async_create_entry(title="", data=user_input)
            else:
                # Re-render the options again, now with the recommended options shown/hidden
                self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

                options = {
                    CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                    CONF_PROMPT: user_input[CONF_PROMPT],
                    CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
                }

        schema = openai_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> VolDictType:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, GROK_SYSTEM_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                default=RECOMMENDED_REASONING_EFFORT,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value="low", label="Low"),
                        SelectOptionDict(value="medium", label="Medium"),
                        SelectOptionDict(value="high", label="High"),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
    )
    return schema