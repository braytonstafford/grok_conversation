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
    CONF_AUTO_MODEL_ROUTING,
    CONF_BUDGET_WARN_USD,
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_FAST_MODEL,
    CONF_HOME_CONTEXT,
    CONF_INTERACTION_MODE,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_OPTIMIZED,
    DOMAIN,
    GROK_SYSTEM_PROMPT,
    MODE_CHAT_ONLY,
    MODE_PIPELINE,
    MODE_TOOLS,
    RECOMMENDED_AUTO_MODEL_ROUTING,
    RECOMMENDED_BUDGET_WARN_USD,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_FAST_MODEL,
    RECOMMENDED_HOME_CONTEXT,
    RECOMMENDED_INTERACTION_MODE,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VOICE_OPTIMIZED,
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
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_SHOW_CITATIONS: RECOMMENDED_SHOW_CITATIONS,
    CONF_SEND_USER_NAME: RECOMMENDED_SEND_USER_NAME,
    CONF_INTERACTION_MODE: RECOMMENDED_INTERACTION_MODE,
    CONF_VOICE_OPTIMIZED: RECOMMENDED_VOICE_OPTIMIZED,
    CONF_AUTO_MODEL_ROUTING: RECOMMENDED_AUTO_MODEL_ROUTING,
    CONF_HOME_CONTEXT: RECOMMENDED_HOME_CONTEXT,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""

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
    except Exception:
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
        except Exception:  # noqa: BLE001
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
    """Grok config flow options handler."""

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
                llm_hass_api = user_input.get(CONF_LLM_HASS_API)
                if llm_hass_api:
                    try:
                        available_apis = llm.async_get_apis(self.hass)
                        available_api_ids = {api.id for api in available_apis}
                    except Exception as err:  # noqa: BLE001
                        _LOGGER.error("Error getting available LLM APIs: %s", err)
                        available_api_ids = set()

                    if isinstance(llm_hass_api, str):
                        api_list = [llm_hass_api]
                    elif isinstance(llm_hass_api, list):
                        api_list = list(llm_hass_api)
                    else:
                        api_list = []

                    if "none" in api_list:
                        api_list.remove("none")

                    if not api_list:
                        user_input.pop(CONF_LLM_HASS_API, None)
                    else:
                        invalid_apis = [
                            api_id
                            for api_id in api_list
                            if api_id not in available_api_ids
                        ]
                        if invalid_apis:
                            errors[CONF_LLM_HASS_API] = "llm_api_not_found"
                        else:
                            user_input[CONF_LLM_HASS_API] = api_list
                else:
                    user_input.pop(CONF_LLM_HASS_API, None)

                if user_input.get(CONF_CHAT_MODEL) in UNSUPPORTED_MODELS:
                    errors[CONF_CHAT_MODEL] = "model_not_supported"
                elif not errors:
                    return self.async_create_entry(title="", data=user_input)
            else:
                self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
                options = {
                    CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                    CONF_PROMPT: user_input.get(CONF_PROMPT, options.get(CONF_PROMPT)),
                    CONF_LLM_HASS_API: user_input.get(CONF_LLM_HASS_API),
                    CONF_LIVE_SEARCH: user_input.get(
                        CONF_LIVE_SEARCH, options.get(CONF_LIVE_SEARCH)
                    ),
                    CONF_INTERACTION_MODE: user_input.get(
                        CONF_INTERACTION_MODE, options.get(CONF_INTERACTION_MODE)
                    ),
                    CONF_SEND_USER_NAME: user_input.get(
                        CONF_SEND_USER_NAME, options.get(CONF_SEND_USER_NAME)
                    ),
                    CONF_VOICE_OPTIMIZED: user_input.get(
                        CONF_VOICE_OPTIMIZED, options.get(CONF_VOICE_OPTIMIZED)
                    ),
                    CONF_HOME_CONTEXT: user_input.get(
                        CONF_HOME_CONTEXT, options.get(CONF_HOME_CONTEXT)
                    ),
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
    """Return a schema for Grok completion options."""
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

    live_search_opts = [
        SelectOptionDict(value="off", label="Off"),
        SelectOptionDict(value="web", label="Web Search"),
        SelectOptionDict(value="x", label="X Search"),
        SelectOptionDict(value="full", label="Full (Web + X)"),
    ]
    mode_opts = [
        SelectOptionDict(value=MODE_TOOLS, label="Tool Control (default)"),
        SelectOptionDict(
            value=MODE_PIPELINE, label="Intelligent Pipeline (HA intent → Grok)"
        ),
        SelectOptionDict(value=MODE_CHAT_ONLY, label="Chat Only (no device control)"),
    ]

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(CONF_PROMPT, GROK_SYSTEM_PROMPT)
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Optional(
            CONF_INTERACTION_MODE,
            description={
                "suggested_value": options.get(
                    CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE
                )
            },
            default=options.get(CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE),
        ): SelectSelector(
            SelectSelectorConfig(
                options=mode_opts, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Optional(
            CONF_LIVE_SEARCH,
            description={
                "suggested_value": options.get(
                    CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH
                )
            },
            default=options.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH),
        ): SelectSelector(
            SelectSelectorConfig(
                options=live_search_opts, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Optional(
            CONF_SHOW_CITATIONS,
            description={
                "suggested_value": options.get(
                    CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS
                )
            },
            default=options.get(CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS),
        ): bool,
        vol.Optional(
            CONF_SEND_USER_NAME,
            description={
                "suggested_value": options.get(
                    CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME
                )
            },
            default=options.get(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME),
        ): bool,
        vol.Optional(
            CONF_LOCATION_CONTEXT,
            description={
                "suggested_value": options.get(CONF_LOCATION_CONTEXT, "")
            },
            default=options.get(CONF_LOCATION_CONTEXT, ""),
        ): TextSelector(TextSelectorConfig(type="text")),
        vol.Optional(
            CONF_VOICE_OPTIMIZED,
            description={
                "suggested_value": options.get(
                    CONF_VOICE_OPTIMIZED, RECOMMENDED_VOICE_OPTIMIZED
                )
            },
            default=options.get(CONF_VOICE_OPTIMIZED, RECOMMENDED_VOICE_OPTIMIZED),
        ): bool,
        vol.Optional(
            CONF_HOME_CONTEXT,
            description={
                "suggested_value": options.get(
                    CONF_HOME_CONTEXT, RECOMMENDED_HOME_CONTEXT
                )
            },
            default=options.get(CONF_HOME_CONTEXT, RECOMMENDED_HOME_CONTEXT),
        ): bool,
        vol.Optional(
            CONF_AUTO_MODEL_ROUTING,
            description={
                "suggested_value": options.get(
                    CONF_AUTO_MODEL_ROUTING, RECOMMENDED_AUTO_MODEL_ROUTING
                )
            },
            default=options.get(
                CONF_AUTO_MODEL_ROUTING, RECOMMENDED_AUTO_MODEL_ROUTING
            ),
        ): bool,
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
                CONF_FAST_MODEL,
                description={"suggested_value": options.get(CONF_FAST_MODEL)},
                default=RECOMMENDED_FAST_MODEL,
            ): str,
            vol.Optional(
                CONF_FALLBACK_MODEL,
                description={"suggested_value": options.get(CONF_FALLBACK_MODEL)},
                default=RECOMMENDED_FALLBACK_MODEL,
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
            vol.Optional(
                CONF_BUDGET_WARN_USD,
                description={
                    "suggested_value": options.get(
                        CONF_BUDGET_WARN_USD, RECOMMENDED_BUDGET_WARN_USD
                    )
                },
                default=options.get(CONF_BUDGET_WARN_USD, RECOMMENDED_BUDGET_WARN_USD),
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=10000, step=0.5, unit_of_measurement="USD")
            ),
        }
    )
    return schema
