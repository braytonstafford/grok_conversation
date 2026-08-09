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
from homeassistant.helpers.aiohttp_client import async_get_clientsession
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
from .api_helpers import async_list_chat_models, is_chat_model_id
from .voice_const import (
    CONF_ENABLE_STT,
    CONF_ENABLE_TTS,
    CONF_STT_LANGUAGE,
    CONF_TTS_LANGUAGE,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_ENABLE_STT,
    RECOMMENDED_ENABLE_TTS,
    RECOMMENDED_STT_LANGUAGE,
    RECOMMENDED_TTS_LANGUAGE,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
    STT_LANGUAGES,
    TTS_LANGUAGES,
    XAI_VOICES,
)
from .voice_api import async_validate_voice_access

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_PROMPT: GROK_SYSTEM_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_FAST_MODEL: RECOMMENDED_FAST_MODEL,
    CONF_FALLBACK_MODEL: RECOMMENDED_FALLBACK_MODEL,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_SHOW_CITATIONS: RECOMMENDED_SHOW_CITATIONS,
    CONF_SEND_USER_NAME: RECOMMENDED_SEND_USER_NAME,
    CONF_INTERACTION_MODE: RECOMMENDED_INTERACTION_MODE,
    CONF_VOICE_OPTIMIZED: RECOMMENDED_VOICE_OPTIMIZED,
    CONF_AUTO_MODEL_ROUTING: RECOMMENDED_AUTO_MODEL_ROUTING,
    CONF_HOME_CONTEXT: RECOMMENDED_HOME_CONTEXT,
    CONF_ENABLE_TTS: RECOMMENDED_ENABLE_TTS,
    CONF_ENABLE_STT: RECOMMENDED_ENABLE_STT,
    CONF_TTS_VOICE: RECOMMENDED_TTS_VOICE,
    CONF_TTS_LANGUAGE: RECOMMENDED_TTS_LANGUAGE,
    CONF_TTS_SPEED: RECOMMENDED_TTS_SPEED,
    CONF_STT_LANGUAGE: RECOMMENDED_STT_LANGUAGE,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect. Returns voice probe info."""

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

    session = async_get_clientsession(hass)
    voice_ok, voice_detail = await async_validate_voice_access(
        session, data[CONF_API_KEY]
    )
    return {"voice_ok": voice_ok, "voice_detail": voice_detail}


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
        description_placeholders: dict[str, str] = {}

        try:
            info = await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:  # noqa: BLE001
            errors["base"] = "unknown"
        else:
            if not info.get("voice_ok"):
                _LOGGER.warning(
                    "xAI key valid for chat but Voice API check failed: %s",
                    info.get("voice_detail"),
                )
                # Still create — conversation works; TTS/STT may need key permissions
            return self.async_create_entry(
                title="xAI Grok",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders=description_placeholders,
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
        self._chat_models: list[str] | None = None

    async def _async_get_chat_models(self) -> list[str]:
        """List chat models from xAI (cached per options flow session)."""
        if self._chat_models is not None:
            return self._chat_models

        api_key = self.config_entry.data.get(CONF_API_KEY, "")
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            http_client=get_async_client(self.hass),
        )
        models = await async_list_chat_models(client)

        # Ensure currently configured models always appear even if filtered out
        options = self.config_entry.options
        for key, default in (
            (CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            (CONF_FAST_MODEL, RECOMMENDED_FAST_MODEL),
            (CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL),
        ):
            current = options.get(key, default)
            if (
                isinstance(current, str)
                and current
                and current not in models
                and is_chat_model_id(current)
            ):
                models = [current, *models]

        self._chat_models = models
        return models

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options
        errors: dict[str, str] = {}
        chat_models = await self._async_get_chat_models()

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                llm_hass_api = user_input.get(CONF_LLM_HASS_API)
                if llm_hass_api:
                    try:
                        available_apis = list(llm.async_get_apis(self.hass))
                        available_api_ids = {api.id for api in available_apis}
                        available_by_name = {
                            (api.name or "").strip().lower(): api.id
                            for api in available_apis
                        }
                    except Exception as err:  # noqa: BLE001
                        _LOGGER.error("Error getting available LLM APIs: %s", err)
                        available_apis = []
                        available_api_ids = set()
                        available_by_name = {}

                    if isinstance(llm_hass_api, str):
                        api_list = [llm_hass_api]
                    elif isinstance(llm_hass_api, list):
                        api_list = list(llm_hass_api)
                    else:
                        api_list = []

                    if "none" in api_list:
                        api_list.remove("none")

                    # Resolve common aliases / display names → API ids
                    resolved: list[str] = []
                    for api_id in api_list:
                        if api_id in available_api_ids:
                            resolved.append(api_id)
                            continue
                        by_name = available_by_name.get(str(api_id).strip().lower())
                        if by_name:
                            resolved.append(by_name)
                            continue
                        # Common HA Assist aliases
                        key = str(api_id).strip().lower()
                        if key in {"assist", "home assistant", "homeassistant"} and available_apis:
                            pick = next(
                                (
                                    a.id
                                    for a in available_apis
                                    if "assist" in a.id.lower()
                                    or "assist" in (a.name or "").lower()
                                    or a.id == "homeassistant"
                                ),
                                available_apis[0].id,
                            )
                            resolved.append(pick)
                            continue
                        resolved.append(api_id)

                    # Dedupe preserve order
                    seen: set[str] = set()
                    api_list = []
                    for a in resolved:
                        if a not in seen:
                            seen.add(a)
                            api_list.append(a)

                    if not api_list:
                        user_input.pop(CONF_LLM_HASS_API, None)
                    else:
                        invalid_apis = [
                            api_id
                            for api_id in api_list
                            if api_id not in available_api_ids
                        ]
                        if invalid_apis and available_api_ids:
                            api_list = [
                                a for a in api_list if a in available_api_ids
                            ]
                            if not api_list:
                                errors[CONF_LLM_HASS_API] = "llm_api_not_found"
                                _LOGGER.warning(
                                    "LLM API(s) not found: %s. Available: %s",
                                    invalid_apis,
                                    available_api_ids,
                                )
                            else:
                                user_input[CONF_LLM_HASS_API] = api_list
                        elif invalid_apis and not available_api_ids:
                            _LOGGER.warning(
                                "No LLM APIs registered yet; saving selection %s",
                                api_list,
                            )
                            user_input[CONF_LLM_HASS_API] = api_list
                        else:
                            user_input[CONF_LLM_HASS_API] = api_list
                else:
                    user_input.pop(CONF_LLM_HASS_API, None)

                # Validate model picks (allow custom values that look chat-capable)
                for model_key in (CONF_CHAT_MODEL, CONF_FAST_MODEL, CONF_FALLBACK_MODEL):
                    model_val = user_input.get(model_key)
                    if not model_val:
                        continue
                    if model_val in UNSUPPORTED_MODELS or not is_chat_model_id(
                        str(model_val)
                    ):
                        errors[model_key] = "model_not_supported"

                if not errors:
                    return self.async_create_entry(title="", data=user_input)
            else:
                # Recommended checkbox toggled — re-render form, keep other values
                self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
                options = {
                    **dict(options),
                    **user_input,
                    CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                }

        schema = openai_config_option_schema(
            self.hass, options, chat_models=chat_models
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )


def _model_select(
    models: list[str],
    current: str | None,
    default: str,
) -> SelectSelector:
    """Build a dropdown of live chat models; allow typing a custom id."""
    opts = list(models)
    cur = current or default
    if cur and cur not in opts:
        opts = [cur, *opts]
    if default not in opts:
        opts = [default, *opts]
    return SelectSelector(
        SelectSelectorConfig(
            options=[SelectOptionDict(value=m, label=m) for m in opts],
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    chat_models: list[str] | None = None,
) -> VolDictType:
    """Return a schema for Grok completion options."""
    models = chat_models or [
        RECOMMENDED_CHAT_MODEL,
        RECOMMENDED_FAST_MODEL,
        RECOMMENDED_FALLBACK_MODEL,
    ]

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

    chat_default = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
    fast_default = options.get(CONF_FAST_MODEL, RECOMMENDED_FAST_MODEL)
    fallback_default = options.get(CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL)

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(CONF_PROMPT, GROK_SYSTEM_PROMPT)
            },
        ): TemplateSelector(),
        # --- Models (always visible; populated from live xAI /v1/models) ---
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": chat_default},
            default=chat_default,
        ): _model_select(models, chat_default, RECOMMENDED_CHAT_MODEL),
        vol.Optional(
            CONF_FAST_MODEL,
            description={"suggested_value": fast_default},
            default=fast_default,
        ): _model_select(models, fast_default, RECOMMENDED_FAST_MODEL),
        vol.Optional(
            CONF_FALLBACK_MODEL,
            description={"suggested_value": fallback_default},
            default=fallback_default,
        ): _model_select(models, fallback_default, RECOMMENDED_FALLBACK_MODEL),
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
        # --- Voice (TTS / STT for Assist pipelines) ---
        vol.Optional(
            CONF_ENABLE_TTS,
            description={
                "suggested_value": options.get(CONF_ENABLE_TTS, RECOMMENDED_ENABLE_TTS)
            },
            default=options.get(CONF_ENABLE_TTS, RECOMMENDED_ENABLE_TTS),
        ): bool,
        vol.Optional(
            CONF_ENABLE_STT,
            description={
                "suggested_value": options.get(CONF_ENABLE_STT, RECOMMENDED_ENABLE_STT)
            },
            default=options.get(CONF_ENABLE_STT, RECOMMENDED_ENABLE_STT),
        ): bool,
        vol.Optional(
            CONF_TTS_VOICE,
            description={
                "suggested_value": options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
            },
            default=options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE),
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=vid, label=label)
                    for vid, label in sorted(XAI_VOICES.items(), key=lambda x: x[1].lower())
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(
            CONF_TTS_LANGUAGE,
            description={
                "suggested_value": options.get(
                    CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE
                )
            },
            default=options.get(CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE),
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=code, label=code)
                    for code in TTS_LANGUAGES
                    if code != "auto"
                ]
                + [SelectOptionDict(value="auto", label="auto (detect)")],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(
            CONF_TTS_SPEED,
            description={
                "suggested_value": options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)
            },
            default=options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
        ): NumberSelector(
            NumberSelectorConfig(min=0.7, max=1.5, step=0.05)
        ),
        vol.Optional(
            CONF_STT_LANGUAGE,
            description={
                "suggested_value": options.get(
                    CONF_STT_LANGUAGE, RECOMMENDED_STT_LANGUAGE
                )
            },
            default=options.get(CONF_STT_LANGUAGE, RECOMMENDED_STT_LANGUAGE),
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=code, label=code) for code in STT_LANGUAGES
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True)
        ): bool,
    }

    # Advanced sampling params only when "Recommended model settings" is off
    if options.get(CONF_RECOMMENDED, True):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                default=options.get(
                    CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                ),
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
