"""Conversation support for xAI Grok."""

from __future__ import annotations

from typing import Any, Literal

import openai

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from . import OpenAIConfigEntry
from .api_helpers import (
    async_responses_completion,
    looks_like_search_query,
    looks_like_simple_query,
    should_use_live_search,
)
from .const import (
    CONF_AUTO_MODEL_ROUTING,
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
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_OPTIMIZED,
    DOMAIN,
    LOGGER,
    MODE_CHAT_ONLY,
    MODE_PIPELINE,
    RECOMMENDED_AUTO_MODEL_ROUTING,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_FAST_MODEL,
    RECOMMENDED_HOME_CONTEXT,
    RECOMMENDED_INTERACTION_MODE,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VOICE_OPTIMIZED,
    VOICE_OPTIMIZED_SUFFIX,
)
from .entity import (
    GrokBaseLLMEntity,
    convert_content_to_param,
    _strip_json_from_response,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


class OpenAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    GrokBaseLLMEntity,
):
    """Grok conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        GrokBaseLLMEntity.__init__(self, entry, None)
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="xAI",
            model="Grok",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._attr_supported_features = conversation.ConversationEntityFeature(0)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        self._update_control_feature()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    def _update_control_feature(self) -> None:
        mode = self.entry.options.get(
            CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE
        )
        llm_hass_api = self.entry.options.get(CONF_LLM_HASS_API)
        if mode == MODE_CHAT_ONLY or not llm_hass_api:
            self._attr_supported_features = conversation.ConversationEntityFeature(0)
            return
        api_ids = llm_hass_api if isinstance(llm_hass_api, list) else [llm_hass_api]
        api_ids = [api_id for api_id in api_ids if api_id != "none"]
        if not api_ids:
            self._attr_supported_features = conversation.ConversationEntityFeature(0)
            return
        try:
            llm.async_get_api(self.hass, api_ids[0])
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        except Exception:  # noqa: BLE001
            self._attr_supported_features = conversation.ConversationEntityFeature(0)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API with function calling support."""
        try:
            return await self._async_handle_message_inner(user_input, chat_log)
        except Exception as err:  # noqa: BLE001
            LOGGER.error(
                "Unexpected error in conversation handler: %s", err, exc_info=True
            )
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(
                "Sorry, I encountered an unexpected error. Please try again."
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id if chat_log else "",
                continue_conversation=False,
            )

    def _is_tool_result_helpful(self, tool_name: str, tool_result: Any) -> bool:
        """Legacy helper — kept for compatibility; always prefer real payloads."""
        return True

    @staticmethod
    def _pick_speech_content(chat_log: conversation.ChatLog) -> str | None:
        """Prefer the last assistant message that is not a tool-call turn."""
        fallback: str | None = None
        for content in reversed(chat_log.content):
            if not isinstance(content, conversation.AssistantContent):
                continue
            text = (content.content or "").strip()
            if not text:
                continue
            if getattr(content, "tool_calls", None):
                if fallback is None:
                    fallback = text
                continue
            return text
        return fallback

    def _build_factual_context(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Location, local time, presence, weather — needed by live search.

        Always attach this to the search pass even when HA tools are present
        (#27). Persona/voice bits are built separately and also attached on
        the search pass (#32).
        """
        options = self.entry.options
        parts: list[str] = []

        location = (options.get(CONF_LOCATION_CONTEXT) or "").strip()
        if location:
            parts.append(
                "Home location context for local queries "
                f"(use this for 'near me', local weather, open now, distance): {location}."
            )
        else:
            tz = self.hass.config.time_zone
            if tz:
                parts.append(f"Home Assistant timezone: {tz}.")

        if options.get(CONF_HOME_CONTEXT, RECOMMENDED_HOME_CONTEXT):
            now = dt_util.now()
            parts.append(
                f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M')}."
            )
            people = []
            for state in self.hass.states.async_all("person"):
                people.append(
                    f"{state.attributes.get('friendly_name', state.entity_id)}={state.state}"
                )
            if people:
                parts.append("Person presence: " + ", ".join(people[:12]) + ".")
            weather = next(
                (
                    s
                    for s in self.hass.states.async_all("weather")
                    if s.state not in ("unavailable", "unknown")
                ),
                None,
            )
            if weather:
                temp = weather.attributes.get("temperature")
                unit = weather.attributes.get("temperature_unit", "")
                parts.append(
                    f"Weather entity {weather.entity_id}: {weather.state}"
                    + (f", {temp}{unit}" if temp is not None else "")
                    + "."
                )

        return "\n".join(parts)

    def _build_persona_context(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Voice-style and user-name hints (not required for search grounding)."""
        options = self.entry.options
        parts: list[str] = []

        if options.get(CONF_VOICE_OPTIMIZED, RECOMMENDED_VOICE_OPTIMIZED):
            parts.append(VOICE_OPTIMIZED_SUFFIX.strip())

        if options.get(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME):
            name = self._resolve_user_name(user_input)
            if name:
                parts.append(
                    f"The current user is named {name}. Address them by name when natural."
                )

        return "\n".join(parts)

    def _build_extra_system_prompt(
        self, user_input: conversation.ConversationInput
    ) -> str:
        """Assemble persona + factual context for the main LLM / tool path."""
        parts = [
            p
            for p in (
                self._build_persona_context(user_input),
                self._build_factual_context(user_input),
            )
            if p
        ]
        return "\n".join(parts)

    def _resolve_user_name(
        self, user_input: conversation.ConversationInput
    ) -> str | None:
        """Best-effort user display name from person entity or HA user."""
        context = user_input.context
        user_id = getattr(context, "user_id", None) if context else None
        if user_id:
            for state in self.hass.states.async_all("person"):
                if state.attributes.get("user_id") == user_id:
                    return state.attributes.get("friendly_name") or state.name
            user = self.hass.auth.async_get_user(user_id) if self.hass.auth else None
            if user and user.name:
                return user.name
        return None

    def _select_model(self, user_text: str, options: dict) -> str:
        """Pick chat/fast model based on auto-routing."""
        primary = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        fast = options.get(CONF_FAST_MODEL, RECOMMENDED_FAST_MODEL)
        if options.get(CONF_AUTO_MODEL_ROUTING, RECOMMENDED_AUTO_MODEL_ROUTING):
            if looks_like_simple_query(user_text) and not looks_like_search_query(
                user_text
            ):
                return fast or primary
        return primary

    async def _try_pipeline(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult | None:
        """Try built-in Home Assistant conversation agent first."""
        try:
            result = await conversation.async_converse(
                self.hass,
                text=user_input.text,
                conversation_id=None,
                context=user_input.context,
                language=user_input.language,
                agent_id="conversation.home_assistant",
                device_id=user_input.device_id,
            )
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("Pipeline HA agent failed: %s", err)
            return None

        speech = ""
        if result and result.response:
            speech = result.response.speech.get("plain", {}).get("speech", "")
            if not speech and hasattr(result.response, "as_dict"):
                data = result.response.as_dict()
                speech = (
                    data.get("speech", {}).get("plain", {}).get("speech", "") or ""
                )
        if not speech:
            return None
        lowered = speech.lower()
        fallback_markers = (
            "sorry",
            "i am not aware",
            "i'm not aware",
            "don't know",
            "do not know",
            "no intent",
            "not sure how",
            "can you rephrase",
        )
        if any(m in lowered for m in fallback_markers):
            return None
        return result

    # Long spoken replies without prewarmed TTS: don't open the mic early (#31)
    _SATELLITE_LONG_REPLY_CHARS = 280

    async def _prewarm_pipeline_tts(
        self,
        message: str,
        user_input: conversation.ConversationInput,
    ) -> bool:
        """Pre-generate Assist pipeline TTS so satellite playback can start ASAP.

        Reads engine/language/voice from the preferred Assist pipeline — not a
        hardcoded voice. Populates the TTS cache used by the next pipeline TTS.
        """
        try:
            from homeassistant.components import assist_pipeline, tts
            from homeassistant.components.tts.media_source import (
                generate_media_source_id,
            )

            pipeline = assist_pipeline.async_get_pipeline(self.hass)
            if not pipeline or not pipeline.tts_engine:
                return False

            options: dict[str, Any] = {}
            if pipeline.tts_voice is not None:
                options[tts.ATTR_VOICE] = pipeline.tts_voice

            media_id = generate_media_source_id(
                self.hass,
                message=message,
                engine=pipeline.tts_engine,
                language=pipeline.tts_language,
                options=options or None,
                cache=True,
            )
            await tts.async_get_media_source_audio(self.hass, media_id)
            LOGGER.debug(
                "Prewarmed pipeline TTS for device_id=%s engine=%s voice=%s",
                user_input.device_id,
                pipeline.tts_engine,
                pipeline.tts_voice,
            )
            return True
        except Exception as err:  # noqa: BLE001
            LOGGER.debug("TTS prewarm failed: %s", err)
            return False

    async def _resolve_continue_conversation(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        speech: str | None,
    ) -> bool:
        """Satellite-aware continue_conversation (#31).

        Text chat (no device_id) keeps ChatLog's question-mark heuristic.
        For Assist satellites, prewarm TTS before returning True so the puck
        does not start listening while xAI TTS is still generating. If prewarm
        fails on a long reply, force False.
        """
        continue_conv = bool(chat_log.continue_conversation)
        if not continue_conv:
            return False
        if not user_input.device_id:
            return True

        speech_text = (speech or "").strip()
        if not speech_text:
            return False

        if await self._prewarm_pipeline_tts(speech_text, user_input):
            return True

        if len(speech_text) >= self._SATELLITE_LONG_REPLY_CHARS:
            LOGGER.debug(
                "TTS prewarm failed for long satellite reply (%s chars); "
                "disabling continue_conversation",
                len(speech_text),
            )
            return False
        return True

    async def _async_handle_message_inner(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Inner method that handles the actual conversation logic."""
        options = self.entry.options
        mode = options.get(CONF_INTERACTION_MODE, RECOMMENDED_INTERACTION_MODE)

        LOGGER.info(
            "Grok handling message mode=%s device_id=%s text=%s",
            mode,
            user_input.device_id,
            user_input.text,
        )

        # Intelligent pipeline: HA intent first
        if mode == MODE_PIPELINE:
            piped = await self._try_pipeline(user_input)
            if piped is not None:
                LOGGER.debug("Served via HA intent pipeline")
                return piped

        # Chat-only: disable LLM HASS API tools
        llm_api_option = None if mode == MODE_CHAT_ONLY else options.get(CONF_LLM_HASS_API)

        extra = self._build_extra_system_prompt(user_input)
        user_extra = user_input.extra_system_prompt or ""
        combined_extra = "\n".join(p for p in (extra, user_extra) if p)

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_api_option,
                options.get(CONF_PROMPT),
                combined_extra or None,
            )
        except conversation.ConverseError as err:
            LOGGER.error("ConverseError in async_provide_llm_data: %s", err)
            return err.as_conversation_result()

        model = self._select_model(user_input.text, options)
        fallback_model = options.get(CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL)
        messages = [
            m
            for content in chat_log.content
            for m in convert_content_to_param(content)
        ]

        # Prefix username on latest user message when enabled
        if options.get(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME):
            name = self._resolve_user_name(user_input)
            if name and messages:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        content = messages[i].get("content")
                        if isinstance(content, str) and not content.startswith(
                            f"[{name}]"
                        ):
                            messages[i] = {
                                **messages[i],
                                "content": f"[{name}] {content}",
                            }
                        break

        client = self.entry.runtime_data
        live_search = options.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
        show_citations = options.get(CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS)
        # Spoken Assist path: never append citation footnotes to speech (#32)
        show_citations_effective = bool(show_citations) and not bool(
            user_input.device_id
        )

        # Live search via Responses API. Previously this was skipped whenever HA
        # tools were present (#26), which made Live Search a no-op for Assist users.
        # Pipeline mode uses an inverted heuristic (#30): deny-list only.
        use_search = should_use_live_search(
            user_input.text,
            interaction_mode=mode,
            live_search=live_search,
        )
        ha_tools_available = bool(chat_log.llm_api and chat_log.llm_api.tools)

        if use_search:
            # Soften overlay: score/facts first, then honor persona (#32)
            search_system = [
                "You have live web/X search. Lead with the key fact or score, "
                "then briefly add context or opinion when the user's prompt "
                "asks for personality.",
                "Be concise. Prefer scores, times, and concrete outcomes first.",
                "Honor the user's personality/system prompt — keep voice and style.",
                "If results are uncertain, say what you found and what is unknown.",
                "When the user says 'near me' / local / open now, use the home "
                "location and local time below — do not ask them for a city.",
            ]
            if options.get(CONF_PROMPT):
                search_system.append(str(options.get(CONF_PROMPT)))
            # Factual context ALWAYS reaches the search pass (#27), even with HA tools
            factual = self._build_factual_context(user_input)
            if factual:
                search_system.append(factual)
            # Always attach persona/voice on the search pass (#32) — previously
            # skipped whenever HA tools were present, which made Assist sound
            # like a recap bot.
            persona = self._build_persona_context(user_input)
            if persona:
                search_system.append(persona)
            if user_extra:
                search_system.append(user_extra)

            search_messages = [
                m for m in messages if m.get("role") != "system"
            ]
            try:
                text, p_tok, c_tok = await async_responses_completion(
                    client,
                    model=model,
                    messages=search_messages,  # type: ignore[arg-type]
                    system_prompt="\n\n".join(search_system) or None,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    temperature=options.get(
                        CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                    ),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    live_search=live_search,
                    show_citations=show_citations_effective,
                    reasoning_effort=options.get(CONF_REASONING_EFFORT),
                )
                text = _strip_json_from_response(text)
                await self._record_usage(
                    model, p_tok, c_tok, service="conversation"
                )

                if text and not ha_tools_available:
                    # Search-only path (chat_only or no LLM HASS API)
                    async for _ in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(
                            agent_id=user_input.agent_id, content=text
                        )
                    ):
                        pass
                    intent_response = intent.IntentResponse(
                        language=user_input.language
                    )
                    intent_response.async_set_speech(text)
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                        continue_conversation=await self._resolve_continue_conversation(
                            user_input, chat_log, text
                        ),
                    )

                if text and ha_tools_available:
                    # Two-pass (#26): inject live findings into the tool loop so
                    # Assist control and live search work together.
                    brief = text.strip()
                    if len(brief) > 4000:
                        brief = brief[:4000] + "…"
                    search_note = (
                        "Live search results for this user question "
                        "(use these facts; do not claim you lack real-time data):\n"
                        f"{brief}"
                    )
                    new_messages: list[dict[str, Any]] = []
                    inserted = False
                    for msg in messages:
                        if (
                            not inserted
                            and isinstance(msg, dict)
                            and msg.get("role") == "system"
                            and isinstance(msg.get("content"), str)
                        ):
                            new_messages.append(
                                {
                                    "role": "system",
                                    "content": f"{msg['content']}\n\n{search_note}",
                                }
                            )
                            inserted = True
                        else:
                            new_messages.append(msg)
                    if not inserted:
                        new_messages.insert(
                            0, {"role": "system", "content": search_note}
                        )
                    messages = new_messages
                    LOGGER.debug(
                        "Injected live search brief (%s chars) into tool loop",
                        len(brief),
                    )
            except Exception as err:  # noqa: BLE001
                LOGGER.warning(
                    "Live search path failed (%s); continuing without search context",
                    err,
                )

        # Shared chat-completions + tool loop (entity base)
        try:
            await self._async_handle_chat_log(
                chat_log,
                model=model,
                options=dict(options),
                messages=list(messages),
                agent_id=user_input.agent_id,
                service="conversation",
                fallback_model=fallback_model,
            )
        except HomeAssistantError:
            raise
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error talking to xAI: {err}") from err

        intent_response = intent.IntentResponse(language=user_input.language)
        speech = self._pick_speech_content(chat_log)
        if speech:
            intent_response.async_set_speech(speech)
        else:
            intent_response.async_set_speech(
                "Sorry, I couldn't generate a response."
            )

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=await self._resolve_continue_conversation(
                user_input, chat_log, speech
            ),
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)
