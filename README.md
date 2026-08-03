![xAI logo](https://brands.home-assistant.io/_/grok_conversation/icon.png)

# xAI Grok Conversation

**Home Assistant conversation agent powered by xAI Grok** — Assist control, live Web/X search, vision, image generation, usage sensors, and automation-ready services.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/braytonstafford/grok_conversation)
[![GitHub release](https://img.shields.io/github/v/release/braytonstafford/grok_conversation)](https://github.com/braytonstafford/grok_conversation/releases)
[![HA](https://img.shields.io/badge/Home%20Assistant-2026.7%2B-blue)](https://www.home-assistant.io/)

Derived from [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) patterns, pointed at the xAI OpenAI-compatible API (`https://api.x.ai/v1`) plus the **Responses API** for server-side Web/X search.

---

## Features

| Feature | Description |
| --- | --- |
| **Assist / LLM tools** | Control exposed HA entities via the standard LLM HASS API |
| **Interaction modes** | `tools` · `pipeline` (HA intent first → Grok) · `chat_only` |
| **Live Search** | Web Search, X Search, or Full — with optional **citations** |
| **User recognition** | Prefixed person/user name so Grok addresses you correctly |
| **Location context** | e.g. `Longview, TX` for local weather/news grounding |
| **Voice-optimized** | Short spoken answers for Assist satellites |
| **Home context** | Injects local time, person presence, and weather snapshot |
| **Auto model routing** | Simple commands → fast model; complex → primary model |
| **Fallback model** | Automatic retry on primary model API errors |
| **Usage sensors** | Tokens, requests, estimated USD cost, last model |
| **Budget warning event** | `grok_conversation_budget_warning` when spend crosses a threshold |
| **Services** | `ask`, `photo_analysis`, `home_briefing`, `generate_image`, `generate_content`, `clear_memory`, `reset_stats` |

### Why this vs other xAI integrations?

- Stays on the **OpenAI-compatible SDK** (simple deps, familiar stack) while still using **Responses live search**.
- **Auto model routing + fallback** for lower latency/cost on routine Assist commands.
- **`home_briefing`** service — one call snapshots HA state and returns a spoken status report (great for morning automations).
- **Budget warning** event for spend guardrails.
- Voice-first defaults without giving up power-user options.

---

## Installation

### HACS (recommended)

1. HACS → Integrations → ⋮ → Custom repositories  
2. URL: `https://github.com/braytonstafford/grok_conversation` · Category: Integration  
3. Download **xAI Grok Conversation**  
4. Restart Home Assistant  

### Manual

Copy `custom_components/grok_conversation` into `<config>/custom_components/` and restart.

### Setup

1. Settings → Devices & Services → Add Integration → **xAI Grok Conversation**  
2. Paste an API key from [console.x.ai](https://console.x.ai/)  
3. Settings → Voice Assistants → edit your assistant → set Conversation agent to **Grok**  
4. Expose entities under Voice Assistants → Expose  

Walkthrough: [blog post](https://braytonstafford.com/home-assistant-xai-grok-conversation-agent/)

---

## Configuration options

| Option | Notes |
| --- | --- |
| Instructions | System prompt (template OK) |
| LLM HASS API | Enable device control (not “No control”) |
| Interaction mode | `tools` / `pipeline` / `chat_only` |
| Live Search | `off` / `web` / `x` / `full` |
| Show citations | Append source URLs after search answers |
| Include username | Person entity name or HA user display name |
| Location context | Free-text home location |
| Voice-optimized | Concise Assist replies |
| Home context | Time + presence + weather injection |
| Auto model routing | Fast model for short commands |
| Recommended toggle | Hide advanced model knobs |
| Chat / Fast / Fallback models | e.g. `grok-4.3-latest`, `grok-4-1-fast-non-reasoning` |
| Budget warning (USD) | `0` disables |

---

## Services

### `grok_conversation.ask`

Stateless one-shot with optional live search — perfect for automations:

```yaml
service: grok_conversation.ask
data:
  config_entry: YOUR_ENTRY_ID
  instructions: "Summarize for a spoken morning briefing in under 60 words."
  input_data: >
    Weather: {{ states('weather.home') }}
    Temp: {{ state_attr('weather.home', 'temperature') }}
  live_search: web
  show_citations: false
response_variable: grok_reply
```

### `grok_conversation.photo_analysis`

```yaml
service: grok_conversation.photo_analysis
data:
  config_entry: YOUR_ENTRY_ID
  prompt: "Is anyone at the front door? Describe the scene."
  images:
    - /config/www/tmp/doorbell.jpg
```

### `grok_conversation.home_briefing`

```yaml
service: grok_conversation.home_briefing
data:
  config_entry: YOUR_ENTRY_ID
  focus: "Security and climate only. Mention if anyone is home."
```

### Other services

- `generate_image` / `generate_content` — creative + multimodal generation  
- `query_image` — legacy alias of photo analysis  
- `clear_memory` — reloads the integration agent  
- `reset_stats` — zeroes usage sensors  

---

## Sensors

Per config entry (diagnostic):

- `sensor.grok_*_total_tokens`
- `sensor.grok_*_prompt_tokens`
- `sensor.grok_*_completion_tokens`
- `sensor.grok_*_api_requests`
- `sensor.grok_*_estimated_cost`
- `sensor.grok_*_last_model` (attributes include per-model / per-service breakdown)

Events:

- `grok_conversation_usage_updated`
- `grok_conversation_budget_warning`

---

## Troubleshooting

**Tools / device control not working**

1. Options → LLM HASS API must not be “No control”  
2. Expose the entities under Voice Assistants  
3. Interaction mode must not be `chat_only`  
4. Enable debug logging:

```yaml
logger:
  logs:
    custom_components.grok_conversation: debug
```

**Live search not triggering**

- Set Live Search to `web`, `x`, or `full`  
- Search is used for chat-only turns, explicit search-y questions, and services that pass `live_search`  
- When HA tools are active on a turn, Grok prioritizes local tool calling unless the query looks like a search

**Slow replies**

- Enable auto-routing + a fast model  
- Lower `max_tokens`  
- Use `grok-4-1-fast-non-reasoning` / `grok-3-mini-fast` for Assist

---

## Development

```text
custom_components/grok_conversation/
  __init__.py          # setup + services
  api_helpers.py       # chat completions + Responses live search
  conversation.py      # Assist agent
  config_flow.py       # config / options
  sensor.py            # usage sensors
  usage.py             # persisted token accounting
  const.py
  services.yaml
  manifest.json
```

CI: Hassfest + HACS validation on push/PR/nightly.

---

## Version

**1.6.0** — Live search, interaction modes, user/location/voice/home context, auto-routing, fallback model, usage sensors, ask/photo/home_briefing services, CI refresh.

## License / trademarks

Unofficial Home Assistant custom integration. xAI / Grok are trademarks of xAI Corp. API use is subject to [xAI terms](https://x.ai/legal/).
