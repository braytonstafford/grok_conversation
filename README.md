# xAI Grok Conversation

**Home Assistant conversation agent + cloud Voice TTS/STT + AI Task, powered by xAI Grok.**

One HACS install → **Conversation agent**, **Speech-to-text**, **Text-to-speech**, and **AI Task** engines that use your existing xAI API key.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/braytonstafford/grok_conversation)
[![GitHub release](https://img.shields.io/github/v/release/braytonstafford/grok_conversation)](https://github.com/braytonstafford/grok_conversation/releases)
[![HA](https://img.shields.io/badge/Home%20Assistant-2026.7%2B-blue)](https://www.home-assistant.io/)

---

## What's included

| Engine | Where it appears | API |
| --- | --- | --- |
| **Conversation** | Voice assistants → Conversation agent → **Grok** | Chat / tools / live search |
| **Speech-to-text** | Voice assistants → Speech-to-text → **xAI Grok** | `POST https://api.x.ai/v1/stt` |
| **Text-to-speech** | Voice assistants → Text-to-speech → **xAI Grok** | `POST https://api.x.ai/v1/tts` |
| **AI Task** | Automations → `ai_task.generate_data` → **Grok AI Task** | Chat Completions + structured output / attachments |

Replace Piper / speech-to-phrase with Grok cloud quality while keeping your Satellite1 (or any Assist pipeline) wake word local.

---

## Features

| Feature | Description |
| --- | --- |
| **Assist / LLM tools** | Control exposed HA entities via the standard LLM HASS API |
| **AI Task** | `ai_task.generate_data` with free text, structured JSON, and image attachments |
| **xAI TTS** | 25+ expressive voices (Eve, Ara, Rex, Luna, …), speed, languages |
| **xAI STT** | Multilingual transcription (25+ languages), PCM/WAV from Assist satellites |
| **Voice API check** | Probes the key for Voice access at setup; warns if chat-only |
| **Interaction modes** | `tools` · `pipeline` · `chat_only` |
| **Live Search** | Web / X / Full + citations |
| **Services** | `ask`, `photo_analysis`, `home_briefing`, image/content generation |

### Example: AI Task automation

```yaml
- action: ai_task.generate_data
  data:
    task_name: Driveway check
    entity_id: ai_task.grok_ai_task
    instructions: How many cars are in the driveway?
    attachments:
      media_content_id: media-source://camera/camera.driveway
      media_content_type: image/jpeg
    structure:
      cars:
        selector:
          number:
  response_variable: result
```

---

## Installation (HACS)

1. HACS → Integrations → ⋮ → **Custom repositories**
2. URL: `https://github.com/braytonstafford/grok_conversation` · Category: **Integration**
3. Download **xAI Grok (Conversation + Voice TTS/STT)**
4. Restart Home Assistant
5. Settings → Devices & Services → Add Integration → **xAI Grok Conversation**
6. Paste API key from [console.x.ai](https://console.x.ai/)

### Wire Assist (Voice assistants UI)

1. **Settings → Voice assistants →** edit *Local Assistant* (or add one)
2. **Conversation agent** → **Grok**
3. **Speech-to-text** → **xAI Grok** · Language e.g. `en` / English
4. **Text-to-speech** → **xAI Grok** · Language e.g. `en` · **Voice** → Eve / Ara / Rex / …
5. **Try voice** to preview TTS
6. Save / Update

Default voice/language can also be set under the integration **Configure** options (TTS voice, TTS language, STT language, speed).

### Voice API access

Setup probes your key against the Voice endpoints. If chat works but TTS/STT fail:

- In the [xAI console](https://console.x.ai/), ensure the key/team has **Voice** enabled
- Check HA logs for `xAI Voice API not available for this key`
- Conversation still loads; only TTS/STT need Voice permissions

---

## Configuration options

| Option | Notes |
| --- | --- |
| LLM HASS API | Assist control (not "No control") |
| Enable xAI TTS / STT | Toggle engines (reload after change) |
| **Chat model** | Primary model — live list from xAI API |
| **Fast model** | Used by auto-routing for short commands |
| **Fallback model** | Tried if the primary errors |
| **Vision model** | Used by `photo_analysis` / `query_image` / image attachments (default `grok-4.3`) |
| Default TTS voice | Eve, Ara, Rex, Luna, … |
| Default TTS language | `en`, `es-ES`, `pt-BR`, … |
| TTS speed | 0.7–1.5 |
| Default STT language | Formatting language for transcripts |
| Live Search | off / web / x / full |
| Interaction mode | tools / pipeline / chat_only |
| AI Task subentry | Name, chat model, optional image model (generate_image deferred) |

---

## Services

### `grok_conversation.generate_image`

Generate images with Grok Imagine. Only documented xAI parameters are sent.

| Field | Notes |
| --- | --- |
| `prompt` | Required |
| `model` | Default `grok-imagine-image`. xAI recommends `grok-imagine-image-2.0` |
| `aspect_ratio` | Optional (`1:1`, `16:9`, `9:16`, …). Omit for xAI `auto` |
| `resolution` | Optional `1k` / `2k` |
| `quality` | Optional `low` / `medium` / `auto` (2.0 only) |
| `n` | 1–10 (default 1) |
| `response_format` | `url` (default) or `b64_json` |

Deprecated (still accepted): `size` (mapped to aspect ratio), `style` (ignored), `quality: standard|hd` (ignored).

```yaml
action: grok_conversation.generate_image
data:
  config_entry: YOUR_ENTRY_ID
  prompt: A cozy living room at dusk
  aspect_ratio: "16:9"
  response_format: b64_json
  model: grok-imagine-image-2.0
```

Response includes `images` (list of `{url}` or `{b64_json}` plus `mime_type` when present). When `response_format=url`, `url` is also set to the first image URL.

### Vision services

`photo_analysis`, `query_image`, `generate_content` (with image files), and AI Task attachments share one resolution rule: an **image-capable chat model always wins**; the **Vision model** option is only the fallback when the chat model cannot take images (default `grok-4.3`). Per-call `model` overrides remap retired ids (`grok-2-vision-*`).

---

## Troubleshooting

**xAI Grok missing from STT/TTS dropdowns**

- Update to **1.7.0+**, restart HA
- Integration → Configure → **Enable TTS** / **Enable STT** on
- Confirm entities under Developer Tools → States (`tts.`, `stt.`)

**TTS/STT errors in logs**

- Key must allow Voice (`/v1/tts`, `/v1/stt`)
- Re-save the integration after enabling Voice on the key

**Tools / device control**

See 1.6.2 notes — real tool payloads + ToolInput fix. Set LLM HASS API to Assist and expose entities.

---

## Development

```text
custom_components/grok_conversation/
  __init__.py          # setup + services + platforms + migration
  entity.py            # shared LLM base (tool loop, structured output, attachments)
  conversation.py      # Assist conversation agent
  ai_task.py           # AI Task entity (generate_data)
  tts.py               # TextToSpeechEntity → /v1/tts
  stt.py               # SpeechToTextEntity → /v1/stt
  voice_api.py         # HTTP client + Voice probe
  voice_const.py       # voices + languages
  config_flow.py       # options + ai_task_data subentry
  sensor.py / usage.py
```

CI: Hassfest + HACS validation + pytest on push/PR/nightly.

---

## Version

**1.10.1** — Pin `openai==3.10.0` to match HA core; return `llm.ToolResult` for parse-error tool seeds (HA 2026.10 chat log).

**1.10.0** — `generate_image` uses documented xAI params (`aspect_ratio` / `resolution` / `quality` / `n` / `response_format`); deprecated `size` / `style` / `quality: standard|hd` still accepted with warnings; `quality` is sent only for `grok-imagine-image-2.0`. Vision default `grok-4.3` with options picker + shared resolution (image-capable chat model wins; vision option is fallback). Replace retired fast/fallback defaults (`grok-4-1-fast-non-reasoning`, `grok-3-mini-fast`) with `grok-4.3` after the [May 15 2026 retirement](https://docs.x.ai/developers/migration/may-15-retirement) (no cheaper/faster documented non-reasoning tool-calling model remains). Options picker rejects retired ids; runtime remap + warning. Minor-3 migration rewrites stored vision/fast/fallback/chat ids on entry options and conversation/ai_task_data subentries.

**1.9.0** — AI Task platform (`generate_data` + structured output + image attachments). Shared LLM entity base. Config subentry + migration (minor 2).

**1.8.0** — Pipeline live-search deny-list (#30); satellite TTS prewarm before continue (#31); persona on search pass + no spoken citations (#32). Thanks @CodyJon.

**1.7.7** — Declare `voluptuous-openapi` so setup works on HA 2026.9 (#28).

**1.7.6** — Search pass always gets location/time/presence (#27); "near me" heuristics.

**1.7.5** — Live Search works with Assist tools (#26 two-pass: search brief → tool loop).

**1.7.4** — Always-visible model pickers (chat / fast / fallback) loaded live from xAI `/v1/models` (non-chat models filtered).

**1.7.3** — Merge all anyOf object-branch properties for timer tools (#25, thanks @CodyJon).

**1.7.2** — Sanitize HA tool JSON schemas for xAI (fix HassStartTimer anyOf/oneOf 400).

**1.7.1** — Fix STT crash (`SAMPLERATE_22050` → valid HA rates); rename engines to **xAI TTS** / **xAI STT**.

**1.7.0** — xAI Voice **TTS** + **STT** engines for Assist (shared API key, voice catalog, languages, Voice API probe).

**1.6.2** — Issue sweep: real tool results, openai pin, Assist API UX.

**1.6.1** — Tool calls on HA 2026.8+.

**1.6.0** — Live search, modes, sensors, services.

## License / trademarks

MIT. Unofficial. xAI / Grok are trademarks of xAI Corp. API use subject to [xAI terms](https://x.ai/legal/).
