# xAI Grok Conversation

**Home Assistant conversation agent + cloud Voice TTS/STT powered by xAI Grok.**

One HACS install → **Conversation agent**, **Speech-to-text**, and **Text-to-speech** engines that use your existing xAI API key.

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

Replace Piper / speech-to-phrase with Grok cloud quality while keeping your Satellite1 (or any Assist pipeline) wake word local.

---

## Features

| Feature | Description |
| --- | --- |
| **Assist / LLM tools** | Control exposed HA entities via the standard LLM HASS API |
| **xAI TTS** | 25+ expressive voices (Eve, Ara, Rex, Luna, …), speed, languages |
| **xAI STT** | Multilingual transcription (25+ languages), PCM/WAV from Assist satellites |
| **Voice API check** | Probes the key for Voice access at setup; warns if chat-only |
| **Interaction modes** | `tools` · `pipeline` · `chat_only` |
| **Live Search** | Web / X / Full + citations |
| **Services** | `ask`, `photo_analysis`, `home_briefing`, image/content generation |

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
| Default TTS voice | Eve, Ara, Rex, Luna, … |
| Default TTS language | `en`, `es-ES`, `pt-BR`, … |
| TTS speed | 0.7–1.5 |
| Default STT language | Formatting language for transcripts |
| Live Search | off / web / x / full |
| Interaction mode | tools / pipeline / chat_only |

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
  __init__.py          # setup + services + platforms
  conversation.py      # Assist conversation agent
  tts.py               # TextToSpeechEntity → /v1/tts
  stt.py               # SpeechToTextEntity → /v1/stt
  voice_api.py         # HTTP client + Voice probe
  voice_const.py       # voices + languages
  config_flow.py
  sensor.py / usage.py
```

CI: Hassfest + HACS validation on push/PR/nightly.

---

## Version

**1.7.1** — Fix STT crash (`SAMPLERATE_22050` → valid HA rates); rename engines to **xAI TTS** / **xAI STT**.

**1.7.0** — xAI Voice **TTS** + **STT** engines for Assist (shared API key, voice catalog, languages, Voice API probe).

**1.6.2** — Issue sweep: real tool results, openai pin, Assist API UX.

**1.6.1** — Tool calls on HA 2026.8+.

**1.6.0** — Live search, modes, sensors, services.

## License / trademarks

MIT. Unofficial. xAI / Grok are trademarks of xAI Corp. API use subject to [xAI terms](https://x.ai/legal/).
