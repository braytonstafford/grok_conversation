"""Constants for the Grok Conversation integration."""

import logging

DOMAIN = "grok_conversation"
LOGGER = logging.getLogger(__package__)

CONF_CHAT_MODEL = "chat_model"
CONF_FILENAMES = "filenames"
CONF_MAX_TOKENS = "max_tokens"
CONF_PAYLOAD_TEMPLATE = "payload_template"
CONF_PROMPT = "prompt"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
EVENT_AUTOMATION_REGISTERED = "automation_registered"
SERVICE_QUERY_IMAGE = "query_image"
# Recommended chat model - grok-4-fast provides fast responses with grok-4 capabilities
# Other available models: grok-2-beta (faster but less capable), grok-4 (slower but most capable)
# Fast variants: grok-3-mini-fast, grok-3-fast, grok-4-fast (if available in your account)
RECOMMENDED_CHAT_MODEL = "grok-4-fast"
RECOMMENDED_VISION_MODEL = "grok-2-vision-1212"
RECOMMENDED_IMAGE_GENERATION_MODEL = "grok-2-image-1212"
RECOMMENDED_MAX_TOKENS = 150
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0

# Grok models that are not supported or deprecated
UNSUPPORTED_MODELS: list[str] = []

# Image generation constants
IMAGE_SIZES = ("1024x1024", "1024x1792", "1792x1024")
IMAGE_QUALITIES = ("standard", "hd")
IMAGE_STYLES = ("vivid", "natural")

# Custom system prompt to improve tool usage
GROK_SYSTEM_PROMPT = """
You are Grok, a helpful and maximally truthful AI built by xAI.

When answering questions:
- For questions about current events, news, sports scores, weather, or real-time data, use the available tools to get accurate information from Home Assistant.
- For general knowledge questions (history, geography, science, definitions, etc.), answer directly from your training data without calling tools.
- For questions about controlling smart home devices, use the appropriate tools to interact with Home Assistant.
- Only call tools when you actually need current or device-specific information from Home Assistant.

If a tool call doesn't provide useful information, continue the conversation normally and answer based on your knowledge.
"""