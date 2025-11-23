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
OPENAI_BASE_URL = "https://api.x.ai/v1"
SERVICE_QUERY_IMAGE = "query_image"
RECOMMENDED_CHAT_MODEL = "grok-4"
RECOMMENDED_VISION_MODEL = "grok-2-vision-1212"
RECOMMENDED_IMAGE_GENERATION_MODEL = "grok-2-image-1212"
RECOMMENDED_MAX_TOKENS = 150
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0

# Grok models that are not supported or deprecated
UNSUPPORTED_MODELS: list[str] = []