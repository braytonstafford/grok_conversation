"""Constants for the Grok Conversation integration."""

import logging

DOMAIN = "grok_conversation"
LOGGER = logging.getLogger(__package__)

# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------
CONF_CHAT_MODEL = "chat_model"
CONF_FAST_MODEL = "fast_model"
CONF_FALLBACK_MODEL = "fallback_model"
CONF_VISION_MODEL = "vision_model"
CONF_IMAGE_MODEL = "image_model"
CONF_FILENAMES = "filenames"
CONF_MAX_TOKENS = "max_tokens"
CONF_PAYLOAD_TEMPLATE = "payload_template"
CONF_PROMPT = "prompt"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"

# Feature options
CONF_LIVE_SEARCH = "live_search"
CONF_SHOW_CITATIONS = "show_citations"
CONF_SEND_USER_NAME = "send_user_name"
CONF_LOCATION_CONTEXT = "location_context"
CONF_INTERACTION_MODE = "interaction_mode"
CONF_VOICE_OPTIMIZED = "voice_optimized"
CONF_AUTO_MODEL_ROUTING = "auto_model_routing"
CONF_HOME_CONTEXT = "home_context"
CONF_BUDGET_WARN_USD = "budget_warn_usd"

EVENT_AUTOMATION_REGISTERED = "automation_registered"
EVENT_USAGE_UPDATED = f"{DOMAIN}_usage_updated"

SERVICE_QUERY_IMAGE = "query_image"
SERVICE_ASK = "ask"
SERVICE_PHOTO_ANALYSIS = "photo_analysis"
SERVICE_CLEAR_MEMORY = "clear_memory"
SERVICE_RESET_STATS = "reset_stats"
SERVICE_HOME_BRIEFING = "home_briefing"
SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_GENERATE_CONTENT = "generate_content"

# ---------------------------------------------------------------------------
# Defaults / recommended
# ---------------------------------------------------------------------------
RECOMMENDED_CHAT_MODEL = "grok-4.3-latest"
RECOMMENDED_FAST_MODEL = "grok-4-1-fast-non-reasoning"
RECOMMENDED_FALLBACK_MODEL = "grok-3-mini-fast"
RECOMMENDED_VISION_MODEL = "grok-2-vision-1212"
RECOMMENDED_IMAGE_GENERATION_MODEL = "grok-imagine-image"
# Keep legacy alias used elsewhere
RECOMMENDED_IMAGE_MODEL = RECOMMENDED_IMAGE_GENERATION_MODEL

RECOMMENDED_MAX_TOKENS = 600
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_LIVE_SEARCH = "off"
RECOMMENDED_SHOW_CITATIONS = True
RECOMMENDED_SEND_USER_NAME = True
RECOMMENDED_LOCATION_CONTEXT = ""
RECOMMENDED_INTERACTION_MODE = "tools"
RECOMMENDED_VOICE_OPTIMIZED = True
RECOMMENDED_AUTO_MODEL_ROUTING = True
RECOMMENDED_HOME_CONTEXT = True
RECOMMENDED_BUDGET_WARN_USD = 0.0  # 0 = disabled

UNSUPPORTED_MODELS: list[str] = []

# Image generation constants
IMAGE_SIZES = ("1024x1024", "1024x1792", "1792x1024")
IMAGE_QUALITIES = ("standard", "hd")
IMAGE_STYLES = ("vivid", "natural")

# Live search modes (xAI Responses API server tools)
LIVE_SEARCH_OFF = "off"
LIVE_SEARCH_WEB = "web"
LIVE_SEARCH_X = "x"
LIVE_SEARCH_FULL = "full"
LIVE_SEARCH_OPTIONS = [
    LIVE_SEARCH_OFF,
    LIVE_SEARCH_WEB,
    LIVE_SEARCH_X,
    LIVE_SEARCH_FULL,
]

# Interaction modes
MODE_TOOLS = "tools"  # Direct tool calling against HA LLM API
MODE_PIPELINE = "pipeline"  # Try HA Assist intent first, fall back to Grok tools
MODE_CHAT_ONLY = "chat_only"  # Pure chat, no device control
INTERACTION_MODE_OPTIONS = [MODE_TOOLS, MODE_PIPELINE, MODE_CHAT_ONLY]

# Rough default USD pricing per 1M tokens (overridden if API exposes pricing)
DEFAULT_INPUT_PRICE_PER_M = 3.0
DEFAULT_OUTPUT_PRICE_PER_M = 15.0

# Custom system prompt to improve tool usage
GROK_SYSTEM_PROMPT = """
You are Grok, a helpful and maximally truthful AI built by xAI, integrated with Home Assistant.

When answering questions:
- For questions about current events, news, sports scores, weather outside HA, or real-time web/X data, use live search when available.
- For general knowledge questions (history, geography, science, definitions, etc.), answer directly from your training data without calling Home Assistant tools.
- For questions about controlling smart home devices or local entity state, use the appropriate Home Assistant tools.
- Only call Home Assistant tools when you actually need current or device-specific information from Home Assistant.

If a tool call doesn't provide useful information, continue the conversation normally and answer based on your knowledge.
Keep spoken Assist replies concise unless the user asks for detail.
"""

VOICE_OPTIMIZED_SUFFIX = (
    "\nYou are responding through a voice assistant. "
    "Keep answers short (1-3 sentences) unless the user asks for detail. "
    "Avoid markdown, bullet lists, and code blocks in spoken replies."
)
