"""Shared xAI API helpers (chat completions + Responses live search)."""

from __future__ import annotations

from typing import Any

import openai

from .const import (
    LIVE_SEARCH_FULL,
    LIVE_SEARCH_OFF,
    LIVE_SEARCH_WEB,
    LIVE_SEARCH_X,
    LOGGER,
)


def build_live_search_tools(live_search: str) -> list[dict[str, Any]]:
    """Return Responses API server-side search tools for the given mode."""
    mode = (live_search or LIVE_SEARCH_OFF).lower().strip()
    tools: list[dict[str, Any]] = []
    if mode in (LIVE_SEARCH_WEB, LIVE_SEARCH_FULL, "web search", "on", "auto"):
        tools.append({"type": "web_search"})
    if mode in (LIVE_SEARCH_X, LIVE_SEARCH_FULL, "x search", "on", "auto"):
        tools.append({"type": "x_search"})
    return tools


def format_citations(citations: Any) -> str:
    """Format citation URLs into a readable footer."""
    if not citations:
        return ""
    urls: list[str] = []
    if isinstance(citations, (list, tuple)):
        for item in citations:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict):
                url = item.get("url") or item.get("uri") or item.get("id")
                if url:
                    urls.append(str(url))
            else:
                url = getattr(item, "url", None) or getattr(item, "uri", None)
                if url:
                    urls.append(str(url))
    elif isinstance(citations, str):
        urls = [citations]
    # Dedupe preserve order
    seen: set[str] = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    if not unique:
        return ""
    lines = "\n".join(f"- {u}" for u in unique[:12])
    return f"\n\nSources:\n{lines}"


def extract_responses_text(response: Any) -> str:
    """Pull assistant text out of a Responses API payload."""
    # Newer SDKs expose output_text
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    chunks: list[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        item_type = getattr(item, "type", None) or (
            item.get("type") if isinstance(item, dict) else None
        )
        if item_type == "message":
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            for part in content or []:
                ptype = getattr(part, "type", None) or (
                    part.get("type") if isinstance(part, dict) else None
                )
                if ptype in ("output_text", "text"):
                    value = getattr(part, "text", None)
                    if value is None and isinstance(part, dict):
                        value = part.get("text")
                    if value:
                        chunks.append(str(value))
        elif item_type in ("output_text", "text"):
            value = getattr(item, "text", None)
            if value is None and isinstance(item, dict):
                value = item.get("text")
            if value:
                chunks.append(str(value))
    return "".join(chunks).strip()


def extract_responses_citations(response: Any) -> list[Any]:
    """Best-effort citation extraction from Responses API result."""
    citations = getattr(response, "citations", None)
    if citations:
        return list(citations)
    # Some payloads nest citations under output annotations
    found: list[Any] = []
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            anns = getattr(part, "annotations", None)
            if anns is None and isinstance(part, dict):
                anns = part.get("annotations")
            for ann in anns or []:
                url = getattr(ann, "url", None)
                if url is None and isinstance(ann, dict):
                    url = ann.get("url")
                if url:
                    found.append(url)
    return found


def extract_usage(response: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from chat or responses result."""
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0) or 0
    return int(prompt or 0), int(completion or 0)


async def async_chat_completion(
    client: openai.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    reasoning_effort: str | None = None,
    user: str | None = None,
) -> Any:
    """Call chat.completions.create with optional reasoning_effort."""
    args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if temperature is not None:
        args["temperature"] = temperature
    if top_p is not None:
        args["top_p"] = top_p
    if user:
        args["user"] = user
    if tools:
        args["tools"] = tools
        args["tool_choice"] = tool_choice or "auto"
    if (
        reasoning_effort
        and reasoning_effort != "none"
        and "reasoning" in model.lower()
    ):
        args["reasoning_effort"] = reasoning_effort
    LOGGER.debug("chat.completions.create model=%s tools=%s", model, bool(tools))
    return await client.chat.completions.create(**args)


async def async_responses_completion(
    client: openai.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    live_search: str = LIVE_SEARCH_OFF,
    show_citations: bool = True,
    reasoning_effort: str | None = None,
) -> tuple[str, int, int]:
    """Call Responses API (supports xAI live web/X search). Returns text, prompt_tok, completion_tok."""
    tools = build_live_search_tools(live_search)
    # Convert chat-style messages to Responses `input`
    input_items: list[dict[str, Any]] = []
    sys_parts: list[str] = []
    if system_prompt:
        sys_parts.append(system_prompt)
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                sys_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, list):
            # multimodal
            input_items.append({"role": role, "content": content})
        else:
            input_items.append({"role": role, "content": str(content or "")})

    args: dict[str, Any] = {
        "model": model,
        "input": input_items,
    }
    if sys_parts:
        args["instructions"] = "\n\n".join(sys_parts)
    if max_tokens is not None:
        # Responses API uses max_output_tokens
        args["max_output_tokens"] = max_tokens
    if temperature is not None:
        args["temperature"] = temperature
    if top_p is not None:
        args["top_p"] = top_p
    if tools:
        args["tools"] = tools
    if (
        reasoning_effort
        and reasoning_effort != "none"
        and "reasoning" in model.lower()
    ):
        args["reasoning"] = {"effort": reasoning_effort}

    LOGGER.debug(
        "responses.create model=%s live_search=%s tools=%s",
        model,
        live_search,
        [t.get("type") for t in tools],
    )
    try:
        response = await client.responses.create(**args)
    except Exception as err:  # noqa: BLE001 - surface as OpenAIError-compatible
        LOGGER.warning("Responses API failed (%s); caller may fall back", err)
        raise

    text = extract_responses_text(response)
    if show_citations:
        text = text + format_citations(extract_responses_citations(response))
    p_tok, c_tok = extract_usage(response)
    return text, p_tok, c_tok


def looks_like_search_query(text: str) -> bool:
    """Heuristic: user wants fresh/web/X info."""
    t = (text or "").lower()
    keywords = (
        "latest",
        "news",
        "headline",
        "today",
        "right now",
        "current",
        "score",
        "stock",
        "price of",
        "weather outside",
        "who won",
        "trending",
        "on x",
        "on twitter",
        "search the web",
        "look up",
        "google",
        "what happened",
    )
    return any(k in t for k in keywords)


def looks_like_simple_query(text: str) -> bool:
    """Heuristic for auto-routing to a fast model."""
    t = (text or "").strip()
    if len(t) > 160:
        return False
    simple_starts = (
        "turn ",
        "switch ",
        "set ",
        "open ",
        "close ",
        "lock ",
        "unlock ",
        "play ",
        "pause ",
        "stop ",
        "what's the",
        "what is the",
        "is the ",
        "are the ",
        "how warm",
        "how cold",
        "temperature",
        "lights",
        "good morning",
        "good night",
        "hello",
        "hi ",
        "thanks",
        "thank you",
    )
    lower = t.lower()
    if any(lower.startswith(s) for s in simple_starts):
        return True
    # Short yes/no or single command
    return len(t.split()) <= 8 and "?" not in t[20:]
