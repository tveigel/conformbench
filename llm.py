"""Centralised LLM configuration.

Configure via .env:
    OPENAI_API_KEY=...        -> uses OpenAI models
    ANTHROPIC_API_KEY=...     -> uses Anthropic models
    OPENAI_BASE_URL=...       -> optional OpenAI-compatible endpoint
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

CHAT_MODEL_OPENAI = "gpt-5.4"
CHAT_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"
MAX_TOKENS = 32_000
TIMEOUT = 300
MAX_RETRIES = 3
_OPENAI_BASE_URL_ENV = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)
_OPENAI_API_KEY_ENV = (
    "OPENAI_API_KEY",
)
_OPENAI_MODEL_ENV = ("OPENAI_MODEL", "OPENAI_DEFAULT_MODEL")


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _detect_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "").strip().lower()
    if backend in ("openai", "anthropic"):
        return backend
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise EnvironmentError(
        "No LLM backend configured. Set one of:\n"
        "  - OPENAI_API_KEY=...\n"
        "  - ANTHROPIC_API_KEY=...\n"
        "in your .env file."
    )


def _default_model(backend: str) -> str:
    if backend == "anthropic":
        return CHAT_MODEL_ANTHROPIC
    if backend == "openai":
        return _first_env(_OPENAI_MODEL_ENV) or CHAT_MODEL_OPENAI
    return CHAT_MODEL_OPENAI


def _parse_model_spec(model: str | None) -> tuple[str | None, str | None]:
    if model and ":" in model:
        provider, name = model.split(":", 1)
        # Treat ``provider:model`` as an explicit routing hint, but leave
        # OpenRouter-style ids such as ``moonshotai/kimi-k2.6:free`` intact.
        if "/" not in provider:
            return provider.strip().lower(), name.strip()
    return None, model


def _select_backend(provider_override: str | None) -> str:
    if not provider_override:
        return _detect_backend()
    if provider_override == "anthropic":
        return "anthropic"
    return "openai"


def _normalize_public_anthropic_model(model: str | None) -> str | None:
    aliases = {
        "claude-sonnet-4-6-20250514": "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001": "claude-3-5-haiku-20241022",
    }
    return aliases.get(model or "", model)


def _anthropic_reasoning_kwargs(model: str | None, reasoning_effort: str | None) -> dict[str, Any]:
    if not reasoning_effort:
        return {}
    effort = reasoning_effort.strip().lower()
    if effort in {"none", "off", "disabled", "false", "0"}:
        return {}
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        return {}

    kwargs: dict[str, Any] = {"effort": effort}
    if (model or "").startswith("claude-opus-4-7"):
        kwargs["thinking"] = {"type": "adaptive"}
    return kwargs


def _build_model(backend: str, **kw):
    kw.setdefault("model", _default_model(backend))
    kw.setdefault("max_tokens", MAX_TOKENS)
    kw.setdefault("timeout", TIMEOUT)
    kw.setdefault("max_retries", MAX_RETRIES)

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kw["model"] = _normalize_public_anthropic_model(kw.get("model"))
        reasoning_effort = kw.pop("reasoning_effort", None)
        kw.update(_anthropic_reasoning_kwargs(kw.get("model"), reasoning_effort))
        kw.pop("streaming", None)
        kw.pop("model_kwargs", None)
        return ChatAnthropic(**kw)

    from langchain_openai import ChatOpenAI

    base_url = _first_env(_OPENAI_BASE_URL_ENV)
    api_key = _first_env(_OPENAI_API_KEY_ENV)
    _apply_openai_compatible_reasoning_kwargs(kw, base_url=base_url or "")
    kw.pop("streaming", None)
    kw.pop("model_kwargs", None)
    if base_url:
        kw["base_url"] = base_url
    if api_key:
        kw["api_key"] = api_key
    return ChatOpenAI(**kw)


def _apply_openai_compatible_reasoning_kwargs(kw: dict[str, Any], *, base_url: str) -> None:
    """Keep provider-specific reasoning controls only where they are supported."""

    effort = str(kw.get("reasoning_effort") or "").strip().lower()
    if effort in {"", "none", "off", "disabled", "false", "0"}:
        kw.pop("reasoning_effort", None)
        return

    if "groq.com" not in base_url.lower():
        kw.pop("reasoning_effort", None)
        return

    model = str(kw.get("model") or "").strip().lower()
    if model == "qwen/qwen3-32b":
        if effort != "default":
            raise ValueError(
                "Groq qwen/qwen3-32b supports reasoning_effort='default' "
                "for thinking mode, or 'none' for non-thinking mode."
            )
        kw["reasoning_effort"] = "default"
        extra_body = dict(kw.get("extra_body") or {})
        extra_body.setdefault("reasoning_format", "hidden")
        kw["extra_body"] = extra_body
        return

    if model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        if effort not in {"low", "medium", "high"}:
            raise ValueError(
                "Groq GPT-OSS models support reasoning_effort low, medium, or high; "
                "use 'none' to disable reasoning."
            )
        kw["reasoning_effort"] = effort
        return

    kw.pop("reasoning_effort", None)


def _routed_model_name(model: str | None, provider: str | None) -> str | None:
    if not model:
        return model
    if "/" in model:
        return model
    if provider == "anthropic" or model.startswith("claude-"):
        return f"anthropic/{model}"
    if provider and provider != "openai":
        return f"{provider}/{model}"
    return model


def get_chat_model_with_config(
    *,
    tools: Sequence[Any] | None = None,
    **overrides: Any,
):
    rotation_key = overrides.pop("rotation_key", None)
    exclude_model = overrides.pop("exclude_model", None)
    requested_model = overrides.get("model")
    provider_override, clean_model = _parse_model_spec(requested_model)
    if clean_model:
        overrides["model"] = clean_model
    backend = _select_backend(provider_override)
    overrides.setdefault("model", _default_model(backend))
    if backend == "anthropic":
        overrides["model"] = _normalize_public_anthropic_model(overrides["model"])
    else:
        overrides["model"] = _routed_model_name(overrides["model"], provider_override)
    config = {
        "provider": backend,
        "model": overrides["model"],
        "requested_model": requested_model or overrides["model"],
        "resolved_model_name": overrides["model"],
        "resolved_model_version": overrides["model"],
        "max_tokens": overrides.get("max_tokens", MAX_TOKENS),
        "timeout": overrides.get("timeout", TIMEOUT),
        "temperature": overrides.get("temperature"),
    }
    if overrides.get("reasoning_effort"):
        config["reasoning_effort"] = overrides["reasoning_effort"]
    if rotation_key is not None:
        config["rotation_key"] = rotation_key
    if exclude_model is not None:
        config["exclude_model"] = exclude_model
    model = _build_model(backend, **overrides)
    if tools:
        model = model.bind_tools(tools)
    return model, config


def get_chat_model(
    *,
    tools: Sequence[Any] | None = None,
    **overrides: Any,
):
    model, _ = get_chat_model_with_config(tools=tools, **overrides)
    return model


def get_judge_model_with_config(**overrides: Any):
    overrides.setdefault("model", CHAT_MODEL_ANTHROPIC)
    overrides.setdefault("timeout", TIMEOUT)
    return get_chat_model_with_config(**overrides)


def get_judge_model(**overrides: Any):
    model, _ = get_judge_model_with_config(**overrides)
    return model


def resolve_chat_model_config(**overrides: Any) -> dict[str, Any]:
    requested_model = overrides.get("model")
    provider_override, clean_model = _parse_model_spec(requested_model)
    backend = _select_backend(provider_override)
    model = clean_model or _default_model(backend)
    if backend == "anthropic":
        model = _normalize_public_anthropic_model(model)
    else:
        model = _routed_model_name(model, provider_override)
    return {
        "provider": backend,
        "model": model,
        "requested_model": requested_model or model,
    }


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)
