"""Provider selection and model discovery for the unified AutoPore GUI."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VALID_PROVIDERS = ("local", "openai", "openrouter")


def provider_defaults(provider: str) -> tuple[str, str]:
    provider = provider.strip().lower()
    if provider == "local":
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        base_url = os.environ.get("OLLAMA_BASE_URL", "") or (
            host if host.endswith("/v1") else f"{host}/v1"
        )
        return os.environ.get("OLLAMA_MODEL", "auto"), base_url
    if provider == "openrouter":
        return (
            os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "https://openrouter.ai/api/v1",
        )
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), ""


@dataclass
class LLMSettings:
    provider: str = "local"
    model: str = "auto"
    base_url: str = "http://127.0.0.1:11434/v1"

    def validate(self) -> None:
        self.provider = self.provider.strip().lower()
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(f"Provider must be one of: {', '.join(VALID_PROVIDERS)}")
        self.model = self.model.strip()
        self.base_url = self.base_url.strip()
        if not self.model:
            self.model, default_url = provider_defaults(self.provider)
            if not self.base_url:
                self.base_url = default_url


def default_settings() -> LLMSettings:
    provider = os.environ.get("AUTOPORE_LLM", "local").strip().lower()
    if provider not in VALID_PROVIDERS:
        provider = "local"
    model, base_url = provider_defaults(provider)
    return LLMSettings(provider=provider, model=model, base_url=base_url)


def load_settings(path: str | Path) -> LLMSettings:
    path = Path(path)
    settings = default_settings()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            settings = LLMSettings(
                provider=str(payload.get("provider", settings.provider)),
                model=str(payload.get("model", settings.model)),
                base_url=str(payload.get("base_url", settings.base_url)),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            settings = default_settings()
    settings.validate()
    return settings


def save_settings(path: str | Path, settings: LLMSettings) -> None:
    settings.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class LLMClient:
    """Create OpenAI-compatible clients and resolve installed Ollama models."""

    def __init__(self, settings: LLMSettings):
        settings.validate()
        self.settings = settings

    def client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for local and API LLM modes."
            ) from exc

        provider = self.settings.provider
        if provider == "local":
            return OpenAI(
                api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
                base_url=self.settings.base_url or provider_defaults("local")[1],
            )
        if provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("Set OPENROUTER_API_KEY before using OpenRouter.")
            return OpenAI(
                api_key=key,
                base_url=self.settings.base_url or provider_defaults("openrouter")[1],
            )
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY before using the OpenAI API.")
        kwargs: dict[str, Any] = {"api_key": key}
        if self.settings.base_url:
            kwargs["base_url"] = self.settings.base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _model_ids(model_page: Any) -> list[str]:
        items = getattr(model_page, "data", model_page)
        models = []
        for item in items or []:
            model_id = getattr(item, "id", None)
            if model_id is None and isinstance(item, dict):
                model_id = item.get("id") or item.get("name")
            if model_id and str(model_id) not in models:
                models.append(str(model_id))
        return models

    @staticmethod
    def _choose_local_model(requested: str, available: list[str]) -> str:
        if not available:
            raise RuntimeError(
                "No local models were found. Install and load an Ollama model first."
            )
        requested = requested.strip()
        if requested and requested.lower() != "auto":
            exact = next(
                (name for name in available if name.lower() == requested.lower()),
                None,
            )
            if exact:
                return exact
            family_match = re.match(r"[a-z_-]+", requested.lower())
            if family_match:
                family = family_match.group(0)
                related = [
                    name for name in available
                    if name.lower().startswith(family)
                ]
                if related:
                    return related[0]
        chat_models = [
            name
            for name in available
            if not any(token in name.lower() for token in ("embed", "embedding"))
        ]
        return (chat_models or available)[0]

    def resolve_model(self, client=None) -> tuple[str, list[str]]:
        if self.settings.provider != "local":
            if not self.settings.model:
                raise RuntimeError("Enter an API model name.")
            return self.settings.model, []
        client = client or self.client()
        try:
            available = self._model_ids(client.models.list())
        except Exception as exc:
            raise RuntimeError(
                f"Could not discover local models from {self.settings.base_url}: {exc}"
            ) from exc
        selected = self._choose_local_model(self.settings.model, available)
        self.settings.model = selected
        return selected, available
