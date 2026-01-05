"""Configuration loader for Deep Research Bot."""
import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """YAML-based configuration with environment variable override support."""

    def __init__(self, path: str = "config.yaml"):
        self._path = Path(path)
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._path}\n"
                "Copy config.example.yaml to config.yaml and fill in your values."
            )

        with open(self._path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def _get(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value with dot notation support."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    # Telegram settings
    @property
    def telegram_token(self) -> str:
        """Telegram bot token (env override: TELEGRAM_BOT_TOKEN)."""
        return os.getenv("TELEGRAM_BOT_TOKEN") or self._get("telegram", "bot_token", default="")

    # LLM settings
    @property
    def llm_model(self) -> str:
        """LLM model name."""
        return os.getenv("MODEL") or self._get("llm", "model", default="claude-haiku-4-5")

    @property
    def llm_api_key(self) -> str:
        """LLM API key (env override: ANTHROPIC_API_KEY)."""
        return os.getenv("ANTHROPIC_API_KEY") or self._get("llm", "api_key", default="")

    @property
    def llm_api_base(self) -> str:
        """LLM API base URL (OpenAI-compatible endpoint)."""
        return os.getenv("API_BASE") or self._get("llm", "api_base", default="https://api.anthropic.com/v1")

    @property
    def llm_temperature(self) -> float:
        """LLM temperature."""
        return self._get("llm", "temperature", default=0.4)

    @property
    def llm_max_tokens(self) -> int:
        """LLM max tokens."""
        return self._get("llm", "max_tokens", default=8000)

    # Search settings
    @property
    def search_provider(self) -> str:
        """Search provider name."""
        return self._get("search", "provider", default="tavily")

    @property
    def tavily_api_key(self) -> str:
        """Tavily API key (env override: TAVILY_API_KEY)."""
        return os.getenv("TAVILY_API_KEY") or self._get("search", "api_key", default="")

    @property
    def tavily_api_url(self) -> str:
        """Tavily API URL."""
        return self._get("search", "api_url", default="https://api.tavily.com")

    @property
    def max_searches(self) -> int:
        """Maximum number of searches per request."""
        return self._get("search", "max_searches", default=4)

    @property
    def max_results(self) -> int:
        """Maximum results per search."""
        return self._get("search", "max_results", default=10)

    # SGR Agent settings (встроенный агент)
    @property
    def sgr_max_iterations(self) -> int:
        """Maximum SGR reasoning iterations."""
        return self._get("sgr", "max_iterations", default=10)

    @property
    def sgr_max_clarifications(self) -> int:
        """Maximum clarification requests."""
        return self._get("sgr", "max_clarifications", default=3)

    # Bot settings
    @property
    def max_history_messages(self) -> int:
        """Maximum messages to keep in history."""
        return self._get("bot", "max_history_messages", default=20)

    @property
    def stream_update_interval(self) -> float:
        """Minimum seconds between stream updates."""
        return self._get("bot", "stream_update_interval", default=1.0)

    @property
    def stream_min_chars(self) -> int:
        """Minimum characters before sending update."""
        return self._get("bot", "stream_min_chars", default=50)

    # Langfuse observability settings
    @property
    def langfuse_enabled(self) -> bool:
        """Whether Langfuse tracing is enabled."""
        env_val = os.getenv("LANGFUSE_ENABLED", "").lower()
        if env_val in ("true", "1", "yes"):
            return True
        if env_val in ("false", "0", "no"):
            return False
        return self._get("langfuse", "enabled", default=False)

    @property
    def langfuse_public_key(self) -> str:
        """Langfuse public key (env override: LANGFUSE_PUBLIC_KEY)."""
        return os.getenv("LANGFUSE_PUBLIC_KEY") or self._get("langfuse", "public_key", default="")

    @property
    def langfuse_secret_key(self) -> str:
        """Langfuse secret key (env override: LANGFUSE_SECRET_KEY)."""
        return os.getenv("LANGFUSE_SECRET_KEY") or self._get("langfuse", "secret_key", default="")

    @property
    def langfuse_host(self) -> str:
        """Langfuse host URL (env override: LANGFUSE_HOST or LANGFUSE_BASE_URL)."""
        return (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or self._get("langfuse", "host", default="https://cloud.langfuse.com")
        )


# Global config instance
_config: Config | None = None


def get_config(path: str = "config.yaml") -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config(path)
    return _config
