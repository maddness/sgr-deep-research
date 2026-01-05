"""Interaction logger for Deep Research Bot."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Logs user interactions to JSON files organized by date and user."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self, user_id: int, timestamp: datetime) -> Path:
        """Get log file path for user and timestamp."""
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        user_dir = self.logs_dir / date_str / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        return user_dir / f"{time_str}.json"

    async def log_interaction(
        self,
        user_id: int,
        username: str | None,
        query: str,
        response: str,
        tools_used: list[str] | None = None,
        tokens: dict[str, int] | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """
        Log a user interaction.

        Args:
            user_id: Telegram user ID
            username: Telegram username
            query: User's query
            response: Bot's response
            tools_used: List of tools used during research
            tokens: Token usage statistics
            error: Error message if failed
            duration_seconds: Request duration
        """
        timestamp = datetime.now()

        log_entry: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "username": username,
            "query": query,
            "response": response[:1000] if len(response) > 1000 else response,
            "response_length": len(response),
        }

        if tools_used:
            log_entry["tools_used"] = tools_used

        if tokens:
            log_entry["tokens"] = tokens

        if error:
            log_entry["error"] = error

        if duration_seconds is not None:
            log_entry["duration_seconds"] = round(duration_seconds, 2)

        log_path = self._get_log_path(user_id, timestamp)

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
            logger.debug(f"Logged interaction to {log_path}")
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def get_user_interactions(
        self,
        user_id: int,
        date: datetime | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get recent interactions for a user.

        Args:
            user_id: Telegram user ID
            date: Specific date (default: today)
            limit: Maximum number of interactions to return

        Returns:
            List of interaction records
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")
        user_dir = self.logs_dir / date_str / str(user_id)

        if not user_dir.exists():
            return []

        interactions = []
        log_files = sorted(user_dir.glob("*.json"), reverse=True)

        for log_file in log_files[:limit]:
            try:
                with open(log_file, encoding="utf-8") as f:
                    interactions.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

        return interactions


def split_message(text: str, max_length: int = 4000) -> list[str]:
    """
    Split a long message into chunks for Telegram.

    Tries to split at natural boundaries (paragraphs, sentences, words).

    Args:
        text: Text to split
        max_length: Maximum chunk length

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find best split point
        split_point = max_length

        # Try paragraph break first
        para_break = remaining.rfind("\n\n", 0, max_length)
        if para_break > max_length // 2:
            split_point = para_break + 2
        else:
            # Try single newline
            line_break = remaining.rfind("\n", 0, max_length)
            if line_break > max_length // 2:
                split_point = line_break + 1
            else:
                # Try sentence break
                for punct in [". ", "! ", "? "]:
                    sent_break = remaining.rfind(punct, 0, max_length)
                    if sent_break > max_length // 2:
                        split_point = sent_break + len(punct)
                        break
                else:
                    # Try word break
                    space = remaining.rfind(" ", 0, max_length)
                    if space > max_length // 2:
                        split_point = space + 1

        chunks.append(remaining[:split_point].rstrip())
        remaining = remaining[split_point:].lstrip()

    return chunks
