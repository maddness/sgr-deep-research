"""Telegram message handlers with conversation memory support."""
import asyncio
import logging
import time
import uuid

from aiogram import Bot, F, Router
from aiogram.types import Message
from aiogram.enums import ParseMode
import telegramify_markdown

from bot.src.sgr.agent import DeepResearchAgent, ResearchResult, ResearchProgress
from bot.src.utils.config import Config
from bot.src.utils.logger import InteractionLogger, split_message
from bot.src.utils.memory import get_memory

logger = logging.getLogger(__name__)

router = Router(name="messages")

# Per-user locks to prevent concurrent requests
user_locks: dict[int, asyncio.Lock] = {}


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Get or create a lock for user."""
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]


async def send_message_draft(
    bot: Bot,
    chat_id: int,
    draft_id: str,
    text: str,
    message_thread_id: int | None = None,
) -> bool:
    """
    Send message draft using Bot API 9.3 sendMessageDraft.

    Returns True if successful, False if fallback needed.
    """
    try:
        result = await bot.session.make_request(
            bot,
            "sendMessageDraft",
            {
                "chat_id": chat_id,
                "draft_id": draft_id,
                "text": text,
                **({"message_thread_id": message_thread_id} if message_thread_id else {}),
            },
        )
        return result is True
    except Exception as e:
        logger.debug(f"sendMessageDraft failed: {e}")
        return False


class ResearchSession:
    """Manages a single research session with conversation memory."""

    def __init__(
        self,
        message: Message,
        bot: Bot,
        agent: DeepResearchAgent,
        config: Config,
        interaction_logger: InteractionLogger,
    ):
        self.message = message
        self.bot = bot
        self.agent = agent
        self.config = config
        self.interaction_logger = interaction_logger

        self.draft_id = str(uuid.uuid4())
        self.status_message: Message | None = None
        self.accumulated_content = ""
        self.last_update_time = 0.0
        self.last_update_length = 0
        self.tools_used: list[str] = []
        self.iterations = 0
        self.start_time = time.time()
        self.use_draft = True
        self.needs_clarification = False
        self.clarification_question: str | None = None
        # Прогресс для интерактивного отображения
        self.progress_steps: list[ResearchProgress] = []

    async def run(self, query: str) -> str | None:
        """
        Execute research with conversation history.

        Returns the assistant response content or None if error/clarification.
        """
        user_id = self.message.from_user.id if self.message.from_user else 0
        username = self.message.from_user.username if self.message.from_user else None

        # Get conversation memory
        memory = get_memory(self.config.max_history_messages)

        # Add user message to memory
        memory.add_user_message(user_id, query)

        # Get full conversation history
        messages = memory.get_messages(user_id)

        # Send initial status
        self.status_message = await self.message.answer(
            "🔬 Начинаю исследование..."
        )

        error_msg: str | None = None

        try:
            async for result in self.agent.research(
                messages=messages,
                user_id=str(user_id),
                session_id=f"tg-{user_id}-{self.draft_id[:8]}",
            ):
                await self._handle_result(result)

        except Exception as e:
            logger.error(f"Research error: {e}")
            error_msg = str(e)
            await self._safe_update(f"Ошибка исследования: {e}")

        finally:
            # Log interaction
            duration = time.time() - self.start_time
            await self.interaction_logger.log_interaction(
                user_id=user_id,
                username=username,
                query=query,
                response=self.accumulated_content,
                tools_used=self.tools_used,
                error=error_msg,
                duration_seconds=duration,
            )

        # Add assistant response to memory (if we got one)
        if self.accumulated_content and not self.needs_clarification:
            memory.add_assistant_message(user_id, self.accumulated_content)
            return self.accumulated_content
        elif self.needs_clarification and self.clarification_question:
            # Add clarification as assistant message
            memory.add_assistant_message(user_id, self.clarification_question)
            return None

        return None

    def _format_progress_message(self) -> str:
        """Форматировать сообщение о прогрессе."""
        if not self.progress_steps:
            return "🔬 Начинаю исследование..."

        lines = ["🔬 Исследование...\n"]

        # Все шаги отображаются одинаково (без галочек)
        for step in self.progress_steps:
            lines.append(f"{step.tool_emoji} {step.description}")

        # Добавляем статистику
        elapsed = time.time() - self.start_time
        searches = self.progress_steps[-1].searches_done if self.progress_steps else 0
        lines.append(f"\n⏱️ {elapsed:.0f} сек")
        if searches > 0:
            lines.append(f"🔍 Поисков: {searches}")

        return "\n".join(lines)

    async def _handle_result(self, result: ResearchResult) -> None:
        """Process research result."""
        # Обработка прогресса
        if result.progress and not result.is_done:
            # Добавляем новый шаг прогресса
            self.progress_steps.append(result.progress)
            progress_msg = self._format_progress_message()
            logger.info(f"Progress update: step={result.progress.step}, tool={result.progress.tool_name}")
            # Обновляем сообщение с прогрессом
            await self._safe_update(progress_msg)
            return

        if result.tools_used:
            self.tools_used = result.tools_used

        if result.iterations:
            self.iterations = result.iterations

        if result.error:
            await self._safe_update(f"❌ Ошибка: {result.error}")
            return

        if result.needs_clarification:
            self.needs_clarification = True
            self.clarification_question = result.clarification_question
            await self._safe_update(
                f"❓ Требуется уточнение:\n\n{result.clarification_question}\n\n"
                "Просто ответьте на вопрос в следующем сообщении."
            )
            return

        if result.content:
            self.accumulated_content = result.content

        if result.is_done:
            await self._send_final_result()

    async def _safe_update(self, text: str) -> None:
        """Safely update message."""
        if not self.status_message:
            return

        # Try sendMessageDraft first
        if self.use_draft:
            success = await send_message_draft(
                self.bot,
                self.message.chat.id,
                self.draft_id,
                text,
                self.message.message_thread_id,
            )
            if success:
                return
            self.use_draft = False
            logger.info("Falling back to edit_message_text")

        # Fallback to edit_message_text
        try:
            if text != self.status_message.text:
                await self.status_message.edit_text(text)
        except Exception as e:
            logger.warning(f"Failed to edit message: {e}")

    async def _send_final_result(self) -> None:
        """Send the complete research result."""
        if not self.accumulated_content:
            await self._safe_update("❌ Ответ не был сгенерирован.")
            return

        # Прогресс-сообщение остаётся как есть (не меняем на "завершено")

        # Отправляем результат отдельным сообщением
        duration = time.time() - self.start_time
        footer = f"\n\n---\n⏱️ {duration:.1f} сек | 🔄 {self.iterations} итераций"
        full_content = self.accumulated_content + footer

        # Конвертируем Markdown в Telegram MarkdownV2
        try:
            converted = telegramify_markdown.markdownify(full_content)
            chunks = split_message(converted)
            parse_mode = ParseMode.MARKDOWN_V2
        except Exception as e:
            logger.warning(f"Failed to convert markdown: {e}, sending as plain text")
            chunks = split_message(full_content)
            parse_mode = None

        # Отправляем все части как новые сообщения
        for chunk_text in chunks:
            try:
                await self.message.answer(chunk_text, parse_mode=parse_mode)
            except Exception as e:
                # Fallback to plain text if markdown fails
                logger.warning(f"Failed to send with markdown: {e}")
                await self.message.answer(chunk_text)


@router.message(F.text)
async def handle_message(
    message: Message,
    bot: Bot,
    agent: DeepResearchAgent,
    config: Config,
    interaction_logger: InteractionLogger,
) -> None:
    """Handle user text messages with conversation memory."""
    if not message.from_user:
        return

    user_id = message.from_user.id
    query = message.text.strip() if message.text else ""

    # Skip empty or too short queries
    if len(query) < 3:
        await message.answer(
            "Пожалуйста, введите более подробный вопрос (минимум 3 символа)."
        )
        return

    # Acquire user lock
    lock = get_user_lock(user_id)

    if lock.locked():
        await message.answer(
            "⏳ Предыдущий запрос ещё обрабатывается. Пожалуйста, подождите."
        )
        return

    async with lock:
        await bot.send_chat_action(message.chat.id, "typing")

        session = ResearchSession(
            message=message,
            bot=bot,
            agent=agent,
            config=config,
            interaction_logger=interaction_logger,
        )

        await session.run(query)
