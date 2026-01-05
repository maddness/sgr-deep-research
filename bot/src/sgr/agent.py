"""
SGR Agent - встроенный Deep Research агент.

Использует sgr-agent-core напрямую без отдельного сервера.
С опциональной интеграцией Langfuse для observability.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Type, Optional

from openai import AsyncOpenAI
from sgr_agent_core import (
    AgentConfig,
    LLMConfig,
    SearchConfig,
    ExecutionConfig,
    AgentStatesEnum,
    BaseTool,
)
from sgr_agent_core.agents import SGRAgent
from sgr_agent_core.tools import (
    WebSearchTool,
    ExtractPageContentTool,
    GeneratePlanTool,
    ReasoningTool,
    ClarificationTool,
    AdaptPlanTool,
    CreateReportTool,
    FinalAnswerTool,
)

logger = logging.getLogger(__name__)

# Try to import langfuse and openinference
try:
    from langfuse import observe, Langfuse
    LANGFUSE_AVAILABLE = True
    langfuse_client = None  # Will be initialized if enabled
except ImportError:
    LANGFUSE_AVAILABLE = False
    observe = lambda *args, **kwargs: lambda f: f  # no-op decorator
    langfuse_client = None
    Langfuse = None

# Try to import OpenAI instrumentor for detailed LLM tracing
try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
    OPENAI_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    OPENAI_INSTRUMENTOR_AVAILABLE = False
    OpenAIInstrumentor = None


def setup_langfuse(
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
) -> bool:
    """
    Setup Langfuse environment variables for tracing.
    Returns True if setup successful.
    """
    global langfuse_client
    try:
        import os
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = host

        if LANGFUSE_AVAILABLE and Langfuse:
            # Initialize client to ensure connection works
            langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info(f"Langfuse configured (host: {host})")

            # Enable OpenAI instrumentation for detailed LLM tracing
            if OPENAI_INSTRUMENTOR_AVAILABLE and OpenAIInstrumentor:
                OpenAIInstrumentor().instrument()
                logger.info("OpenAI instrumentation enabled (OpenInference)")

            return True
        else:
            logger.warning("Langfuse not installed")
            return False
    except Exception as e:
        logger.warning(f"Failed to setup Langfuse: {e}")
        return False


def create_openai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Create standard OpenAI client."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


@dataclass
class ResearchProgress:
    """Прогресс исследования для отображения пользователю."""

    step: int = 0
    tool_name: str = ""
    tool_emoji: str = ""
    description: str = ""
    searches_done: int = 0
    is_final: bool = False


# Маппинг инструментов на эмодзи и описания
TOOL_INFO = {
    "generateplantool": ("📋", "Планирование исследования"),
    "websearchtool": ("🔍", "Поиск в интернете"),
    "extractpagecontenttool": ("📄", "Извлечение контента"),
    "reasoningtool": ("🧠", "Анализ информации"),
    "clarificationtool": ("❓", "Запрос уточнения"),
    "adaptplantool": ("🔄", "Корректировка плана"),
    "createreporttool": ("📝", "Создание отчёта"),
    "finalanswertool": ("✅", "Финализация ответа"),
}


def get_tool_info(tool_name: str) -> tuple[str, str]:
    """Получить эмодзи и описание для инструмента."""
    key = tool_name.lower().replace("_", "")
    return TOOL_INFO.get(key, ("🔧", tool_name))


@dataclass
class ResearchResult:
    """Результат исследования."""

    content: str = ""
    is_done: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    progress: ResearchProgress | None = None  # Добавлено для прогресса


class DeepResearchAgent:
    """
    Deep Research агент на базе SGR Agent Core.

    Использует Schema-Guided Reasoning для глубокого исследования
    с поиском в интернете через Tavily.
    """

    # Системный промпт для агента
    SYSTEM_PROMPT = (
        "Ты - исследовательский AI-ассистент. "
        "ВСЕГДА отвечай на русском языке, независимо от языка запроса или найденных источников. "
        "Структурируй ответы с заголовками и списками для удобства чтения."
    )

    # Стандартный набор tools для deep research (список классов)
    DEFAULT_TOOLS: list[Type[BaseTool]] = [
        WebSearchTool,
        ExtractPageContentTool,
        GeneratePlanTool,
        ReasoningTool,
        ClarificationTool,
        AdaptPlanTool,
        CreateReportTool,
        FinalAnswerTool,
    ]

    def __init__(
        self,
        anthropic_api_key: str,
        tavily_api_key: str,
        model: str = "claude-haiku-4-5",
        api_base: str = "https://api.anthropic.com/v1",
        temperature: float = 0.4,
        max_tokens: int = 8000,
        max_iterations: int = 10,
        max_searches: int = 4,
        max_clarifications: int = 3,
        # Langfuse settings
        langfuse_enabled: bool = False,
        langfuse_public_key: str = "",
        langfuse_secret_key: str = "",
        langfuse_host: str = "https://cloud.langfuse.com",
    ):
        """
        Инициализация агента.

        Args:
            anthropic_api_key: API ключ Anthropic
            tavily_api_key: API ключ Tavily для веб-поиска
            model: Модель Claude (по умолчанию Haiku 4.5)
            api_base: Base URL API
            temperature: Температура генерации
            max_tokens: Максимум токенов
            max_iterations: Максимум итераций агента
            max_searches: Максимум поисковых запросов
            max_clarifications: Максимум уточняющих вопросов
            langfuse_enabled: Включить трейсинг Langfuse
            langfuse_public_key: Публичный ключ Langfuse
            langfuse_secret_key: Секретный ключ Langfuse
            langfuse_host: URL хоста Langfuse
        """
        self.anthropic_api_key = anthropic_api_key
        self.tavily_api_key = tavily_api_key
        self.langfuse_enabled = langfuse_enabled

        # Конфигурация агента
        self.config = AgentConfig(
            llm=LLMConfig(
                api_key=anthropic_api_key,
                model=model,
                base_url=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            search=SearchConfig(
                tavily_api_key=tavily_api_key,
                max_searches=max_searches,
                max_results=10,
            ),
            execution=ExecutionConfig(
                max_iterations=max_iterations,
                max_clarifications=max_clarifications,
            ),
        )

        # Setup Langfuse если включено
        if langfuse_enabled and langfuse_public_key and langfuse_secret_key:
            setup_langfuse(langfuse_public_key, langfuse_secret_key, langfuse_host)

        # OpenAI-совместимый клиент
        self.client = create_openai_client(api_key=anthropic_api_key, base_url=api_base)

        # Toolkit - список классов tools
        self.toolkit = self.DEFAULT_TOOLS

    @observe(name="deep_research")
    async def research(
        self,
        messages: list[dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[ResearchResult]:
        """
        Провести исследование по запросу с учётом истории разговора.

        Args:
            messages: История сообщений в формате OpenAI [{"role": "user/assistant", "content": "..."}]
            user_id: ID пользователя для трейсинга (для Langfuse)
            session_id: ID сессии для трейсинга (для Langfuse)

        Yields:
            ResearchResult с прогрессом и финальным результатом
        """
        tools_used: list[str] = []

        # Get last user message for logging
        last_query = messages[-1]["content"] if messages else ""

        # Log Langfuse context info
        if self.langfuse_enabled and LANGFUSE_AVAILABLE:
            logger.debug(f"Langfuse tracing active for query: {last_query[:50]}...")

        try:
            # Добавляем системное сообщение в начало
            messages_with_system = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                *messages,
            ]

            # Создаём агента с полной историей разговора
            agent = SGRAgent(
                task_messages=messages_with_system,
                openai_client=self.client,
                agent_config=self.config,
                toolkit=self.toolkit,
            )

            # Запускаем выполнение
            logger.info(f"Starting research ({len(messages)} messages): {last_query[:200]}...")

            # Выполняем агента с мониторингом состояния
            context = agent._context
            execute_task = asyncio.create_task(agent.execute())

            # Отслеживаем изменения для yield прогресса
            last_iteration = 0
            last_tool = ""

            # Мониторим состояние пока выполняется
            while not execute_task.done():
                await asyncio.sleep(0.3)

                # Проверяем не ждёт ли агент clarification
                if context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
                    logger.info("Agent waiting for clarification, cancelling execute task")
                    execute_task.cancel()
                    try:
                        await execute_task
                    except asyncio.CancelledError:
                        pass
                    break

                # Проверяем изменился ли шаг или инструмент
                current_iteration = context.iteration
                current_tool = ""
                tool_detail = ""

                # Извлекаем информацию о текущем инструменте
                if hasattr(context, 'current_step_reasoning') and context.current_step_reasoning:
                    reasoning = context.current_step_reasoning
                    # Название инструмента в function.tool_name_discriminator
                    if hasattr(reasoning, 'function') and reasoning.function:
                        func = reasoning.function
                        if hasattr(func, 'tool_name_discriminator'):
                            current_tool = func.tool_name_discriminator
                        elif hasattr(func, 'tool_name'):
                            current_tool = func.tool_name
                        # Для поиска показываем query
                        if hasattr(func, 'query') and func.query:
                            tool_detail = func.query[:50] + "..." if len(func.query) > 50 else func.query
                        # Для плана показываем цель
                        elif hasattr(func, 'research_goal') and func.research_goal:
                            tool_detail = func.research_goal[:50] + "..." if len(func.research_goal) > 50 else func.research_goal
                        # Для отчёта показываем title
                        elif hasattr(func, 'title') and func.title:
                            tool_detail = func.title[:50] + "..." if len(func.title) > 50 else func.title

                # Yield прогресс если что-то изменилось
                if current_iteration != last_iteration or current_tool != last_tool:
                    if current_tool:
                        emoji, description = get_tool_info(current_tool)
                        if tool_detail:
                            description = f"{description}: {tool_detail}"

                        progress = ResearchProgress(
                            step=current_iteration,
                            tool_name=current_tool,
                            tool_emoji=emoji,
                            description=description,
                            searches_done=context.searches_used,
                            is_final=False,
                        )
                        logger.info(f"Yielding progress: step={current_iteration}, tool={current_tool}, detail={tool_detail[:30] if tool_detail else 'none'}")
                        yield ResearchResult(progress=progress)

                        last_iteration = current_iteration
                        last_tool = current_tool

            # Debug logging
            logger.info(f"Agent state: {context.state}")
            logger.info(f"Execution result: {context.execution_result[:200] if context.execution_result else 'None'}...")
            if hasattr(context, 'current_step_reasoning'):
                logger.info(f"Current step reasoning: {context.current_step_reasoning}")
            if hasattr(context, 'clarification_received'):
                logger.info(f"Clarification received: {context.clarification_received}")

            # Собираем использованные tools
            if hasattr(context, 'tools_called'):
                tools_used = list(context.tools_called)

            # Проверяем на ошибку
            if context.state == AgentStatesEnum.FAILED:
                error_msg = context.execution_result or "Агент завершился с ошибкой"
                logger.error(f"Agent failed: {error_msg}")
                yield ResearchResult(
                    is_done=True,
                    error=error_msg,
                    tools_used=tools_used,
                    iterations=context.iteration if hasattr(context, 'iteration') else 0,
                )
                return

            # Проверяем нужно ли уточнение
            if context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
                # Извлекаем вопросы из current_step_reasoning.function.questions
                clarification_questions = None
                if hasattr(context, 'current_step_reasoning') and context.current_step_reasoning:
                    reasoning = context.current_step_reasoning
                    if hasattr(reasoning, 'function') and reasoning.function:
                        func = reasoning.function
                        if hasattr(func, 'questions') and func.questions:
                            clarification_questions = "\n".join(func.questions)

                logger.info(f"Clarification questions extracted: {clarification_questions}")

                yield ResearchResult(
                    needs_clarification=True,
                    clarification_question=clarification_questions or "Пожалуйста, уточните ваш запрос.",
                    tools_used=tools_used,
                    iterations=context.iteration if hasattr(context, 'iteration') else 0,
                )
                return

            # Возвращаем результат
            result_content = context.execution_result or ""
            iterations = context.iteration if hasattr(context, 'iteration') else 0

            yield ResearchResult(
                content=result_content,
                is_done=True,
                tools_used=tools_used,
                iterations=iterations,
            )

            logger.info(f"Research completed. Tools used: {tools_used}")

            # Flush Langfuse traces
            if self.langfuse_enabled and LANGFUSE_AVAILABLE and langfuse_client:
                try:
                    langfuse_client.flush()
                    logger.debug("Langfuse traces flushed")
                except Exception as flush_err:
                    logger.debug(f"Failed to flush Langfuse: {flush_err}")

        except Exception as e:
            logger.error(f"Research failed: {e}")
            yield ResearchResult(
                is_done=True,
                error=str(e),
                tools_used=tools_used,
            )



def create_agent_from_config(config) -> DeepResearchAgent:
    """
    Создать агента из конфигурации бота.

    Args:
        config: Config объект с настройками

    Returns:
        DeepResearchAgent инстанс
    """
    return DeepResearchAgent(
        anthropic_api_key=config.llm_api_key,
        tavily_api_key=config.tavily_api_key,
        model=config.llm_model,
        api_base=config.llm_api_base,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        max_iterations=config.sgr_max_iterations,
        max_searches=config.max_searches,
        max_clarifications=config.sgr_max_clarifications,
        langfuse_enabled=config.langfuse_enabled,
        langfuse_public_key=config.langfuse_public_key,
        langfuse_secret_key=config.langfuse_secret_key,
        langfuse_host=config.langfuse_host,
    )
