"""
SGR Agent - встроенный Deep Research агент.

Использует sgr-agent-core напрямую без отдельного сервера.
С опциональной интеграцией Langfuse для observability.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Type, Optional

from openai import AsyncOpenAI
from sgr_agent_core import (
    AgentConfig,
    LLMConfig,
    SearchConfig,
    ExecutionConfig,
    PromptsConfig,
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
    NextStepToolsBuilder,
)

logger = logging.getLogger(__name__)

# Registry: tool name -> tool class
TOOL_REGISTRY: dict[str, Type[BaseTool]] = {
    "WebSearchTool": WebSearchTool,
    "ExtractPageContentTool": ExtractPageContentTool,
    "GeneratePlanTool": GeneratePlanTool,
    "ReasoningTool": ReasoningTool,
    "ClarificationTool": ClarificationTool,
    "AdaptPlanTool": AdaptPlanTool,
    "CreateReportTool": CreateReportTool,
    "FinalAnswerTool": FinalAnswerTool,
}

# Base system prompt from sgr-agent-core + validation instructions
SYSTEM_PROMPT_WITH_VALIDATION = """<MAIN_TASK_GUIDELINES>
You are an expert researcher with adaptive planning and schema-guided-reasoning capabilities. You get the research task and you neeed to do research and genrete answer
</MAIN_TASK_GUIDELINES>

<DATE_GUIDELINES>
PAY ATTENTION TO THE DATE INSIDE THE USER REQUEST
DATE FORMAT: YYYY-MM-DD HH:MM:SS (ISO 8601)
IMPORTANT: The date above is in YYYY-MM-DD format (Year-Month-Day). For example, 2025-10-03 means October 3rd, 2025, NOT March 10th.
</DATE_GUIDELINES>

<IMPORTANT_LANGUAGE_GUIDELINES>: Detect the language from user request and use this LANGUAGE for all responses, searches, and result finalanswertool
LANGUAGE ADAPTATION: Always respond and create reports in the SAME LANGUAGE as the user's request.
If user writes in Russian - respond in Russian, if in English - respond in English.
</IMPORTANT_LANGUAGE_GUIDELINES>:

<CORE_PRINCIPLES>:
1. Memorize plan you generated in first step and follow the task inside your plan.
1. Adapt plan when new data contradicts initial assumptions
2. Search queries in SAME LANGUAGE as user request
3. Final Answer ENTIRELY in SAME LANGUAGE as user request
</CORE_PRINCIPLES>

<REASONING_GUIDELINES>
ADAPTIVITY: Actively change plan when discovering new data.
ANALYSIS EXTRACT DATA: Always analyze data that you took in extractpagecontenttool
</REASONING_GUIDELINES>

<PRECISION_GUIDELINES>
CRITICAL FOR FACTUAL ACCURACY:
When answering questions about specific dates, numbers, versions, or names:
1. EXACT VALUES: Extract the EXACT value from sources (day, month, year for dates; precise numbers for quantities)
2. VERIFY YEAR: If question mentions a specific year (e.g., "in 2022"), verify extracted content is about that SAME year
3. CROSS-VERIFICATION: When sources provide contradictory information, prefer:
   - Official sources and primary documentation over secondary sources
   - Search result snippets that DIRECTLY answer the question over extracted page content
   - Multiple independent sources confirming the same fact
4. DATE PRECISION: Pay special attention to exact dates - day matters (October 21 ≠ October 22)
5. NUMBER PRECISION: For numbers/versions, exact match required (6.88b ≠ 6.88c, Episode 31 ≠ Episode 32)
6. SNIPPET PRIORITY: If search snippet clearly states the answer, trust it unless extract proves it wrong
7. TEMPORAL VALIDATION: When extracting page content, check if the page shows data for correct time period
</PRECISION_GUIDELINES>

<DATA_VALIDATION_GUIDELINES>
MANDATORY: Before using CreateReportTool, you MUST use ReasoningTool to validate collected data.

In ReasoningTool validation step, check:

1. SANITY CHECK - Are numbers realistic?
   - Compare with your general knowledge about typical ranges
   - If a number seems unusually high/low, flag it or search for verification

2. DATA CONFLICTS - Do sources agree?
   - List key numbers found with their sources
   - If numbers differ significantly, note the discrepancy in report

3. CONCLUSION ALIGNMENT - Does your conclusion match the data?
   - State which option is best BASED ON THE DATA you collected
   - If data shows X is best but you recommend Y, you MUST explain WHY

4. SOURCE RELIABILITY - Are sources trustworthy?
   - Prefer official statistics, established platforms (Glassdoor, LinkedIn, govt data)
   - Flag data from unknown or potentially unreliable sources

ONLY after ReasoningTool validation, proceed to CreateReportTool.
</DATA_VALIDATION_GUIDELINES>

<AGENT_TOOL_USAGE_GUIDELINES>:
{available_tools}
</AGENT_TOOL_USAGE_GUIDELINES>

<CRITICAL_LANGUAGE_REQUIREMENT>
IMPORTANT: Your ENTIRE response (CreateReportTool, FinalAnswerTool) MUST be in the SAME LANGUAGE as the user's question.
- User writes in Russian → ALL your output in Russian
- User writes in English → ALL your output in English
- This applies to: report content, conclusions, recommendations, all text
DO NOT mix languages. DO NOT respond in English if user asked in Russian.
</CRITICAL_LANGUAGE_REQUIREMENT>"""


def resolve_tools(tool_names: list[str]) -> list[Type[BaseTool]]:
    """Resolve tool names to tool classes."""
    tools = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            tools.append(TOOL_REGISTRY[name])
        else:
            logger.warning(f"Unknown tool: {name}, skipping")
    return tools


def parse_sse_tool_call(sse_data: str) -> tuple[str, str, dict] | None:
    """
    Parse SSE event to extract tool call info.

    Returns (tool_call_id, tool_name, arguments) or None if not a tool call.
    """
    if not sse_data.startswith("data: "):
        return None

    json_str = sse_data[6:].strip()  # Remove "data: " prefix
    if json_str == "[DONE]":
        return None

    try:
        data = json.loads(json_str)
        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        tool_calls = delta.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            tool_id = tc.get("id", "")
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            arguments_str = func.get("arguments", "{}")

            # Parse arguments JSON
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                arguments = {}

            if tool_name:
                return (tool_id, tool_name, arguments)

    except json.JSONDecodeError:
        pass

    return None

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


class TelegramResearchAgent(SGRAgent):
    """
    Кастомный SGR агент с динамическим контролем tools.

    Переопределяет _prepare_tools() для принудительного завершения
    при достижении лимитов итераций, clarifications и поисков.

    Tools передаются через config, а не хардкодятся.
    """

    def __init__(
        self,
        task_messages: list[dict],
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        **kwargs,
    ):
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            **kwargs,
        )

    async def _prepare_tools(self):
        """
        Подготовить доступные tools на основе текущего состояния.

        Динамически убирает tools при достижении лимитов:
        - max_iterations → только CreateReportTool, FinalAnswerTool
        - max_clarifications → убрать ClarificationTool
        - max_searches → убрать WebSearchTool
        """
        tools = set(self.toolkit)

        # При достижении лимита итераций - только завершающие tools
        if self._context.iteration >= self.config.execution.max_iterations:
            logger.info(f"Max iterations reached ({self._context.iteration}), limiting to final tools")
            tools = {CreateReportTool, FinalAnswerTool}

        # Исчерпали clarifications - убрать ClarificationTool
        if self._context.clarifications_used >= self.config.execution.max_clarifications:
            logger.debug(f"Max clarifications reached ({self._context.clarifications_used})")
            tools -= {ClarificationTool}

        # Исчерпали поиски - убрать WebSearchTool и GeneratePlanTool
        # (GeneratePlanTool убираем чтобы агент не застрял в цикле планирования)
        if self._context.searches_used >= self.config.search.max_searches:
            logger.info(f"Max searches reached ({self._context.searches_used}), limiting to synthesis tools")
            tools -= {WebSearchTool, GeneratePlanTool}

        return NextStepToolsBuilder.build_NextStepTools(list(tools))


class DeepResearchAgent:
    """
    Deep Research агент на базе SGR Agent Core.

    Использует Schema-Guided Reasoning для глубокого исследования
    с поиском в интернете через Tavily.

    Конфигурация tools загружается из config.yaml.
    Системный промпт берётся из sgr-agent-core (полный, оптимизированный).
    Язык ответа определяется автоматически по языку запроса.
    """

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
        # Tools from config
        toolkit: list[Type[BaseTool]] | None = None,
        # Langfuse settings
        langfuse_enabled: bool = False,
        langfuse_public_key: str = "",
        langfuse_secret_key: str = "",
        langfuse_host: str = "https://cloud.langfuse.com",
    ):
        """
        Инициализация агента.

        Args:
            anthropic_api_key: API ключ LLM провайдера
            tavily_api_key: API ключ Tavily для веб-поиска
            model: Модель LLM
            api_base: Base URL API
            temperature: Температура генерации
            max_tokens: Максимум токенов
            max_iterations: Максимум итераций агента
            max_searches: Максимум поисковых запросов
            max_clarifications: Максимум уточняющих вопросов
            toolkit: Список классов tools (из конфига)
            langfuse_enabled: Включить трейсинг Langfuse
            langfuse_public_key: Публичный ключ Langfuse
            langfuse_secret_key: Секретный ключ Langfuse
            langfuse_host: URL хоста Langfuse
        """
        self.anthropic_api_key = anthropic_api_key
        self.tavily_api_key = tavily_api_key
        self.langfuse_enabled = langfuse_enabled

        # Tools из конфига или дефолтные
        self.toolkit = toolkit or list(TOOL_REGISTRY.values())

        # Конфигурация агента
        # Системный промпт = базовый из sgr-agent-core + инструкции по валидации данных
        self.agent_config = AgentConfig(
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
            prompts=PromptsConfig(
                system_prompt_str=SYSTEM_PROMPT_WITH_VALIDATION,
            ),
        )

        # Setup Langfuse если включено
        if langfuse_enabled and langfuse_public_key and langfuse_secret_key:
            setup_langfuse(langfuse_public_key, langfuse_secret_key, langfuse_host)

        # OpenAI-совместимый клиент
        self.client = create_openai_client(api_key=anthropic_api_key, base_url=api_base)

        logger.info(f"Agent initialized with {len(self.toolkit)} tools")
        logger.info(f"Using custom system prompt with data validation instructions")

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
            # Создаём агента с историей разговора
            # Системный промпт берётся из sgr-agent-core автоматически
            # Дополнительные инструкции переданы через PromptsConfig
            agent = TelegramResearchAgent(
                task_messages=messages,  # только user/assistant сообщения
                openai_client=self.client,
                agent_config=self.agent_config,
                toolkit=self.toolkit,
            )

            # Запускаем выполнение
            logger.info(f"Starting research ({len(messages)} messages): {last_query[:200]}...")

            context = agent._context
            step_counter = 0
            last_tool_id = ""

            # Запускаем execute() в фоне
            execute_task = asyncio.create_task(agent.execute())

            # Стримим события через streaming_generator
            try:
                async for sse_event in agent.streaming_generator.stream():
                    # Парсим tool_call из SSE события
                    tool_info = parse_sse_tool_call(sse_event)

                    if tool_info:
                        tool_id, tool_name, arguments = tool_info

                        # Пропускаем reasoning events (они имеют id с "-reasoning")
                        # Показываем только action events (id с "-action")
                        if "-action" in tool_id and tool_id != last_tool_id:
                            last_tool_id = tool_id
                            step_counter += 1

                            # Извлекаем детали из arguments
                            tool_detail = ""
                            if "query" in arguments:
                                q = arguments["query"]
                                tool_detail = q[:50] + "..." if len(q) > 50 else q
                            elif "research_goal" in arguments:
                                g = arguments["research_goal"]
                                tool_detail = g[:50] + "..." if len(g) > 50 else g
                            elif "title" in arguments:
                                t = arguments["title"]
                                tool_detail = t[:50] + "..." if len(t) > 50 else t

                            emoji, description = get_tool_info(tool_name)
                            if tool_detail:
                                description = f"{description}: {tool_detail}"

                            progress = ResearchProgress(
                                step=step_counter,
                                tool_name=tool_name,
                                tool_emoji=emoji,
                                description=description,
                                searches_done=context.searches_used,
                                is_final=False,
                            )

                            logger.info(f"[Stream] Progress: step={step_counter}, tool={tool_name}")
                            tools_used.append(tool_name)
                            yield ResearchResult(progress=progress)

                    # Проверяем clarification
                    if context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
                        logger.info("Agent waiting for clarification")
                        # Не ждём execute_task - агент приостановлен
                        execute_task.cancel()
                        break

            except asyncio.CancelledError:
                logger.info("Streaming cancelled")

            # Ждём завершения execute если ещё не завершился и не clarification
            if context.state != AgentStatesEnum.WAITING_FOR_CLARIFICATION and not execute_task.done():
                try:
                    await execute_task
                except asyncio.CancelledError:
                    pass

            # Debug logging
            logger.info(f"Agent state: {context.state}")
            logger.info(f"Execution result: {context.execution_result[:200] if context.execution_result else 'None'}...")
            if hasattr(context, 'current_step_reasoning'):
                logger.info(f"Current step reasoning: {context.current_step_reasoning}")
            if hasattr(context, 'clarification_received'):
                logger.info(f"Clarification received: {context.clarification_received}")

            # tools_used уже собраны во время streaming

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
