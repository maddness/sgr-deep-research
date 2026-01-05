"""Deep Research Telegram Bot - Main entry point."""
import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from fastapi import FastAPI
import uvicorn

from bot.src.handlers.commands import router as commands_router
from bot.src.handlers.messages import router as messages_router
from bot.src.api.routes import router as api_router, set_agent
from bot.src.sgr.agent import DeepResearchAgent, resolve_tools
from bot.src.utils.config import get_config
from bot.src.utils.logger import InteractionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Deep Research API",
    description="HTTP API for testing Deep Research Agent",
    version="1.0.0",
)
app.include_router(api_router)


async def run_telegram_bot(agent: DeepResearchAgent, config) -> None:
    """Run the Telegram bot."""
    interaction_logger = InteractionLogger(logs_dir="logs")

    bot = Bot(
        token=config.telegram_token,
        default=DefaultBotProperties(parse_mode=None),
    )

    dp = Dispatcher()
    dp.include_router(commands_router)
    dp.include_router(messages_router)

    dp["agent"] = agent
    dp["config"] = config
    dp["interaction_logger"] = interaction_logger

    logger.info("Starting Telegram Bot polling...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("Telegram Bot stopped")


async def run_api_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the FastAPI server."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting API server on http://{host}:{port}")
    logger.info(f"Swagger UI available at http://{host}:{port}/docs")
    await server.serve()


async def main() -> None:
    """Initialize and start both bot and API server."""
    # Load configuration
    try:
        config = get_config()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate required settings
    if not config.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not configured")
        sys.exit(1)

    if not config.llm_api_key:
        logger.error("ANTHROPIC_API_KEY not configured")
        sys.exit(1)

    if not config.tavily_api_key:
        logger.error("TAVILY_API_KEY not configured")
        sys.exit(1)

    # Resolve tools from config
    toolkit = resolve_tools(config.agent_tools)
    logger.info(f"Resolved {len(toolkit)} tools from config: {config.agent_tools}")

    # Load system prompt from config
    system_prompt = config.system_prompt
    logger.info(f"System prompt loaded ({len(system_prompt)} chars)")

    # Initialize Deep Research Agent
    logger.info("Initializing Deep Research Agent...")
    agent = DeepResearchAgent(
        anthropic_api_key=config.llm_api_key,
        tavily_api_key=config.tavily_api_key,
        model=config.llm_model,
        api_base=config.llm_api_base,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        max_iterations=config.sgr_max_iterations,
        max_searches=config.max_searches,
        max_clarifications=config.sgr_max_clarifications,
        # Tools and prompts from config
        toolkit=toolkit,
        system_prompt=system_prompt,
        # Langfuse observability
        langfuse_enabled=config.langfuse_enabled,
        langfuse_public_key=config.langfuse_public_key,
        langfuse_secret_key=config.langfuse_secret_key,
        langfuse_host=config.langfuse_host,
    )
    logger.info(f"Agent initialized with model: {config.llm_model}")
    logger.info(f"Langfuse tracing: {'enabled' if config.langfuse_enabled else 'disabled'}")

    # Set agent for API routes
    set_agent(agent)

    # Run both services concurrently
    logger.info("Starting services...")
    await asyncio.gather(
        run_telegram_bot(agent, config),
        run_api_server(host="0.0.0.0", port=8080),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Services stopped by user")
