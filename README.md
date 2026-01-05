# Deep Research Telegram Bot

Telegram-бот для глубокого исследования вопросов с использованием AI-агента и веб-поиска.

Построен на базе **[SGR (Schema-Guided Reasoning)](https://github.com/vamplabai/sgr-agent-core)** — фреймворка для создания исследовательских AI-агентов с поддержкой планирования, веб-поиска и структурированных ответов.

## Возможности

- **Глубокое исследование** — агент анализирует вопрос, ищет информацию в интернете, строит план и генерирует структурированный отчёт
- **Веб-поиск** — интеграция с Tavily API для актуальной информации
- **Память разговора** — бот помнит контекст беседы и позволяет задавать уточняющие вопросы
- **Уточнения** — агент может запросить дополнительную информацию если вопрос неясен
- **HTTP API** — Swagger UI на порту 8080 для тестирования агента
- **Observability** — опциональная интеграция с Langfuse для трейсинга

## Технологии

| Компонент | Технология |
|-----------|------------|
| LLM | OpenAI-совместимый API (GPT-4o-mini, Claude и др.) |
| Агент | [sgr-agent-core](https://github.com/vamplabai/sgr-agent-core) |
| Поиск | Tavily API |
| Telegram | aiogram 3.24+ |
| API | FastAPI + Uvicorn |
| Контейнеризация | Docker / Podman |

## Структура проекта

```
sgr-deep-research/
├── app.py                 # Точка входа
├── config.yaml            # Конфигурация (создать из примера)
├── .env                   # API ключи (создать из примера)
├── bot/src/
│   ├── handlers/          # Telegram команды и сообщения
│   ├── sgr/               # Deep Research агент
│   ├── api/               # HTTP API endpoints
│   │   └── routes.py      # /api/chat, /api/health
│   └── utils/             # Конфиг, логгер, память
├── logs/                  # Логи взаимодействий
└── db/                    # База данных (зарезервировано)
```

## Первоначальная настройка

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd sgr-deep-research
```

### 2. Создать конфигурацию

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

### 3. Получить API ключи

| Ключ | Где получить |
|------|--------------|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) в Telegram |
| `ANTHROPIC_API_KEY` | Ваш OpenAI-совместимый провайдер |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |
| `LANGFUSE_*` (опционально) | [cloud.langfuse.com](https://cloud.langfuse.com) |

### 4. Заполнить `.env`

```bash
ANTHROPIC_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
TELEGRAM_BOT_TOKEN=123456789:ABC...

# Опционально (по умолчанию выключено)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 5. Настроить `config.yaml`

Основные параметры:
- `llm.model` — модель (например, `gpt-4o-mini`)
- `llm.api_base` — URL API провайдера
- `sgr.max_iterations` — максимум итераций агента
- `bot.max_history_messages` — размер памяти разговора

## Запуск

### Docker / Podman (рекомендуется)

```bash
# Запустить
docker compose up -d --build

# Или с Podman
podman compose up -d --build
```

### Локально (для разработки)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить
python app.py
```

## Остановка

```bash
# Docker
docker compose down

# Podman
podman compose down
```

## Использование

### Telegram команды

| Команда | Описание |
|---------|----------|
| `/start` | Начать работу с ботом |
| `/new` | Начать новую сессию (очистить историю) |
| `/help` | Справка |
| `/cancel` | Отменить текущий запрос |

### HTTP API

После запуска доступен Swagger UI: http://localhost:8080/docs

```bash
# Пример запроса
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Что нового в Python 3.13?"}'
```

## Логи

Просмотр логов контейнера:

```bash
docker compose logs -f

# Или
podman logs -f deep-research-bot
```

Логи взаимодействий сохраняются в `logs/` с разбивкой по дате.

## Разработка

При запуске через Docker Compose код монтируется в контейнер — изменения применяются при перезапуске:

```bash
docker compose restart
```

Для полной пересборки:

```bash
docker compose up -d --build
```
