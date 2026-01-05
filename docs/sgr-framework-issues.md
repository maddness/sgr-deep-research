# SGR Framework: Отчёт о проблемах и решениях

## Обзор
Документ описывает проблемы, обнаруженные при разработке Deep Research Telegram Bot на базе sgr-agent-core, и применённые решения.

---

## Проблема 1: Агент застревает в цикле планирования

### Описание
После исчерпания лимита поисков (`max_searches: 4`) агент застрял в цикле вызовов `GeneratePlanTool` — 19 раз подряд без прогресса.

**Трейс выполнения:**
```
шаги 1-10: Нормальная работа (plan, search x4, extract x3)
шаги 11-29: GeneratePlanTool × 19 (ЦИКЛ)
шаг 30: FinalAnswerTool (принудительно по max_iterations)
```

### Причина
Когда поиски исчерпаны, `WebSearchTool` убирался из доступных инструментов, но `GeneratePlanTool` оставался. Агент продолжал создавать планы с поисками, но не мог их выполнить.

### Решение
Модифицирован `_prepare_tools()` — теперь `GeneratePlanTool` тоже убирается при исчерпании поисков:

```python
if self._context.searches_used >= self.config.search.max_searches:
    logger.info(f"Max searches reached, limiting to synthesis tools")
    tools -= {WebSearchTool, GeneratePlanTool}
```

**Файл:** `bot/src/sgr/agent.py`, строки 260-264

---

## Проблема 2: Сообщение о кларификации не отправляется в Telegram

### Описание
Когда агент переходил в состояние `WAITING_FOR_CLARIFICATION`, сообщение с уточняющими вопросами не появлялось в Telegram.

### Причина
После обнаружения состояния кларификации в streaming-цикле, код продолжал ожидать `execute_task`, который никогда не завершится, потому что агент приостановлен в ожидании ответа пользователя.

### Решение
Отменять `execute_task` и пропускать ожидание при обнаружении кларификации:

```python
if context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
    logger.info("Agent waiting for clarification")
    execute_task.cancel()
    break

# Пропускаем await если в состоянии кларификации
if context.state != AgentStatesEnum.WAITING_FOR_CLARIFICATION and not execute_task.done():
    await execute_task
```

**Файл:** `bot/src/sgr/agent.py`, строки 448-460

---

## Проблема 3: Нереалистичные данные принимаются без валидации

### Описание
Агент принимал явно неверные данные (например, $230,500 зарплата AI/ML инженера в России) без какой-либо проверки. Модель gpt-4o-mini не имеет достаточных способностей для валидации качества данных.

**Пример вывода:**
```
- Россия: $230,500/год
- Германия: ~$100,000/год
- США: $130,000-160,000/год
Вывод: "Германия — оптимальный вариант" (игнорируя собственные данные!)
```

### Причина
1. gpt-4o-mini не сверяет данные со здравым смыслом
2. Нет инструкции использовать `ReasoningTool` для валидации перед отчётом
3. Выводы не соответствуют представленным данным

### Решение (частичное)
Добавлены `<DATA_VALIDATION_GUIDELINES>` в системный промпт:
1. **Обязательный ReasoningTool** перед CreateReportTool
2. **Sanity check** — проверка на реалистичность
3. **Проверка конфликтов** между источниками
4. **Соответствие вывода** представленным данным

```
<DATA_VALIDATION_GUIDELINES>
MANDATORY: Before using CreateReportTool, you MUST use ReasoningTool to validate collected data.

In ReasoningTool validation step, check:
1. SANITY CHECK - Are numbers realistic?
2. DATA CONFLICTS - Do sources agree?
3. CONCLUSION ALIGNMENT - Does your conclusion match the data?
4. SOURCE RELIABILITY - Are sources trustworthy?

ONLY after ReasoningTool validation, proceed to CreateReportTool.
</DATA_VALIDATION_GUIDELINES>
```

### Ограничения
gpt-4o-mini всё ещё плохо справляется с этими инструкциями. Для сложных исследовательских задач рекомендуется более мощная модель (gpt-4o, Claude).

**Файл:** `bot/src/sgr/agent.py`, строки 93-115

---

## Проблема 4: Агент отвечает на неправильном языке

### Описание
Агент отвечал на английском, когда вопрос был задан на русском.

### Причина
Инструкции по языку были в промпте, но gpt-4o-mini их игнорировал.

### Решение
Добавлен `<CRITICAL_LANGUAGE_REQUIREMENT>` в **конец** системного промпта (модели лучше следуют инструкциям в начале и конце):

```
<CRITICAL_LANGUAGE_REQUIREMENT>
IMPORTANT: Your ENTIRE response MUST be in the SAME LANGUAGE as the user's question.
- User writes in Russian → ALL your output in Russian
- User writes in English → ALL your output in English
DO NOT mix languages.
</CRITICAL_LANGUAGE_REQUIREMENT>
```

**Файл:** `bot/src/sgr/agent.py`, строки 121-127

---

## Проблема 5: Tavily не может извлечь контент

### Описание
Многие URL не удалось обработать (Glassdoor, ZipRecruiter блокируют scraping):
```
⚠️ Failed to extract 4 URLs: [glassdoor.com, ziprecruiter.com...]
```

### Причина
Популярные сайты с зарплатами блокируют автоматический scraping.

### Влияние
Агент получает неполные данные, что приводит к ненадёжным выводам.

### Рекомендации
- Увеличить `max_searches` для компенсации неудачных извлечений
- Рассмотреть альтернативные источники данных
- Это ограничение Tavily API, не sgr-agent-core

---

## Изменения конфигурации

### config.yaml
```yaml
search:
  max_searches: 10  # Увеличено с 4

sgr:
  max_iterations: 30  # Увеличено с 10

llm:
  max_tokens: 16000  # Увеличено с 8000
```

---

## Рекомендации для SGR Framework

1. **Встроенный шаг валидации данных** — добавить опциональную фазу валидации перед генерацией отчёта
2. **Детекция циклов** — обнаруживать когда агент вызывает один и тот же инструмент многократно без прогресса
3. **Улучшенное ограничение инструментов** — когда ресурс X исчерпан, ограничивать и инструменты, зависящие от X
4. **Усиленный контроль языка** — определение и контроль языка на уровне фреймворка
5. **Профили для разных моделей** — разные промпты/стратегии для моделей с разными возможностями

---

## Изменённые файлы

| Файл | Изменения |
|------|-----------|
| `bot/src/sgr/agent.py` | Ограничение инструментов, фикс кларификации, кастомный системный промпт |
| `config.yaml` | Увеличены лимиты |
| `requirements.txt` | Добавлен telegramify-markdown |
| `bot/src/handlers/messages.py` | Отображение прогресса, конвертация markdown |

---

*Последнее обновление: 2026-01-06*
