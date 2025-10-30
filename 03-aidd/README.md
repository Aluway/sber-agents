# AI Financial Advisor Bot

Telegram-бот с LLM в роли финансового советника.

## Быстрый старт

### 1. Установка зависимостей

```bash
make setup
```

### 2. Настройка

Скопируйте `.env.example` в `.env` и заполните ваши ключи:

```bash
copy .env.example .env
```

Отредактируйте `.env`:
- `TELEGRAM_BOT_TOKEN` - токен от @BotFather в Telegram
- `OPENROUTER_API_KEY` - API ключ от [openrouter.ai](https://openrouter.ai)

### 3. Запуск

```bash
make run
```

## Доступные команды

- `make setup` - установить зависимости
- `make run` - запустить бота
- `make clean` - очистить временные файлы
- `make help` - показать справку

## Проект

- **Идея:** `docs/idea.md`
- **Видение:** `docs/vision.md`
- **Конвенции:** `docs/conventions.md`
- **План:** `docs/tasklist.md`
- **Workflow:** `docs/workflow.md`

