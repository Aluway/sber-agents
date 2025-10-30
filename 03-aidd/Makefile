.PHONY: setup run clean help

help:
	@echo "Доступные команды:"
	@echo "  make setup  - Установить зависимости через uv"
	@echo "  make run    - Запустить Telegram-бот"
	@echo "  make clean  - Очистить временные файлы"

setup:
	@echo "Установка зависимостей..."
	uv sync
	@echo "✓ Зависимости установлены"
	@echo ""
	@echo "Не забудьте создать .env файл на основе .env.example"
	@echo "  copy .env.example .env"
	@echo "  # затем отредактируйте .env и добавьте ваши ключи"

run:
	uv run python src/bot.py

clean:
	@echo "Очистка временных файлов..."
	@if exist __pycache__ rmdir /s /q __pycache__ 2>nul || true
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul || true
	@for /r . %%f in (*.pyc) do @del /q "%%f" 2>nul || true
	@for /r . %%f in (*.pyo) do @del /q "%%f" 2>nul || true
	@for /r . %%f in (*.log) do @del /q "%%f" 2>nul || true
	@echo "✓ Очистка завершена"

