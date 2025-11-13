.PHONY: install run clean

install:
	uv sync

run:
	uv run python src/bot.py

clean:
	rm -rf __pycache__ .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

