#!/usr/bin/env python3
"""
Telegram бот с эхо-ответами на aiogram.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram бот с эхо-ответами."""
    
    def __init__(self):
        """Инициализация бота."""
        load_dotenv()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN не найден в .env файле!")
            raise ValueError("Необходим TELEGRAM_BOT_TOKEN")
        
        self.bot = Bot(token=self.bot_token)
        self.dp = Dispatcher()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Настройка обработчиков сообщений."""
        self.dp.message.register(self.handle_start, Command("start"))
        self.dp.message.register(self.handle_message, F.text)
    
    async def handle_start(self, message: Message):
        """Обработчик команды /start."""
        await message.answer("Привет! Я финансовый советник. Задай мне вопрос!")
    
    async def handle_message(self, message: Message):
        """Обработчик текстовых сообщений."""
        await message.answer(message.text)
    
    async def run(self):
        """Запуск бота."""
        logger.info("Бот запущен")
        await self.dp.start_polling(self.bot)


def main():
    """Точка входа в программу."""
    try:
        bot = TelegramBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
    except Exception as e:
        logger.error(f"Ошибка при работе бота: {e}")


if __name__ == "__main__":
    main()

