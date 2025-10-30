#!/usr/bin/env python3
"""
Telegram бот с LLM в роли финансового советника.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from openai import OpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Системный промпт для финансового советника
SYSTEM_PROMPT = """Ты — профессиональный финансовый советник. 
Твоя задача — помогать пользователям с финансовыми вопросами, 
планированием бюджета и накоплениями. Отвечай кратко, по делу и дружелюбно."""


class TelegramBot:
    """Telegram бот с LLM финансовым советником."""
    
    def __init__(self):
        """Инициализация бота."""
        load_dotenv()
        
        # Загрузка конфигурации
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        
        # Проверка обязательных параметров
        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN не найден в .env файле!")
            raise ValueError("Необходим TELEGRAM_BOT_TOKEN")
        
        if not self.openrouter_key:
            logger.error("OPENROUTER_API_KEY не найден в .env файле!")
            raise ValueError("Необходим OPENROUTER_API_KEY")
        
        # Инициализация Telegram бота
        self.bot = Bot(token=self.bot_token)
        self.dp = Dispatcher()
        
        # Инициализация OpenAI клиента для OpenRouter
        self.llm_client = OpenAI(
            api_key=self.openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/aidd-bot",
                "X-Title": "AI Financial Advisor Bot"
            }
        )
        
        # История диалога
        self.conversation_history = []
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Настройка обработчиков сообщений."""
        self.dp.message.register(self.handle_start, Command("start"))
        self.dp.message.register(self.handle_message, F.text)
    
    async def handle_start(self, message: Message):
        """Обработчик команды /start."""
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        await message.answer("Привет! Я финансовый советник. Задай мне вопрос!")
    
    async def handle_message(self, message: Message):
        """Обработчик текстовых сообщений."""
        try:
            # Добавляем сообщение пользователя в историю
            self.conversation_history.append({"role": "user", "content": message.text})
            
            # Получаем ответ от LLM
            response_text = await self.get_llm_response()
            
            # Добавляем ответ ассистента в историю
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            await message.answer(response_text)
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            await message.answer("Извините, произошла ошибка. Попробуйте позже.")
    
    async def get_llm_response(self) -> str:
        """Получить ответ от LLM."""
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history
        )
        
        return response.choices[0].message.content
    
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

