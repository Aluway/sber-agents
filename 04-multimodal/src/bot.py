import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinanceBot:
    """Финансовый бот для учета доходов и расходов"""
    
    def __init__(self):
        # Загрузка переменных окружения
        load_dotenv()
        
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_TOKEN not found in .env")
        
        # Инициализация aiogram
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        
        # Регистрация хэндлеров
        self.dp.message.register(self.handle_message)
        
        logger.info("FinanceBot initialized")
    
    async def handle_message(self, message: Message):
        """Обработка текстового сообщения (эхо-бот)"""
        logger.info(f"Message received from user {message.from_user.id}")
        
        try:
            # Эхо-ответ
            await message.answer(f"Вы написали: {message.text}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await message.answer("Произошла ошибка при обработке сообщения")
    
    async def run(self):
        """Запуск бота через polling"""
        logger.info("Bot started")
        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Error in polling: {e}")
        finally:
            await self.bot.session.close()


if __name__ == "__main__":
    import asyncio
    
    bot = FinanceBot()
    asyncio.run(bot.run())
