import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from openai import OpenAI

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
        # Ищем .env в корне проекта (на уровень выше src/)
        env_path = Path(__file__).parent.parent / '.env'
        
        # Загружаем переменные
        if env_path.exists():
            # Читаем файл вручную и устанавливаем переменные
            with open(env_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig убирает BOM
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            raise ValueError(f"TELEGRAM_TOKEN not found. Checked: {env_path}")
        
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError(f"OPENROUTER_API_KEY not found. Checked: {env_path}")
        
        # Инициализация aiogram
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        
        # Инициализация LLM
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        
        # Модель LLM
        self.llm_model = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
        
        # История диалога
        self.chat_history = []
        
        # Системный промпт
        self.system_prompt = """Ты - дружелюбный финансовый помощник.
Отвечай кратко и по делу. Будь вежливым и полезным."""
        
        # Регистрация хэндлеров
        self.dp.message.register(self.handle_message)
        
        logger.info("FinanceBot initialized")
    
    async def call_llm(self, user_message: str) -> str:
        """Вызов LLM через OpenRouter"""
        try:
            # Добавляем сообщение пользователя в историю
            self.chat_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Формируем полную историю с системным промптом
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.chat_history
            
            # Вызов LLM
            logger.info(f"Calling LLM ({self.llm_model})...")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            
            # Получаем ответ
            assistant_message = response.choices[0].message.content
            logger.info("LLM call successful")
            
            # Добавляем ответ в историю
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return "Извините, произошла ошибка при обработке запроса."
    
    async def handle_message(self, message: Message):
        """Обработка текстового сообщения"""
        logger.info(f"Message received from user {message.from_user.id}")
        
        try:
            # Вызываем LLM
            response = await self.call_llm(message.text)
            await message.answer(response)
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
