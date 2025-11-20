import os
import asyncio
import logging
from pathlib import Path

# Отключаем предупреждение tokenizers о параллелизме (для HuggingFace)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Устанавливаем UTF-8 кодировку для консоли и файлов
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Отключаем прокси для HuggingFace Hub (если прокси недоступен)
# Это решает проблему "Unable to connect to proxy" при загрузке моделей
# Проверяем, настроен ли прокси в системе
http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

if not http_proxy and not https_proxy:
    # Явно отключаем прокси для HuggingFace, если он не настроен
    os.environ['NO_PROXY'] = 'huggingface.co,*.huggingface.co'
    # Устанавливаем пустые значения для прокси, чтобы отключить его
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    # Также отключаем для huggingface_hub
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    
    # Отключаем прокси для requests/urllib3 (используется huggingface_hub)
    try:
        import urllib3
        urllib3.disable_warnings()
        # Отключаем прокси для urllib3
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['CURL_CA_BUNDLE'] = ''
    except ImportError:
        pass

from aiogram import Bot, Dispatcher
from handlers import router
from config import config
import indexer
import rag

# Создаем директорию для логов
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка логирования в консоль и файл
# Создаем StreamHandler с правильной кодировкой для Windows
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_dir / "bot.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        console_handler,  # Вывод в консоль
        file_handler  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 70)
    logger.info("🤖 Advanced Hybrid RAG Bot Starting...")
    logger.info("=" * 70)
    
    # Логирование конфигурации
    logger.info("📋 Configuration:")
    logger.info(f"  Retrieval mode: {config.RETRIEVAL_MODE}")
    logger.info(f"  Embedding provider: {config.EMBEDDING_PROVIDER}")
    if config.EMBEDDING_PROVIDER == "openai":
        logger.info(f"  Embedding model: {config.EMBEDDING_MODEL}")
    elif config.EMBEDDING_PROVIDER == "huggingface":
        logger.info(f"  Embedding model: {config.HUGGINGFACE_EMBEDDING_MODEL}")
        logger.info(f"  Device: {config.HUGGINGFACE_DEVICE}")
    
    if config.RETRIEVAL_MODE in ["hybrid", "hybrid_reranker"]:
        logger.info(f"  Semantic k: {config.SEMANTIC_RETRIEVER_K}, BM25 k: {config.BM25_RETRIEVER_K}")
        logger.info(f"  Ensemble weights: {config.ENSEMBLE_SEMANTIC_WEIGHT}/{config.ENSEMBLE_BM25_WEIGHT}")
    if config.RETRIEVAL_MODE == "hybrid_reranker":
        logger.info(f"  Cross-encoder: {config.CROSS_ENCODER_MODEL}")
        logger.info(f"  Reranker top-k: {config.RERANKER_TOP_K}")
    
    logger.info(f"  LangSmith tracing: {config.LANGSMITH_TRACING_V2}")
    logger.info(f"  Show sources: {config.SHOW_SOURCES}")
    logger.info("-" * 70)
    
    # Индексация при старте
    logger.info("📚 Starting indexing...")
    result = await indexer.reindex_all()
    if result and result[0] is not None:
        rag.vector_store, rag.chunks = result
        # Инициализируем retriever
        rag.initialize_retriever()
        stats = rag.get_vector_store_stats()
        logger.info(f"✅ Indexing completed: {stats['count']} documents indexed")
    else:
        logger.warning("⚠️  Indexing completed with no documents - bot will run but cannot answer questions")
    
    bot = Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("-" * 70)
    logger.info("🚀 Starting bot polling...")
    logger.info("=" * 70)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except asyncio.CancelledError:
        logger.info("👋 Bot polling cancelled")
    except Exception as e:
        logger.error(f"❌ Bot stopped with error: {e}", exc_info=True)
        raise
    finally:
        await bot.session.close()
        logger.info("=" * 70)
        logger.info("🛑 Bot shutdown complete")
        logger.info("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())

