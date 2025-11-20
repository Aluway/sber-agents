import os
import logging
import time
from typing import Optional, Dict, Any, List
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
# Импортируем pipeline лениво, чтобы избежать проблем с torchvision при импорте модуля
from datasets import Dataset

# Отключаем прокси для HuggingFace Hub, если он недоступен
if 'HTTP_PROXY' not in os.environ and 'HTTPS_PROXY' not in os.environ:
    os.environ.setdefault('NO_PROXY', 'huggingface.co,*.huggingface.co')

# Импортируем accelerate заранее, чтобы он был доступен для device_map
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import config
import rag

logger = logging.getLogger(__name__)

# Глобальные инициализированные метрики
_ragas_metrics = None
_ragas_run_config = None
_cached_provider = None  # Кэшируем провайдер, чтобы переинициализировать при изменении

# Глобальная переменная для отслеживания времени последнего запроса (rate limiting)
_last_request_time = 0.0

class RateLimitedLLM(BaseChatModel):
    """
    Обертка для LLM с задержкой между запросами для предотвращения rate limit ошибок
    """
    def __init__(self, llm: BaseChatModel, delay: float = 2.0, **kwargs):
        # Инициализируем базовый класс без передачи llm как поля модели
        super().__init__(**kwargs)
        # Сохраняем llm как обычный атрибут (не pydantic поле)
        object.__setattr__(self, 'llm', llm)
        object.__setattr__(self, 'delay', delay)
        object.__setattr__(self, '_last_request_time', 0.0)
    
    def __getattr__(self, name):
        """Делегируем все остальные атрибуты к внутреннему LLM"""
        if name == 'llm':
            return object.__getattribute__(self, 'llm')
        try:
            return getattr(self.llm, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Перехватываем установку атрибутов для llm"""
        if name in ('llm', 'delay', '_last_request_time'):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    
    def _enforce_rate_limit(self):
        """Добавляет задержку между запросами"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s before next request")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def invoke(self, input, config=None, **kwargs):
        """Синхронный вызов с rate limiting"""
        self._enforce_rate_limit()
        return self.llm.invoke(input, config=config, **kwargs)
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Асинхронный вызов с rate limiting"""
        self._enforce_rate_limit()
        return await self.llm.ainvoke(input, config=config, **kwargs)
    
    def batch(self, inputs, config=None, **kwargs):
        """Batch вызов с rate limiting"""
        results = []
        for input_item in inputs:
            self._enforce_rate_limit()
            results.append(self.llm.invoke(input_item, config=config, **kwargs))
        return results
    
    async def abatch(self, inputs, config=None, **kwargs):
        """Асинхронный batch вызов с rate limiting"""
        results = []
        for input_item in inputs:
            self._enforce_rate_limit()
            results.append(await self.llm.ainvoke(input_item, config=config, **kwargs))
        return results
    
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Генерация с rate limiting"""
        self._enforce_rate_limit()
        return self.llm._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)
    
    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Асинхронная генерация с rate limiting"""
        self._enforce_rate_limit()
        return await self.llm._agenerate(prompts, stop=stop, run_manager=run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return f"rate_limited_{self.llm._llm_type}"

def create_ragas_llm():
    """
    Фабрика для создания RAGAS LLM по провайдеру из конфига
    Поддерживает: openai (внешний API), huggingface (локальная модель)
    """
    provider = config.RAGAS_LLM_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI LLM: {config.RAGAS_LLM_MODEL}")
        llm_kwargs = {
            "model": config.RAGAS_LLM_MODEL,
            "temperature": 0,
            # Увеличиваем количество повторных попыток для обработки rate limit
            "max_retries": config.RAGAS_MAX_RETRIES,
            # Таймаут для запросов (в секундах)
            "timeout": config.RAGAS_REQUEST_TIMEOUT,
        }
        # Передаем base_url и api_key, если они заданы в конфиге
        if config.OPENAI_BASE_URL:
            llm_kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.OPENAI_API_KEY:
            llm_kwargs["api_key"] = config.OPENAI_API_KEY
        
        base_llm = ChatOpenAI(**llm_kwargs)
        
        # Оборачиваем в RateLimitedLLM для добавления задержки между запросами
        # Используем обертку, которая правильно делегирует атрибуты
        rate_limited_llm = RateLimitedLLM(base_llm, delay=config.RAGAS_REQUEST_DELAY)
        
        logger.info(f"Rate limit protection: max_retries={config.RAGAS_MAX_RETRIES}, timeout={config.RAGAS_REQUEST_TIMEOUT}s, max_wait={config.RAGAS_MAX_WAIT}s, request_delay={config.RAGAS_REQUEST_DELAY}s")
        return rate_limited_llm
    
    elif provider == "huggingface":
        # Проверяем доступность CUDA
        import torch
        cuda_available = torch.cuda.is_available()
        
        # Определяем фактическое устройство с учетом доступности CUDA
        requested_device = config.RAGAS_HUGGINGFACE_LLM_DEVICE
        if requested_device in ["cuda", "auto"] and not cuda_available:
            logger.warning(f"CUDA requested ({requested_device}) but not available. Falling back to CPU.")
            actual_device = "cpu"
        elif requested_device == "auto":
            actual_device = "cuda" if cuda_available else "cpu"
        else:
            actual_device = requested_device
        
        # Предупреждение о больших моделях на CPU
        large_models = ["qwen2.5-7b", "qwen2.5-14b", "saiga2_7b", "saiga2_13b", "llama-7b", "llama-13b"]
        model_name_lower = config.RAGAS_HUGGINGFACE_LLM_MODEL.lower()
        is_large_model = any(lm in model_name_lower for lm in large_models)
        
        if actual_device == "cpu" and is_large_model:
            logger.warning(
                f"⚠️  Large model '{config.RAGAS_HUGGINGFACE_LLM_MODEL}' on CPU may require 14-26GB RAM. "
                f"Consider using a smaller model like 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' (~3GB RAM) "
                f"or enable quantization (8bit/4bit) if CUDA is available."
            )
        
        logger.info(f"Creating RAGAS HuggingFace LLM: {config.RAGAS_HUGGINGFACE_LLM_MODEL} on {actual_device} (requested: {requested_device}, CUDA available: {cuda_available})")
        
        # Настройка model_kwargs в зависимости от quantization
        model_kwargs = {}
        quantization_config = None
        
        # Добавляем quantization, если указано (используем новый API через BitsAndBytesConfig)
        # Примечание: quantization работает только на CUDA, поэтому проверяем actual_device
        if config.RAGAS_HUGGINGFACE_LLM_QUANTIZATION == "4bit":
            if actual_device == "cpu":
                logger.warning("4-bit quantization requires CUDA, but device is CPU. Loading without quantization.")
            else:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                    )
                    logger.info("Using 4-bit quantization for RAGAS LLM")
                except Exception as e:
                    logger.warning(f"Failed to create 4-bit quantization config: {e}. Loading without quantization.")
        elif config.RAGAS_HUGGINGFACE_LLM_QUANTIZATION == "8bit":
            if actual_device == "cpu":
                logger.warning("8-bit quantization requires CUDA, but device is CPU. Loading without quantization.")
            else:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("Using 8-bit quantization for RAGAS LLM")
                except Exception as e:
                    logger.warning(f"Failed to create 8-bit quantization config: {e}. Loading without quantization.")
        else:
            # Для CPU используем float32, для GPU - float16
            if actual_device == "cpu":
                model_kwargs["torch_dtype"] = "float32"
            else:
                model_kwargs["torch_dtype"] = "float16"
        
        # Добавляем quantization_config в model_kwargs, если он создан
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Определяем device_map для модели (используем actual_device вместо config)
        device_map = actual_device if actual_device != "auto" else ("cuda" if cuda_available else "cpu")
        
        # Определяем device для pipeline (если device_map="auto", используем None для автоматического определения)
        if device_map == "auto":
            pipeline_device = None  # Pipeline сам определит устройство
        elif device_map == "cuda":
            pipeline_device = 0
        else:
            pipeline_device = -1  # CPU
        
        # Загружаем модель и токенайзер
        logger.info(f"Loading model {config.RAGAS_HUGGINGFACE_LLM_MODEL}...")
        try:
            # Определяем тип модели по имени для выбора правильного токенайзера
            model_name_lower = config.RAGAS_HUGGINGFACE_LLM_MODEL.lower()
            
            # Для моделей на базе Llama (Saiga) используем LlamaTokenizer
            # Для остальных (Qwen, DeepSeek и т.д.) используем AutoTokenizer
            if "saiga" in model_name_lower or "llama" in model_name_lower:
                try:
                    tokenizer = LlamaTokenizer.from_pretrained(
                        config.RAGAS_HUGGINGFACE_LLM_MODEL,
                        use_fast=False,  # Медленный токенайзер обходит проблемы с tiktoken
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded tokenizer with LlamaTokenizer")
                except Exception as e1:
                    logger.warning(f"LlamaTokenizer failed: {e1}, trying AutoTokenizer")
                    tokenizer = AutoTokenizer.from_pretrained(
                        config.RAGAS_HUGGINGFACE_LLM_MODEL,
                        use_fast=True,  # Для не-Llama моделей можно использовать fast
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded tokenizer with AutoTokenizer (fallback)")
            else:
                # Для Qwen, DeepSeek и других моделей используем AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.RAGAS_HUGGINGFACE_LLM_MODEL,
                    use_fast=True,  # Fast tokenizer для современных моделей
                    trust_remote_code=True
                )
                logger.info("Successfully loaded tokenizer with AutoTokenizer")
            
            # Подготавливаем параметры загрузки модели
            model_load_kwargs = {
                "trust_remote_code": True,
                **model_kwargs
            }
            
            # Используем device_map только если accelerate доступен и device_map не "cpu"
            # accelerate должен быть импортирован в начале файла
            use_device_map = False
            if ACCELERATE_AVAILABLE and device_map not in ["cpu", None, -1]:
                model_load_kwargs["device_map"] = device_map
                use_device_map = True
                logger.info(f"Using device_map={device_map} with accelerate")
            else:
                # Не используем device_map - загрузим модель и переместим вручную
                if not ACCELERATE_AVAILABLE:
                    logger.warning("accelerate not available, loading model without device_map")
                logger.info(f"Loading model without device_map (target device: {device_map})")
            
            model = AutoModelForCausalLM.from_pretrained(
                config.RAGAS_HUGGINGFACE_LLM_MODEL,
                **model_load_kwargs
            )
            
            # Если не использовали device_map, перемещаем модель на нужное устройство вручную
            if not use_device_map:
                target_device = actual_device  # Используем actual_device, который уже проверен
                model = model.to(target_device)
                logger.info(f"Model moved to {target_device}")
            logger.info("Model loaded successfully")
        except MemoryError as e:
            logger.error(f"❌ Out of memory while loading model: {e}")
            logger.error(f"Model: {config.RAGAS_HUGGINGFACE_LLM_MODEL}")
            logger.error("💡 Solutions:")
            logger.error("  1. Use a smaller model: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' (~3GB RAM)")
            logger.error("  2. Enable quantization: set RAGAS_HUGGINGFACE_LLM_QUANTIZATION=8bit (requires CUDA)")
            logger.error("  3. Use CUDA if available: set RAGAS_HUGGINGFACE_LLM_DEVICE=cuda")
            logger.error("  4. Close other applications to free RAM")
            raise
        except (OSError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "memory" in error_str:
                logger.error(f"❌ Out of memory error: {e}")
                logger.error(f"Model: {config.RAGAS_HUGGINGFACE_LLM_MODEL}")
                logger.error("💡 Solutions:")
                logger.error("  1. Use a smaller model: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' (~3GB RAM)")
                logger.error("  2. Enable quantization: set RAGAS_HUGGINGFACE_LLM_QUANTIZATION=8bit (requires CUDA)")
                logger.error("  3. Use CUDA if available: set RAGAS_HUGGINGFACE_LLM_DEVICE=cuda")
                logger.error("  4. Close other applications to free RAM")
            else:
                logger.error(f"Error loading model: {e}")
                logger.error("Try using a different model or check if the model requires special configuration")
                logger.error(f"Model: {config.RAGAS_HUGGINGFACE_LLM_MODEL}")
                logger.error("Recommended alternatives: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (lightweight) or Qwen/Qwen2.5-7B-Instruct (better quality, needs more RAM)")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Try using a different model or check if the model requires special configuration")
            logger.error(f"Model: {config.RAGAS_HUGGINGFACE_LLM_MODEL}")
            logger.error("Recommended alternatives: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (lightweight) or Qwen/Qwen2.5-7B-Instruct (better quality, needs more RAM)")
            raise
        
        # Создаем pipeline
        # Импортируем pipeline локально, чтобы избежать проблем с torchvision при импорте модуля
        from transformers import pipeline
        
        # Если модель загружена с device_map (accelerate), не передаем device в pipeline
        # Иначе передаем device для ручного управления
        pipeline_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "max_new_tokens": 512,
            "temperature": 0,
            "do_sample": False,  # Для evaluation нужна детерминированность
        }
        
        # Если модель загружена с device_map, не передаем device
        # Иначе передаем device для ручного управления
        if not use_device_map:
            pipeline_kwargs["device"] = pipeline_device
            logger.info(f"Creating pipeline with device={pipeline_device}")
        else:
            logger.info("Creating pipeline without device argument (model uses device_map)")
        
        pipe = pipeline("text-generation", **pipeline_kwargs)
        
        # Обертываем в Langchain LLM (HuggingFacePipeline работает как обычный LLM)
        # RAGAS может работать с обычным LLM через LangchainLLMWrapper
        return HuggingFacePipeline(pipeline=pipe)
    
    else:
        raise ValueError(f"Unknown RAGAS LLM provider: {provider}. Use 'openai' or 'huggingface'")

def create_ragas_embeddings():
    """
    Фабрика для создания RAGAS embeddings по провайдеру из конфига
    Поддерживает: openai, huggingface
    """
    provider = config.RAGAS_EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI embeddings: {config.RAGAS_EMBEDDING_MODEL}")
        embedding_kwargs = {"model": config.RAGAS_EMBEDDING_MODEL}
        # Передаем base_url и api_key, если они заданы в конфиге
        if config.OPENAI_BASE_URL:
            embedding_kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.OPENAI_API_KEY:
            embedding_kwargs["api_key"] = config.OPENAI_API_KEY
        return OpenAIEmbeddings(**embedding_kwargs)
    
    elif provider == "huggingface":
        logger.info(f"Creating RAGAS HuggingFace embeddings: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
        return HuggingFaceEmbeddings(
            model_name=config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={'device': config.RAGAS_HUGGINGFACE_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"Unknown RAGAS embedding provider: {provider}. Use 'openai' or 'huggingface'")

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config, _cached_provider
    
    # Проверяем, изменился ли провайдер - если да, переинициализируем метрики
    current_provider = config.RAGAS_LLM_PROVIDER.lower()
    if _ragas_metrics is not None and _cached_provider == current_provider:
        return _ragas_metrics, _ragas_run_config
    
    # Если провайдер изменился, сбрасываем кэш
    if _cached_provider is not None and _cached_provider != current_provider:
        logger.info(f"RAGAS LLM provider changed from '{_cached_provider}' to '{current_provider}'. Reinitializing metrics...")
        _ragas_metrics = None
        _ragas_run_config = None
    
    logger.info("Initializing RAGAS metrics...")
    
    # Проверяем, что модель настроена правильно для Fireworks (только если используется OpenAI провайдер)
    if config.RAGAS_LLM_PROVIDER.lower() == "openai":
        if config.OPENAI_BASE_URL and "fireworks" in config.OPENAI_BASE_URL.lower():
            if not config.RAGAS_LLM_MODEL or config.RAGAS_LLM_MODEL == "gpt-4o":
                logger.warning(
                    f"RAGAS_LLM_MODEL is set to '{config.RAGAS_LLM_MODEL}' but using Fireworks API. "
                    f"This may cause 404 errors. Set RAGAS_LLM_MODEL to a Fireworks model (e.g., "
                    f"'accounts/fireworks/models/gpt-oss-120b') in your .env file."
                )
    
    # Настройка LLM и embeddings для RAGAS (фиксированные модели для единообразной оценки)
    langchain_llm = create_ragas_llm()
    
    if config.RAGAS_LLM_PROVIDER.lower() == "openai":
        logger.info(f"RAGAS LLM configured: {config.RAGAS_LLM_MODEL} (provider: openai, base_url: {config.OPENAI_BASE_URL or 'default'})")
    else:
        logger.info(f"RAGAS LLM configured: {config.RAGAS_HUGGINGFACE_LLM_MODEL} (provider: huggingface, device: {config.RAGAS_HUGGINGFACE_LLM_DEVICE})")
    
    langchain_embeddings = create_ragas_embeddings()
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    # Инициализируем метрики
    # Если langchain_llm - это RateLimitedLLM, передаем базовый LLM в LangchainLLMWrapper
    # чтобы избежать проблем с доступом к полю llm в pydantic
    # Rate limiting будет работать через переопределенные методы в RateLimitedLLM
    if isinstance(langchain_llm, RateLimitedLLM):
        # Извлекаем базовый LLM для LangchainLLMWrapper
        base_llm_for_ragas = langchain_llm.llm
        logger.info("Using base LLM (extracted from RateLimitedLLM) for LangchainLLMWrapper to avoid pydantic field access issues")
        # Создаем wrapper с базовым LLM
        ragas_llm = LangchainLLMWrapper(base_llm_for_ragas)
        # Заменяем внутренний LLM на RateLimitedLLM, чтобы rate limiting работал
        # LangchainLLMWrapper будет использовать RateLimitedLLM для вызовов
        ragas_llm.llm = langchain_llm
    else:
        ragas_llm = LangchainLLMWrapper(langchain_llm)
    
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = ragas_llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = ragas_embeddings
        run_config = RunConfig()
        metric.init(run_config)
    
    # Настройки для выполнения
    # Увеличиваем параметры для обработки rate limit ошибок
    run_config = RunConfig(
        max_workers=1,  # Последовательная обработка для стабильности
        max_wait=config.RAGAS_MAX_WAIT,  # Время ожидания между попытками (из конфига)
        max_retries=config.RAGAS_MAX_RETRIES  # Количество попыток при ошибках (из конфига)
    )
    logger.info(f"RAGAS RunConfig: max_workers=1, max_wait={config.RAGAS_MAX_WAIT}s, max_retries={config.RAGAS_MAX_RETRIES}")
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    _cached_provider = current_provider  # Сохраняем текущий провайдер
    
    logger.info(f"✓ RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
    logger.info(f"✓ RAGAS LLM Provider: {config.RAGAS_LLM_PROVIDER}")
    if config.RAGAS_LLM_PROVIDER.lower() == "openai":
        logger.info(f"✓ RAGAS LLM: {config.RAGAS_LLM_MODEL}")
    else:
        logger.info(f"✓ RAGAS LLM: {config.RAGAS_HUGGINGFACE_LLM_MODEL} (device: {config.RAGAS_HUGGINGFACE_LLM_DEVICE}, quantization: {config.RAGAS_HUGGINGFACE_LLM_QUANTIZATION})")
    logger.info(f"✓ RAGAS Embedding Provider: {config.RAGAS_EMBEDDING_PROVIDER}")
    if config.RAGAS_EMBEDDING_PROVIDER == "openai":
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
    
    return _ragas_metrics, _ragas_run_config

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Проверка существования датасета в LangSmith
    
    Args:
        dataset_name: имя датасета
    
    Returns:
        True если датасет существует
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def evaluate_dataset(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Главная функция evaluation RAG системы
    
    По образцу референсного ноутбука (раздел 5.2):
    1. Запуск эксперимента в LangSmith с blocking=False и сбор данных
    2. RAGAS batch evaluation
    3. Загрузка метрик как feedback в LangSmith
    
    Args:
        dataset_name: имя датасета (по умолчанию из конфига)
    
    Returns:
        dict с результатами evaluation
    """
    if not config.LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set. Cannot run evaluation.")
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
    
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        # Используем существующую RAG цепочку
        # Передаем только вопрос (без истории для evaluation)
        from langchain_core.messages import HumanMessage
        result = rag.get_rag_chain().invoke({"messages": [HumanMessage(content=question)]})
        
        return {
            "answer": result["answer"],
            "documents": result["documents"]
        }
    
    # Собираем данные во время выполнения evaluate
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    run_ids = []
    
    # evaluate() с blocking=False возвращает итератор
    for result in client.evaluate(
        target,
        data=dataset_name,
        evaluators=[],
        experiment_prefix="rag-evaluation",
        metadata={
            "approach": "RAGAS batch evaluation + LangSmith feedback",
            "model": config.MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        blocking=False,
    ):
        run = result["run"]
        example = result["example"]
        
        # Получаем данные
        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        documents = run.outputs.get("documents", [])
        contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        ground_truth = example.outputs.get("answer", "") if example else ""
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation с обработкой ошибок
    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=ragas_metrics,
            run_config=ragas_run_config,
        )
        
        ragas_df = ragas_result.to_pandas()
        
        logger.info("RAGAS evaluation completed")
    except Exception as e:
        error_msg = str(e)
        # Проверяем, является ли это rate limit ошибкой
        if "429" in error_msg or "rate limit" in error_msg.lower() or "RateLimitError" in str(type(e)):
            logger.error(f"❌ Rate limit error during RAGAS evaluation: {e}")
            logger.error("💡 Solutions:")
            logger.error(f"  1. Wait a few minutes and try again (current max_wait={config.RAGAS_MAX_WAIT}s)")
            logger.error(f"  2. Increase RAGAS_MAX_RETRIES in .env (current: {config.RAGAS_MAX_RETRIES})")
            logger.error(f"  3. Increase RAGAS_MAX_WAIT in .env (current: {config.RAGAS_MAX_WAIT}s)")
            logger.error("  4. Consider using a local HuggingFace model (RAGAS_LLM_PROVIDER=huggingface)")
            logger.error("  5. Reduce the dataset size or split evaluation into smaller batches")
        else:
            logger.error(f"❌ Error during RAGAS evaluation: {e}")
        raise
    
    # Вычисляем средние значения метрик
    metrics_summary = {}
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            avg_score = ragas_df[metric.name].mean()
            metrics_summary[metric.name] = avg_score
            logger.info(f"  {metric.name}: {avg_score:.3f}")
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                client.create_feedback(
                    run_id=run_id,
                    key=metric.name,
                    score=float(score),
                    comment=f"RAGAS metric: {metric.name}"
                )
    
    logger.info(f"Feedback uploaded ({len(run_ids)} runs)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

