import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("MODEL")
    MODEL_QUERY_TRANSFORM = os.getenv("MODEL_QUERY_TRANSFORM", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
    CONVERSATION_SYSTEM_PROMPT_FILE = os.getenv("CONVERSATION_SYSTEM_PROMPT_FILE", "conversation_system.txt")
    QUERY_TRANSFORM_PROMPT_FILE = os.getenv("QUERY_TRANSFORM_PROMPT_FILE", "query_transform.txt")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai/huggingface
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    # Очищаем значение от комментариев (если они попали в переменную)
    _hf_device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
    HUGGINGFACE_DEVICE = _hf_device.split('#')[0].strip() if _hf_device else "cpu"  # cpu/cuda/mps
    
    # Retrieval Configuration
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "semantic")  # semantic/hybrid/hybrid_reranker
    SEMANTIC_RETRIEVER_K = int(os.getenv("SEMANTIC_RETRIEVER_K", "10"))
    BM25_RETRIEVER_K = int(os.getenv("BM25_RETRIEVER_K", "10"))
    ENSEMBLE_SEMANTIC_WEIGHT = float(os.getenv("ENSEMBLE_SEMANTIC_WEIGHT", "0.5"))
    ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.5"))
    
    # Cross-Encoder Reranking Configuration
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    
    # Отображение источников
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    
    # LangSmith настройки
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # RAGAS evaluation настройки (фиксированные модели для единообразной оценки)
    # Провайдер LLM для RAGAS: "openai" (внешний API) или "huggingface" (локальная модель)
    RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", "openai")  # openai/huggingface
    # Если RAGAS_LLM_MODEL не задан, используем основную модель (MODEL)
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL") or MODEL or "gpt-4o"
    # Настройки для обработки rate limit (для внешних API)
    RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "20"))  # Количество повторных попыток (увеличено до 20)
    RAGAS_MAX_WAIT = int(os.getenv("RAGAS_MAX_WAIT", "1800"))  # Максимальное время ожидания между попытками (30 минут)
    RAGAS_REQUEST_TIMEOUT = float(os.getenv("RAGAS_REQUEST_TIMEOUT", "120.0"))  # Таймаут запроса (2 минуты)
    RAGAS_REQUEST_DELAY = float(os.getenv("RAGAS_REQUEST_DELAY", "2.0"))  # Задержка между запросами в секундах (для предотвращения rate limit)
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)  # По умолчанию = основному провайдеру
    # Для HuggingFace используем те же настройки что и для основных embeddings
    RAGAS_HUGGINGFACE_EMBEDDING_MODEL = os.getenv("RAGAS_HUGGINGFACE_EMBEDDING_MODEL", HUGGINGFACE_EMBEDDING_MODEL)
    # Очищаем значение от комментариев (если они попали в переменную)
    _ragas_hf_device = os.getenv("RAGAS_HUGGINGFACE_DEVICE", HUGGINGFACE_DEVICE)
    RAGAS_HUGGINGFACE_DEVICE = _ragas_hf_device.split('#')[0].strip() if _ragas_hf_device else HUGGINGFACE_DEVICE
    # Настройки для HuggingFace LLM (если RAGAS_LLM_PROVIDER=huggingface)
    RAGAS_HUGGINGFACE_LLM_MODEL = os.getenv("RAGAS_HUGGINGFACE_LLM_MODEL", "IlyaGusev/saiga2_7b_lora")
    RAGAS_HUGGINGFACE_LLM_DEVICE = os.getenv("RAGAS_HUGGINGFACE_LLM_DEVICE", "auto")  # auto/cpu/cuda
    RAGAS_HUGGINGFACE_LLM_QUANTIZATION = os.getenv("RAGAS_HUGGINGFACE_LLM_QUANTIZATION", "none")  # none/4bit/8bit
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        prompt_path = Path(cls.PROMPTS_DIR) / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')
    
    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        # Валидация RETRIEVAL_MODE
        valid_retrieval_modes = ["semantic", "hybrid", "hybrid_reranker"]
        if cls.RETRIEVAL_MODE not in valid_retrieval_modes:
            raise ValueError(
                f"Invalid RETRIEVAL_MODE: {cls.RETRIEVAL_MODE}. "
                f"Must be one of: {', '.join(valid_retrieval_modes)}"
            )
        
        # Валидация EMBEDDING_PROVIDER
        valid_embedding_providers = ["openai", "huggingface"]
        if cls.EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {cls.EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )
        
        # Валидация RAGAS_EMBEDDING_PROVIDER
        if cls.RAGAS_EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid RAGAS_EMBEDDING_PROVIDER: {cls.RAGAS_EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )

config = Config()
# Валидация конфигурации при загрузке
config.validate()

