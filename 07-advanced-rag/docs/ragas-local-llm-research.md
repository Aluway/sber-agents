# Исследование: Локальные HuggingFace модели для RAGAS LLM

## Цель
Найти оптимальную локальную HuggingFace модель для использования в качестве LLM в RAGAS evaluation вместо внешних провайдеров (OpenAI/Fireworks).

## Требования

### Функциональные
- ✅ Поддержка русского языка
- ✅ Способность к reasoning и оценке качества ответов
- ✅ Работа с RAGAS через LangchainLLMWrapper
- ✅ Стабильная работа в evaluation режиме

### Технические
- ✅ Совместимость с Langchain
- ✅ Работа на CPU (желательно) или GPU
- ✅ Разумное потребление памяти
- ✅ Приемлемая скорость инференса

## Рекомендуемые модели

### 1. **Saiga2 7B/13B** (Рекомендуется для начала)
- **Модель**: `IlyaGusev/saiga2_7b_lora` или `IlyaGusev/saiga2_13b_lora`
- **Размер**: 7B или 13B параметров
- **Плюсы**:
  - ✅ Отличная поддержка русского языка
  - ✅ Специально обучена для русскоязычных задач
  - ✅ Хорошее качество reasoning
  - ✅ Популярная в русскоязычном сообществе
- **Минусы**:
  - ⚠️ 7B требует ~14GB RAM, 13B ~26GB RAM
  - ⚠️ Медленнее на CPU
- **Требования**: GPU рекомендуется для 13B, CPU возможно для 7B с quantization

### 2. **Qwen2.5 7B/14B** (Альтернатива)
- **Модель**: `Qwen/Qwen2.5-7B-Instruct` или `Qwen/Qwen2.5-14B-Instruct`
- **Размер**: 7B или 14B параметров
- **Плюсы**:
  - ✅ Хорошая мультиязычная поддержка (включая русский)
  - ✅ Современная архитектура
  - ✅ Хорошее качество для evaluation задач
  - ✅ Активно поддерживается
- **Минусы**:
  - ⚠️ Может быть слабее в русском, чем Saiga
  - ⚠️ Требует больше ресурсов
- **Требования**: GPU рекомендуется

### 3. **Llama 3.1 8B** (Если доступна)
- **Модель**: `meta-llama/Llama-3.1-8B-Instruct`
- **Размер**: 8B параметров
- **Плюсы**:
  - ✅ Хорошее качество reasoning
  - ✅ Поддержка русского (через fine-tuning)
  - ✅ Оптимизированная архитектура
- **Минусы**:
  - ⚠️ Требует запрос на доступ в Meta
  - ⚠️ Может быть слабее в русском без fine-tuning
- **Требования**: GPU рекомендуется

### 4. **DeepSeek-R1 1.5B/7B** (Для быстрого тестирования)
- **Модель**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` или `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Размер**: 1.5B или 7B параметров
- **Плюсы**:
  - ✅ Специально для reasoning задач
  - ✅ Меньший размер (1.5B вариант)
  - ✅ Быстрая работа
- **Минусы**:
  - ⚠️ Может быть слабее в русском
  - ⚠️ Меньший размер = меньше качество
- **Требования**: CPU возможно для 1.5B

## Рекомендация по выбору

### Для CPU (ограниченные ресурсы):
1. **Saiga2 7B** с quantization (4-bit или 8-bit)
2. **DeepSeek-R1 1.5B** (быстро, но слабее)

### Для GPU (есть видеокарта):
1. **Saiga2 13B** (лучшее качество для русского)
2. **Qwen2.5 14B** (хороший баланс)
3. **Saiga2 7B** (быстрее, но качество ниже)

### Для начала (быстрый старт):
**Saiga2 7B** - оптимальный баланс качества, размера и поддержки русского языка.

## Интеграция с Langchain

RAGAS использует `LangchainLLMWrapper`, который работает с любым Langchain LLM. Для HuggingFace доступны два основных способа:

### 1. ChatHuggingFace (Рекомендуется)
```python
from langchain_huggingface import ChatHuggingFace

llm = ChatHuggingFace(
    model_id="IlyaGusev/saiga2_7b_lora",
    task="text-generation",
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": "float16",  # или "bfloat16"
        "load_in_4bit": True,  # для quantization
    }
)
```

### 2. HuggingFacePipeline (Альтернатива)
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="IlyaGusev/saiga2_7b_lora",
    device_map="auto",
    model_kwargs={"torch_dtype": "float16"}
)

llm = HuggingFacePipeline(pipeline=pipe)
```

## Конфигурация

Рекомендуется добавить в конфиг:
- `RAGAS_LLM_PROVIDER`: "openai" | "huggingface"
- `RAGAS_HUGGINGFACE_LLM_MODEL`: путь к модели
- `RAGAS_HUGGINGFACE_LLM_DEVICE`: "cpu" | "cuda" | "auto"
- `RAGAS_HUGGINGFACE_LLM_QUANTIZATION`: "4bit" | "8bit" | "none"

## Производительность

### Ожидаемая скорость (примерно):
- **Saiga2 7B на CPU**: ~1-3 сек на запрос
- **Saiga2 7B на GPU**: ~0.5-1 сек на запрос
- **Saiga2 13B на GPU**: ~1-2 сек на запрос

### Память:
- **7B FP16**: ~14GB RAM/VRAM
- **7B 8-bit**: ~7GB RAM/VRAM
- **7B 4-bit**: ~4GB RAM/VRAM
- **13B FP16**: ~26GB RAM/VRAM

## Выводы

**Рекомендуемая конфигурация для начала:**
- Модель: `IlyaGusev/saiga2_7b_lora`
- Провайдер: HuggingFace через ChatHuggingFace
- Устройство: GPU (если есть), иначе CPU с quantization
- Quantization: 8-bit для GPU, 4-bit для CPU

**Альтернатива (если нет GPU):**
- Модель: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Устройство: CPU
- Quantization: не требуется (модель маленькая)

## Настройка в .env

```env
# Использовать локальную HuggingFace модель
RAGAS_LLM_PROVIDER=huggingface
RAGAS_HUGGINGFACE_LLM_MODEL=IlyaGusev/saiga2_7b_lora
RAGAS_HUGGINGFACE_LLM_DEVICE=auto  # или cpu/cuda
RAGAS_HUGGINGFACE_LLM_QUANTIZATION=8bit  # для экономии памяти
```

## Установка зависимостей

Для quantization может потребоваться:
```bash
pip install bitsandbytes  # для 4-bit и 8-bit quantization
```

Для GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Тестирование

После настройки запустите evaluation и проверьте логи:
- Модель должна загрузиться при инициализации RAGAS
- Первый запрос может быть медленным (загрузка модели)
- Последующие запросы должны быть быстрее

## Производительность

Ожидаемое время на evaluation (29 примеров):
- **Saiga2 7B на GPU**: ~30-60 секунд
- **Saiga2 7B на CPU**: ~2-5 минут
- **DeepSeek-R1 1.5B на CPU**: ~1-2 минуты

