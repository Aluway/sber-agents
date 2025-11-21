"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from langchain_core.tools import tool
import rag
from config import config

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests library not installed. Install it with: pip install requests")

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую используя актуальные курсы валют.
    
    Args:
        amount: Сумма для конвертации
        from_currency: Исходная валюта (например: USD, EUR, RUB, GBP, JPY)
        to_currency: Целевая валюта (например: USD, EUR, RUB, GBP, JPY)
    
    Returns:
        Строка с результатом конвертации в формате: "100.0 USD = 92.50 EUR"
    """
    if requests is None:
        return "Ошибка: библиотека requests не установлена. Установите её командой: pip install requests"
    
    try:
        # Нормализуем коды валют (верхний регистр)
        from_currency = from_currency.upper().strip()
        to_currency = to_currency.upper().strip()
        
        # Проверка на одинаковые валюты
        if from_currency == to_currency:
            return f"{amount:.2f} {from_currency} = {amount:.2f} {to_currency} (валюта не изменилась)"
        
        # Используем API ключ если он есть, иначе бесплатный endpoint
        api_key = getattr(config, 'EXCHANGERATE_API_KEY', None)
        
        if api_key:
            # Используем API с ключом (v6 API)
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}"
        else:
            # Бесплатный endpoint без ключа (v4 API, ограниченный функционал)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        
        # Выполняем запрос к API
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Обработка ответа в зависимости от версии API
        if api_key:
            # v6 API с ключом
            if data.get('result') != 'success':
                error_type = data.get('error-type', 'unknown')
                logger.error(f"ExchangeRate API error: {error_type}")
                return f"Ошибка при получении курса валют: {error_type}"
            
            rates = data.get('conversion_rates', {})
        else:
            # v4 API без ключа
            rates = data.get('rates', {})
        
        # Проверяем наличие целевой валюты
        if to_currency not in rates:
            return f"Ошибка: валюта {to_currency} не найдена. Доступные валюты: {', '.join(sorted(rates.keys())[:10])}..."
        
        # Вычисляем конвертированную сумму
        rate = rates[to_currency]
        converted_amount = amount * rate
        
        # Форматируем результат
        result = f"{amount:.2f} {from_currency} = {converted_amount:.2f} {to_currency} (курс: 1 {from_currency} = {rate:.4f} {to_currency})"
        
        logger.info(f"Currency conversion: {amount} {from_currency} -> {converted_amount:.2f} {to_currency}")
        return result
        
    except requests.exceptions.Timeout:
        logger.error("ExchangeRate API timeout")
        return "Ошибка: превышено время ожидания ответа от сервиса конвертации валют. Попробуйте позже."
    except requests.exceptions.RequestException as e:
        logger.error(f"ExchangeRate API request error: {e}")
        return f"Ошибка при подключении к сервису конвертации валют: {str(e)}"
    except KeyError as e:
        logger.error(f"ExchangeRate API response format error: {e}")
        return f"Ошибка: неверный формат ответа от API. Проверьте правильность кодов валют."
    except Exception as e:
        logger.error(f"Unexpected error in currency_converter: {e}", exc_info=True)
        return f"Неожиданная ошибка при конвертации валют: {str(e)}"