"""
Інструменти для аналізу ринкової тональності та роботи з ринковими даними
"""
import asyncio
import openai
import backoff
import httpx
import re
import json
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Константа для правил аналізу тональності
SENTIMENT_ANALYSIS_RULES = """Проаналізуй тональність економічної новини для ринку.

Класифікуй як:
- bullish: позитивний вплив на ринок (зростання, позитивні показники)
- bearish: негативний вплив на ринок (падіння, негативні показники)  
- neutral: нейтральний або неоднозначний вплив

Поверни результат у JSON форматі:
{
  "sentiment": "bullish/bearish/neutral",
  "explanation": "коротке пояснення чому така оцінка (до 50 слів)",
  "confidence": 0.8
}
"""

async def analyze_market_sentiment(news_text: str) -> str:
    """
    Аналізує тональність новини за допомогою OpenAI GPT-3.5-turbo
    
    Args:
        news_text: Текст новини для аналізу
        
    Returns:
        JSON рядок з результатами аналізу
    """
    
    @backoff.on_exception(
        backoff.expo,
        (openai.APIError, openai.RateLimitError, asyncio.TimeoutError),
        max_tries=3
    )
    async def call_openai():
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SENTIMENT_ANALYSIS_RULES},
                        {"role": "user", "content": f"Новина: {news_text[:2000]}"}
                    ],
                    temperature=0,
                    max_tokens=512,
                    stream=False
                ),
                timeout=30.0
            )
            return response.choices[0].message.content.strip()
        finally:
            await client.close()
    
    return await call_openai()


async def analyze_multiple_news(news_list: list) -> list:
    """
    Аналізує список новин паралельно
    
    Args:
        news_list: Список текстів новин
        
    Returns:
        Список результатів аналізу
    """
    semaphore = asyncio.Semaphore(5)  # Максимум 5 одночасних запитів
    
    async def analyze_single(news_text):
        async with semaphore:
            return await analyze_market_sentiment(news_text)
    
    tasks = [analyze_single(news) for news in news_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Обробляємо помилки
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "sentiment": "neutral",
                "explanation": f"Помилка аналізу: {str(result)[:50]}",
                "confidence": 0.0
            })
        else:
            try:
                import json
                processed_results.append(json.loads(result))
            except:
                processed_results.append({
                    "sentiment": "neutral", 
                    "explanation": "Не вдалося розпарсити відповідь",
                    "confidence": 0.0
                })
    
    return processed_results


# Константа для детекції інтентів
INTENT_DETECTION_RULES = """Ти асистент, який визначає, що саме хоче користувач.

Проаналізуй запит користувача і визнач тип:

- analyze_news: якщо просить оцінити ринкову ситуацію, новини, загальний стан ринку
- get_price: якщо просить дізнатись ціну конкретних акцій/валюти/криптовалюти  
- investment_advice: якщо питає "куди вкладати", "що купити", "як інвестувати"
- general_chat: якщо звичайне спілкування, вітання, подяки
- fallback: якщо незрозуміло або не стосується фінансів

Відповідай тільки одним словом з списку вище.

Приклади:
"Що там на ринку?" → analyze_news
"Скільки коштує AAPL?" → get_price  
"Куди краще вкладати гроші?" → investment_advice
"Привіт" → general_chat
"Як справи?" → general_chat
"""

# Популярні тикери для розпізнавання
POPULAR_TICKERS = {
    # Американські акції
    "apple": "AAPL", "aapl": "AAPL", "яблоко": "AAPL", "епл": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "мікрософт": "MSFT", "майкрософт": "MSFT",
    "google": "GOOGL", "googl": "GOOGL", "гугл": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN", "амазон": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA", "тесла": "TSLA", "илон маск": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA", "нвідіа": "NVDA", "нвидіа": "NVDA",
    "meta": "META", "facebook": "META", "фейсбук": "META", "мета": "META",
    "netflix": "NFLX", "нетфлікс": "NFLX", "нетфликс": "NFLX",
    "amd": "AMD", "advanced micro": "AMD",
    "intel": "INTC", "интел": "INTC",
    "coinbase": "COIN", "коинбейс": "COIN",
    "zoom": "ZM", "зум": "ZM",
    "uber": "UBER", "убер": "UBER",
    "airbnb": "ABNB", "аирбнб": "ABNB",
    
    # Криптовалюти
    "bitcoin": "BTCUSD", "btc": "BTCUSD", "біткоїн": "BTCUSD", "биткоин": "BTCUSD",
    "ethereum": "ETHUSD", "eth": "ETHUSD", "ефіріум": "ETHUSD", "эфириум": "ETHUSD",
    "cardano": "ADAUSD", "ada": "ADAUSD", "кардано": "ADAUSD",
    "solana": "SOLUSD", "sol": "SOLUSD", "солана": "SOLUSD",
    "dogecoin": "DOGEUSD", "doge": "DOGEUSD", "доге": "DOGEUSD",
    
    # Валюти  
    "eurusd": "EURUSD", "євро": "EURUSD", "euro": "EURUSD",
    "gbpusd": "GBPUSD", "фунт": "GBPUSD", "pound": "GBPUSD",
    "usdjpy": "USDJPY", "йена": "USDJPY", "yen": "USDJPY",
    "usdcad": "USDCAD", "канадський долар": "USDCAD"
}

# Категорії активів
ASSET_CATEGORIES = {
    "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", "COIN", "ZM", "UBER", "ABNB"],
    "crypto": ["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOGEUSD"],
    "forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD"]
}

# Ключові слова для розпізнавання типу запиту  
NON_FINANCIAL_KEYWORDS = [
    "rays", "baseball", "football", "soccer", "game", "sport", "team", "player",
    "weather", "погода", "температура", "дождь", "снег",
    "cat", "dog", "кот", "собака", "домашні тварини",
    "recipe", "рецепт", "готувати", "їжа", "food",
    "movie", "фільм", "кино", "серіал"
]


async def analyze_intent(user_input: str) -> str:
    """
    Визначає намір користувача на основі його запиту
    
    Args:
        user_input: Текст запиту користувача
        
    Returns:
        Тип наміру: analyze_news, get_price, investment_advice, general_chat, fallback
    """
    
    @backoff.on_exception(
        backoff.expo,
        (openai.APIError, openai.RateLimitError, asyncio.TimeoutError),
        max_tries=3
    )
    async def call_openai():
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": INTENT_DETECTION_RULES},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0,
                    max_tokens=512,
                    stream=False
                ),
                timeout=15.0
            )
            return response.choices[0].message.content.strip().lower()
        finally:
            await client.close()
    
    try:
        intent = await call_openai()
        
        # Валідація відповіді
        valid_intents = ["analyze_news", "get_price", "investment_advice", "general_chat", "fallback"]
        if intent in valid_intents:
            return intent
        else:
            return "fallback"
            
    except Exception as e:
        print(f"Помилка детекції інтенту: {e}")
        return "fallback"


def extract_ticker(user_input: str) -> Optional[str]:
    """
    Витягує тикер акції з тексту користувача
    
    Args:
        user_input: Текст запиту користувача
        
    Returns:
        Тикер акції або None якщо не знайдено
    """
    user_input_lower = user_input.lower()
    
    # Спочатку шукаємо в популярних тикерах
    for key, ticker in POPULAR_TICKERS.items():
        if key in user_input_lower:
            return ticker
    
    # Шукаємо паттерни типу AAPL, TSLA (верхній регістр, 3-5 букв)
    ticker_pattern = r'\b[A-Z]{3,5}\b'
    matches = re.findall(ticker_pattern, user_input)
    if matches:
        return matches[0]
    
    return None


async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """
    Отримує поточну ціну акції через Twelve Data API
    
    Args:
        symbol: Тикер акції (наприклад, AAPL)
        
    Returns:
        Словник з інформацією про ціну
    """
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        return {"error": "API ключ Twelve Data не налаштований"}
    
    url = "https://api.twelvedata.com/price"
    params = {
        "symbol": symbol.upper(),
        "apikey": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "price" in data:
                return {
                    "symbol": symbol.upper(),
                    "price": float(data["price"]),
                    "currency": "USD"
                }
            else:
                return {"error": f"Ціна для {symbol} не знайдена. {data.get('message', '')}"}
                
    except httpx.TimeoutException:
        return {"error": "Таймаут при отриманні даних"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP помилка: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Помилка отримання ціни: {str(e)}"}


async def get_crypto_price(symbol: str) -> Dict[str, Any]:
    """
    Отримує поточну ціну криптовалюти через Twelve Data API
    
    Args:
        symbol: Тикер криптовалюти (наприклад, BTC/USD або BTCUSD)
        
    Returns:
        Словник з інформацією про ціну
    """
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        return {"error": "API ключ Twelve Data не налаштований"}
    
    # Конвертуємо формат для Twelve Data API (BTC/USD)
    if symbol.upper().endswith("USD") and len(symbol) > 3:
        crypto_base = symbol.upper()[:-3]
        formatted_symbol = f"{crypto_base}/USD"
    else:
        formatted_symbol = symbol.upper()
    
    url = "https://api.twelvedata.com/price"
    params = {
        "symbol": formatted_symbol,
        "apikey": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "price" in data:
                return {
                    "symbol": formatted_symbol,
                    "price": float(data["price"]),
                    "currency": "USD"
                }
            else:
                return {"error": f"Ціна для {symbol} не знайдена. {data.get('message', '')}"}
                
    except Exception as e:
        return {"error": f"Помилка отримання ціни криптовалюти: {str(e)}"}


# Константа для детекції сутностей
ENTITY_DETECTION_RULES = """Ти експерт з розпізнавання фінансових активів у тексті.

Проаналізуй запит користувача і визнач:

1. Чи йдеться про фінансовий актив? (так/ні)
2. Якщо так - який саме тикер?

Фінансові активи:
- Акції: Apple (AAPL), Tesla (TSLA), Google (GOOGL), Microsoft (MSFT), Amazon (AMZN), Meta (META), Netflix (NFLX), NVIDIA (NVDA), AMD, Intel (INTC), Coinbase (COIN), Zoom (ZM), Uber, Airbnb (ABNB)
- Криптовалюти: Bitcoin (BTCUSD), Ethereum (ETHUSD), Cardano (ADAUSD), Solana (SOLUSD), Dogecoin (DOGEUSD)
- Валютні пари: EUR/USD (EURUSD), GBP/USD (GBPUSD), USD/JPY (USDJPY), USD/CAD (USDCAD)

НЕ фінансові: спорт (rays, baseball), погода, їжа, тварини, фільми.

Поверни результат у JSON:
{
  "is_financial": true/false,
  "entity": "AAPL" / null,
  "entity_type": "stock" / "crypto" / "forex" / null,
  "confidence": 0.9
}

Приклади:
"Ціна Apple" → {"is_financial": true, "entity": "AAPL", "entity_type": "stock", "confidence": 0.95}
"Bitcoin зростає" → {"is_financial": true, "entity": "BTCUSD", "entity_type": "crypto", "confidence": 0.9}
"Rays trade deadline" → {"is_financial": false, "entity": null, "entity_type": null, "confidence": 0.8}
"""

# Константа для планування дій
ACTION_PLANNING_RULES = """Ти плануєш дії для фінансового агента на основі запиту користувача.

Доступні дії:
1. get_price(symbol) - отримати поточну ціну активу
2. get_news_general() - загальні ринкові новини  
3. get_news_targeted(query) - новини по конкретному активу
4. analyze_sentiment(news) - аналіз тональності новин
5. generate_advice(context) - генерувати інвестиційні поради

Проаналізуй запит і поверни план дій у JSON:
{
  "actions": [
    {"action": "get_price", "params": {"symbol": "AAPL"}},
    {"action": "get_news_targeted", "params": {"query": "Apple"}}
  ],
  "reasoning": "Користувач питає про Apple, потрібні ціна та новини"
}

Приклади:
"Ціна Tesla" → [{"action": "get_price", "params": {"symbol": "TSLA"}}]
"Варто купляти Apple?" → [{"action": "get_price", "params": {"symbol": "AAPL"}}, {"action": "get_news_targeted", "params": {"query": "Apple"}}, {"action": "generate_advice", "params": {"context": "AAPL analysis"}}]
"Ринкова ситуація" → [{"action": "get_news_general"}, {"action": "analyze_sentiment"}]
"""


async def detect_entity(user_input: str) -> Dict[str, Any]:
    """
    Розпізнає фінансові сутності у запиті користувача
    
    Args:
        user_input: Текст запиту користувача
        
    Returns:
        Словник з інформацією про сутність
    """
    # Спочатку швидка перевірка через словник
    user_input_lower = user_input.lower()
    
    # Перевіряємо чи є не-фінансові ключові слова
    for keyword in NON_FINANCIAL_KEYWORDS:
        if keyword in user_input_lower:
            return {
                "is_financial": False,
                "entity": None,
                "entity_type": None,
                "confidence": 0.8
            }
    
    # Шукаємо в словнику популярних тикерів
    for key, ticker in POPULAR_TICKERS.items():
        if key in user_input_lower:
            # Визначаємо тип активу
            entity_type = "stock"
            if ticker in ASSET_CATEGORIES["crypto"]:
                entity_type = "crypto"
            elif ticker in ASSET_CATEGORIES["forex"]:
                entity_type = "forex"
            
            return {
                "is_financial": True,
                "entity": ticker,
                "entity_type": entity_type,
                "confidence": 0.9
            }
    
    # Шукаємо тикери у верхньому регістрі (AAPL, TSLA, тощо)
    ticker_pattern = r'\b[A-Z]{3,5}\b'
    matches = re.findall(ticker_pattern, user_input)
    if matches:
        ticker = matches[0]
        entity_type = "stock"  # За замовчуванням
        
        for category, tickers in ASSET_CATEGORIES.items():
            if ticker in tickers:
                entity_type = category.rstrip('s')  # stocks -> stock
                break
        
        return {
            "is_financial": True,
            "entity": ticker,
            "entity_type": entity_type,
            "confidence": 0.7
        }
    
    # Якщо нічого не знайшли через словник, використовуємо LLM
    try:
        @backoff.on_exception(
            backoff.expo,
            (openai.APIError, openai.RateLimitError, asyncio.TimeoutError),
            max_tries=3
        )
        async def call_openai():
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": ENTITY_DETECTION_RULES},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0,
                        max_tokens=512,
                        stream=False
                    ),
                    timeout=15.0
                )
                return response.choices[0].message.content.strip()
            finally:
                await client.close()
        
        result = await call_openai()
        return json.loads(result)
        
    except Exception as e:
        print(f"Помилка детекції сутності: {e}")
        return {
            "is_financial": False,
            "entity": None,
            "entity_type": None,
            "confidence": 0.0
        }


async def plan_actions(user_input: str, entity_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Планує дії агента на основі запиту та розпізнаної сутності
    
    Args:
        user_input: Текст запиту користувача
        entity_info: Інформація про розпізнану сутність
        
    Returns:
        План дій у JSON форматі
    """
    
    # Простий плануваль на основі правил для швидкості
    actions = []
    
    # Аналізуємо тип запиту
    user_lower = user_input.lower()
    
    if entity_info["is_financial"] and entity_info["entity"]:
        entity = entity_info["entity"]
        
        # Якщо питають про ціну
        if any(word in user_lower for word in ["ціна", "скільки", "коштує", "price", "cost"]):
            actions.append({"action": "get_price", "params": {"symbol": entity}})
        
        # Якщо питають про інвестиції/рекомендації
        elif any(word in user_lower for word in ["варто", "купити", "вкладати", "инвестировать", "buy", "invest"]):
            actions.extend([
                {"action": "get_price", "params": {"symbol": entity}},
                {"action": "get_news_targeted", "params": {"query": entity.replace("USD", "")}},
                {"action": "generate_advice", "params": {"context": f"{entity} investment analysis"}}
            ])
        
        # Якщо питають про новини/ситуацію з активом
        elif any(word in user_lower for word in ["новини", "що", "ситуація", "стан", "news", "situation"]):
            actions.extend([
                {"action": "get_price", "params": {"symbol": entity}},
                {"action": "get_news_targeted", "params": {"query": entity.replace("USD", "")}}
            ])
        
        # За замовчуванням - ціна та новини
        else:
            actions.extend([
                {"action": "get_price", "params": {"symbol": entity}},
                {"action": "get_news_targeted", "params": {"query": entity.replace("USD", "")}}
            ])
    
    # Якщо немає конкретного активу
    else:
        if any(word in user_lower for word in ["ринок", "ситуація", "новини", "market", "news", "sentiment"]):
            actions.extend([
                {"action": "get_news_general"},
                {"action": "analyze_sentiment"}
            ])
        
        elif any(word in user_lower for word in ["вкладати", "купити", "инвестировать", "invest", "buy"]):
            actions.extend([
                {"action": "get_news_general"},
                {"action": "analyze_sentiment"},
                {"action": "generate_advice", "params": {"context": "general market analysis"}}
            ])
    
    return {
        "actions": actions,
        "reasoning": f"Запит містить сутність: {entity_info.get('entity', 'загальний')}, тип: {entity_info.get('entity_type', 'загальний')}"
    }


async def get_price(symbol: str) -> Dict[str, Any]:
    """
    Універсальна функція для отримання ціни (акції, крипто або форекс)
    
    Args:
        symbol: Тикер активу
        
    Returns:
        Словник з інформацією про ціну
    """
    # Визначаємо тип активу та API endpoint
    if symbol in ASSET_CATEGORIES["stocks"]:
        return await get_stock_price(symbol)
    elif symbol in ASSET_CATEGORIES["crypto"]:
        return await get_crypto_price(symbol)
    elif symbol in ASSET_CATEGORIES["forex"]:
        return await get_forex_price(symbol)
    else:
        # Спробуємо вгадати тип за форматом
        if "USD" in symbol and len(symbol) <= 7:
            return await get_crypto_price(symbol)
        elif len(symbol) == 6 and symbol.isupper():
            return await get_forex_price(symbol)
        else:
            return await get_stock_price(symbol)


async def get_forex_price(symbol: str) -> Dict[str, Any]:
    """
    Отримує поточну ціну валютної пари через Finage API
    
    Args:
        symbol: Тикер валютної пари (наприклад, EURUSD)
        
    Returns:
        Словник з інформацією про ціну
    """
    api_key = os.getenv("FINAGE_API_KEY")
    if not api_key:
        return {"error": "API ключ Finage не налаштований"}
    
    url = f"https://api.finage.co.uk/last/forex/{symbol}"
    params = {"apikey": api_key}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "price" in data or "ask" in data:
                price = data.get("price") or data.get("ask")
                return {
                    "symbol": symbol,
                    "price": price,
                    "currency": "USD",
                    "time": data.get("time", ""),
                    "bid": data.get("bid"),
                    "ask": data.get("ask")
                }
            else:
                return {"error": f"Ціна для {symbol} не знайдена"}
                
    except Exception as e:
        return {"error": f"Помилка отримання ціни валютної пари: {str(e)}"}


async def get_news_targeted(query: str, limit: int = 5) -> list:
    """
    Отримує новини по конкретному активу
    
    Args:
        query: Пошуковий запит (наприклад, "Apple", "Bitcoin")
        limit: Кількість новин
        
    Returns:
        Список новин
    """
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return []
    
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "q": f"{query} finance stock market investment",
        "category": "business",
        "language": "en",
        "size": str(limit)
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for article in data.get("results", [])[:limit]:
                if article.get("title") and article.get("description"):
                    news_items.append({
                        "title": article["title"],
                        "description": article["description"],
                        "source": "NewsData.io",
                        "published_at": article.get("pubDate", ""),
                        "full_text": f"{article['title']}. {article['description']}"
                    })
            
            return news_items
    except Exception as e:
        print(f"Помилка отримання таргетованих новин: {e}")
        return []


async def execute_action_plan(action_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Виконує план дій та збирає результати
    
    Args:
        action_plan: План дій з функції plan_actions
        
    Returns:
        Результати виконання всіх дій
    """
    results = {"actions": [], "data": {}}
    
    for action in action_plan["actions"]:
        action_type = action["action"]
        params = action.get("params", {})
        
        try:
            if action_type == "get_price":
                symbol = params["symbol"]
                price_data = await get_price(symbol)
                results["data"]["price"] = price_data
                results["actions"].append(f"Отримано ціну для {symbol}")
                
            elif action_type == "get_news_targeted":
                query = params["query"]
                news_data = await get_news_targeted(query)
                results["data"]["targeted_news"] = news_data
                results["actions"].append(f"Отримано новини для {query}")
                
            elif action_type == "get_news_general":
                # Використовуємо існуючу функцію
                from main import get_news
                news_data = await get_news()
                results["data"]["general_news"] = news_data
                results["actions"].append("Отримано загальні новини")
                
            elif action_type == "analyze_sentiment":
                # Аналізуємо тональність отриманих новин
                news_to_analyze = results["data"].get("targeted_news") or results["data"].get("general_news", [])
                if news_to_analyze:
                    news_texts = [item["full_text"] for item in news_to_analyze]
                    sentiment_data = await analyze_multiple_news(news_texts[:3])  # Обмежуємо для швидкості
                    results["data"]["sentiment"] = sentiment_data
                    results["actions"].append("Проаналізовано тональність новин")
                
            elif action_type == "generate_advice":
                # Генеруємо поради на основі зібраних даних
                context = params.get("context", "general")
                advice = await generate_investment_advice(results["data"], context)
                results["data"]["advice"] = advice
                results["actions"].append("Згенеровано інвестиційні поради")
                
        except Exception as e:
            results["actions"].append(f"Помилка виконання {action_type}: {str(e)}")
    
    return results


async def generate_investment_advice(data: Dict[str, Any], context: str) -> str:
    """
    Генерує інвестиційні поради на основі зібраних даних
    
    Args:
        data: Зібрані дані (ціна, новини, тональність)
        context: Контекст для порад
        
    Returns:
        Текст з порадами
    """
    
    # Формуємо контекст для LLM
    context_parts = []
    
    if "price" in data and not data["price"].get("error"):
        price_info = data["price"]
        symbol = price_info.get("symbol", "N/A")
        price = price_info.get("price", "N/A")
        change = price_info.get("change")
        context_parts.append(f"Поточна ціна {symbol}: ${price}")
        if change is not None:
            direction = "зростання" if change >= 0 else "падіння"
            context_parts.append(f"Зміна ціни: {change:.2f} ({direction})")
    
    if "sentiment" in data:
        sentiment_data = data["sentiment"]
        bullish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bullish")
        bearish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bearish")
        total = len(sentiment_data)
        
        if bullish > bearish:
            context_parts.append(f"Новини переважно позитивні ({bullish}/{total})")
        elif bearish > bullish:
            context_parts.append(f"Новини переважно негативні ({bearish}/{total})")
        else:
            context_parts.append(f"Новини збалансовані ({total} новин)")
    
    if "targeted_news" in data or "general_news" in data:
        news = data.get("targeted_news") or data.get("general_news", [])
        if news:
            latest_headlines = [item["title"][:50] + "..." for item in news[:2]]
            context_parts.append(f"Останні новини: {'; '.join(latest_headlines)}")
    
    market_context = "\n".join(context_parts) if context_parts else "Обмежені дані для аналізу"
    
    INVESTMENT_ADVICE_LOCAL = """Ти експерт з інвестицій, який надає поради на основі аналізу новин та цін.

На основі наданих даних:
- Ринкової тональності новин
- Поточних цін активів
- Загальної ситуації на ринку

Дай короткі, практичні поради щодо інвестування.

Структура відповіді:
📊 **Короткий аналіз ситуації**
💡 **Рекомендації:**
⚠️ **Ризики:**

Будь конкретним але обережним. Згадай що це не фінансова консультація."""

    advice_prompt = f"""Контекст ринку:
{market_context}

{INVESTMENT_ADVICE_LOCAL}

Дай конкретні поради для контексту: {context}"""
    
    try:
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": advice_prompt},
                    {"role": "user", "content": "Дай інвестиційні поради на основі наданого контексту"}
                ],
                temperature=0.3,
                max_tokens=4096,
                stream=False
            ),
            timeout=25.0
        )
        
        advice = response.choices[0].message.content.strip()
        await client.close()
        return advice
        
    except Exception as e:
        return f"Не вдалося згенерувати поради: {str(e)}"