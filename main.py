"""
MarketAnalystAgent - GenAI агент для аналізу ринкової тональності
"""
import asyncio
import json
import os
from typing import List, Dict, Any
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from tools import (
    analyze_multiple_news, analyze_intent, extract_ticker, get_price,
    detect_entity, plan_actions, execute_action_plan, get_news_targeted
)
from agent_tools import AVAILABLE_TOOLS
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

app = FastAPI(
    title="Market Analyst Agent",
    description="GenAI агент для аналізу ринкової тональності на основі новин",
    version="1.0.0"
)

# Pydantic моделі для API
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    message: Message

class NewsItem(BaseModel):
    title: str
    description: str
    source: str
    published_at: str

# Кеш для новин (простий в пам'яті кеш)
news_cache = {"data": [], "timestamp": None}
CACHE_DURATION = 300  # 5 хвилин

# Налаштування LangChain агента
AGENT_SYSTEM_PROMPT = """Ти експертний фінансовий аналітик з доступом до реальних даних через API.

🎯 **Твоя роль:**
- Надавай точні, основані на даних поради щодо інвестицій
- Завжди використовуй актуальні дані перед відповіддю
- Будь конкретним та практичним у рекомендаціях

🔧 **Доступні інструменти:**
- get_stock_price: ціни акцій (AAPL, TSLA, GOOGL, MSFT, AMZN, META, NVDA, тощо)
- get_crypto_price: ціни криптовалют (BTCUSD, ETHUSD, тощо)
- get_stock_news: новини про конкретні компанії
- get_market_summary: загальна ринкова ситуація
- analyze_sentiment: аналіз тональності новин

🧠 **Алгоритм роботи:**
1. **Для інвестиційних запитів** ("варто купляти X?"):
   - Отримай поточну ціну активу
   - Знайди останні новини про нього
   - Проаналізуй тональність новин
   - Дай конкретну рекомендацію з обґрунтуванням

2. **Для цінових запитів** ("скільки коштує X?"):
   - Просто отримай та поверни поточну ціну

3. **Для ринкових запитів** ("що на ринку?"):
   - Отримай загальний огляд ринку
   - Проаналізуй тональність

⚠️ **Важливо:**
- Завжди згадуй що це не фінансова консультація
- Базуй поради тільки на отриманих даних
- Будь чесним щодо обмежень та ризиків

🚫 **НЕ відповідай на не-фінансові запити** (спорт, погода, рецепти, тощо)"""

# Створення LangChain агента
def create_financial_agent():
    """Створює LangChain агента з фінансовими інструментами"""
    
    # Налаштування LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Створення промпту
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Створення агента
    agent = create_openai_functions_agent(llm, AVAILABLE_TOOLS, prompt)
    
    # Створення виконавця агента
    agent_executor = AgentExecutor(
        agent=agent,
        tools=AVAILABLE_TOOLS,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        return_intermediate_steps=False
    )
    
    return agent_executor

# Глобальний агент (створюється один раз)
financial_agent = None

# Константи для відповідей агента
INVESTMENT_ADVICE_RULES = """Ти експерт з інвестицій, який надає поради на основі аналізу новин та цін.

На основі наданих даних:
- Ринкової тональності новин
- Поточних цін активів
- Загальної ситуації на ринку

Дай короткі, практичні поради щодо інвестування.

Структура відповіді:
📊 **Короткий аналіз ситуації**
💡 **Рекомендації:**
⚠️ **Ризики:**

Будь конкретним але обережним. Згадай що це не фінансова консультація.
"""

GENERAL_CHAT_RESPONSES = {
    "привіт": "Вітаю! Я MarketAnalyst Agent. Можу допомогти з аналізом ринку, цінами акцій або інвестиційними порадами. Що вас цікавить?",
    "дякую": "Будь ласка! Завжди радий допомогти з фінансовими питаннями 😊",
    "як справи": "У мене все добре! Слідкую за ринками та готовий надати актуальну аналітику. Що бажаєте дізнатися?",
    "допомога": "Я можу:\n🔍 Аналізувати ринкові новини\n💰 Показувати ціни акцій\n📈 Давати інвестиційні поради\n\nПросто напишіть що вас цікавить!",
}


async def fetch_newsdata_io_news() -> List[Dict]:
    """Отримує новини з NewsData.io API"""
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return []
    
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "category": "business,top",
        "language": "en",
        "size": "3"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            for article in data.get("results", [])[:3]:
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
        print(f"Помилка при отриманні новин з NewsData.io: {e}")
        return []


async def fetch_finage_news() -> List[Dict]:
    """Отримує новини з Finage API"""
    api_key = os.getenv("FINAGE_API_KEY")
    if not api_key:
        return []
    
    url = "https://api.finage.co.uk/news/forex"
    params = {
        "apikey": api_key,
        "limit": "3"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_items = []
            articles = data if isinstance(data, list) else data.get("news", [])
            
            for article in articles[:3]:
                if article.get("title"):
                    description = article.get("description", "") or article.get("summary", "")
                    news_items.append({
                        "title": article["title"],
                        "description": description,
                        "source": "Finage",
                        "published_at": article.get("date", ""),
                        "full_text": f"{article['title']}. {description}"
                    })
            return news_items
    except Exception as e:
        print(f"Помилка при отриманні новин з Finage: {e}")
        return []


@app.get("/news", response_model=List[NewsItem])
async def get_news():
    """
    Отримує останні 3 економічні новини з різних джерел
    """
    global news_cache
    
    # Перевіряємо кеш
    current_time = datetime.now().timestamp()
    if (news_cache["timestamp"] and 
        current_time - news_cache["timestamp"] < CACHE_DURATION and 
        news_cache["data"]):
        return news_cache["data"]
    
    try:
        # Отримуємо новини з обох джерел паралельно
        newsdata_task = fetch_newsdata_io_news()
        finage_task = fetch_finage_news()
        
        newsdata_results, finage_results = await asyncio.gather(
            newsdata_task, finage_task, return_exceptions=True
        )
        
        # Обробляємо результати
        all_news = []
        
        if not isinstance(newsdata_results, Exception):
            all_news.extend(newsdata_results)
        
        if not isinstance(finage_results, Exception):
            all_news.extend(finage_results)
        
        # Обмежуємо до 6 новин і сортуємо
        all_news = all_news[:6]
        
        # Оновлюємо кеш
        news_cache = {
            "data": all_news,
            "timestamp": current_time
        }
        
        return all_news
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка отримання новин: {str(e)}")


async def get_financial_agent():
    """Отримує або створює фінансового агента"""
    global financial_agent
    if financial_agent is None:
        print("🤖 Створюю LangChain агента...")
        financial_agent = create_financial_agent()
    return financial_agent


def truncate_response(response: str, max_length: int = 3500) -> str:
    """Обрізає відповідь до безпечної довжини для Telegram"""
    if len(response) <= max_length:
        return response
    
    # Знаходимо останній повний рядок
    truncated = response[:max_length]
    last_newline = truncated.rfind('\n')
    
    if last_newline > max_length * 0.7:  # Якщо є хороше місце для обрізання
        truncated = truncated[:last_newline]
    
    return truncated + "\n\n💬 [Відповідь обрізана для Telegram]"


@app.post("/run", response_model=ChatResponse)
async def run_agent(request: ChatRequest):
    """
    MarketAnalyst Agent v4.0 - LangChain AgentExecutor з повним планування дій
    """
    try:
        # Отримуємо текст запиту користувача
        user_input = request.messages[-1].content if request.messages else ""
        
        if not user_input:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content="Будь ласка, напишіть ваш запит."
                )
            )
        
        print(f"🔍 Обробляю запит через LangChain агента: {user_input}")
        
        # Отримуємо LangChain агента
        agent = await get_financial_agent()
        
        # Виконуємо запит через агента
        try:
            # LangChain AgentExecutor автоматично планує та виконує дії
            result = await asyncio.wait_for(
                asyncio.to_thread(agent.invoke, {"input": user_input}),
                timeout=60.0
            )
            
            response_content = result.get("output", "Не вдалося отримати відповідь від агента.")
            print(f"✅ Агент відповів: {response_content[:100]}...")
            
        except asyncio.TimeoutError:
            response_content = "⏰ Запит займає більше часу, ніж очікувалося. Спробуйте пізніше або запитайте щось простіше."
            
        except Exception as e:
            print(f"❌ Помилка виконання агента: {e}")
            response_content = "Вибачте, сталася помилка при аналізі. Спробуйте перефразувати запит."
        
        # Обрізаємо надто довгі відповіді для Telegram сумісності
        if len(response_content) > 3500:
            response_content = truncate_response(response_content)
        
        return ChatResponse(
            message=Message(
                role="assistant",
                content=response_content
            )
        )
        
    except Exception as e:
        print(f"❌ Критична помилка в run_agent: {e}")
        return ChatResponse(
            message=Message(
                role="assistant",
                content="Вибачте, сталася технічна помилка. Спробуйте ще раз."
            )
        )


async def format_intelligent_response(
    user_input: str, 
    entity_info: Dict[str, Any], 
    execution_results: Dict[str, Any]
) -> str:
    """
    Форматує природну відповідь на основі виконаних дій та зібраних даних
    
    Args:
        user_input: Оригінальний запит користувача
        entity_info: Інформація про розпізнану сутність
        execution_results: Результати виконання дій
        
    Returns:
        Форматована відповідь
    """
    
    data = execution_results.get("data", {})
    user_lower = user_input.lower()
    
    # Якщо є конкретний актив
    if entity_info["is_financial"] and entity_info["entity"]:
        entity = entity_info["entity"]
        entity_name = get_entity_display_name(entity)
        
        # Запит про ціну
        if any(word in user_lower for word in ["ціна", "скільки", "коштує", "price", "cost"]):
            return format_price_response(entity_name, data.get("price"))
            
        # Запит про інвестиції
        elif any(word in user_lower for word in ["варто", "купити", "вкладати", "інвестувати", "buy", "invest"]):
            return await format_investment_response(entity_name, data)
            
        # Запит про новини/ситуацію
        elif any(word in user_lower for word in ["новини", "що", "ситуація", "стан", "news", "situation"]):
            return format_news_response(entity_name, data)
            
        # Загальний запит про актив
        else:
            return format_general_asset_response(entity_name, data)
    
    # Загальні ринкові запити
    else:
        if any(word in user_lower for word in ["ринок", "ситуація", "новини", "market", "news"]):
            return format_market_overview_response(data)
        elif any(word in user_lower for word in ["вкладати", "купити", "інвестувати", "invest", "buy"]):
            return format_general_investment_response(data)
    
    # За замовчуванням
    return "Обробив ваш запит, але не знайшов специфічних даних для відповіді."


def get_entity_display_name(entity: str) -> str:
    """Повертає читабельну назву активу"""
    display_names = {
        "AAPL": "Apple",
        "TSLA": "Tesla", 
        "GOOGL": "Google",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "META": "Meta",
        "NVDA": "NVIDIA",
        "BTCUSD": "Bitcoin",
        "ETHUSD": "Ethereum",
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD"
    }
    return display_names.get(entity, entity)


def format_price_response(entity_name: str, price_data: Dict[str, Any]) -> str:
    """Форматує відповідь про ціну"""
    if not price_data or price_data.get("error"):
        return f"❌ Не вдалося отримати ціну для {entity_name}. {price_data.get('error', '')}"
    
    symbol = price_data.get("symbol", entity_name)
    price = price_data.get("price")
    change = price_data.get("change")
    change_percent = price_data.get("change_percent")
    
    if price is None:
        return f"❌ Ціна для {entity_name} недоступна"
    
    response_parts = [f"💰 **{entity_name} ({symbol}): ${price:.2f}**"]
    
    if change is not None and change_percent is not None:
        change_emoji = "📈" if change >= 0 else "📉"
        sign = "+" if change >= 0 else ""
        response_parts.append(f"{change_emoji} {sign}{change:.2f} ({sign}{change_percent:.2f}%)")
    
    response_parts.append("⏰ Оновлено: щойно")
    
    return "\n".join(response_parts)


async def format_investment_response(entity_name: str, data: Dict[str, Any]) -> str:
    """Форматує відповідь з інвестиційними порадами"""
    response_parts = [f"📊 **Аналіз {entity_name}**\n"]
    
    # Ціна
    if "price" in data and not data["price"].get("error"):
        price_info = data["price"]
        price = price_info.get("price")
        change = price_info.get("change")
        
        if price:
            response_parts.append(f"💰 Поточна ціна: ${price:.2f}")
            if change is not None:
                trend = "зростання" if change >= 0 else "падіння"
                response_parts.append(f"📈 Динаміка: {trend} ({change:+.2f})")
    
    # Новини та тональність
    if "sentiment" in data:
        sentiment_data = data["sentiment"]
        bullish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bullish")
        bearish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bearish")
        
        if bullish > bearish:
            response_parts.append("📈 Новини переважно позитивні")
        elif bearish > bullish:
            response_parts.append("📉 Новини переважно негативні")
        else:
            response_parts.append("⚖️ Новини збалансовані")
    
    # Поради
    if "advice" in data:
        response_parts.append("\n" + data["advice"])
    
    return "\n".join(response_parts)


def format_news_response(entity_name: str, data: Dict[str, Any]) -> str:
    """Форматує відповідь з новинами про актив"""
    response_parts = [f"📰 **Новини про {entity_name}**\n"]
    
    # Ціна для контексту
    if "price" in data and not data["price"].get("error"):
        price = data["price"].get("price")
        if price:
            response_parts.append(f"💰 Поточна ціна: ${price:.2f}")
    
    # Новини
    news = data.get("targeted_news", [])
    if news:
        response_parts.append(f"\n🔍 **Останні новини ({len(news)}):**")
        for i, item in enumerate(news[:3], 1):
            title = item["title"][:70] + "..." if len(item["title"]) > 70 else item["title"]
            response_parts.append(f"{i}. {title}")
    else:
        response_parts.append("ℹ️ Актуальні новини не знайдені")
    
    return "\n".join(response_parts)


def format_general_asset_response(entity_name: str, data: Dict[str, Any]) -> str:
    """Форматує загальну відповідь про актив"""
    response_parts = [f"📋 **{entity_name}**\n"]
    
    # Ціна
    if "price" in data and not data["price"].get("error"):
        price_info = data["price"]
        price = price_info.get("price")
        change = price_info.get("change")
        
        if price:
            response_parts.append(f"💰 Ціна: ${price:.2f}")
            if change is not None:
                emoji = "📈" if change >= 0 else "📉"
                response_parts.append(f"{emoji} Зміна: {change:+.2f}")
    
    # Короткі новини
    news = data.get("targeted_news", [])
    if news:
        latest_news = news[0]["title"][:80] + "..." if len(news[0]["title"]) > 80 else news[0]["title"]
        response_parts.append(f"📰 Останні новини: {latest_news}")
    
    return "\n".join(response_parts)


def format_market_overview_response(data: Dict[str, Any]) -> str:
    """Форматує загальний огляд ринку"""
    response_parts = ["📊 **Ринкова ситуація**\n"]
    
    if "sentiment" in data:
        sentiment_data = data["sentiment"]
        bullish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bullish")
        bearish = sum(1 for s in sentiment_data if s["sentiment"].lower() == "bearish")
        total = len(sentiment_data)
        
        if bullish > bearish:
            response_parts.append(f"📈 **Позитивний настрій** ({bullish}/{total} новин)")
            response_parts.append("Ринок схильний до зростання")
        elif bearish > bullish:
            response_parts.append(f"📉 **Негативний настрій** ({bearish}/{total} новин)")
            response_parts.append("Ринок під тиском")
        else:
            response_parts.append(f"⚖️ **Нейтральний настрій** ({total} новин)")
            response_parts.append("Ринок в очікуванні")
        
        # Топ новини
        news = data.get("general_news", [])
        if news:
            response_parts.append(f"\n🔍 **Ключові новини:**")
            for i, item in enumerate(news[:3], 1):
                title = item["title"][:60] + "..." if len(item["title"]) > 60 else item["title"]
                response_parts.append(f"{i}. {title}")
    
    return "\n".join(response_parts)


def format_general_investment_response(data: Dict[str, Any]) -> str:
    """Форматує загальні інвестиційні поради"""
    if "advice" in data:
        return data["advice"]
    else:
        return """📊 **Загальні інвестиційні поради**

💡 **Базові принципи:**
• Диверсифікуйте портфель між різними активами
• Інвестуйте довгостроково (від 1 року)
• Не вкладайте більше 5-10% в один актив

⚠️ **Ризики:**
• Ринки волатильні та непередбачувані
• Минулі результати не гарантують майбутніх

🔍 **Рекомендація:** Проаналізуйте конкретні активи для точніших порад.

**Дисклеймер:** Це не фінансова консультація."""


def handle_fallback_response(user_input: str, entity_info: Dict[str, Any]) -> str:
    """Обробляє випадки, коли не вдалося зрозуміти запит"""
    if not entity_info["is_financial"]:
        return """Вибачте, здається ваш запит не стосується фінансів. 

Я спеціалізуюся на:
🔍 Аналізі ринкових новин
💰 Цінах акцій та криптовалют  
📈 Інвестиційних порадах

Спробуйте запитати щось на кшталт:
• "Ціна Apple"
• "Варто купляти Tesla?"
• "Що там на ринку?"
"""
    else:
        entity = entity_info.get("entity", "активу")
        return f"""Розумію, що ви питаєте про {entity}, але не зміг точно визначити, що саме вас цікавить.

Спробуйте уточнити:
💰 "Ціна {entity}"
📊 "Новини про {entity}"  
📈 "Варто купляти {entity}?"

Або задайте більш конкретне питання."""


# Залишаю старі функції для сумісності, але вони тепер не використовуються
async def handle_news_analysis() -> str:
    """Обробляє запити на аналіз новин"""
    try:
        news_items = await get_news()
        
        if not news_items:
            return "На жаль, не вдалося отримати актуальні новини для аналізу."
        
        # Аналізуємо тональність новин
        news_texts = [item["full_text"] for item in news_items]
        sentiment_results = await analyze_multiple_news(news_texts)
        
        # Формуємо компактний звіт
        bullish_count = sum(1 for s in sentiment_results if s["sentiment"].lower() == "bullish")
        bearish_count = sum(1 for s in sentiment_results if s["sentiment"].lower() == "bearish")
        neutral_count = len(sentiment_results) - bullish_count - bearish_count
        
        report_parts = ["📊 **Ринкова тональність:**\n"]
        
        if bullish_count > bearish_count:
            report_parts.append(f"📈 **Позитивний настрій** ({bullish_count} з {len(sentiment_results)} новин)")
            report_parts.append("Ринок схильний до зростання.")
        elif bearish_count > bullish_count:
            report_parts.append(f"📉 **Негативний настрій** ({bearish_count} з {len(sentiment_results)} новин)")  
            report_parts.append("Ринок під тиском.")
        else:
            report_parts.append(f"⚖️ **Нейтральний настрій** (збалансовано)")
            report_parts.append("Ринок в очікуванні.")
        
        # Додаємо найважливіші новини
        report_parts.append(f"\n🔍 **Ключові новини:**")
        for i, (news_item, sentiment) in enumerate(zip(news_items[:3], sentiment_results[:3])):
            emoji = "📈" if sentiment["sentiment"].lower() == "bullish" else "📉" if sentiment["sentiment"].lower() == "bearish" else "⚖️"
            title = news_item["title"][:60] + "..." if len(news_item["title"]) > 60 else news_item["title"]
            report_parts.append(f"{emoji} {title}")
        
        return "\n".join(report_parts)
        
    except Exception as e:
        return f"Помилка аналізу новин: {str(e)}"


async def handle_price_request(user_input: str) -> str:
    """Обробляє запити на отримання цін"""
    try:
        # Витягуємо тикер з запиту
        ticker = extract_ticker(user_input)
        
        if not ticker:
            return """Не зміг визначити тикер акції. 
            
Спробуйте вказати конкретну назву:
• "Ціна Apple" або "AAPL"
• "Скільки Tesla?" або "TSLA" 
• "Bitcoin ціна" або "BTC"

Підтримую популярні акції та криптовалюти."""
        
        # Отримуємо ціну
        price_data = await get_price(ticker)
        
        if "error" in price_data:
            return f"❌ {price_data['error']}\n\nПеревірте правильність тикера або спробуйте пізніше."
        
        # Форматуємо відповідь
        price = price_data.get("price", "N/A")
        change = price_data.get("change")
        change_percent = price_data.get("change_percent")
        
        response_parts = [f"💰 **{ticker}: ${price:.2f}**"]
        
        if change is not None and change_percent is not None:
            change_emoji = "📈" if change >= 0 else "📉"
            sign = "+" if change >= 0 else ""
            response_parts.append(f"{change_emoji} {sign}{change:.2f} ({sign}{change_percent:.2f}%)")
        
        response_parts.append(f"⏰ Оновлено: щойно")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Помилка отримання ціни: {str(e)}"


async def handle_investment_advice() -> str:
    """Обробляє запити на інвестиційні поради"""
    try:
        # Отримуємо дані для аналізу
        news_items = await get_news()
        
        if not news_items:
            return """📊 **Інвестиційні поради**

Зараз немає актуальних новин для аналізу, але ось загальні рекомендації:

💡 **Базові принципи:**
• Диверсифікуйте портфель
• Інвестуйте довгостроково
• Не вкладайте всі кошти в одне місце

⚠️ **Пам'ятайте:** Це не фінансова консультація. Завжди консультуйтеся з професіоналами."""
        
        # Швидкий аналіз тональності
        news_texts = [item["full_text"] for item in news_items[:3]]
        sentiment_results = await analyze_multiple_news(news_texts)
        
        # Аналізуємо настрій ринку
        bullish_count = sum(1 for s in sentiment_results if s["sentiment"].lower() == "bullish")
        bearish_count = sum(1 for s in sentiment_results if s["sentiment"].lower() == "bearish")
        
        # Формуємо поради
        context_data = {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "total_news": len(sentiment_results),
            "top_news": [item["title"] for item in news_items[:2]]
        }
        
        advice_prompt = f"""Контекст ринку:
- Позитивних новин: {bullish_count}
- Негативних новин: {bearish_count}
- Загалом новин: {len(sentiment_results)}
- Ключові теми: {', '.join(context_data['top_news'])}

{INVESTMENT_ADVICE_RULES}"""
        
        # Генеруємо поради через LLM
        import openai
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": advice_prompt},
                        {"role": "user", "content": "Дай інвестиційні поради на основі поточної ситуації"}
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                    stream=False
                ),
                timeout=20.0
            )
            
            advice = response.choices[0].message.content.strip()
            advice += "\n\n⚠️ **Дисклеймер:** Це не фінансова консультація. Завжди консультуйтеся з професіоналами."
            
            await client.close()
            return advice
            
        except Exception as e:
            await client.close()
            return f"Помилка генерації порад: {str(e)}"
        
    except Exception as e:
        return f"Помилка аналізу для інвестиційних порад: {str(e)}"


def handle_general_chat(user_input: str) -> str:
    """Обробляє загальне спілкування"""
    user_input_lower = user_input.lower()
    
    # Шукаємо відповідність у заготовлених відповідях
    for key, response in GENERAL_CHAT_RESPONSES.items():
        if key in user_input_lower:
            return response
    
    # За замовчуванням
    return """Привіт! Я MarketAnalyst Agent 🤖

Можу допомогти з:
🔍 Аналізом ринкових новин
💰 Цінами акцій та криптовалют  
📈 Інвестиційними порадами

Просто напишіть що вас цікавить!"""


@app.get("/")
async def root():
    """Базовий endpoint з інформацією про API"""
    return {
        "name": "Market Analyst Agent",
        "version": "1.0.0",
        "description": "GenAI агент для аналізу ринкової тональності",
        "endpoints": {
            "POST /run": "Запустити аналіз ринку",
            "GET /news": "Отримати останні новини",
            "GET /docs": "API документація"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)