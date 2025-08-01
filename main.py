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

from tools import analyze_multiple_news

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


@app.post("/run", response_model=ChatResponse)
async def run_agent(request: ChatRequest):
    """
    Основний endpoint агента - аналізує ринкові новини та повертає звіт
    """
    try:
        # Отримуємо останні новини
        news_items = await get_news()
        
        if not news_items:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content="На жаль, не вдалося отримати актуальні новини для аналізу."
                )
            )
        
        # Аналізуємо тональність кожної новини
        news_texts = [item["full_text"] for item in news_items]
        sentiment_results = await analyze_multiple_news(news_texts)
        
        # Формуємо звіт
        bullish_news = []
        bearish_news = []
        neutral_news = []
        
        for i, (news_item, sentiment) in enumerate(zip(news_items, sentiment_results)):
            news_summary = {
                "title": news_item["title"][:80] + "..." if len(news_item["title"]) > 80 else news_item["title"],
                "source": news_item["source"],
                "explanation": sentiment["explanation"],
                "confidence": sentiment.get("confidence", 0.0)
            }
            
            sentiment_type = sentiment["sentiment"].lower()
            if sentiment_type == "bullish":
                bullish_news.append(news_summary)
            elif sentiment_type == "bearish":
                bearish_news.append(news_summary)
            else:
                neutral_news.append(news_summary)
        
        # Створюємо фінальний звіт
        report_parts = ["🔍 **Аналіз ринкової тональності**\n"]
        
        if bullish_news:
            report_parts.append(f"📈 **BULLISH сигнали ({len(bullish_news)}):**")
            for news in bullish_news:
                report_parts.append(f"• {news['title']} ({news['source']})")
                report_parts.append(f"  └ {news['explanation']}\n")
        
        if bearish_news:
            report_parts.append(f"📉 **BEARISH сигнали ({len(bearish_news)}):**")
            for news in bearish_news:
                report_parts.append(f"• {news['title']} ({news['source']})")
                report_parts.append(f"  └ {news['explanation']}\n")
        
        if neutral_news:
            report_parts.append(f"⚖️ **NEUTRAL сигнали ({len(neutral_news)}):**")
            for news in neutral_news:
                report_parts.append(f"• {news['title']} ({news['source']})")
                report_parts.append(f"  └ {news['explanation']}\n")
        
        # Загальний висновок
        total_news = len(news_items)
        report_parts.append("📊 **Загальний висновок:**")
        
        if len(bullish_news) > len(bearish_news):
            report_parts.append("Переважають позитивні сигнали для ринку.")
        elif len(bearish_news) > len(bullish_news):
            report_parts.append("Переважають негативні сигнали для ринку.")
        else:
            report_parts.append("Ринкові сигнали збалансовані.")
        
        final_report = "\n".join(report_parts)
        
        return ChatResponse(
            message=Message(
                role="assistant",
                content=final_report
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки запиту: {str(e)}")


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