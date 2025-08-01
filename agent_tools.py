"""
LangChain Tools для MarketAnalystAgent з повноцінним планування дій
"""
import os
import asyncio
import httpx
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------- STOCK PRICE ----------
class GetStockPriceInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)")

class GetStockPriceTool(BaseTool):
    name: str = "get_stock_price"
    description: str = "Отримує поточну ціну акції через Twelve Data API."
    args_schema: Type[BaseModel] = GetStockPriceInput

    def _run(self, symbol: str) -> str:
        return asyncio.run(self._arun(symbol))

    async def _arun(self, symbol: str) -> str:
        api_key = os.getenv("TWELVE_DATA_API_KEY")
        if not api_key:
            return "❌ API ключ Twelve Data не налаштований"
        
        url = "https://api.twelvedata.com/price"
        params = {
            "symbol": symbol.upper(),
            "apikey": api_key
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                
                if "price" in data:
                    price = float(data["price"])
                    return f"💰 {symbol.upper()}: ${price:.2f}"
                else:
                    return f"❌ Не вдалося знайти ціну для {symbol}. Помилка: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Помилка отримання ціни {symbol}: {str(e)}"


# ---------- CRYPTO ----------
class GetCryptoPriceInput(BaseModel):
    symbol: str = Field(description="Crypto symbol (e.g., BTCUSD, ETHUSD)")

class GetCryptoPriceTool(BaseTool):
    name: str = "get_crypto_price"
    description: str = "Отримує поточну ціну криптовалюти через Twelve Data API."
    args_schema: Type[BaseModel] = GetCryptoPriceInput

    def _run(self, symbol: str) -> str:
        return asyncio.run(self._arun(symbol))

    async def _arun(self, symbol: str) -> str:
        api_key = os.getenv("TWELVE_DATA_API_KEY")
        if not api_key:
            return "❌ API ключ Twelve Data не налаштований"
        
        # Twelve Data використовує формат BTC/USD замість BTCUSD
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
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                
                if "price" in data:
                    price = float(data["price"])
                    return f"💰 {formatted_symbol}: ${price:,.2f}"
                else:
                    return f"❌ Не вдалося знайти ціну для {symbol}. Помилка: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"❌ Помилка отримання ціни {symbol}: {str(e)}"


# ---------- NEWS BY COMPANY ----------
class GetStockNewsInput(BaseModel):
    query: str = Field(description="Назва компанії або тикер")
    limit: int = Field(default=3, description="Кількість новин")

class GetStockNewsTool(BaseTool):
    name: str = "get_stock_news"
    description: str = "Повертає новини про конкретну компанію через NewsData.io"
    args_schema: Type[BaseModel] = GetStockNewsInput

    def _run(self, query: str, limit: int = 3) -> str:
        return asyncio.run(self._arun(query, limit))

    async def _arun(self, query: str, limit: int = 3) -> str:
        api_key = os.getenv("NEWSDATA_API_KEY")
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": api_key,
            "q": query,
            "category": "business",
            "language": "en",
            "size": str(min(limit, 5))
        }

        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                articles = data.get("results", [])
                if not articles:
                    return f"📰 Немає новин про {query}"
                return "\n".join([f"{i+1}. {a['title']}" for i, a in enumerate(articles)])
        except Exception as e:
            return f"❌ Помилка: {str(e)}"


# ---------- MARKET SUMMARY ----------
class GetMarketSummaryInput(BaseModel):
    category: str = Field(default="business")

class GetMarketSummaryTool(BaseTool):
    name: str = "get_market_summary"
    description: str = "Огляд загальної ситуації на ринку"
    args_schema: Type[BaseModel] = GetMarketSummaryInput

    def _run(self, category: str = "business") -> str:
        return asyncio.run(self._arun(category))

    async def _arun(self, category: str = "business") -> str:
        api_key = os.getenv("NEWSDATA_API_KEY")
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": api_key,
            "category": category,
            "q": "market",
            "language": "en",
            "size": "5"
        }

        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                articles = data.get("results", [])
                if not articles:
                    return "📊 Немає актуальних ринкових новин"
                return "\n".join([f"{i+1}. {a['title']}" for i, a in enumerate(articles)])
        except Exception as e:
            return f"❌ Помилка: {str(e)}"


# ---------- SENTIMENT ----------
class AnalyzeSentimentInput(BaseModel):
    text: str = Field(description="Текст для аналізу")

class AnalyzeSentimentTool(BaseTool):
    name: str = "analyze_sentiment"
    description: str = "Аналізує тональність тексту: bullish / bearish / neutral"
    args_schema: Type[BaseModel] = AnalyzeSentimentInput

    def _run(self, text: str) -> str:
        return asyncio.run(self._arun(text))

    async def _arun(self, text: str) -> str:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Проаналізуй тональність тексту для ринку. Визнач: bullish (позитивно), bearish (негативно), neutral (нейтрально). Дай коротке пояснення (1-2 речення)."},
                    {"role": "user", "content": text[:1000]}
                ],
                temperature=0,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            await client.close()
            return f"📈 **Тональність:** {result}"
        except Exception as e:
            return f"❌ Помилка аналізу: {str(e)}"


# ---------- REGISTER TOOLS ----------
AVAILABLE_TOOLS = [
    GetStockPriceTool(),
    GetCryptoPriceTool(),
    GetStockNewsTool(),
    GetMarketSummaryTool(),
    AnalyzeSentimentTool()
]
