"""
Інструменти для аналізу ринкової тональності
"""
import asyncio
import openai
import backoff
from typing import Dict, Any
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
                    max_tokens=300,
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