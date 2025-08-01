"""
Приклади використання MarketAnalystAgent API
"""
import asyncio
import httpx
import json


async def test_news_endpoint():
    """Тестує endpoint отримання новин"""
    print("🔍 Тестуємо отримання новин...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/news")
            response.raise_for_status()
            
            news = response.json()
            print(f"✅ Отримано {len(news)} новин:")
            
            for i, item in enumerate(news, 1):
                print(f"{i}. {item['title'][:60]}... ({item['source']})")
                
        except Exception as e:
            print(f"❌ Помилка: {e}")


async def test_run_endpoint():
    """Тестує основний endpoint аналізу"""
    print("\n🚀 Тестуємо аналіз ринку...")
    
    request_data = {
        "messages": [
            {"role": "user", "content": "Проаналізуй останні новини ринку"}
        ]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/run",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            print("✅ Аналіз завершено:")
            print("-" * 50)
            print(result["message"]["content"])
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Помилка: {e}")


async def test_api_info():
    """Тестує базовий endpoint з інформацією"""
    print("\n📋 Отримуємо інформацію про API...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/")
            response.raise_for_status()
            
            info = response.json()
            print("✅ Інформація про API:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ Помилка: {e}")


async def main():
    """Головна функція тестування"""
    print("🧪 Запуск тестів MarketAnalystAgent API")
    print("=" * 60)
    
    # Перевіряємо доступність сервера
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/")
            if response.status_code == 200:
                print("✅ Сервер доступний\n")
            else:
                print("❌ Сервер недоступний")
                return
    except:
        print("❌ Не вдалося підключитися до сервера")
        print("💡 Переконайтеся що сервер запущено: python main.py")
        return
    
    # Запускаємо тести
    await test_api_info()
    await test_news_endpoint()
    await test_run_endpoint()
    
    print("\n🎉 Тестування завершено!")


if __name__ == "__main__":
    asyncio.run(main())