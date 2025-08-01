"""
Приклади використання MarketAnalystAgent API з новим інтелектуальним роутингом
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


async def test_intent_detection():
    """Тестує різні типи запитів для перевірки інтент-детекції"""
    test_queries = [
        # Аналіз новин
        "Що там на ринку?",
        "Проаналізуй новини",
        "Ринкова ситуація",
        
        # Ціни
        "Скільки коштує Apple?",
        "Ціна TSLA",
        "Bitcoin ціна",
        
        # Інвестиційні поради
        "Куди вкладати гроші?",
        "Що краще купити?",
        "Інвестиційні поради",
        
        # Загальне спілкування
        "Привіт",
        "Дякую",
        "Як справи?",
        
        # Fallback
        "Що робити з котом?",
        "Погода сьогодні",
    ]
    
    print("\n🧠 Тестуємо інтент-детекцію...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Запит: '{query}'")
        
        request_data = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["message"]["content"]
                
                # Показуємо тільки початок відповіді
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"   Відповідь: {preview}")
                
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
        
        # Невелика затримка між запитами
        await asyncio.sleep(1)


async def test_price_requests():
    """Тестує запити на отримання цін"""
    price_queries = [
        "Ціна Apple",
        "Скільки коштує GOOGL?", 
        "Tesla акція",
        "Bitcoin ціна",
        "Ethereum",
        "MSFT"
    ]
    
    print("\n💰 Тестуємо запити цін...")
    
    for query in price_queries:
        print(f"\nЗапит: '{query}'")
        
        request_data = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                print(f"Відповідь: {result['message']['content']}")
                
        except Exception as e:
            print(f"❌ Помилка: {e}")


async def test_investment_advice():
    """Тестує інвестиційні поради"""
    print("\n📈 Тестуємо інвестиційні поради...")
    
    advice_queries = [
        "Куди вкладати гроші цього місяця?",
        "Що краще купити зараз?",
        "Дай інвестиційні поради"
    ]
    
    for query in advice_queries:
        print(f"\nЗапит: '{query}'")
        
        request_data = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["message"]["content"]
                
                # Показуємо повну відповідь для порад
                print("Відповідь:")
                print("-" * 40)
                print(content)
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Помилка: {e}")
        
        break  # Тестуємо тільки один запит, бо це довго


async def test_entity_recognition():
    """Тестує розпізнавання сутностей та планування дій"""
    test_cases = [
        # Правильне розпізнавання
        {"query": "Ціна Apple", "expected": "AAPL"},
        {"query": "Варто купляти Tesla?", "expected": "TSLA"},
        {"query": "Bitcoin новини", "expected": "BTCUSD"},
        {"query": "MSFT акції", "expected": "MSFT"},
        {"query": "Що з євро?", "expected": "EURUSD"},
        
        # Неправильне розпізнавання (edge cases)
        {"query": "Rays trade deadline", "expected": "не фінансовий"},
        {"query": "Apple pie рецепт", "expected": "не фінансовий"},
        {"query": "Tesla машина купити", "expected": "TSLA або не фінансовий"},
    ]
    
    print("\n🧠 Тестуємо розпізнавання сутностей...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. '{case['query']}'")
        print(f"   Очікується: {case['expected']}")
        
        request_data = {
            "messages": [
                {"role": "user", "content": case['query']}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["message"]["content"]
                
                # Аналізуємо відповідь
                print(f"   Відповідь: {content[:80]}...")
                
                # Перевіряємо чи правильно розпізнав
                if case["expected"] == "не фінансовий":
                    if "не стосується фінансів" in content or "спеціалізуюся на" in content:
                        print("   ✅ Правильно відхилив не-фінансовий запит")
                    else:
                        print("   ❌ Не розпізнав не-фінансовий запит")
                else:
                    if any(symbol in content for symbol in ["$", "USD", "₴", "💰"]):
                        print("   ✅ Розпізнав як фінансовий запит")
                    else:
                        print("   ⚠️ Не впевнений в розпізнаванні")
                
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
        
        await asyncio.sleep(0.5)


async def test_edge_cases():
    """Тестує edge cases та non-financial запити"""
    edge_cases = [
        "Rays trade deadline baseball",
        "Apple pie recipe cooking",
        "Tesla car price (не акція)",
        "Dog training tips",
        "Weather forecast today",
        "Movie recommendations",
        "Як налаштувати Wi-Fi?",
        "Bitcoin майнінг дома",  # межовий випадок
    ]
    
    print("\n🚨 Тестуємо edge cases...")
    
    for query in edge_cases:
        print(f"\nТест: '{query}'")
        
        request_data = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["message"]["content"]
                
                # Аналізуємо відповідь
                if "не стосується фінансів" in content or "спеціалізуюся на" in content:
                    print("✅ Правильно відхилив")
                elif any(symbol in content for symbol in ["$", "💰", "📈", "📉"]):
                    print("⚠️ Потенційно розпізнав як фінансовий")
                else:
                    print("❓ Неоднозначна відповідь")
                
                print(f"Відповідь: {content[:100]}...")
                
        except Exception as e:
            print(f"❌ Помилка: {e}")
        
        await asyncio.sleep(0.5)


async def test_langchain_agent():
    """Тестує нового LangChain агента з планування дій"""
    test_scenarios = [
        {
            "query": "Варто купляти акції Apple зараз?",
            "expected_actions": ["get_stock_price", "get_stock_news", "analyze_sentiment"],
            "description": "Повний інвестиційний аналіз"
        },
        {
            "query": "Скільки коштує Tesla?",
            "expected_actions": ["get_stock_price"],
            "description": "Простий запит ціни"
        },
        {
            "query": "Що там на ринку сьогодні?",
            "expected_actions": ["get_market_summary"],
            "description": "Загальний огляд ринку"
        },
        {
            "query": "Bitcoin впав чи зріс?",
            "expected_actions": ["get_crypto_price"],
            "description": "Криптовалютна ціна"
        }
    ]
    
    print("\n🤖 Тестуємо LangChain агента з планування дій...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Сценарій: {scenario['description']}")
        print(f"   Запит: '{scenario['query']}'")
        print(f"   Очікувані дії: {', '.join(scenario['expected_actions'])}")
        
        request_data = {
            "messages": [
                {"role": "user", "content": scenario['query']}
            ]
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    "http://localhost:8000/run",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["message"]["content"]
                
                end_time = asyncio.get_event_loop().time()
                execution_time = end_time - start_time
                
                print(f"   ⏱️ Час виконання: {execution_time:.2f}с")
                print(f"   📝 Відповідь ({len(content)} символів):")
                print(f"   {content[:200]}...")
                
                # Аналіз якості відповіді
                quality_indicators = []
                if "$" in content:
                    quality_indicators.append("💰 Ціна")
                if any(word in content.lower() for word in ["новини", "news"]):
                    quality_indicators.append("📰 Новини")
                if any(word in content.lower() for word in ["рекомендую", "радить", "варто", "не варто"]):
                    quality_indicators.append("💡 Поради")
                if "не фінансова консультація" in content.lower() or "дисклеймер" in content.lower():
                    quality_indicators.append("⚠️ Дисклеймер")
                
                if quality_indicators:
                    print(f"   ✅ Якість: {', '.join(quality_indicators)}")
                else:
                    print(f"   ⚠️ Базова відповідь без специфічних індикаторів")
                
        except asyncio.TimeoutError:
            print(f"   ⏰ Таймаут - агент працює довше 90 секунд")
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
        
        # Затримка між тестами
        await asyncio.sleep(2)


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
    print("🧪 Запуск тестів MarketAnalystAgent API v4.0 (LangChain AgentExecutor)")
    print("=" * 70)
    
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
    
    # Меню тестування
    print("Виберіть тест для запуску:")
    print("1. 🔍 Базова інформація про API")
    print("2. 🤖 Тест LangChain агента (НОВИЙ!)")
    print("3. 📰 Тест отримання новин")  
    print("4. 🧠 Тест розпізнавання сутностей")
    print("5. 💰 Тест запитів цін (контекстні)")
    print("6. 📈 Тест інвестиційних порад (повний цикл)")
    print("7. 🚨 Тест edge cases (спорт, не-фінанси)")
    print("8. 🚀 Повний набір тестів")
    print("0. Вихід")
    
    try:
        choice = input("\nВведіть номер тесту (0-8): ").strip()
        
        if choice == "1":
            await test_api_info()
        elif choice == "2":
            await test_langchain_agent()
        elif choice == "3":
            await test_news_endpoint()
        elif choice == "4":
            await test_entity_recognition()
        elif choice == "5":
            await test_price_requests()
        elif choice == "6":
            await test_investment_advice()
        elif choice == "7":
            await test_edge_cases()
        elif choice == "8":
            print("\n🚀 Запуск повного набору тестів...")
            await test_api_info()
            await test_langchain_agent()
            await test_news_endpoint()
            await test_entity_recognition()
            await test_price_requests()
            await test_investment_advice()
            await test_edge_cases()
        elif choice == "0":
            print("👋 До побачення!")
            return
        else:
            print("❌ Невірний вибір")
            return
            
        print("\n🎉 Тестування завершено!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Тестування перервано користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")


if __name__ == "__main__":
    asyncio.run(main())