"""
Скрипт для швидкого запуску MarketAnalystAgent сервера
"""
import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Перевіряє наявність необхідних змінних середовища"""
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "NEWSDATA_API_KEY", 
        "FINAGE_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == "your_openai_api_key_here":
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Відсутні або некоректні змінні середовища:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Відредагуйте файл .env та додайте правильні API ключі")
        return False
    
    print("✅ Всі змінні середовища налаштовані")
    return True


def main():
    """Головна функція запуску"""
    print("🚀 MarketAnalystAgent - Запуск сервера")
    print("=" * 50)
    
    # Перевіряємо середовище
    if not check_environment():
        sys.exit(1)
    
    # Запускаємо сервер
    print("🌐 Запускаємо FastAPI сервер...")
    print("📍 URL: http://localhost:8000")
    print("📚 Документація: http://localhost:8000/docs")
    print("🛑 Для зупинки натисніть Ctrl+C\n")
    
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Сервер зупинено")
    except ImportError:
        print("❌ Помилка імпорту. Встановіть залежності: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Помилка запуску: {e}")


if __name__ == "__main__":
    main()