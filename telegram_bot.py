import os
import asyncio
import logging
import time
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import httpx

load_dotenv()

# Token із .env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = "http://localhost:8000/run"

# Ініціалізація бота
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Логування
logging.basicConfig(level=logging.INFO)

# Кеш для відслідковування типів запитів (для аналітики)
request_stats = {"total": 0, "by_intent": {}}

# Емодзі для різних типів відповідей
INTENT_EMOJIS = {
    "analyze_news": "📊",
    "get_price": "💰", 
    "investment_advice": "📈",
    "general_chat": "💬",
    "fallback": "❓"
}


# Команди бота
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обробляє команду /start"""
    welcome_text = """🤖 **Вітаю! Я MarketAnalyst Agent**

Можу допомогти з:
📊 Аналізом ринкових новин
💰 Цінами акцій та криптовалют
📈 Інвестиційними порадами

**Приклади запитів:**
• "Що там на ринку?"
• "Ціна Apple"
• "Куди вкладати гроші?"
• "Скільки коштує Bitcoin?"

Просто напишіть ваш запит!

/help - допомога
/stats - статистика"""
    
    await message.answer(welcome_text, parse_mode="Markdown")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Обробляє команду /help"""
    help_text = """ℹ️ **Довідка MarketAnalyst Agent**

**Типи запитів:**
📊 **Аналіз новин:** "ринок", "новини", "ситуація"
💰 **Ціни:** "ціна Apple", "скільки TSLA", "Bitcoin"
📈 **Поради:** "куди вкладати", "що купити", "інвестиції"
💬 **Спілкування:** "привіт", "дякую", "як справи"

**Підтримувані активи:**
• Акції: AAPL, TSLA, GOOGL, MSFT, AMZN, META...
• Криптовалюти: Bitcoin, Ethereum...
• Можна писати як тикерами, так і назвами

**Команди:**
/start - почати роботу
/help - ця довідка
/stats - статистика запитів"""
    
    await message.answer(help_text, parse_mode="Markdown")


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    """Показує статистику використання"""
    if request_stats["total"] == 0:
        await message.answer("📊 Поки що запитів не було.")
        return
    
    stats_text = f"📊 **Статистика запитів:**\n\n"
    stats_text += f"Загальна кількість: {request_stats['total']}\n\n"
    
    if request_stats["by_intent"]:
        stats_text += "**По типах:**\n"
        for intent, count in request_stats["by_intent"].items():
            emoji = INTENT_EMOJIS.get(intent, "❓")
            percentage = (count / request_stats["total"]) * 100
            stats_text += f"{emoji} {intent}: {count} ({percentage:.1f}%)\n"
    
    await message.answer(stats_text, parse_mode="Markdown")


@dp.message()
async def handle_message(message: types.Message):
    """Обробляє звичайні повідомлення користувача"""
    user_input = message.text.strip()
    start_time = time.time()
    
    # Відправляємо повідомлення про початок обробки
    processing_msg = await message.answer("🔍 Аналізую запит...")

    payload = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["message"]["content"]

        # Видаляємо повідомлення про обробку
        await processing_msg.delete()
        
        # Оновлюємо статистику
        request_stats["total"] += 1
        
        # Розбиваємо відповідь, якщо вона надто довга
        max_length = 4096
        if len(content) <= max_length:
            await message.answer(content, parse_mode="Markdown")
        else:
            # Розбиваємо на частини
            parts = []
            current_part = ""
            
            for line in content.split('\n'):
                if len(current_part + line + '\n') > max_length:
                    if current_part:
                        parts.append(current_part)
                        current_part = line + '\n'
                    else:
                        # Якщо один рядок довший за ліміт
                        parts.append(line[:max_length])
                        current_part = line[max_length:] + '\n'
                else:
                    current_part += line + '\n'
            
            if current_part:
                parts.append(current_part)
            
            for i, part in enumerate(parts):
                if i == 0:
                    await message.answer(part, parse_mode="Markdown")
                else:
                    await message.answer(f"📄 **Частина {i+1}:**\n\n{part}", parse_mode="Markdown")

        # Логування часу обробки
        processing_time = time.time() - start_time
        logging.info(f"✅ Запит оброблено за {processing_time:.2f}с: {user_input[:50]}...")

    except httpx.TimeoutException:
        await processing_msg.edit_text("⏰ Запит займає більше часу, ніж очікувалося. Спробуйте пізніше.")
        logging.warning("⏰ Таймаут запиту")
        
    except httpx.HTTPStatusError as e:
        await processing_msg.edit_text("❌ Сервер тимчасово недоступний. Спробуйте пізніше.")
        logging.error(f"❌ HTTP помилка: {e.response.status_code}")
        
    except Exception as e:
        await processing_msg.edit_text("❌ Виникла помилка при обробці запиту.")
        logging.exception("❌ Помилка обробки повідомлення:")
        logging.error(f"Запит: {user_input}")
        logging.error(f"Помилка: {str(e)}")


async def main():
    print("🚀 Telegram-бот запущено. Очікування повідомлень...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
