from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse

app = FastAPI()

# Отключаем блокировки CORS, чтобы Android-приложение беспрепятственно получало ответы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root_endpoint():
    return {"status": "Новый сервер Inter AI успешно запущен и работает!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # Читаем любые входящие JSON-данные от смартфона в сыром виде
        data = await request.json()
        
        # Безопасно извлекаем текст сообщения (если поля нет, подставим пустую строку)
        user_text = data.get("text", "")
        
        # Ищем картинку в любых возможных вариациях имени поля (Android передает image_base64)
        image_base64 = data.get("image_base64") or data.get("imageBase64") or data.get("image")

        # 1. Логика для МУЛЬТИМОДАЛЬНОГО запроса (если пользователь прикрепил фотографию)
        if image_base64:
            content_structure = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
            
            payload = {
                "model": "p1",  # Бесплатная мультимодальная модель со зрением на Pollinations
                "messages": [
                    {
                        "role": "user",
                        "content": content_structure
                    }
                ]
            }
            
            # Отправляем POST-запрос на продвинутый шлюз ИИ
            response = requests.post(
                "https://pollinations.ai",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Сбой ИИ при анализе фотографии")
                
            result = response.json()
            ai_reply = result["choices"][0]["message"]["content"]
            return {"reply": ai_reply}
            
        # 2. Логика для ОБЫЧНОГО ТЕКСТА (когда картинки нет)
        else:
            if not user_text:
                return {"reply": "Привет! Напиши что-нибудь..."}
                
            # Безопасно кодируем русский текст для URL-запроса (чтобы пробелы не ломали ссылку)
            encoded_prompt = urllib.parse.quote(user_text)
            
            # ЖЕЛЕЗОБЕТОННЫЙ URL: домен строго отделен от кириллицы с помощью /?prompt=
            url = f"https://pollinations.ai{encoded_prompt}&model=search"
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Сбой текстового ИИ шлюза")
                
            # Возвращаем JSON-объект, который ожидает получить ваше Android-приложение
            return {"reply": response.text}

    except Exception as e:
        # Если произойдет внутренний сбой, сервер вернет текст ошибки прямо в Logcat смартфона
        raise HTTPException(status_code=500, detail=str(e))
