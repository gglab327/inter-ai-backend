from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse

app = FastAPI()

# Отключаем блокировки CORS для Android-приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root_endpoint():
    return {"status": "Новый сервер Inter AI успешно запущен!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # Читаем любые входящие JSON-данные в сыром виде
        data = await request.json()
        user_text = data.get("text", "")
        image_base64 = data.get("image_base64") or data.get("imageBase64") or data.get("image")

        # 1. Логика для МУЛЬТИМОДАЛЬНОГО запроса (если прикреплено фото автомобиля)
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
                "model": "p1",  # Бесплатная мультимодальная модель со зрением
                "messages": [{"role": "user", "content": content_structure}]
            }
            response = requests.post(
                "https://pollinations.ai",
                json=payload,
                timeout=30
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Сбой ИИ Vision")
            result = response.json()
            return {"reply": result["choices"]["message"]["content"]}
            
        # 2. Логика для ОБЫЧНОГО ТЕКСТА (когда отправлен только текст)
        else:
            if not user_text:
                return {"reply": "Привет! Напиши что-нибудь..."}
            
            # ИДЕАЛЬНАЯ СВЯЗКА ЧЕРЕЗ URL-ПАРАМЕТР (разделяем домен и текст кириллицы)
            encoded_prompt = urllib.parse.quote(user_text)
            url = f"https://pollinations.ai{encoded_prompt}&model=search"
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Сбой текстового ИИ")
            return {"reply": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
