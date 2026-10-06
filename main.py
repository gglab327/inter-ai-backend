from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@app.get("/")
async def root_endpoint():
    return {"status": "Сервер работает с поддержкой контекста!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        
        # 1. Если Android передал готовый массив истории сообщений
        raw_messages = data.get("messages")
        
        # 2. Если Android передал одиночный текст (старый формат)
        user_text = data.get("text", "")
        image_base64 = data.get("image_base64") or data.get("imageBase64") or data.get("image")

        formatted_messages = []

        if raw_messages and isinstance(raw_messages, list):
            # Используем готовую историю диалога из приложения
            formatted_messages = raw_messages
        else:
            # Формируем структуру из одного сообщения
            if image_base64:
                content = [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            else:
                content = user_text if user_text else "Привет!"
                
            formatted_messages = [{"role": "user", "content": content}]

        # Формируем тело запроса для Pollinations OpenAI API
        payload = {
            "model": "openai",  # Можно использовать: "openai", "qwen-coder", "p1"
            "messages": formatted_messages
        }

        # Отправляем весь контекст в Pollinations
        response = requests.post(
            "https://text.pollinations.ai/openai",
            json=payload,
            headers=HEADERS,
            timeout=40
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Ошибка Pollinations AI: {response.text}"
            )

        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]
        
        return {"reply": ai_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
