from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import traceback

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

def clean_base64(b64_str: str) -> str:
    """Очищает base64 от спецсимволов и повторных префиксов"""
    if not b64_str:
        return ""
    b64_str = b64_str.strip().replace("\n", "").replace("\r", "")
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[1]
    return b64_str

@app.get("/")
async def root_endpoint():
    return {"status": "Сервер работает!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        
        raw_messages = data.get("messages")
        user_text = data.get("text", "")
        image_base64 = data.get("image_base64") or data.get("imageBase64") or data.get("image")
        
        cleaned_b64 = clean_base64(image_base64) if image_base64 else None
        formatted_messages = []

        # Если передана история сообщений
        if raw_messages and isinstance(raw_messages, list) and len(raw_messages) > 0:
            formatted_messages = [dict(m) for m in raw_messages]
            
            # Если прикреплено фото, объединяем его с последним сообщением пользователя
            if cleaned_b64:
                last_msg = formatted_messages[-1]
                last_text = last_msg.get("content", "")
                
                # Если content был обычной строкой, переводим его в массив мультимодального формата
                if isinstance(last_text, str):
                    last_msg["content"] = [
                        {"type": "text", "text": last_text if last_text else "Что на этой картинке?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{cleaned_b64}"}
                        }
                    ]
        else:
            # Одиночный запрос без истории
            if cleaned_b64:
                content = [
                    {"type": "text", "text": user_text if user_text else "Что на этой картинке?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{cleaned_b64}"}
                    }
                ]
            else:
                content = user_text if user_text else "Привет!"
                
            formatted_messages = [{"role": "user", "content": content}]

        payload = {
            "model": "openai-large",  # Модель с отличным распознаванием изображений
            "messages": formatted_messages
        }

        # Отправляем запрос в Pollinations AI
        response = requests.post(
            "https://text.pollinations.ai/openai",
            json=payload,
            headers=HEADERS,
            timeout=60
        )

        # Если модель openai-large не ответила, пробуем стандартную модель openai
        if response.status_code != 200:
            payload["model"] = "openai"
            response = requests.post(
                "https://text.pollinations.ai/openai",
                json=payload,
                headers=HEADERS,
                timeout=60
            )

        if response.status_code != 200:
            print(f"Pollinations Error {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=500, 
                detail=f"Ошибка ИИ-сервиса ({response.status_code}): {response.text[:150]}"
            )

        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]
        
        return {"reply": ai_reply}

    except HTTPException as he:
        raise he
    except Exception as e:
        print("Ошибка сервера:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")
