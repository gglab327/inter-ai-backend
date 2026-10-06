import base64
import os
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()

# Клиент автоматически берет ключ из переменной окружения GEMINI_API_KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = "Что изображено на этом фото?"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
  try:
    contents = []

    # 1. Если передана картинка в Base64 — декодируем её в байты
    if request.image_base64:
      # Очищаем от возможных заголовков data:image/jpeg;base64,
      clean_b64 = request.image_base64.split(",")[-1]
      image_bytes = base64.b64decode(clean_b64)

      # Формируем объект изображения для Gemini
      image_part = types.Part.from_bytes(
          data=image_bytes, mime_type="image/jpeg"
      )
      contents.append(image_part)

    # 2. Добавляем текстовый промпт
    prompt_text = request.prompt or "Опиши подробно, что ты видишь на снимке."
    contents.append(prompt_text)

    # 3. Отправляем запрос в модель Gemini 1.5 Flash
    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=contents
    )

    # Как проверить успешность: вернуть статус 200 и текст ответа ИИ
    return {"reply": response.text}

  except Exception as e:
    # Безопасная перехватка ошибок — сервер не вылетит с 500 без объяснений
    raise HTTPException(status_code=500, detail=f"Ошибка Gemini API: {str(e)}")
