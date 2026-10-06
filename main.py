import base64
import os
import random
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Inter AI Backend")

# Считываем ключи из переменной окружения GEMINI_API_KEYS (через запятую)
RAW_KEYS = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = None


def get_gemini_response(contents: list) -> str:
  """Отправляет запрос в Gemini API с перебором ключей при ошибках."""
  if not API_KEYS:
    raise HTTPException(
        status_code=500,
        detail=(
            "Переменная GEMINI_API_KEYS не задана или не содержит API-ключей."
        ),
    )

  # Перемешиваем список ключей для равномерной нагрузки
  shuffled_keys = API_KEYS.copy()
  random.shuffle(shuffled_keys)

  last_error = None
  for key in shuffled_keys:
    try:
      client = genai.Client(api_key=key)
      # Используется указанная модель gemini-3.8-flash
      response = client.models.generate_content(
          model="gemini-3.8-flash", contents=contents
      )
      if response.text:
        return response.text
    except Exception as e:
      last_error = e
      continue

  # Если ни один из ключей не сработал
  raise HTTPException(
      status_code=500, detail=f"Все API-ключи завершились ошибкой: {last_error}"
  )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
  try:
    contents = []

    # 1. Если передано изображение в формате Base64
    if request.image_base64 and request.image_base64.strip():
      clean_b64 = request.image_base64.split(",")[-1]
      try:
        image_bytes = base64.b64decode(clean_b64)
      except Exception:
        raise HTTPException(
            status_code=400, detail="Ошибка декодирования Base64"
        )

      image_part = types.Part.from_bytes(
          data=image_bytes, mime_type="image/jpeg"
      )
      contents.append(image_part)

    # 2. Текстовый промпт
    prompt_text = (
        request.prompt
        if request.prompt
        else "Опиши подробно, что изображено на этом снимке."
    )
    contents.append(prompt_text)

    # 3. Вызов модели с ротацией ключей
    reply_text = get_gemini_response(contents)

    return {"reply": reply_text}

  except HTTPException as http_ex:
    raise http_ex
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
  return {"status": "ok", "total_keys_loaded": len(API_KEYS)}
