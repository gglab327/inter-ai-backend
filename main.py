import base64
import os
import random
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Inter AI Backend")

# Считываем API-ключи из переменной GEMINI_API_KEYS
RAW_KEYS = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

# Актуальные модели Gemini в порядке приоритета
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
]


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = None
  text: str | None = None


def get_gemini_response(contents: list) -> str:
  """Отправляет запрос в Gemini API с перебором ключей и моделей."""
  if not API_KEYS:
    raise HTTPException(
        status_code=500, detail="Переменная GEMINI_API_KEYS пуста."
    )

  shuffled_keys = API_KEYS.copy()
  random.shuffle(shuffled_keys)

  errors_log = []

  # 1. Перебираем ключи
  for key in shuffled_keys:
    try:
      client = genai.Client(api_key=key)

      # 2. Перебираем поддерживаемые модели
      for model_name in MODELS_TO_TRY:
        try:
          response = client.models.generate_content(
              model=model_name, contents=contents
          )
          if response.text:
            return response.text
        except Exception as model_err:
          errors_log.append(f"[{model_name}]: {str(model_err)}")
          continue

    except Exception as key_err:
      errors_log.append(f"[Key error]: {str(key_err)}")
      continue

  # Если ни одна модель не сработала
  last_err_detail = errors_log[-1] if errors_log else "Неизвестная ошибка"
  raise HTTPException(
      status_code=500,
      detail=f"Ошибка Gemini API. Последняя ошибка: {last_err_detail}",
  )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
  try:
    contents = []

    # 1. Обработка изображения Base64
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
    user_prompt = request.prompt or request.text
    if not user_prompt:
      user_prompt = (
          "Опиши подробно, что изображено на этом снимке."
          if request.image_base64
          else "Привет!"
      )

    contents.append(user_prompt)

    # 3. Вызов генерации
    reply_text = get_gemini_response(contents)

    return {"reply": reply_text}

  except HTTPException as http_ex:
    raise http_ex
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
  return {"status": "ok", "total_keys_loaded": len(API_KEYS)}
