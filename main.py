import base64
import os
import random
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Inter AI Backend")

# Считываем список API-ключей из переменной окружения GEMINI_API_KEYS (через запятую)
RAW_KEYS = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

# Список стабильных моделей в порядке приоритета (для обхода ошибки 503)
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = None
  text: str | None = None  # Поддержка поля "text" из запроса приложения


def get_gemini_response(contents: list) -> str:
  """Запрос к Gemini с ротацией ключей и автоматическим переключением моделей при перегрузке (503)."""
  if not API_KEYS:
    raise HTTPException(
        status_code=500,
        detail=(
            "Переменная GEMINI_API_KEYS не задана или не содержит"
            " API-ключей."
        ),
    )

  # Перемешиваем ключи для равномерной нагрузки
  shuffled_keys = API_KEYS.copy()
  random.shuffle(shuffled_keys)

  last_error = None

  # 1. Перебираем ключи
  for key in shuffled_keys:
    try:
      client = genai.Client(api_key=key)

      # 2. Если модель перегружена (503), переключаемся на резервную модель
      for model_name in MODELS_TO_TRY:
        try:
          response = client.models.generate_content(
              model=model_name, contents=contents
          )
          if response.text:
            return response.text
        except Exception as model_err:
          last_error = model_err
          # Переходим к следующей модели при ошибках 503 / 404
          continue

    except Exception as key_err:
      last_error = key_err
      continue

  # Если ни один ключ и ни одна модель не сработали
  raise HTTPException(
      status_code=500,
      detail=(
          "Не удалось получить ответ от Gemini. Все ключи и модели"
          f" завершились ошибкой: {last_error}"
      ),
  )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
  try:
    contents = []

    # 1. Если передана картинка в формате Base64
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

    # 2. Извлекаем текст (поддерживаются и "prompt", и "text")
    user_prompt = request.prompt or request.text
    if not user_prompt:
      user_prompt = (
          "Опиши подробно, что изображено на этом снимке."
          if request.image_base64
          else "Привет!"
      )

    contents.append(user_prompt)

    # 3. Вызов функции генерации
    reply_text = get_gemini_response(contents)

    return {"reply": reply_text}

  except HTTPException as http_ex:
    raise http_ex
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
  return {"status": "ok", "total_keys_loaded": len(API_KEYS)}
