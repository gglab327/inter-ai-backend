import base64
import os
import random
import time
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Inter AI Backend")

# Считываем API-ключи из переменной GEMINI_API_KEYS
RAW_KEYS = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

# Актуальная модель согласно требованию Google API
MODELS_TO_TRY = [
    "gemini-3.8-flash",
]


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = None
  text: str | None = None


def get_gemini_response(contents: list) -> str:
  """Отправляет запрос в Gemini API с ротацией ключей и автоматическими повторами при 503."""
  if not API_KEYS:
    raise HTTPException(
        status_code=500, detail="Переменная GEMINI_API_KEYS пуста."
    )

  shuffled_keys = API_KEYS.copy()
  random.shuffle(shuffled_keys)

  errors_log = []

  # 1. Перебор API-ключей
  for key in shuffled_keys:
    try:
      client = genai.Client(api_key=key)

      for model_name in MODELS_TO_TRY:
        # 2. До 3 повторных попыток на случай временной перегрузки (503 / 429)
        for attempt in range(3):
          try:
            response = client.models.generate_content(
                model=model_name, contents=contents
            )
            if response.text:
              return response.text
          except Exception as model_err:
            err_str = str(model_err)
            errors_log.append(f"[{model_name} attempt {attempt+1}]: {err_str}")

            # Если модель перегружена (503 / 429), ждем 1 сек и пробуем снова
            if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
              time.sleep(1)
              continue
            else:
              break

    except Exception as key_err:
      errors_log.append(f"[Key error]: {key_err}")
      continue

  detailed_errors = " | ".join(errors_log)
  raise HTTPException(
      status_code=500,
      detail=f"Ошибка Gemini API. Детали попыток: {detailed_errors}",
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
