import asyncio
import base64
import os
import random
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Inter AI Backend")

RAW_KEYS = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

# Единственная актуальная и поддерживаемая модель
MODEL_NAME = "gemini-3.8-flash"


class ChatRequest(BaseModel):
  image_base64: str | None = None
  prompt: str | None = None
  text: str | None = None


def _sync_generate(client, model, contents):
  return client.models.generate_content(model=model, contents=contents)


async def get_gemini_response(contents: list) -> str:
  if not API_KEYS:
    raise HTTPException(
        status_code=500, detail="Переменная GEMINI_API_KEYS пуста."
    )

  shuffled_keys = API_KEYS.copy()
  random.shuffle(shuffled_keys)
  errors_log = []

  # 1. Перебираем доступные API-ключи
  for key in shuffled_keys:
    try:
      client = genai.Client(api_key=key)

      # 2. Делаем до 5 повторных попыток с паузой при 503 / 429
      for attempt in range(1, 6):
        try:
          response = await asyncio.to_thread(
              _sync_generate, client, MODEL_NAME, contents
          )
          if response and response.text:
            return response.text
        except Exception as model_err:
          err_str = str(model_err)
          errors_log.append(f"[Attempt {attempt}]: {err_str}")

          # Если сервер перегружен (503 / 429), ждем перед повторной попыткой
          if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
            wait_time = attempt * 1.5  # Задержка 1.5s, 3.0s, 4.5s...
            await asyncio.sleep(wait_time)
            continue
          else:
            break

    except Exception as key_err:
      errors_log.append(f"[Key error]: {key_err}")
      continue

  detailed_errors = " | ".join(errors_log)
  raise HTTPException(
      status_code=500,
      detail=(
          f"Модель {MODEL_NAME} временно перегружена. Детали попыток:"
          f" {detailed_errors}"
      ),
  )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
  try:
    contents = []

    # Обработка изображения Base64
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

    # Извлечение промпта
    user_prompt = request.prompt or request.text
    if not user_prompt:
      user_prompt = (
          "Опиши подробно, что изображено на этом снимке."
          if request.image_base64
          else "Привет!"
      )

    contents.append(user_prompt)

    reply_text = await get_gemini_response(contents)
    return {"reply": reply_text}

  except HTTPException as http_ex:
    raise http_ex
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
  return {"status": "ok", "total_keys_loaded": len(API_KEYS)}
