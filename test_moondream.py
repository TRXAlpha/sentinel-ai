import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ESP_IP = "10.62.241.20"
IMG_URL = f"http://{ESP_IP}/jpg"

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

# Try GPU first. If it crashes with CUDA out of memory, change this to "cpu".
USE_GPU = torch.cuda.is_available()

print("[moondream] CUDA:", torch.cuda.is_available())
print("[moondream] Loading model...")

if USE_GPU:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        device_map={"": "cuda"},
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
)

print("[moondream] Capturing image from ESP32...")
r = requests.get(IMG_URL, timeout=2)
r.raise_for_status()

image = Image.open(BytesIO(r.content)).convert("RGB")

print("[moondream] Caption:")
print(model.caption(image, length="short")["caption"])

question = (
    "You are controlling a small desktop sentinel camera. "
    "Choose the most interesting visible thing to track. "
    "Prefer a person, face, hand, phone, laptop, keyboard, mouse, bottle, cup, or tool. "
    "Answer with one short object name only."
)

print("[moondream] Query:")
answer = model.query(image, question)["answer"]
print(answer)