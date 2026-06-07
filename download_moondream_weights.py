import requests
from io import BytesIO
from PIL import Image
import moondream as md

ESP_IP = "10.62.241.20"
IMG_URL = f"http://{ESP_IP}/jpg"

MODEL_PATH = "weights/moondream-0_5b-int8.mf.gz"

print("[moondream-0.5b] loading...")
model = md.vl(model=MODEL_PATH)

print("[moondream-0.5b] getting image...")
r = requests.get(IMG_URL, timeout=2)
r.raise_for_status()
image = Image.open(BytesIO(r.content)).convert("RGB")

print("[moondream-0.5b] caption:")
print(model.caption(image, length="short"))

print("[moondream-0.5b] query:")
answer = model.query(
    image,
    "Choose the most interesting visible thing to track. Answer with one short object name only."
)
print(answer)