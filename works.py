import time
import math
import threading
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import requests
import torch
from PIL import Image
from ultralytics import YOLO
import moondream as md

# ============================================================
# CONFIG
# ============================================================

ESP_IP = "YOUR_ESP32_IP"
ESP_BASE = f"http://{ESP_IP}"
ESP_STREAM = f"http://{ESP_IP}:81/stream"

YOLO_MODEL = "yolo11n.pt"

# Default target until Moondream chooses something.
DEFAULT_TARGET_CLASS = "person"

# Moondream 0.5B lightweight local runtime.
# Download with:
# python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='vikhyatk/moondream2', filename='moondream-0_5b-int8.mf.gz', revision='onnx', local_dir='weights'))"
ENABLE_MOONDREAM = True
BASE_DIR = Path(__file__).resolve().parent
MOONDREAM_MODEL_PATH = str(BASE_DIR / "weights" / "moondream-0_5b-int8.mf.gz")

# How often Moondream rethinks the target.
# Keep this fairly slow so it stays a curiosity layer, not a servo-control menace.
MOONDREAM_INTERVAL = 8.0

# Servo limits
PAN_MIN = 30
PAN_MAX = 150
PAN_CENTER = 90

# ============================================================
# CONTROL TUNING FOR ~3 FPS ESP32-CAM
# ============================================================

pan_angle = PAN_CENTER

GAIN = 14.0
DEADZONE = 0.08
MAX_STEP = 2.5
COMMAND_INTERVAL = 0.25

# If the servo moves the wrong direction, flip this between -1 and +1.
SERVO_DIRECTION = -1

# Detection
CONF_MIN = 0.35
LOST_TIMEOUT = 1.5

# Display
SHOW_WINDOW = True

# Camera reader
BAD_FRAME_SLEEP = 0.03
STREAM_RECONNECT_AFTER_SEC = 3.0

# Moondream 0.5B is chatty, so force/parse into this tiny command vocabulary.
ALLOWED_TARGETS = [
    "person",
    "cell phone",
    "laptop",
    "keyboard",
    "mouse",
    "bottle",
    "cup",
    "book",
    "chair",
    "backpack",
    "tv",
]


# ============================================================
# TORCH / YOLO DEVICE
# ============================================================

YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("[system] Torch:", torch.__version__)
print("[system] CUDA available:", torch.cuda.is_available())
print("[system] YOLO device:", YOLO_DEVICE)

if torch.cuda.is_available():
    print("[system] GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TYPES
# ============================================================

@dataclass
class TargetBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str


# ============================================================
# CAMERA READER
# ============================================================

class LatestFrameReader:
    """
    Reads ESP32-CAM MJPEG stream in the background.

    Every genuinely new frame increments frame_id.
    The YOLO loop only processes each frame_id once.

    It also reconnects if frames stop arriving. FFmpeg may print warnings like
    "overread 5" on imperfect MJPEG chunks; that warning is usually not fatal.
    """

    def __init__(self, url: str):
        self.url = url
        self.cap = None

        self.frame = None
        self.frame_id = 0
        self.ok = False
        self.last_good_frame_time = 0.0

        self.running = True
        self.lock = threading.Lock()

        self._open_capture()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _open_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open ESP32-CAM stream: {self.url}")

        self.last_good_frame_time = time.time()
        print("[camera] stream opened")

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            now = time.time()

            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.ok = True
                    self.last_good_frame_time = now
            else:
                time.sleep(BAD_FRAME_SLEEP)

                # If the MJPEG stream silently wedges, reconnect.
                if now - self.last_good_frame_time > STREAM_RECONNECT_AFTER_SEC:
                    print("[camera] no frames recently, reconnecting stream...")
                    try:
                        self._open_capture()
                    except Exception as e:
                        print("[camera] reconnect failed:", repr(e))
                        time.sleep(1.0)

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.frame_id

            return self.ok, self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ============================================================
# ESP32 HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def send_get(path, params=None, timeout=0.25):
    try:
        return requests.get(f"{ESP_BASE}{path}", params=params, timeout=timeout)
    except requests.RequestException:
        return None


def send_pan(angle):
    angle = int(clamp(angle, PAN_MIN, PAN_MAX))
    r = send_get("/pan", params={"angle": angle}, timeout=0.25)
    return r is not None and r.ok


def center_servo():
    send_get("/center", timeout=0.5)


scan_is_on = None

def set_scan(on: bool):
    """
    Only sends scan command when scan state changes.
    No HTTP spam. Civilization improves by millimeters.
    """
    global scan_is_on

    if scan_is_on == on:
        return

    params = {"on": 1 if on else 0}

    if on:
        params.update({
            "min": 40,
            "max": 140,
            "step": 2,
            "delay": 100,
        })

    r = send_get("/scan", params=params, timeout=0.4)

    if r is not None and r.ok:
        scan_is_on = on
        print(f"[scan] {'ON' if on else 'OFF'}")
    else:
        print("[scan] failed to set scan mode")


# ============================================================
# YOLO TARGETING
# ============================================================

def get_best_target(results, class_name: str) -> Optional[TargetBox]:
    """
    Returns best matching YOLO box for class_name.
    """
    if not results:
        return None

    r = results[0]
    names = r.names

    if r.boxes is None:
        return None

    best = None
    best_score = -1.0

    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        label = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0].item())

        if label != class_name:
            continue

        if conf < CONF_MIN:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = max(1.0, (x2 - x1) * (y2 - y1))

        # Prefer confident, larger targets.
        score = conf * math.sqrt(area)

        if score > best_score:
            best_score = score
            best = TargetBox(x1, y1, x2, y2, conf, label)

    return best


# ============================================================
# MOONDREAM
# ============================================================

def cv2_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def normalize_moondream_answer(answer: str) -> str:
    """
    Moondream 0.5B often answers like a tiny essay goblin.
    This crushes its prose into one allowed YOLO class.
    """
    text = answer.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Phrase aliases first, because "cell phone" should survive before "phone".
    aliases = [
        ("cell phone", "cell phone"),
        ("mobile phone", "cell phone"),
        ("smartphone", "cell phone"),
        ("phone", "cell phone"),

        ("water bottle", "bottle"),
        ("bottle", "bottle"),
        ("mug", "cup"),
        ("cup", "cup"),

        ("keyboard", "keyboard"),
        ("mouse", "mouse"),
        ("laptop", "laptop"),
        ("computer", "laptop"),
        ("monitor", "tv"),
        ("screen", "tv"),
        ("tv", "tv"),

        ("backpack", "backpack"),
        ("bag", "backpack"),
        ("book", "book"),
        ("chair", "chair"),

        # Humans. Put these after objects, then add a priority rule below.
        ("person", "person"),
        ("human", "person"),
        ("face", "person"),
        ("head", "person"),
        ("hand", "person"),
        ("man", "person"),
        ("woman", "person"),
        ("boy", "person"),
        ("girl", "person"),
        ("body", "person"),

        # COCO YOLO has no generic screwdriver/tool class in the default model.
        ("tool", "person"),
        ("screwdriver", "person"),
    ]

    found = []
    for phrase, yolo_class in aliases:
        if phrase in text:
            found.append(yolo_class)

    if not found:
        return DEFAULT_TARGET_CLASS

    # If a human is present in the answer, prefer person. The sentinel is supposed
    # to care about people unless you deliberately swap priorities later.
    if "person" in found:
        return "person"

    for cls in ALLOWED_TARGETS:
        if cls in found:
            return cls

    return DEFAULT_TARGET_CLASS


def map_moondream_to_yolo(answer: str) -> str:
    return normalize_moondream_answer(answer)


class MoondreamBrain:
    """
    Background curiosity worker using Moondream 0.5B .mf/.mf.gz.

    It receives the newest frame from the YOLO loop.
    Every MOONDREAM_INTERVAL seconds, it asks:
    "what should the sentinel track?"

    It does NOT block YOLO tracking.
    """

    def __init__(self):
        self.enabled = ENABLE_MOONDREAM
        self.running = False

        self.lock = threading.Lock()
        self.latest_frame = None

        self.target_class = DEFAULT_TARGET_CLASS
        self.raw_answer = "default"
        self.last_query_time = 0.0
        self.busy = False

        self.model = None

        if self.enabled:
            self._load_model()
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def _load_model(self):
        import os

        print(f"[moondream-0.5b] Loading local model: {MOONDREAM_MODEL_PATH}")

        if not os.path.exists(MOONDREAM_MODEL_PATH):
            raise FileNotFoundError(
                f"Moondream model file not found: {MOONDREAM_MODEL_PATH}"
            )

        size_mb = os.path.getsize(MOONDREAM_MODEL_PATH) / 1024 / 1024
        print(f"[moondream-0.5b] Model size: {size_mb:.1f} MB")

        self.model = md.vl(
            local=True,
            model=MOONDREAM_MODEL_PATH,
        )

        print("[moondream-0.5b] Loaded local model")

    def update_frame(self, frame):
        if not self.enabled:
            return

        with self.lock:
            # QVGA is already small, but keep a copy so the background worker
            # doesn't touch a frame while OpenCV is displaying/processing it.
            self.latest_frame = frame.copy()

    def get_target_class(self) -> Tuple[str, str]:
        with self.lock:
            return self.target_class, self.raw_answer

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def _worker_loop(self):
        while self.running:
            now = time.time()

            if now - self.last_query_time < MOONDREAM_INTERVAL:
                time.sleep(0.1)
                continue

            with self.lock:
                if self.latest_frame is None:
                    time.sleep(0.1)
                    continue

                frame = self.latest_frame.copy()

            self.busy = True
            self.last_query_time = now

            try:
                yolo_class, answer = self._choose_target(frame)

                with self.lock:
                    self.target_class = yolo_class
                    self.raw_answer = answer

                print(f"[moondream-0.5b] answer='{answer}' -> target='{yolo_class}'")

            except Exception as e:
                # Keep YOLO running even if Moondream has a tantrum.
                print("[moondream-0.5b] failed:", repr(e))

            self.busy = False

    def _choose_target(self, frame) -> Tuple[str, str]:
        image = cv2_to_pil(frame)

        question = (
            "Pick exactly ONE label from this list and output only that label: "
            "person, cell phone, laptop, keyboard, mouse, bottle, cup, book, chair, backpack, tv. "
            "Prefer person if any human, face, head, body, or hand is visible. "
            "No numbering. No sentence. No explanation. One label only."
        )

        result = self.model.query(image, question)

        if isinstance(result, dict):
            raw_answer = str(result.get("answer", "")).strip()
        else:
            raw_answer = str(result).strip()

        if not raw_answer:
            raw_answer = DEFAULT_TARGET_CLASS

        yolo_class = map_moondream_to_yolo(raw_answer)

        # Show a clean answer in the overlay, but keep the raw model babble in logs.
        cleaned_answer = yolo_class
        return yolo_class, cleaned_answer


# ============================================================
# DRAWING
# ============================================================

def draw_target(frame, target: Optional[TargetBox], state_text: str, md_text: str):
    h, w = frame.shape[:2]

    # Center guide
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 1)

    if target is not None:
        x1, y1, x2, y2 = map(int, [target.x1, target.y1, target.x2, target.y2])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Error line
        cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"{target.label} {target.conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        state_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        md_text,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global pan_angle

    print("[sentinel] Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL)

    print("[sentinel] Starting Moondream 0.5B brain...")
    brain = MoondreamBrain()

    print("[sentinel] Opening ESP32-CAM stream:", ESP_STREAM)
    cam = LatestFrameReader(ESP_STREAM)

    center_servo()
    send_pan(PAN_CENTER)
    set_scan(False)

    last_processed_frame_id = -1
    last_command_time = 0.0
    last_seen_time = 0.0

    processed_frame_count = 0
    fps_t0 = time.time()
    yolo_fps = 0.0

    state = "BOOT"

    print("[sentinel] Running. Press Q in the OpenCV window to quit.")

    try:
        while True:
            ok, frame, frame_id = cam.read_latest()

            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Critical: do not run YOLO multiple times on the same frame.
            if frame_id == last_processed_frame_id:
                time.sleep(0.003)
                continue

            last_processed_frame_id = frame_id

            now = time.time()
            h, w = frame.shape[:2]

            # Give Moondream the newest frame, but do not wait for it.
            brain.update_frame(frame)

            target_class, md_answer = brain.get_target_class()

            processed_frame_count += 1
            if now - fps_t0 >= 1.0:
                yolo_fps = processed_frame_count / (now - fps_t0)
                processed_frame_count = 0
                fps_t0 = now

            # YOLO inference, once per new frame.
            results = yolo.predict(
                source=frame,
                imgsz=320,
                conf=CONF_MIN,
                verbose=False,
                device=YOLO_DEVICE,
            )

            target = get_best_target(results, target_class)

            if target is not None:
                last_seen_time = now
                set_scan(False)

                cx = (target.x1 + target.x2) / 2.0
                x_norm = cx / w
                error = x_norm - 0.5

                state = "LOCKED"

                if abs(error) > DEADZONE and (now - last_command_time) >= COMMAND_INTERVAL:
                    raw_delta = error * GAIN
                    delta = clamp(raw_delta, -MAX_STEP, MAX_STEP)

                    new_pan = clamp(
                        pan_angle + (SERVO_DIRECTION * delta),
                        PAN_MIN,
                        PAN_MAX,
                    )

                    if abs(new_pan - pan_angle) >= 1.0:
                        old_pan = pan_angle
                        pan_angle = new_pan

                        if send_pan(pan_angle):
                            last_command_time = now
                            print(
                                f"[pan] frame={frame_id} "
                                f"target={target_class} "
                                f"x={x_norm:.2f} err={error:+.2f} "
                                f"delta={delta:+.2f} "
                                f"pan={old_pan:.1f}->{pan_angle:.1f}"
                            )
                        else:
                            print("[sentinel] failed to send pan command")

                elif abs(error) <= DEADZONE:
                    state = "CENTERED"

            else:
                time_since_seen = now - last_seen_time

                if time_since_seen > LOST_TIMEOUT:
                    state = "SEARCHING"
                    set_scan(True)
                else:
                    state = "LOST"

            if SHOW_WINDOW:
                state_text = (
                    f"{state} | pan={pan_angle:.1f} | "
                    f"fps={yolo_fps:.1f} | frame={frame_id} | target={target_class}"
                )

                md_text = f"Moondream: {md_answer}"

                draw_target(frame, target, state_text, md_text)
                cv2.imshow("Sentinel: YOLO + Moondream", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        print("[sentinel] shutting down...")
        cam.stop()
        brain.stop()
        set_scan(False)
        cv2.destroyAllWindows()
        print("[sentinel] stopped")


if __name__ == "__main__":
    main()
