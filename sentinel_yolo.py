import time
import math
import threading

import cv2
import requests
import torch
from ultralytics import YOLO


# ==========================
# CONFIG
# ==========================

ESP_IP = "10.62.241.20"
ESP_BASE = f"http://{ESP_IP}"
ESP_STREAM = f"http://{ESP_IP}:81/stream"

YOLO_MODEL = "yolo11n.pt"
TARGET_CLASS = "person"

PAN_MIN = 30
PAN_MAX = 150
PAN_CENTER = 90

# ==========================
# CONTROL TUNING FOR ~3 FPS CAMERA
# ==========================
# At 3 FPS, do NOT move too aggressively. The camera feedback is slow.
pan_angle = PAN_CENTER

GAIN = 14.0
DEADZONE = 0.08
MAX_STEP = 2.5
COMMAND_INTERVAL = 0.25

# If your servo moves away from the target, flip this to +1.
# Current formula uses pan_angle - delta.
SERVO_DIRECTION = -1

# Detection filtering
CONF_MIN = 0.35
LOST_TIMEOUT = 1.5

# Display
SHOW_WINDOW = True

# Stream behavior
BAD_FRAME_SLEEP = 0.03


# ==========================
# DEVICE
# ==========================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("[sentinel] Torch:", torch.__version__)
print("[sentinel] CUDA available:", torch.cuda.is_available())
print("[sentinel] Using device:", DEVICE)
if torch.cuda.is_available():
    print("[sentinel] GPU:", torch.cuda.get_device_name(0))


# ==========================
# CAMERA READER
# ==========================

class LatestFrameReader:
    """
    Continuously reads from the ESP32-CAM stream in a background thread.

    Important:
    - Every truly new frame increments frame_id.
    - Main YOLO loop processes each frame_id only once.
    - This prevents YOLO from running multiple times on the same stale frame.
    """

    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # May or may not work depending on backend, but harmless.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open ESP32-CAM stream: {url}")

        self.frame = None
        self.frame_id = 0
        self.ok = False
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()

            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.ok = True
            else:
                time.sleep(BAD_FRAME_SLEEP)

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.frame_id

            return self.ok, self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ==========================
# HELPERS
# ==========================

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
    Only send scan command when state changes.
    No more /scan?on=0 spam every frame, because apparently HTTP abuse is not a control strategy.
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
            "delay": 100
        })

    r = send_get("/scan", params=params, timeout=0.4)

    if r is not None and r.ok:
        scan_is_on = on
        print(f"[scan] {'ON' if on else 'OFF'}")
    else:
        print("[scan] failed to set scan mode")


def get_best_target(results, class_name):
    """
    Returns best matching YOLO box:
    (x1, y1, x2, y2, conf, label)
    or None.
    """
    if not results:
        return None

    r = results[0]
    names = r.names
    best = None
    best_score = -1

    if r.boxes is None:
        return None

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

        # Larger + more confident target wins.
        score = conf * math.sqrt(area)

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf, label)

    return best


def draw_target(frame, target, state_text):
    h, w = frame.shape[:2]

    # Center guide
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 1)

    if target is not None:
        x1, y1, x2, y2, conf, label = target
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Error line from center to target
        cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        state_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )


# ==========================
# MAIN
# ==========================

def main():
    global pan_angle

    print("[sentinel] Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

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

            # THE IMPORTANT PART:
            # Do not run YOLO twice on the same camera frame.
            if frame_id == last_processed_frame_id:
                time.sleep(0.003)
                continue

            last_processed_frame_id = frame_id

            now = time.time()
            h, w = frame.shape[:2]

            processed_frame_count += 1
            if now - fps_t0 >= 1.0:
                yolo_fps = processed_frame_count / (now - fps_t0)
                processed_frame_count = 0
                fps_t0 = now

            # YOLO inference, once per new frame
            results = model.predict(
                source=frame,
                imgsz=320,
                conf=CONF_MIN,
                verbose=False,
                device=DEVICE
            )

            target = get_best_target(results, TARGET_CLASS)

            if target is not None:
                last_seen_time = now
                set_scan(False)

                x1, y1, x2, y2, conf, label = target

                cx = (x1 + x2) / 2.0
                x_norm = cx / w
                error = x_norm - 0.5

                state = "LOCKED"

                # At 3 FPS, each new frame already includes a delay.
                # Still keep command interval so the servo does not get spammed.
                if abs(error) > DEADZONE and (now - last_command_time) >= COMMAND_INTERVAL:
                    raw_delta = error * GAIN
                    delta = clamp(raw_delta, -MAX_STEP, MAX_STEP)

                    # SERVO_DIRECTION = -1 means:
                    # target right -> decrease pan, based on your current working sign.
                    new_pan = clamp(
                        pan_angle + (SERVO_DIRECTION * delta),
                        PAN_MIN,
                        PAN_MAX
                    )

                    # Don't send meaningless tiny angle changes
                    if abs(new_pan - pan_angle) >= 1.0:
                        old_pan = pan_angle
                        pan_angle = new_pan

                        if send_pan(pan_angle):
                            last_command_time = now
                            print(
                                f"[pan] frame={frame_id} "
                                f"x={x_norm:.2f} err={error:+.2f} "
                                f"delta={delta:+.2f} "
                                f"pan={old_pan:.1f}->{pan_angle:.1f}"
                            )
                        else:
                            print("[sentinel] Failed to send pan command")

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
                    f"fps={yolo_fps:.1f} | frame={frame_id} | target={TARGET_CLASS}"
                )

                draw_target(frame, target, state_text)
                cv2.imshow("Sentinel YOLO", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        set_scan(False)
        print("[sentinel] stopped")


if __name__ == "__main__":
    main()