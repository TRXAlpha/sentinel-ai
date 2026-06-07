import time
import math
import re
import json
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import requests
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

ESP_IP = "10.62.241.20"
ESP_BASE = f"http://{ESP_IP}"
ESP_STREAM = f"http://{ESP_IP}:81/stream"

# Highest standard YOLO11 COCO detector.
# If too slow/VRAM-heavy, change to yolo11l.pt, yolo11m.pt, yolo11s.pt.
YOLO_MODEL = "yolo11x.pt"

USE_TRACKING = True
TRACKER = "bytetrack.yaml"

IMGSZ = 640
CONF_MIN = 0.30

# None = allow all built-in COCO classes from yolo11x.pt.
INTEREST_CLASSES = None

# Trim candidate list before sending to tiny SLM.
# Not a semantic priority table; just avoids sending a wall of noisy boxes.
MAX_CANDIDATES_FOR_OLLAMA = 12

# Ollama local SLM policy.
ENABLE_OLLAMA = True
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:270m"

# How often the SLM chooses a new target.
OLLAMA_INTERVAL = 3.0
OLLAMA_TIMEOUT = 12.0
OLLAMA_LOAD_TIMEOUT = 60.0

# If no valid model answer yet:
# False = wait/show boxes, no secret fallback.
# True = tracks first trimmed candidate until Ollama replies.
AUTO_SELECT_FIRST_UNTIL_OLLAMA = False

# Servo limits.
PAN_MIN = 10
PAN_MAX = 170
PAN_CENTER = 90

# Servo tuning.
pan_angle = PAN_CENTER
GAIN = 16.0
DEADZONE = 0.025
MAX_STEP = 2.5
COMMAND_INTERVAL = 0.20
SERVO_DIRECTION = -1  # flip to +1 if servo moves away from target

LOST_TIMEOUT = 2.0

SHOW_WINDOW = True
BAD_FRAME_SLEEP = 0.03


# ============================================================
# DEVICE
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
class CandidateBox:
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(1.0, (self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass
class OllamaSelection:
    selected_idx: Optional[int] = None
    selected_track_id: Optional[int] = None
    raw_answer: str = "none"
    reason: str = ""
    last_update_time: float = 0.0


# ============================================================
# CAMERA READER
# ============================================================

class LatestFrameReader:
    def __init__(self, url: str):
        self.url = url
        self.cap = None

        self.frame = None
        self.frame_id = 0
        self.ok = False

        self.running = True
        self.lock = threading.Lock()

        self._open_stream()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _open_stream(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open ESP32-CAM stream: {self.url}")

        print("[camera] stream opened")

    def _reader_loop(self):
        bad_count = 0

        while self.running:
            ok, frame = self.cap.read()

            if ok and frame is not None:
                bad_count = 0
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.ok = True
            else:
                bad_count += 1
                time.sleep(BAD_FRAME_SLEEP)

                if bad_count > 80:
                    print("[camera] too many bad frames, reconnecting...")
                    try:
                        self._open_stream()
                    except Exception as e:
                        print("[camera] reconnect failed:", repr(e))
                        time.sleep(1.0)
                    bad_count = 0

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
# YOLO / CANDIDATES
# ============================================================

def extract_candidates(results, conf_min=CONF_MIN) -> List[CandidateBox]:
    candidates = []

    if not results:
        return candidates

    r = results[0]
    names = r.names

    if r.boxes is None:
        return candidates

    for box in r.boxes:
        conf = float(box.conf[0].item())
        if conf < conf_min:
            continue

        cls_id = int(box.cls[0].item())
        label = names.get(cls_id, str(cls_id))

        if INTEREST_CLASSES is not None and label not in INTEREST_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        track_id = None
        if getattr(box, "id", None) is not None and box.id is not None:
            track_id = int(box.id[0].item())

        candidates.append(
            CandidateBox(
                idx=len(candidates),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=conf,
                label=label,
                track_id=track_id,
            )
        )

    return candidates


def trim_candidates_for_ollama(
    candidates: List[CandidateBox],
    frame_w: int,
    frame_h: int,
    max_count: int = MAX_CANDIDATES_FOR_OLLAMA,
) -> List[CandidateBox]:
    if len(candidates) <= max_count:
        return candidates

    frame_area = max(1, frame_w * frame_h)

    def score(c: CandidateBox):
        area_norm = c.area / frame_area
        return c.conf + area_norm

    top = sorted(candidates, key=score, reverse=True)[:max_count]

    reindexed = []
    for i, c in enumerate(top):
        reindexed.append(
            CandidateBox(
                idx=i,
                x1=c.x1,
                y1=c.y1,
                x2=c.x2,
                y2=c.y2,
                conf=c.conf,
                label=c.label,
                track_id=c.track_id,
            )
        )

    return reindexed


def candidates_to_prompt(candidates: List[CandidateBox], frame_w: int, frame_h: int) -> str:
    lines = []

    for c in candidates:
        cx = c.cx / frame_w
        cy = c.cy / frame_h
        area = c.area / max(1, frame_w * frame_h)
        tid = c.track_id if c.track_id is not None else -1

        lines.append(
            f"{c.idx}: label={c.label}, conf={c.conf:.2f}, "
            f"x={cx:.2f}, y={cy:.2f}, size={area:.3f}, track_id={tid}"
        )

    return "\n".join(lines)


def parse_ollama_choice(text: str, max_idx: int) -> Tuple[Optional[int], str]:
    raw = str(text).strip()

    # Small models love wrapping JSON in Markdown despite being told not to,
    # because apparently brackets need cosplay.
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    # Try strict JSON first.
    try:
        data = json.loads(raw)
        idx = int(data["id"])
        reason = str(data.get("reason", "")).strip()
        if 0 <= idx <= max_idx:
            return idx, reason
    except Exception:
        pass

    # Try to find JSON inside extra text, because small models love adding junk.
    try:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            idx = int(data["id"])
            reason = str(data.get("reason", "")).strip()
            if 0 <= idx <= max_idx:
                return idx, reason
    except Exception:
        pass

    # Fallback: first integer in the answer.
    m = re.search(r"\d+", raw)
    if not m:
        return None, "no parse"

    idx = int(m.group(0))
    if 0 <= idx <= max_idx:
        return idx, "parsed integer"

    return None, "id out of range"


def find_candidate_by_track_or_idx(
    candidates: List[CandidateBox],
    selected_track_id: Optional[int],
    selected_label: Optional[str],
    selected_idx: Optional[int],
) -> Optional[CandidateBox]:
    if not candidates:
        return None

    if selected_track_id is not None:
        for c in candidates:
            if c.track_id == selected_track_id:
                return c

    if selected_idx is not None and 0 <= selected_idx < len(candidates):
        c = candidates[selected_idx]
        if selected_label is None or c.label == selected_label:
            return c

    return None


# ============================================================
# OLLAMA BRAIN
# ============================================================

class OllamaBrain:
    """
    Background SLM attention policy.

    YOLO detects boxes.
    Ollama sees only text candidate boxes.
    Ollama chooses a candidate number.
    Python validates that the chosen ID exists.
    """

    def __init__(self):
        self.enabled = ENABLE_OLLAMA
        self.running = False

        self.lock = threading.Lock()
        self.latest_candidates: List[CandidateBox] = []
        self.frame_w = 1
        self.frame_h = 1

        self.selection = OllamaSelection()
        self.selected_label: Optional[str] = None

        if self.enabled:
            self._test_ollama()
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def _test_ollama(self):
        print(f"[ollama] using model: {OLLAMA_MODEL}")
        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "Return only: {\"id\":0}",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0,
                        "num_predict": 20,
                    },
                },
                timeout=OLLAMA_LOAD_TIMEOUT,
            )
            if not r.ok:
                print("[ollama] warning: test request failed:", r.status_code, r.text[:200])
            else:
                print("[ollama] reachable")
        except requests.RequestException as e:
            print("[ollama] warning: could not reach Ollama:", repr(e))
            print("[ollama] script will keep running, but no SLM decisions will arrive.")

    def update_candidates(self, candidates: List[CandidateBox], frame_w: int, frame_h: int):
        if not self.enabled:
            return

        prompt_candidates = trim_candidates_for_ollama(candidates, frame_w, frame_h)

        with self.lock:
            self.latest_candidates = prompt_candidates
            self.frame_w = frame_w
            self.frame_h = frame_h

    def get_selection(self) -> Tuple[OllamaSelection, Optional[str]]:
        with self.lock:
            return (
                OllamaSelection(
                    selected_idx=self.selection.selected_idx,
                    selected_track_id=self.selection.selected_track_id,
                    raw_answer=self.selection.raw_answer,
                    reason=self.selection.reason,
                    last_update_time=self.selection.last_update_time,
                ),
                self.selected_label,
            )

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

    def _worker_loop(self):
        while self.running:
            now = time.time()

            if now - self.selection.last_update_time < OLLAMA_INTERVAL:
                time.sleep(0.1)
                continue

            with self.lock:
                candidates = list(self.latest_candidates)
                frame_w = self.frame_w
                frame_h = self.frame_h

            if not candidates:
                with self.lock:
                    self.selection = OllamaSelection(
                        selected_idx=None,
                        selected_track_id=None,
                        raw_answer="no candidates",
                        reason="no boxes",
                        last_update_time=now,
                    )
                    self.selected_label = None
                time.sleep(0.1)
                continue

            try:
                selected_idx, raw_answer, reason = self._choose_candidate(candidates, frame_w, frame_h)

                selected_track_id = None
                selected_label = None

                if selected_idx is not None:
                    chosen = candidates[selected_idx]
                    selected_track_id = chosen.track_id
                    selected_label = chosen.label

                with self.lock:
                    self.selection = OllamaSelection(
                        selected_idx=selected_idx,
                        selected_track_id=selected_track_id,
                        raw_answer=raw_answer,
                        reason=reason,
                        last_update_time=now,
                    )
                    self.selected_label = selected_label

                print(
                    f"[ollama] answer={raw_answer!r} "
                    f"-> idx={selected_idx} label={selected_label} "
                    f"track_id={selected_track_id} reason={reason!r}"
                )

            except Exception as e:
                print("[ollama] failed:", repr(e))
                with self.lock:
                    self.selection.last_update_time = time.time()
                time.sleep(0.5)

    def _choose_candidate(
        self,
        candidates: List[CandidateBox],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Optional[int], str, str]:
        candidate_text = candidates_to_prompt(candidates, frame_w, frame_h)

        prompt = (
            "You are the attention policy for a small desktop sentinel robot.\n"
            "YOLO has already detected real objects. You do NOT see the image.\n"
            "Your job is to choose ONE candidate ID from the list.\n\n"
            "Rules:\n"
            "1. Choose only an ID that appears in the candidate list.\n"
            "2. Do not always choose 0. Compare all candidates before choosing.\n"
            "3. Compare label, confidence, position, size, and track_id.\n"
            "4. Interesting means: interactive, unusual, likely to move, useful to watch, or visually salient.\n"
            "5. Avoid a boring background object if another object is more interactive.\n"
            "6. Return only JSON. No markdown. No code block. No extra words.\n\n"
            "Required JSON fields:\n"
            "- id: the chosen candidate number from the list\n"
            "- reason: a very short reason\n\n"
            "Candidates:\n"
            f"{candidate_text}\n"
        )

        # Useful debug: see exactly what the SLM was asked to choose from.
        print("[ollama] candidates sent:\n" + candidate_text)

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 50,
                    "top_p": 0.9,
                    "num_ctx": 1024,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()

        data = r.json()
        answer = str(data.get("response", "")).strip()

        selected_idx, reason = parse_ollama_choice(answer, len(candidates) - 1)
        return selected_idx, answer, reason


# ============================================================
# DRAWING
# ============================================================

def draw(frame, candidates: List[CandidateBox], selected: Optional[CandidateBox], state_text: str, policy_text: str):
    h, w = frame.shape[:2]

    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 1)

    for c in candidates:
        x1, y1, x2, y2 = map(int, [c.x1, c.y1, c.x2, c.y2])
        cx, cy = int(c.cx), int(c.cy)

        is_selected = selected is not None and (
            (selected.track_id is not None and c.track_id == selected.track_id)
            or (selected.track_id is None and c.idx == selected.idx)
        )

        color = (0, 255, 0) if is_selected else (170, 170, 170)
        thickness = 3 if is_selected else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        tid = c.track_id if c.track_id is not None else "-"
        text = f"{c.idx}: {c.label} {c.conf:.2f} tid={tid}"

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            2,
        )

    if selected is not None:
        cv2.line(
            frame,
            (w // 2, h // 2),
            (int(selected.cx), int(selected.cy)),
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        state_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        policy_text,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global pan_angle

    print("[sentinel] Loading YOLO model:", YOLO_MODEL)
    yolo = YOLO(YOLO_MODEL)

    print("[yolo] built-in class names:")
    print(yolo.names)

    print("[sentinel] Starting Ollama attention policy...")
    brain = OllamaBrain() if ENABLE_OLLAMA else None

    print("[sentinel] Opening ESP32-CAM stream:", ESP_STREAM)
    cam = LatestFrameReader(ESP_STREAM)

    center_servo()
    send_pan(PAN_CENTER)
    set_scan(False)

    last_processed_frame_id = -1
    last_command_time = 0.0
    last_seen_time = 0.0

    processed_count = 0
    fps_t0 = time.time()
    fps = 0.0

    state = "BOOT"

    print("[sentinel] Running YOLO boxes + Ollama chooser. Press Q to quit.")
    print(
        f"[control] GAIN={GAIN} DEADZONE={DEADZONE} "
        f"MAX_STEP={MAX_STEP} COMMAND_INTERVAL={COMMAND_INTERVAL} "
        f"SERVO_DIRECTION={SERVO_DIRECTION} "
        f"OLLAMA_TIMEOUT={OLLAMA_TIMEOUT} LOAD_TIMEOUT={OLLAMA_LOAD_TIMEOUT}"
    )

    try:
        while True:
            ok, frame, frame_id = cam.read_latest()

            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if frame_id == last_processed_frame_id:
                time.sleep(0.003)
                continue

            last_processed_frame_id = frame_id
            now = time.time()
            h, w = frame.shape[:2]

            processed_count += 1
            if now - fps_t0 >= 1.0:
                fps = processed_count / (now - fps_t0)
                processed_count = 0
                fps_t0 = now

            if USE_TRACKING:
                results = yolo.track(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONF_MIN,
                    verbose=False,
                    device=YOLO_DEVICE,
                    persist=True,
                    tracker=TRACKER,
                )
            else:
                results = yolo.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONF_MIN,
                    verbose=False,
                    device=YOLO_DEVICE,
                )

            candidates = extract_candidates(results, CONF_MIN)

            if brain is not None:
                brain.update_candidates(candidates, w, h)
                selection, selected_label = brain.get_selection()
            else:
                selection = OllamaSelection(raw_answer="ollama disabled")
                selected_label = None

            selected = find_candidate_by_track_or_idx(
                candidates,
                selection.selected_track_id,
                selected_label,
                selection.selected_idx,
            )

            if selected is None and AUTO_SELECT_FIRST_UNTIL_OLLAMA and candidates:
                selected = candidates[0]

            if selected is not None:
                last_seen_time = now
                set_scan(False)

                x_norm = selected.cx / w
                error = x_norm - 0.5

                state = f"TRACK idx={selected.idx} {selected.label}"
                if selected.track_id is not None:
                    state += f" tid={selected.track_id}"

                if abs(error) <= DEADZONE:
                    state = "CENTERED " + state

                if abs(error) > DEADZONE and (now - last_command_time) >= COMMAND_INTERVAL:
                    raw_delta = error * GAIN
                    delta = clamp(raw_delta, -MAX_STEP, MAX_STEP)

                    if abs(delta) < 0.8:
                        delta = 0.8 if delta > 0 else -0.8

                    new_pan = clamp(
                        pan_angle + (SERVO_DIRECTION * delta),
                        PAN_MIN,
                        PAN_MAX,
                    )

                    if abs(new_pan - pan_angle) >= 0.5:
                        old_pan = pan_angle
                        pan_angle = new_pan

                        if send_pan(pan_angle):
                            last_command_time = now
                            print(
                                f"[pan] frame={frame_id} "
                                f"idx={selected.idx} label={selected.label} "
                                f"tid={selected.track_id} "
                                f"x_norm={x_norm:.3f} err={error:+.3f} "
                                f"raw_delta={raw_delta:+.2f} delta={delta:+.2f} "
                                f"pan={old_pan:.1f}->{pan_angle:.1f}"
                            )
                        else:
                            print("[sentinel] failed to send pan command")

            else:
                time_since_seen = now - last_seen_time

                if candidates:
                    state = "WAITING_FOR_OLLAMA"
                    set_scan(False)
                elif time_since_seen > LOST_TIMEOUT:
                    state = "SCANNING_NO_BOXES"
                    set_scan(True)
                else:
                    state = "LOST"

            if SHOW_WINDOW:
                policy_text = (
                    f"Ollama={OLLAMA_MODEL} ans={selection.raw_answer!r} "
                    f"idx={selection.selected_idx} tid={selection.selected_track_id}"
                )

                state_text = (
                    f"{state} | pan={pan_angle:.1f} | "
                    f"fps={fps:.1f} | boxes={len(candidates)} | frame={frame_id}"
                )

                draw(frame, candidates, selected, state_text, policy_text)
                cv2.imshow("Sentinel YOLO11x + Ollama Picker", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("[sentinel] KeyboardInterrupt")

    finally:
        print("[sentinel] shutting down...")
        cam.stop()
        if brain is not None:
            brain.stop()
        set_scan(False)
        cv2.destroyAllWindows()
        print("[sentinel] stopped")


if __name__ == "__main__":
    main()
