# Sentinel AI

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32--CAM-AI%20Thinker-E7352C?logo=espressif&logoColor=white)
![Arduino](https://img.shields.io/badge/Firmware-Arduino-00979D?logo=arduino&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Detection-YOLO11-111111)
![Moondream](https://img.shields.io/badge/VLM-Moondream%200.5B-6A5ACD)
![Ollama](https://img.shields.io/badge/Policy-Ollama-000000?logo=ollama&logoColor=white)
![Status](https://img.shields.io/badge/status-prototype-orange)

Sentinel AI is a tiny ESP32-CAM pan sentinel with a laptop-side vision brain.
The ESP32-CAM streams frames and exposes simple HTTP controls. The laptop runs
YOLO, Moondream, or Ollama policy code, then sends pan commands back to the
servo so the device can search, lock, center, and follow.

## Get It Running

### 1. Prepare the ESP32-CAM firmware

Open `main.cpp`, set your Wi-Fi credentials, and flash it to an AI Thinker
ESP32-CAM with the Arduino ESP32 toolchain.

```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

Required Arduino-side pieces:

- AI Thinker ESP32-CAM board support
- `ESP32Servo`
- Servo signal wired to GPIO 13
- Camera module: OV2640

After flashing, open the serial monitor and copy the board IP. The firmware
prints links similar to:

```text
Control: http://<esp-ip>/
Status:  http://<esp-ip>/status
Stream:  http://<esp-ip>:81/stream
```

### 2. Verify the board

From a browser or terminal, check:

```text
http://<esp-ip>/status
http://<esp-ip>/jpg
http://<esp-ip>:81/stream
http://<esp-ip>/pan?angle=90
```

If those work, the hardware side is ready.

### 3. Set up Python on the laptop

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the pinned CUDA Torch packages do not match your machine, install the correct
Torch build for your GPU or CPU first, then reinstall the remaining
requirements.

### 4. Point the scripts at your ESP32

Each controller currently uses:

```python
ESP_IP = "YOUR_ESP32_IP"
```

Change that constant in the script you want to run, or reserve the ESP32-CAM IP
on your router.

### 5. Run a controller

Start with the simplest loop:

```powershell
python sentinel_yolo.py
```

Then try the richer policies:

```powershell
python works.py
```

```powershell
ollama pull gemma3:270m
python sentinel_brain.py
```

Press `q` in the OpenCV window to stop a controller.

## What It Does

```text
ESP32-CAM body
  - hosts /jpg snapshots
  - hosts /stream MJPEG video
  - accepts /pan, /center, /scan, /flash commands

Laptop brain
  - reads the latest camera frame
  - detects objects with YOLO
  - optionally asks Moondream or Ollama what to watch
  - sends small pan updates back to the ESP32
```

The ESP32 stays simple and responsive. The laptop does the expensive vision and
policy work.

## Controller Modes

| Mode | Command | Behavior |
| --- | --- | --- |
| Basic YOLO | `python sentinel_yolo.py` | Tracks the best visible `person` box. Best first test. |
| Moondream + YOLO | `python works.py` | Moondream periodically chooses a target class, then YOLO tracks that class. |
| Ollama + YOLO11 | `python sentinel_brain.py` | YOLO tracks boxes and Ollama chooses the most interesting candidate ID. |

## Firmware Endpoints

| Endpoint | Description |
| --- | --- |
| `/` | Plain-text device help and current state. |
| `/status` | JSON status with IP, stream URL, pan angle, RSSI, heap, and camera settings. |
| `/jpg` | Single JPEG snapshot. |
| `/pan?angle=90` | Move pan servo to an absolute angle. |
| `/pan?delta=5` | Move pan servo by a relative amount. |
| `/center` | Center the pan servo. |
| `/scan?on=1` | Enable scan mode. Optional: `min`, `max`, `step`, `delay`. |
| `/scan?on=0` | Disable scan mode. |
| `/flash?on=1` | Turn the flash LED on. |
| `/flash?on=0` | Turn the flash LED off. |
| `/camera?framesize=qvga&quality=12` | Change camera resolution or JPEG quality. |
| `/restart` | Restart the ESP32-CAM. |

## Repository Map

| File | Purpose |
| --- | --- |
| `main.cpp` | ESP32-CAM firmware for camera streaming, HTTP control, servo pan, scan mode, flash, and camera settings. |
| `sentinel_yolo.py` | Minimal laptop controller: YOLO tracks the best detected person. |
| `works.py` | YOLO controller with a background local Moondream 0.5B target-class selector. |
| `sentinel_brain.py` | YOLO11 controller with an Ollama policy that chooses from detected candidate boxes. |
| `test_moondream.py` | One-shot snapshot test using Hugging Face Moondream. |
| `download_moondream_weights.py` | Local `.mf.gz` Moondream smoke test against an ESP32 snapshot. |
| `requirements.txt` | Python dependencies for the laptop vision scripts. |

## Moondream Weights

`works.py` expects the local model file here:

```text
weights/moondream-0_5b-int8.mf.gz
```

Download it with:

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='vikhyatk/moondream2', filename='moondream-0_5b-int8.mf.gz', revision='onnx', local_dir='weights'))"
```

Smoke test the local file:

```powershell
python download_moondream_weights.py
```

## Tuning

Common Python constants:

| Constant | Meaning |
| --- | --- |
| `ESP_IP` | ESP32-CAM address. |
| `YOLO_MODEL` | Ultralytics model, such as `yolo11n.pt` or `yolo11x.pt`. |
| `CONF_MIN` | Minimum detection confidence. |
| `GAIN` | How strongly pan angle reacts to horizontal error. |
| `DEADZONE` | Center tolerance before the servo moves. |
| `MAX_STEP` | Maximum servo angle change per command. |
| `COMMAND_INTERVAL` | Minimum delay between servo commands. |
| `SERVO_DIRECTION` | Flip between `-1` and `+1` if the servo moves away from targets. |

Common firmware constants:

| Constant | Meaning |
| --- | --- |
| `SERVO_SAFE_MIN`, `SERVO_SAFE_MAX` | Hard pan limits. |
| `currentFrameSize` | Default stream resolution. QVGA is used for low latency. |
| `currentJpegQuality` | JPEG quality. Lower values improve quality but use more bandwidth. |
| `scanMin`, `scanMax`, `scanStep`, `scanDelayMs` | Default scan behavior. |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| OpenCV cannot open the stream | Confirm `http://<esp-ip>:81/stream` works and `ESP_IP` is correct. |
| Pan commands do nothing | Test `http://<esp-ip>/pan?angle=90` directly. |
| Servo moves away from the target | Flip `SERVO_DIRECTION`. |
| Stream is unstable | Keep `framesize=qvga`, improve Wi-Fi, or increase the JPEG quality number slightly. |
| Ollama decisions never arrive | Confirm Ollama is running at `http://localhost:11434` and `gemma3:270m` is installed. |
| Moondream fails to load | Confirm `weights/moondream-0_5b-int8.mf.gz` exists. |

## Notes

- The control loop processes each camera frame once, avoiding stale-frame servo
  commands.
- If no target is visible for long enough, the Python controller can enable
  ESP32 scan mode until a target returns.
- Servo limits are intentionally conservative for small printed mounts and
  lightweight assemblies.
