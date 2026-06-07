#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <ESP32Servo.h>
#include <ESP32PWM.h>
#include <Arduino.h>

// ======================================================
// WiFi settings
// ======================================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ======================================================
// Board: AI Thinker ESP32-CAM + OV2640
// ======================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ======================================================
// Hardware settings
// ======================================================
#define SERVO_PIN 13
#define FLASH_LED_PIN 4

#define SERVO_MIN_US 500
#define SERVO_MAX_US 2400

// Keep these conservative so your glued case does not become a sad clicking box.
int SERVO_SAFE_MIN = 30;
int SERVO_SAFE_MAX = 150;
int SERVO_CENTER   = 90;

// ======================================================
// Camera defaults
// ======================================================
framesize_t currentFrameSize = FRAMESIZE_QVGA;  // 320x240, good for low latency YOLO
int currentJpegQuality = 12;                    // lower = better quality, higher = smaller file

// ======================================================
// Runtime state
// ======================================================
Servo panServo;
int currentPan = 90;

bool scanEnabled = false;
int scanMin = 40;
int scanMax = 140;
int scanStep = 2;
int scanDirection = 1;
unsigned long scanDelayMs = 70;
unsigned long lastScanMoveMs = 0;

bool flashState = false;

httpd_handle_t control_httpd = NULL;
httpd_handle_t stream_httpd = NULL;

unsigned long bootTimeMs = 0;
unsigned long lastWifiCheckMs = 0;

// ======================================================
// Utility helpers
// ======================================================

int clampInt(int value, int minVal, int maxVal) {
  if (value < minVal) return minVal;
  if (value > maxVal) return maxVal;
  return value;
}

bool parseIntStrict(const char* str, int* out) {
  if (str == NULL || str[0] == '\0') return false;

  char* endptr;
  long value = strtol(str, &endptr, 10);

  if (*endptr != '\0') return false;
  if (value < -32768 || value > 32767) return false;

  *out = (int)value;
  return true;
}

bool getQueryValue(httpd_req_t* req, const char* key, char* value, size_t valueSize) {
  char query[256];

  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
    return false;
  }

  if (httpd_query_key_value(query, key, value, valueSize) != ESP_OK) {
    return false;
  }

  return true;
}

void addCorsHeaders(httpd_req_t* req) {
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, OPTIONS");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "*");
}

void movePanTo(int angle) {
  angle = clampInt(angle, SERVO_SAFE_MIN, SERVO_SAFE_MAX);
  currentPan = angle;
  panServo.write(currentPan);
}

const char* frameSizeToString(framesize_t size) {
  switch (size) {
    case FRAMESIZE_QQVGA: return "qqvga";
    case FRAMESIZE_QVGA:  return "qvga";
    case FRAMESIZE_VGA:   return "vga";
    case FRAMESIZE_SVGA:  return "svga";
    case FRAMESIZE_XGA:   return "xga";
    default:              return "unknown";
  }
}

framesize_t stringToFrameSize(const char* value, bool* ok) {
  *ok = true;

  if (strcmp(value, "qqvga") == 0) return FRAMESIZE_QQVGA; // 160x120
  if (strcmp(value, "qvga")  == 0) return FRAMESIZE_QVGA;  // 320x240
  if (strcmp(value, "vga")   == 0) return FRAMESIZE_VGA;   // 640x480
  if (strcmp(value, "svga")  == 0) return FRAMESIZE_SVGA;  // 800x600
  if (strcmp(value, "xga")   == 0) return FRAMESIZE_XGA;   // 1024x768

  *ok = false;
  return currentFrameSize;
}

String makeStatusJson() {
  String ip = WiFi.localIP().toString();

  String json = "{";
  json += "\"ok\":true,";
  json += "\"name\":\"ESP32-CAM Sentinel\",";
  json += "\"ip\":\"" + ip + "\",";
  json += "\"control_port\":80,";
  json += "\"stream_port\":81,";
  json += "\"jpg_url\":\"http://" + ip + "/jpg\",";
  json += "\"stream_url\":\"http://" + ip + ":81/stream\",";
  json += "\"pan\":" + String(currentPan) + ",";
  json += "\"servo_min\":" + String(SERVO_SAFE_MIN) + ",";
  json += "\"servo_max\":" + String(SERVO_SAFE_MAX) + ",";
  json += "\"scan\":" + String(scanEnabled ? "true" : "false") + ",";
  json += "\"flash\":" + String(flashState ? "true" : "false") + ",";
  json += "\"framesize\":\"" + String(frameSizeToString(currentFrameSize)) + "\",";
  json += "\"jpeg_quality\":" + String(currentJpegQuality) + ",";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
  json += "\"psram\":" + String(psramFound() ? "true" : "false") + ",";
  json += "\"free_psram\":" + String(psramFound() ? ESP.getFreePsram() : 0) + ",";
  json += "\"uptime_ms\":" + String(millis() - bootTimeMs);
  json += "}";

  return json;
}

// ======================================================
// HTTP handlers: control server, port 80
// ======================================================

static esp_err_t root_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  String ip = WiFi.localIP().toString();

  String text = "";
  text += "ESP32-CAM Sentinel online\n\n";
  text += "Control endpoints:\n";
  text += "  http://" + ip + "/status\n";
  text += "  http://" + ip + "/jpg\n";
  text += "  http://" + ip + "/pan?angle=90\n";
  text += "  http://" + ip + "/pan?delta=5\n";
  text += "  http://" + ip + "/center\n";
  text += "  http://" + ip + "/scan?on=1\n";
  text += "  http://" + ip + "/scan?on=0\n";
  text += "  http://" + ip + "/flash?on=1\n";
  text += "  http://" + ip + "/flash?on=0\n";
  text += "  http://" + ip + "/camera?framesize=qvga&quality=12\n";
  text += "  http://" + ip + "/restart\n\n";
  text += "Stream endpoint:\n";
  text += "  http://" + ip + ":81/stream\n\n";
  text += "Current pan: " + String(currentPan) + "\n";
  text += "Frame size: " + String(frameSizeToString(currentFrameSize)) + "\n";
  text += "JPEG quality: " + String(currentJpegQuality) + "\n";
  text += "RSSI: " + String(WiFi.RSSI()) + " dBm\n";

  httpd_resp_set_type(req, "text/plain");
  httpd_resp_send(req, text.c_str(), text.length());
  return ESP_OK;
}

static esp_err_t status_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  String json = makeStatusJson();

  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());
  return ESP_OK;
}

static esp_err_t jpg_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  camera_fb_t* fb = esp_camera_fb_get();

  if (!fb) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, "Camera capture failed", HTTPD_RESP_USE_STRLEN);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  esp_err_t result = httpd_resp_send(req, (const char*)fb->buf, fb->len);

  esp_camera_fb_return(fb);
  return result;
}

static esp_err_t pan_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  char value[32];

  if (getQueryValue(req, "angle", value, sizeof(value))) {
    int angle;

    if (!parseIntStrict(value, &angle)) {
      httpd_resp_set_status(req, "400 Bad Request");
      httpd_resp_send(req, "Invalid angle", HTTPD_RESP_USE_STRLEN);
      return ESP_FAIL;
    }

    scanEnabled = false;
    movePanTo(angle);
  }
  else if (getQueryValue(req, "delta", value, sizeof(value))) {
    int delta;

    if (!parseIntStrict(value, &delta)) {
      httpd_resp_set_status(req, "400 Bad Request");
      httpd_resp_send(req, "Invalid delta", HTTPD_RESP_USE_STRLEN);
      return ESP_FAIL;
    }

    scanEnabled = false;
    movePanTo(currentPan + delta);
  }
  else {
    httpd_resp_set_status(req, "400 Bad Request");
    httpd_resp_send(req, "Use /pan?angle=90 or /pan?delta=5", HTTPD_RESP_USE_STRLEN);
    return ESP_FAIL;
  }

  String json = "{\"ok\":true,\"pan\":" + String(currentPan) + "}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());

  return ESP_OK;
}

static esp_err_t center_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  scanEnabled = false;
  movePanTo(SERVO_CENTER);

  String json = "{\"ok\":true,\"pan\":" + String(currentPan) + "}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());

  return ESP_OK;
}

static esp_err_t scan_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  char value[32];

  if (getQueryValue(req, "min", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      scanMin = clampInt(v, SERVO_SAFE_MIN, SERVO_SAFE_MAX);
    }
  }

  if (getQueryValue(req, "max", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      scanMax = clampInt(v, SERVO_SAFE_MIN, SERVO_SAFE_MAX);
    }
  }

  if (scanMin > scanMax) {
    int temp = scanMin;
    scanMin = scanMax;
    scanMax = temp;
  }

  if (getQueryValue(req, "step", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      scanStep = clampInt(abs(v), 1, 20);
    }
  }

  if (getQueryValue(req, "delay", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      scanDelayMs = clampInt(v, 20, 2000);
    }
  }

  if (getQueryValue(req, "on", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      scanEnabled = (v != 0);
    }
  } else {
    scanEnabled = !scanEnabled;
  }

  String json = "{";
  json += "\"ok\":true,";
  json += "\"scan\":" + String(scanEnabled ? "true" : "false") + ",";
  json += "\"min\":" + String(scanMin) + ",";
  json += "\"max\":" + String(scanMax) + ",";
  json += "\"step\":" + String(scanStep) + ",";
  json += "\"delay_ms\":" + String(scanDelayMs);
  json += "}";

  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());

  return ESP_OK;
}

static esp_err_t flash_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  char value[16];

  if (getQueryValue(req, "on", value, sizeof(value))) {
    int on;
    if (parseIntStrict(value, &on)) {
      flashState = (on != 0);
    }
  } else {
    flashState = !flashState;
  }

  digitalWrite(FLASH_LED_PIN, flashState ? HIGH : LOW);

  String json = "{\"ok\":true,\"flash\":" + String(flashState ? "true" : "false") + "}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());

  return ESP_OK;
}

static esp_err_t camera_settings_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  sensor_t* s = esp_camera_sensor_get();

  if (!s) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    httpd_resp_send(req, "Sensor not available", HTTPD_RESP_USE_STRLEN);
    return ESP_FAIL;
  }

  char value[32];

  if (getQueryValue(req, "framesize", value, sizeof(value))) {
    bool ok = false;
    framesize_t newSize = stringToFrameSize(value, &ok);

    if (ok) {
      s->set_framesize(s, newSize);
      currentFrameSize = newSize;
    }
  }

  if (getQueryValue(req, "quality", value, sizeof(value))) {
    int q;

    if (parseIntStrict(value, &q)) {
      q = clampInt(q, 8, 30);
      s->set_quality(s, q);
      currentJpegQuality = q;
    }
  }

  if (getQueryValue(req, "brightness", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      s->set_brightness(s, clampInt(v, -2, 2));
    }
  }

  if (getQueryValue(req, "contrast", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      s->set_contrast(s, clampInt(v, -2, 2));
    }
  }

  if (getQueryValue(req, "saturation", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      s->set_saturation(s, clampInt(v, -2, 2));
    }
  }

  if (getQueryValue(req, "vflip", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      s->set_vflip(s, v != 0);
    }
  }

  if (getQueryValue(req, "hmirror", value, sizeof(value))) {
    int v;
    if (parseIntStrict(value, &v)) {
      s->set_hmirror(s, v != 0);
    }
  }

  String json = "{";
  json += "\"ok\":true,";
  json += "\"framesize\":\"" + String(frameSizeToString(currentFrameSize)) + "\",";
  json += "\"quality\":" + String(currentJpegQuality);
  json += "}";

  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json.c_str(), json.length());

  return ESP_OK;
}

static esp_err_t restart_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  httpd_resp_set_type(req, "text/plain");
  httpd_resp_send(req, "Restarting ESP32-CAM...", HTTPD_RESP_USE_STRLEN);

  delay(300);
  ESP.restart();

  return ESP_OK;
}

static esp_err_t options_handler(httpd_req_t* req) {
  addCorsHeaders(req);
  httpd_resp_send(req, "", 0);
  return ESP_OK;
}

// ======================================================
// HTTP handler: stream server, port 81
// ======================================================

static esp_err_t stream_handler(httpd_req_t* req) {
  addCorsHeaders(req);

  httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
  httpd_resp_set_hdr(req, "Connection", "close");

  char partHeader[96];

  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();

    if (!fb) {
      Serial.println("Stream capture failed");
      return ESP_FAIL;
    }

    esp_err_t res = httpd_resp_send_chunk(req, "--frame\r\n", strlen("--frame\r\n"));

    if (res == ESP_OK) {
      size_t headerLen = snprintf(
        partHeader,
        sizeof(partHeader),
        "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
        (unsigned int)fb->len
      );

      res = httpd_resp_send_chunk(req, partHeader, headerLen);
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, "\r\n", strlen("\r\n"));
    }

    esp_camera_fb_return(fb);

    if (res != ESP_OK) {
      break;
    }

    // QVGA + 20-40ms is a decent practical range.
    // The laptop can drop frames. The ESP32 should not be tortured.
    delay(30);
  }

  return ESP_OK;
}

// ======================================================
// Camera setup
// ======================================================

bool setupCamera() {
  Serial.println("Creating camera config...");

  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;

  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size   = currentFrameSize;
  config.jpeg_quality = currentJpegQuality;

  if (psramFound()) {
    Serial.println("PSRAM found");
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("No PSRAM found");
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  Serial.println("Calling esp_camera_init...");
  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  Serial.println("Camera init OK");

  sensor_t* s = esp_camera_sensor_get();

  if (s) {
    s->set_framesize(s, currentFrameSize);
    s->set_quality(s, currentJpegQuality);

    // Tune these via /camera later if needed.
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);

    // If your case makes the image upside down, call:
    // /camera?vflip=1
    // /camera?hmirror=1
  }

  return true;
}

// ======================================================
// WiFi setup
// ======================================================

bool setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // reduces stream stutter, uses more power, because physics charges rent

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  unsigned long startAttemptTime = millis();
  const unsigned long timeoutMs = 25000;

  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < timeoutMs) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection failed");
    return false;
  }

  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());

  return true;
}

// ======================================================
// Server setup
// ======================================================

bool startControlServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 16;
  config.stack_size = 8192;

  if (httpd_start(&control_httpd, &config) != ESP_OK) {
    Serial.println("Failed to start control server");
    return false;
  }

  httpd_uri_t root_uri = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = root_handler,
    .user_ctx = NULL
  };

  httpd_uri_t status_uri = {
    .uri = "/status",
    .method = HTTP_GET,
    .handler = status_handler,
    .user_ctx = NULL
  };

  httpd_uri_t jpg_uri = {
    .uri = "/jpg",
    .method = HTTP_GET,
    .handler = jpg_handler,
    .user_ctx = NULL
  };

  httpd_uri_t pan_uri = {
    .uri = "/pan",
    .method = HTTP_GET,
    .handler = pan_handler,
    .user_ctx = NULL
  };

  httpd_uri_t center_uri = {
    .uri = "/center",
    .method = HTTP_GET,
    .handler = center_handler,
    .user_ctx = NULL
  };

  httpd_uri_t scan_uri = {
    .uri = "/scan",
    .method = HTTP_GET,
    .handler = scan_handler,
    .user_ctx = NULL
  };

  httpd_uri_t flash_uri = {
    .uri = "/flash",
    .method = HTTP_GET,
    .handler = flash_handler,
    .user_ctx = NULL
  };

  httpd_uri_t camera_uri = {
    .uri = "/camera",
    .method = HTTP_GET,
    .handler = camera_settings_handler,
    .user_ctx = NULL
  };

  httpd_uri_t restart_uri = {
    .uri = "/restart",
    .method = HTTP_GET,
    .handler = restart_handler,
    .user_ctx = NULL
  };

  httpd_register_uri_handler(control_httpd, &root_uri);
  httpd_register_uri_handler(control_httpd, &status_uri);
  httpd_register_uri_handler(control_httpd, &jpg_uri);
  httpd_register_uri_handler(control_httpd, &pan_uri);
  httpd_register_uri_handler(control_httpd, &center_uri);
  httpd_register_uri_handler(control_httpd, &scan_uri);
  httpd_register_uri_handler(control_httpd, &flash_uri);
  httpd_register_uri_handler(control_httpd, &camera_uri);
  httpd_register_uri_handler(control_httpd, &restart_uri);

  Serial.println("Control server started on port 80");
  return true;
}

bool startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port = 81;
  config.ctrl_port = 32769;
  config.max_uri_handlers = 4;
  config.stack_size = 8192;

  if (httpd_start(&stream_httpd, &config) != ESP_OK) {
    Serial.println("Failed to start stream server");
    return false;
  }

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  httpd_register_uri_handler(stream_httpd, &stream_uri);

  Serial.println("Stream server started on port 81");
  return true;
}

// ======================================================
// Background tasks
// ======================================================

void updateScan() {
  if (!scanEnabled) return;

  unsigned long now = millis();

  if (now - lastScanMoveMs < scanDelayMs) return;

  lastScanMoveMs = now;

  int nextPan = currentPan + (scanDirection * scanStep);

  if (nextPan >= scanMax) {
    nextPan = scanMax;
    scanDirection = -1;
  } else if (nextPan <= scanMin) {
    nextPan = scanMin;
    scanDirection = 1;
  }

  movePanTo(nextPan);
}

void updateWiFi() {
  unsigned long now = millis();

  if (now - lastWifiCheckMs < 5000) return;
  lastWifiCheckMs = now;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    WiFi.disconnect();
    WiFi.begin(ssid, password);
  }
}

// ======================================================
// Main setup / loop
// ======================================================

void setup() {
  bootTimeMs = millis();

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=======================================");
  Serial.println("Starting ESP32-CAM Sentinel firmware");
  Serial.println("Board: AI Thinker ESP32-CAM");
  Serial.println("Camera: OV2640");
  Serial.println("Servo: GPIO13");
  Serial.println("=======================================");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  Serial.println("Setting up camera first...");
  if (!setupCamera()) {
    Serial.println("Camera setup failed. Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("Attaching servo after camera init...");

  // Camera uses LEDC timer/channel 0 for XCLK.
  // Allocate another timer for servo to reduce PWM nonsense.
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  panServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  movePanTo(SERVO_CENTER);

  Serial.println("Setting up WiFi...");
  if (!setupWiFi()) {
    Serial.println("WiFi failed. Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  if (!startControlServer()) {
    Serial.println("Control server failed. Restarting...");
    delay(3000);
    ESP.restart();
  }

  if (!startStreamServer()) {
    Serial.println("Stream server failed. Restarting...");
    delay(3000);
    ESP.restart();
  }

  String ip = WiFi.localIP().toString();

  Serial.println();
  Serial.println("Ready.");
  Serial.println("Control:");
  Serial.println("  http://" + ip + "/");
  Serial.println("  http://" + ip + "/status");
  Serial.println("  http://" + ip + "/jpg");
  Serial.println("  http://" + ip + "/pan?angle=90");
  Serial.println("  http://" + ip + "/scan?on=1");
  Serial.println();
  Serial.println("Stream:");
  Serial.println("  http://" + ip + ":81/stream");
  Serial.println();
}

void loop() {
  updateScan();
  updateWiFi();

  // No server.handleClient() here.
  // esp_http_server runs handlers in its own tasks.
  delay(5);
}
