# 👁️ AI Vision Navigation Assistant (XIAO ESP32-S3 + YOLO26 Depth + Gemini Live)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Hardware: Seeed Studio XIAO ESP32-S3](https://img.shields.io/badge/Hardware-XIAO%20ESP32--S3%20Sense-orange.svg)](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html)
[![AI Engine: Gemini Live](https://img.shields.io/badge/AI-Gemini%20Live%20Streaming-purple.svg)](https://ai.google.dev/)

An advanced, real-time multimodal AI environmental and navigation assistant designed to help blind or visually impaired individuals safely navigate physical environments. 

The system pairs an ultra-compact **Seeed Studio XIAO ESP32-S3 Sense** wearable camera with local **YOLO26 metric depth estimation** and **Google Gemini Live (Bidirectional Audio + Real-time Video Streaming)** with configurable reasoning thinking levels.

---

## 🌟 Key Features

- **Wearable Wireless Vision (ESP32-S3 Sense)**:
  - Low-latency single-frame HTTP **Pull Mode** (`GET /capture` @ 1 FPS, ~240 ms round-trip).
  - Optimized camera settings (`FRAMESIZE_VGA` 640x480, `jpeg_quality = 14`, `CAMERA_GRAB_LATEST`).
  - Active WiFi power-save bypass (`WiFi.setSleep(false)`) preventing WiFi freezes and dropouts.
  - Automatic fault tolerance: Session recreation after 3 consecutive failures, `CAMERA_LOST` dispatch after 5 failures, and automatic `CAMERA_OK` recovery.

- **Real-Time Metric Depth Estimation (YOLO26 Depth Engine)**:
  - Sub-80ms depth map inference (`imgsz=288`, temporal alpha smoothing).
  - Calculates real-world metric distance in meters: Center corridor depth (`C`) and nearest obstacle minimum distance (`MIN`).
  - Stale frame dropping (> 4.0s) and skip-if-slow (> 1.0s) protection to prevent pipeline backlogs.

- **Synchronized Dual-View Composite Streaming**:
  - Encodes a single unified frame per physical instant: **Left Half = Normal RGB Image**, **Right Half = Colorized Metric Depth Map + Visual Badges**.
  - Transmits via native `session.send_realtime_input(video=...)` into Gemini Live without blocking audio or event threads.

- **Gemini Live Multimodal Interaction**:
  - Bidirectional, full-duplex conversational voice interaction (16 kHz mic input, 24 kHz speech output).
  - Configurable thinking levels (`GEMINI_THINKING_LEVEL=low` / `medium` / `high`) powered by `gemini-3.8-live-extended-thinking`.
  - Directional, concise spoken responses tailored for safety ("Chair directly in front at 0.8 meters, step to your right").

- **2-Layer Hazard State Machine & Speech Avoidance**:
  - **Layer 1 (Near Gate)**: Continuous hysteresis tracking (`ALERT_DISTANCE_M`, `CLEAR_DISTANCE_M`, `URGENT_DISTANCE_M`) with rate-limited updates.
  - **Speech Avoidance**: Holds alert turns while Gemini is actively speaking and releases them cleanly once speech drains, preventing audio interruptions.

- **Persistent Memory System**:
  - Long-term vector and graph memory tracking for visual landmarks, personal items, user preferences, and navigational protocols (`store_memory`, `retrieve_memory`, `update_protocol`).

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │ Seeed XIAO ESP32-S3    │
       │ Wearable Camera (OV2640│
       └───────────┬────────────┘
                   │ Wi-Fi HTTP (GET /capture @ 1 FPS)
                   ▼
       ┌────────────────────────┐
       │ CameraStreamReader     │  <-- Non-blocking pull loop (~240ms)
       │ (Auto-Reconnect)       │
       └───────────┬────────────┘
                   │ Raw BGR Frame
                   ▼
       ┌────────────────────────┐
       │ YOLO26 Metric Depth    │  <-- Metric Depth Engine (~73ms inference)
       │ Temporal Smoothing     │
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         │ Composite Builder │  <-- Dual-View Side-by-Side Frame
         │ RGB + Depth Map   │      (Center & Corridor Min Distance Badges)
         └─────────┬─────────┘
                   │
      ┌────────────┼────────────────────────┐
      │            │                        │
      ▼            ▼                        ▼
┌───────────┐ ┌───────────────┐  ┌──────────────────────────────┐
│ Near Gate │ │ Async Queue   │  │ PyAudio Realtime Mic & Audio │
│ State M/C │ │ (Latest Only) │  └──────────────┬───────────────┘
└─────┬─────┘ └────────┬──────┘                 │
      │                │                        │
      │ Alert Turn     │ Video Blob             │ 16kHz PCM
      ▼                ▼                        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            Google Gemini Live Bidirectional Session         │
 │         (WebSocket Audio In/Out + Native Realtime Video)    │
 └─────────────────────────────┬───────────────────────────────┘
                               │ 24kHz Spoken Audio Output
                               ▼
                   ┌───────────────────────┐
                   │ Speaker / Earphones   │
                   │ Directional Guidance  │
                   └───────────────────────┘
```

---

## 📁 Repository Structure

```text
xiao-esp32s3-vision/
├── Cameranew/                      # ESP32-S3 Arduino firmware
│   ├── Cameranew.ino              # Main sketch (VGA, PSRAM=opi, WiFi.setSleep(false))
│   ├── app_httpd.cpp              # HTTP server (/capture on port 80, /stream on port 81)
│   ├── camera_index.h             # Web portal assets
│   ├── camera_pins.h              # Pin definitions for XIAO ESP32S3 Sense
│   └── partitions.csv             # Custom partition layout
├── memory/                         # Memory graphs, logs, and analytics
├── diagnose_esp32.py               # Step 1 diagnostic script (verifies 10 frame captures)
├── test_pull_mode.py               # Step 7 endurance test runner & recovery validator
├── memory_tools.py                 # Persistent memory tools (vector + graph)
├── new1.py                         # Main assistant pipeline (Gemini Live + YOLO26)
├── yolo1.py                        # YOLO26 depth engine wrapper
├── YOLO.py                         # Standalone YOLO inference script
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git exclusion rules for large weights and secrets
└── README.md                       # Documentation
```

---

## ⚙️ Hardware Setup

### Components
1. **Seeed Studio XIAO ESP32-S3 Sense** (ESP32-S3 with OV2640 camera board and microphone module).
2. **Power Supply**: 5V / 1A+ battery pack or USB-C cable.
3. **Wi-Fi Network**: 2.4 GHz mobile hotspot or local Wi-Fi router.

### Flashing Firmware
1. Open the [Cameranew](Cameranew/Cameranew.ino) folder in Arduino IDE or compile via `arduino-cli`.
2. Configure your Wi-Fi credentials in `Cameranew.ino`:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Board Settings:
   - **Board**: `XIAO_ESP32S3` (`esp32:esp32:XIAO_ESP32S3`)
   - **PSRAM**: `OPI PSRAM` (`PSRAM=opi`)
   - **Flash Size**: `8MB (64Mb)`
   - **USB CDC On Boot**: `Enabled`
4. Upload to the board. When booted, the serial monitor will display:
   ```text
   WiFi connected
   Camera Ready! Use 'http://10.24.69.1' to connect
   ```

---

## 🚀 Software Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Arun-Raj-C-R/xiao-esp32s3-vision.git
cd xiao-esp32s3-vision
```

### 2. Install Python Dependencies
Ensure you are using Python 3.10 or 3.11:
```bash
pip install -r requirements.txt
```
*(Key libraries: `google-genai`, `opencv-python`, `pyaudio`, `requests`, `numpy`, `python-dotenv`, `ultralytics`)*

### 3. Model Weights
Place the YOLO26 depth model weights in the project root:
- `yolo26n-depth.pt`

### 4. Configuration (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your settings:
```env
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_LIVE_MODEL="gemini-3.8-live-extended-thinking"
GEMINI_THINKING_LEVEL="low"

# ESP32 Camera Configuration
CAMERA_MODE="capture"
ESP32_CAM_URL="http://10.24.69.1/capture"
CAMERA_FPS=1.0

# Near Gate State Machine (in meters)
ALERT_DISTANCE_M=2.0
CLEAR_DISTANCE_M=2.3
URGENT_DISTANCE_M=0.5
```

---

## 🧪 Testing & Diagnostics

### 1. Verify ESP32 Camera Pull Mode
Run the standalone diagnostic script to verify 10 consecutive frame captures from the ESP32:
```bash
python diagnose_esp32.py
```
Expected output:
```text
============================================================
STEP 1 DIAGNOSTIC: Testing ESP32 Capture Endpoint
Target URL: http://10.24.69.1/capture
Testing 10 requests, 1 per second...
============================================================
Request #01: status=200 | bytes=4304 | time=204.7 ms
...
Results: 10/10 successful (100%)
Average time: 241.2 ms | Min: 166.5 ms | Max: 513.7 ms
Saved first_capture.jpg and last_capture.jpg successfully.
============================================================
```

### 2. Endurance & Recovery Test
Run a 2-minute endurance test that verifies session recreation and alerts:
```bash
python test_pull_mode.py
```

---

## 🏃 Running the Assistant

Run the main pipeline:
```bash
python new1.py
```

### Interactive Commands
- Speak naturally to the AI through your microphone.
- Type in terminal during live streaming:
  - `/camera`: Pause or resume camera streaming dynamically.
  - `/exit` or `quit`: Gracefully shutdown all threads and connections.

---

## 🔒 Safety & Privacy

- **No Secret Leaks**: The [.gitignore](.gitignore) prevents `.env` keys, large neural models (`*.pt`, `*.tflite`), and private image dumps from being tracked or pushed.
- **Fail-Safe Warnings**: When camera connectivity drops, Gemini is notified with a high-priority spoken alert (`CAMERA_LOST`) and alerted upon reconnection (`CAMERA_OK`).

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
