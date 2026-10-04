"""
Test & Diagnostic Utility for ESP32-S3 Camera Stream (from Cameranew)
Connects directly to the ESP32-S3 MJPEG stream or snapshot endpoint and displays live video.

Usage:
    python test_esp32_cam.py                    # Reads ESP32_CAM_URL from .env
    python test_esp32_cam.py 192.168.1.100       # Tests specific IP
    python test_esp32_cam.py http://192.168.1.100:81/stream
"""

import os
import sys
import time
import cv2
import requests
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env
load_dotenv()


def resolve_camera_source(source):
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if not s:
        return 0

    if not s.startswith(("http://", "https://")):
        s = "http://" + s

    parsed = urlparse(s)
    clean = s.rstrip("/")
    if not parsed.port and parsed.path in ("", "/"):
        return f"{parsed.scheme}://{parsed.hostname}:81/stream"
    elif parsed.path in ("", "/"):
        return f"{clean}/stream"
    return s


def test_esp32_stream(url):
    print("=" * 60)
    print(f"📡 Testing ESP32-S3 Camera Stream at: {url}")
    print("=" * 60)
    print("Connecting... Press 'q' in the video window to quit, or 's' to save IP to .env.")

    fps_count = 0
    fps_start = time.time()
    current_fps = 0.0

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ Failed to open video stream at {url}")
        return False

    print("✅ Stream opened successfully! Receiving frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            fps_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                current_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            # Draw diagnostic HUD
            h, w = frame.shape[:2]
            hud_text = f"ESP32-S3 Cam | {w}x{h} | {current_fps:.1f} FPS"
            cv2.rectangle(frame, (10, 10), (10 + len(hud_text) * 11, 40), (0, 0, 0), -1)
            cv2.putText(frame, hud_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("ESP32-S3 Camera Stream Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting preview.")
                break
            elif key == ord('s'):
                save_to_env(url)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return True


def save_to_env(url):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"ESP32_CAM_URL={url}\n")
    else:
        with open(env_path, "r") as f:
            content = f.read()
        import re
        if re.search(r"^ESP32_CAM_URL=.*$", content, flags=re.MULTILINE):
            content = re.sub(r"^ESP32_CAM_URL=.*$", f"ESP32_CAM_URL={url}", content, flags=re.MULTILINE)
        else:
            content += f"\nESP32_CAM_URL={url}\n"
        with open(env_path, "w") as f:
            f.write(content)
    print(f"💾 Saved ESP32_CAM_URL={url} into .env successfully!")


if __name__ == "__main__":
    target = ""
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.getenv("ESP32_CAM_URL", "").strip()

    if not target:
        print("No ESP32 camera URL specified in .env or arguments.")
        try:
            target = input("Enter your ESP32-S3 IP or stream URL (e.g. 192.168.1.100): ").strip()
        except EOFError:
            target = ""

    if not target:
        print("❌ No URL provided. Exiting.")
        sys.exit(1)

    resolved = resolve_camera_source(target)
    test_esp32_stream(resolved)
