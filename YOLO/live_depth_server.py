"""
YOLO26 Real-Time Depth Web Server & Desktop Viewer
==================================================
Provides:
1. Real-time browser stream at http://localhost:5050
2. Always-on-top desktop OpenCV preview
3. 1-second interval auto-capture to D:\\Hackthon\\YOLO Images
"""

import os
import sys
import time
import threading
import webbrowser
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26s-depth.pt"
SAVE_DIR = r"D:\Hackthon\YOLO Images"
RAW_DIR = os.path.join(SAVE_DIR, "raw")

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

SAVE_INTERVAL = 1.0
INFERENCE_IMGSZ = 768
TEMPORAL_ALPHA = 0.70

WEB_PORT = 5050

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

app = Flask(__name__)

# Global frame buffer for web streaming
latest_composite_jpeg = None
lock = threading.Lock()

current_stats = {
    "center_dist": 0.0,
    "min_obstacle": 0.0,
    "fps": 0.0,
    "saved_count": 0,
    "last_saved": "None",
}


def compute_robust_distance(depth_map: np.ndarray, cx: int, cy: int, radius: int = 25) -> float:
    """Trimmed inlier distance around (cx, cy) to filter noise and specular glints."""
    h, w = depth_map.shape
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    region = depth_map[y1:y2, x1:x2]

    valid = region[region > 0]
    if valid.size == 0:
        return float(depth_map[cy, cx])

    p_low, p_high = np.percentile(valid, (15, 85))
    inliers = valid[(valid >= p_low) & (valid <= p_high)]
    if inliers.size > 0:
        return float(np.mean(inliers))
    return float(np.median(valid))


def camera_loop():
    global latest_composite_jpeg, current_stats

    print(f"[Model] Loading YOLO26 weights ({MODEL_PATH}) ...")
    model = YOLO(MODEL_PATH)
    print("[Model] YOLO26 depth model loaded successfully!")

    print(f"[Camera] Opening camera {CAMERA_INDEX} with CAP_DSHOW...")
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    print(f"[Camera] Camera ready ({CAMERA_WIDTH}x{CAMERA_HEIGHT}).")

    last_save_time = 0.0
    saved_count = 0
    smoothed_depth = None
    fps_start = time.time()
    frames = 0
    fps = 0.0

    # Create top-most desktop window
    win_name = "YOLO26 Live Depth (RGB | Depth)"
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 400)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frames += 1
            if time.time() - fps_start >= 1.0:
                fps = frames / (time.time() - fps_start)
                frames = 0
                fps_start = time.time()

            # YOLO26 Depth Inference
            results = model(frame, imgsz=INFERENCE_IMGSZ, verbose=False)
            res = results[0]
            if res.depth is None:
                continue

            # (H, W) metric depth in meters
            depth = res.depth.data.cpu().numpy().astype(np.float32)
            depth = np.squeeze(depth)

            # Resize to frame size
            if depth.shape != (frame.shape[0], frame.shape[1]):
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Temporal EMA Smoothing
            if smoothed_depth is None:
                smoothed_depth = depth.copy()
            else:
                smoothed_depth = TEMPORAL_ALPHA * depth + (1.0 - TEMPORAL_ALPHA) * smoothed_depth

            h, w = smoothed_depth.shape
            cx, cy = w // 2, h // 2

            # Compute Target Distance & Corridor Minimum
            center_dist = compute_robust_distance(smoothed_depth, cx, cy, radius=25)

            corridor = smoothed_depth[int(h * 0.35):h, int(w * 0.25):int(w * 0.75)]
            valid_c = corridor[corridor > 0]
            min_obstacle = float(np.percentile(valid_c, 3)) if valid_c.size > 0 else center_dist

            # Colorize with disparity (near = bright yellow/warm, far = dark purple)
            depth_color = colorize_depth(smoothed_depth, cmap="inferno", mode="disparity")

            # Auto-save 1 frame every second
            now = time.time()
            if now - last_save_time >= SAVE_INTERVAL:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(SAVE_DIR, f"depth_{ts}.png"), depth_color)
                np.save(os.path.join(RAW_DIR, f"depth_{ts}.npy"), smoothed_depth)
                cv2.imwrite(os.path.join(SAVE_DIR, f"rgb_{ts}.jpg"), frame)
                saved_count += 1
                last_save_time = now

                current_stats["saved_count"] = saved_count
                current_stats["last_saved"] = ts
                print(f"[SAVED] {ts} | Target: {center_dist:.2f}m | Obstacle: {min_obstacle:.2f}m")

            current_stats["center_dist"] = round(center_dist, 2)
            current_stats["min_obstacle"] = round(min_obstacle, 2)
            current_stats["fps"] = round(fps, 1)

            # Draw HUD overlays on frames
            frame_draw = frame.copy()
            depth_draw = depth_color.copy()

            # Target Box & Crosshairs
            box_r = 25
            cv2.rectangle(depth_draw, (cx - box_r, cy - box_r), (cx + box_r, cy + box_r), (255, 255, 255), 2)
            cv2.drawMarker(depth_draw, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(frame_draw, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # Distance overlays
            cv2.rectangle(depth_draw, (15, 15), (420, 95), (0, 0, 0), -1)
            cv2.rectangle(depth_draw, (15, 15), (420, 95), (80, 80, 80), 1)
            cv2.putText(depth_draw, f"Target Depth: {center_dist:.2f} m", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(depth_draw, f"Nearest: {min_obstacle:.2f} m", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0) if min_obstacle > 1.2 else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame_draw, f"AUTO-SAVED: {saved_count} frames", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Composite side-by-side preview (640x360 each)
            preview_w, preview_h = 640, 360
            f_small = cv2.resize(frame_draw, (preview_w, preview_h))
            d_small = cv2.resize(depth_draw, (preview_w, preview_h))
            composite = np.hstack((f_small, d_small))

            # Encode for web streaming
            _, buf = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with lock:
                latest_composite_jpeg = buf.tobytes()

            # Show desktop window
            cv2.imshow(win_name, composite)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Shutdown] Camera released.")


# ============================================================
# FLASK WEB INTERFACE
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26 Live Depth Estimation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 26px;
            color: #58a6ff;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .badge {
            background: rgba(56, 139, 253, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(56, 139, 253, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            display: inline-block;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            width: 100%;
            max-width: 1280px;
            margin-bottom: 20px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .card .label {
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .card .value {
            font-size: 28px;
            font-weight: 700;
            color: #f0f6fc;
        }
        .card .value.highlight { color: #f0883e; }
        .card .value.green { color: #3fb950; }
        .stream-container {
            width: 100%;
            max-width: 1280px;
            background: #000;
            border: 2px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            position: relative;
        }
        .stream-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .legend {
            margin-top: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            font-size: 14px;
            color: #8b949e;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot.yellow { background: #ffe600; box-shadow: 0 0 8px #ffe600; }
        .dot.purple { background: #551a8b; }
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLO26 High-Accuracy Monocular Depth Stream</h1>
        <span class="badge">● Live 1 FPS Auto-Capture Active</span>
    </div>

    <div class="stats-grid">
        <div class="card">
            <div class="label">Target Center Distance</div>
            <div class="value highlight" id="center-dist">-- m</div>
        </div>
        <div class="card">
            <div class="label">Nearest Corridor Obstacle</div>
            <div class="value green" id="min-obstacle">-- m</div>
        </div>
        <div class="card">
            <div class="label">Auto-Saved Captures</div>
            <div class="value" id="saved-count">0</div>
        </div>
        <div class="card">
            <div class="label">Inference Speed</div>
            <div class="value" id="fps">-- FPS</div>
        </div>
    </div>

    <div class="stream-container">
        <img src="/video_feed" alt="YOLO26 Dual Video Stream">
    </div>

    <div class="legend">
        <div class="legend-item"><span class="dot yellow"></span> Bright Yellow/Orange = Close Obstacle (Warning Zone)</div>
        <div class="legend-item"><span class="dot purple"></span> Dark Purple/Black = Distant Background</div>
        <div class="legend-item">💾 Saving every 1.0s to <code>D:\\Hackthon\\YOLO Images</code></div>
    </div>

    <script>
        setInterval(async () => {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                document.getElementById('center-dist').textContent = data.center_dist.toFixed(2) + ' m';
                const obsEl = document.getElementById('min-obstacle');
                obsEl.textContent = data.min_obstacle.toFixed(2) + ' m';
                obsEl.className = data.min_obstacle < 1.0 ? 'value highlight' : 'value green';
                document.getElementById('saved-count').textContent = data.saved_count;
                document.getElementById('fps').textContent = data.fps.toFixed(1) + ' FPS';
            } catch (e) {}
        }, 500);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/stats")
def stats():
    return current_stats


def generate_frames():
    global latest_composite_jpeg
    while True:
        with lock:
            if latest_composite_jpeg is None:
                frame_data = None
            else:
                frame_data = latest_composite_jpeg

        if frame_data is None:
            time.sleep(0.03)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    # Start camera worker thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Open browser automatically after 2 seconds
    def _open_browser():
        time.sleep(2.5)
        webbrowser.open(f"http://localhost:{WEB_PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()

    print("=" * 60)
    print(f" Web Stream available at: http://localhost:{WEB_PORT}")
    print(f" Saving 1 frame/sec to: {SAVE_DIR}")
    print("=" * 60)

    # Run web server
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
