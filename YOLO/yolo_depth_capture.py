"""
============================================================
YOLO26 HIGH-ACCURACY MONOCULAR DEPTH ESTIMATION & AUTO-CAPTURE
============================================================

Accuracy Enhancements:
1. Native 768px Resolution: Runs inference at imgsz=768 (native training resolution of YOLO26-depth).
2. Model Capacity: Uses yolo26s-depth.pt (higher accuracy, better boundary delineation than nano).
3. Temporal Depth Smoothing (EMA): Eliminates frame-to-frame distance jitter and sensor noise.
4. Robust Trimmed Inlier Sampling: Filters out noise spikes and specular glints in the target zone.
5. Calibrated Disparity Colorization: Uses 2nd-98th percentile inverse depth mapping (near = warm, far = dark).
6. Multi-Zone Spatial Awareness: Tracks Center Distance + Closest Obstacle in the forward corridor.
7. Automated 1-Second Sync Capture: Saves depth PNG, raw metric float NPY, and RGB frame to D:\\Hackthon\\YOLO Images.
"""

import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

# ============================================================
# CONFIGURATION
# ============================================================

# Model selection: 'yolo26s-depth.pt' (high accuracy) or 'yolo26n-depth.pt' (ultralight)
MODEL_PATH = "yolo26s-depth.pt"

# Output directories
SAVE_DIR = r"D:\Hackthon\YOLO Images"
RAW_DIR = os.path.join(SAVE_DIR, "raw")

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Save one frame every second
SAVE_INTERVAL = 1.0

# Inference image size (768 is native training size for YOLO26 depth weights)
INFERENCE_IMGSZ = 768

# Temporal smoothing factor (0.0 = no smoothing, 0.7 = balanced responsive & stable)
TEMPORAL_ALPHA = 0.70

# Metric calibration (adjust if physical tape measurement differs by a fixed ratio)
SCALE_FACTOR = 1.0
OFFSET_METERS = 0.0

# Visualization Colormap: 'inferno' (yellow=near, purple=far), 'jet', or 'spectral'
COLORMAP = "inferno"

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("       YOLO26 HIGH-ACCURACY DEPTH ESTIMATION")
print("=" * 60)
print(f"[1/3] Loading YOLO26 model ({MODEL_PATH}) ...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# OPEN CAMERA
# ============================================================

print(f"[2/3] Opening camera {CAMERA_INDEX}...")

# Use CAP_DSHOW on Windows for instant, lag-free hardware initialization
cap = None
if sys.platform.startswith("win"):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"ERROR: Could not open camera {CAMERA_INDEX}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Camera opened successfully.")
print()


# ============================================================
# START
# ============================================================

print("[3/3] Starting depth estimation...")
print(f"Saving every {SAVE_INTERVAL} second to: {SAVE_DIR}")
print("Press 'q' or ESC in preview window to quit.")
print()

last_save_time = 0.0
smoothed_depth = None


# ============================================================
# HELPER: ACCURATE DISTANCE ESTIMATION
# ============================================================

def compute_robust_distance(depth_map: np.ndarray, cx: int, cy: int, radius: int = 20) -> float:
    """
    Computes a robust trimmed inlier distance around (cx, cy)
    to ignore specular highlights, hair, or sharp edge spikes.
    """
    h, w = depth_map.shape
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    region = depth_map[y1:y2, x1:x2]

    # Filter out non-positive readings
    valid = region[region > 0]
    if valid.size == 0:
        return float(depth_map[cy, cx])

    # 15th to 85th percentile trimmed mean
    p_low, p_high = np.percentile(valid, (15, 85))
    inliers = valid[(valid >= p_low) & (valid <= p_high)]
    if inliers.size > 0:
        return float(np.mean(inliers))
    return float(np.median(valid))


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret or frame is None:
        print("ERROR: Failed to read camera frame.")
        time.sleep(0.05)
        continue


    # --------------------------------------------------------
    # YOLO26 HIGH-RES DEPTH INFERENCE
    # --------------------------------------------------------

    results = model(
        frame,
        imgsz=INFERENCE_IMGSZ,
        verbose=False
    )

    result = results[0]

    if result.depth is None:
        print("WARNING: No depth output received.")
        continue


    # --------------------------------------------------------
    # GET METRIC DEPTH MAP
    # --------------------------------------------------------

    depth = result.depth.data.cpu().numpy().astype(np.float32)
    depth = np.squeeze(depth)

    # Resize depth map to exact camera frame resolution
    depth = cv2.resize(
        depth,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Apply physical metric calibration
    depth = depth * SCALE_FACTOR + OFFSET_METERS


    # --------------------------------------------------------
    # TEMPORAL SMOOTHING (ELIMINATES FRAME-TO-FRAME NOISE)
    # --------------------------------------------------------

    if smoothed_depth is None:
        smoothed_depth = depth.copy()
    else:
        smoothed_depth = TEMPORAL_ALPHA * depth + (1.0 - TEMPORAL_ALPHA) * smoothed_depth


    # --------------------------------------------------------
    # ACCURATE CENTER DISTANCE & NEAREST OBSTACLE
    # --------------------------------------------------------

    height, width = smoothed_depth.shape
    center_x = width // 2
    center_y = height // 2

    # Robust center distance (trimmed sample)
    center_dist = compute_robust_distance(smoothed_depth, center_x, center_y, radius=25)

    # Nearest obstacle in central walking corridor (middle 50% width, lower 60% height)
    corridor_y1, corridor_y2 = int(height * 0.4), height
    corridor_x1, corridor_x2 = int(width * 0.25), int(width * 0.75)
    corridor = smoothed_depth[corridor_y1:corridor_y2, corridor_x1:corridor_x2]
    valid_corridor = corridor[corridor > 0]
    min_obstacle_dist = float(np.percentile(valid_corridor, 3)) if valid_corridor.size > 0 else center_dist


    # --------------------------------------------------------
    # CREATE ACCURATE DEPTH VISUALIZATION
    # --------------------------------------------------------

    # Disparity mode clips 2nd-98th percentiles so close objects pop with warm colors
    depth_color = colorize_depth(
        smoothed_depth,
        cmap=COLORMAP,
        mode="disparity"
    )


    # --------------------------------------------------------
    # DRAW HUD / TARGET CROSSHAIR
    # --------------------------------------------------------

    # Target box around center measurement region
    box_r = 25
    cv2.rectangle(
        depth_color,
        (center_x - box_r, center_y - box_r),
        (center_x + box_r, center_y + box_r),
        (255, 255, 255),
        2
    )
    cv2.drawMarker(
        depth_color,
        (center_x, center_y),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2
    )

    # Overlay Telemetry HUD
    hud_bg = (0, 0, 0)
    cv2.rectangle(depth_color, (15, 15), (420, 100), hud_bg, -1)
    cv2.rectangle(depth_color, (15, 15), (420, 100), (80, 80, 80), 1)

    text_center = f"Target Depth : {center_dist:.2f} m"
    text_nearest = f"Min Corridor : {min_obstacle_dist:.2f} m"

    cv2.putText(
        depth_color,
        text_center,
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        depth_color,
        text_nearest,
        (25, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 200, 0) if min_obstacle_dist > 1.2 else (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # SAVE EVERY SECOND
    # --------------------------------------------------------

    current_time = time.time()

    if current_time - last_save_time >= SAVE_INTERVAL:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Color depth image
        image_path = os.path.join(SAVE_DIR, f"depth_{timestamp}.png")
        cv2.imwrite(image_path, depth_color)

        # Raw float depth data (smoothed in meters)
        raw_path = os.path.join(RAW_DIR, f"depth_{timestamp}.npy")
        np.save(raw_path, smoothed_depth)

        # RGB camera image
        rgb_path = os.path.join(SAVE_DIR, f"rgb_{timestamp}.jpg")
        cv2.imwrite(rgb_path, frame)

        print(
            f"[SAVED] {timestamp} | "
            f"Center: {center_dist:.2f} m | "
            f"Nearest Obstacle: {min_obstacle_dist:.2f} m"
        )

        last_save_time = current_time


    # --------------------------------------------------------
    # SHOW WINDOWS
    # --------------------------------------------------------

    cv2.imshow("Camera", frame)
    cv2.imshow("YOLO26 High-Accuracy Depth", depth_color)


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("YOLO26 DEPTH ESTIMATION STOPPED")
print("========================================")
print()
