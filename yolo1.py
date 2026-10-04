"""
yolo1.py - High-Performance Real-Time YOLO Depth Module
======================================================
Modular, low-latency monocular depth estimation engine for integration into
robotics, navigation, or AI assistant pipelines (e.g., new1.py).

Usage Example:
--------------
    from yolo1 import YOLODepth

    detector = YOLODepth()
    depth_map, stats = detector.estimate(frame)

    print(f"Center Distance: {stats['center_depth']:.2f} m")
    print(f"Nearest Obstacle: {stats['corridor_min']:.2f} m")
    print(f"Inference Latency: {stats['latency_ms']:.1f} ms")
"""

import os
import sys
import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth


class YOLODepth:
    """
    Ultra-low latency real-time monocular depth estimation engine.
    """

    def __init__(
        self,
        model_path: str = "yolo26n-depth.pt",
        imgsz: int = 448,
        temporal_alpha: float = 0.70,
        scale_factor: float = 1.0,
        offset_meters: float = 0.0,
        warmup: bool = True,
    ):
        """
        Args:
            model_path: Path or name of YOLO depth model ('yolo26n-depth.pt' or 'yolo26s-depth.pt').
            imgsz: Input resolution for inference (384/448 for low latency CPU, 640/768 for high res).
            temporal_alpha: Exponential moving average smoothing (0.0=off, 0.7=balanced, 1.0=instant).
            scale_factor: Multiplier for physical metric calibration.
            offset_meters: Additive offset in meters for physical metric calibration.
            warmup: If True, performs dummy inference upon init to eliminate first-call delay.
        """
        self.imgsz = imgsz
        self.temporal_alpha = temporal_alpha
        self.scale_factor = scale_factor
        self.offset_meters = offset_meters
        self._smoothed_depth: Optional[np.ndarray] = None

        resolved_path = self._resolve_model_path(model_path)
        self.model = YOLO(resolved_path)

        if warmup:
            self._warmup()

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        candidates = [
            model_path,
            os.path.join(os.path.dirname(__file__), model_path),
            os.path.join(r"D:\Hackthon", model_path),
            os.path.join(r"D:\Hackthon\YOLO", model_path),
            "yolo26n-depth.pt",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return model_path

    def _warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.imgsz, verbose=False)

    def get_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculates raw 2D metric depth map in meters (matching input frame resolution).
        """
        h, w = frame.shape[:2]
        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        res = results[0]

        if res.depth is None:
            return np.zeros((h, w), dtype=np.float32)

        depth = res.depth.data.cpu().numpy().astype(np.float32)
        depth = np.squeeze(depth)

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.scale_factor != 1.0 or self.offset_meters != 0.0:
            depth = depth * self.scale_factor + self.offset_meters

        if self.temporal_alpha < 1.0:
            if self._smoothed_depth is None or self._smoothed_depth.shape != depth.shape:
                self._smoothed_depth = depth.copy()
            else:
                self._smoothed_depth = (
                    self.temporal_alpha * depth
                    + (1.0 - self.temporal_alpha) * self._smoothed_depth
                )
            return self._smoothed_depth

        return depth

    def get_distance_stats(self, depth: np.ndarray) -> Dict[str, float]:
        """
        Computes robust distance telemetry from a 2D depth map:
        - center_depth: Target distance in meters at center crosshair.
        - corridor_min: Closest obstacle in the forward walking corridor.
        - min_depth: Overall closest surface in the camera frame.
        """
        h, w = depth.shape
        cx, cy = w // 2, h // 2

        # 1. Target center inlier distance (20px radius around center)
        r = 20
        c_region = depth[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
        c_valid = c_region[c_region > 0]
        if c_valid.size > 0:
            p_lo, p_hi = np.percentile(c_valid, (15, 85))
            inliers = c_valid[(c_valid >= p_lo) & (c_valid <= p_hi)]
            center_depth = float(np.mean(inliers)) if inliers.size > 0 else float(np.median(c_valid))
        else:
            center_depth = float(depth[cy, cx])

        # 2. Forward corridor closest obstacle (middle 50% width, lower 65% height)
        corridor = depth[int(h * 0.35):h, int(w * 0.25):int(w * 0.75)]
        corr_valid = corridor[corridor > 0]
        corridor_min = float(np.percentile(corr_valid, 3)) if corr_valid.size > 0 else center_depth

        # 3. Overall minimum valid depth
        all_valid = depth[depth > 0]
        min_depth = float(np.percentile(all_valid, 1)) if all_valid.size > 0 else center_depth

        return {
            "center_depth": round(center_depth, 2),
            "corridor_min": round(corridor_min, 2),
            "min_depth": round(min_depth, 2),
        }

    def colorize(
        self,
        depth: np.ndarray,
        cmap: str = "inferno",
        mode: str = "disparity",
    ) -> np.ndarray:
        """
        Converts float depth map (in meters) to high-contrast BGR image (uint8).
        """
        return colorize_depth(depth, cmap=cmap, mode=mode)

    def estimate(
        self,
        frame: np.ndarray,
        return_color: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        """
        Unified single-call method:
        Returns: (depth_map, stats_dict, optional_color_image)
        """
        t0 = time.time()
        depth = self.get_depth(frame)
        latency_ms = (time.time() - t0) * 1000

        stats = self.get_distance_stats(depth)
        stats["latency_ms"] = round(latency_ms, 1)
        stats["fps"] = round(1000.0 / max(latency_ms, 0.001), 1)

        color_img = self.colorize(depth) if return_color else None
        return depth, stats, color_img

    def reset_smoothing(self):
        """Clears temporal buffer (e.g. after camera switch or sudden scene cut)."""
        self._smoothed_depth = None


# ============================================================
# STANDALONE TEST RUNNER
# ============================================================

def main():
    """Runs a fast live camera test loop when executed directly."""
    print("=" * 60)
    print("    YOLO1 LOW-LATENCY DEPTH MODULE - TEST MODE")
    print("=" * 60)

    # Initialize low-latency depth engine
    engine = YOLODepth(model_path="yolo26n-depth.pt", imgsz=448)
    print("Engine ready. Opening camera 0...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera running. Press 'q' or ESC in preview window to exit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Core module call: Fast real-time estimation
            depth, stats, depth_color = engine.estimate(frame, return_color=True)

            # Draw telemetry HUD
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.drawMarker(depth_color, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            text1 = f"Center: {stats['center_depth']:.2f}m | Obstacle: {stats['corridor_min']:.2f}m"
            text2 = f"Latency: {stats['latency_ms']:.1f}ms ({stats['fps']:.0f} FPS)"

            cv2.rectangle(depth_color, (10, 10), (450, 75), (0, 0, 0), -1)
            cv2.putText(depth_color, text1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(depth_color, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

            cv2.imshow("YOLO1 Real-Time Depth (Press Q to quit)", depth_color)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
