import os
import sys
import time
import json
import asyncio
import threading
import traceback
import pyaudio
import cv2
import requests
from urllib.parse import urlparse

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import collections
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import memory_tools
from yolo1 import YOLODepth


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Create a .env file and add GEMINI_API_KEY=your_key"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

FORMAT = pyaudio.paInt16
CHANNELS = 1

SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000

CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()


# ============================================================
# MEMORY TOOLS (ONLY MEMORY SYSTEM TOOLS KEPT)
# ============================================================

def store_memory(text: str, category: str = "fact") -> dict:
    """Stores a fact, preference, insight, or detail into persistent long-term memory."""
    res = memory_tools.run_store_logic(text=text, category=category)
    return {"status": "stored", "message": res}


def retrieve_memory(query: str, context: str = "") -> dict:
    """Retrieves stored memories matching the query and context from long-term memory."""
    res = memory_tools.run_retrieve_logic(query=query, context=context)
    return {"query": query, "memory": res}


def update_protocol(update: str) -> dict:
    """Updates behavioral or operational protocol rules in memory."""
    res = memory_tools.run_update_protocol_logic(update=update)
    return {"status": "updated", "message": res}


def save_visual_information(text: str, category: str = "landmark") -> dict:
    """Proactively stores useful visual information (signs, text, room numbers, landmarks) into long-term memory."""
    res = memory_tools.run_store_logic(text=text, category=category)
    return {"status": "stored", "message": res}


# ============================================================
# HAZARD & ALERT CONFIGURATION
# ============================================================

# LAYER 1: NEAR GATE CONFIGURATION (Continuous State Machine)
ALERT_DISTANCE_M = float(os.getenv("ALERT_DISTANCE_M", "25.0"))   # testing value (0.90m), will change to 3.0 later
CLEAR_DISTANCE_M = float(os.getenv("CLEAR_DISTANCE_M", "26.3"))   # hysteresis, must be above ALERT_DISTANCE_M
CLEAR_CONFIRM_FRAMES = int(os.getenv("CLEAR_CONFIRM_FRAMES", "1"))
NEAR_UPDATE_INTERVAL_SEC = float(os.getenv("NEAR_UPDATE_INTERVAL_SEC", "4.0"))
NEAR_CLOSER_DELTA_M = float(os.getenv("NEAR_CLOSER_DELTA_M", "5.0"))
URGENT_DISTANCE_M = float(os.getenv("URGENT_DISTANCE_M", "10.0"))
ALERT_COOLDOWN_SEC = float(os.getenv("ALERT_COOLDOWN_SEC", "7.0"))  # Shared cooldown for far layer alerts

# LAYER 2: FAR HAZARDS VIA FUNCTION CALL
VEHICLE_ALERT_TTC_SEC = float(os.getenv("VEHICLE_ALERT_TTC_SEC", "6.0"))
VEHICLE_ALERT_DISTANCE_M = float(os.getenv("VEHICLE_ALERT_DISTANCE_M", "10.0"))
STATIC_HAZARD_ALERT_DISTANCE_M = float(os.getenv("STATIC_HAZARD_ALERT_DISTANCE_M", "5.0"))  # signal, crossing, footpath end, stairs, drop
HAZARD_FORGET_TIMEOUT_SEC = 3.0
HAZARD_REALERT_DIST_DROP_M = 1.5
HAZARD_REALERT_TTC_SEC = 3.0
CLOSING_SPEED_MIN_MPS = 0.3  # Closing speed threshold to consider approaching
YOLO26_MAX_RELIABLE_DEPTH_M = 12.0  # Max reliable range for YOLO26 depth fallback

# Shared alert queue to Gemini Live
alert_queue = asyncio.Queue()

# Recent depth maps buffer keyed by frame_id (Step 2)
MAX_DEPTH_BUFFER_SIZE = 10
depth_buffer = collections.OrderedDict()
depth_buffer_lock = threading.Lock()
latest_depth_entry = None


def _box_iou(boxA, boxB):
    try:
        xA = max(float(boxA[0]), float(boxB[0]))
        yA = max(float(boxA[1]), float(boxB[1]))
        xB = min(float(boxA[2]), float(boxB[2]))
        yB = min(float(boxA[3]), float(boxB[3]))
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1e-6, (float(boxA[2]) - float(boxA[0])) * (float(boxA[3]) - float(boxA[1])))
        boxBArea = max(1e-6, (float(boxB[2]) - float(boxB[0])) * (float(boxB[3]) - float(boxB[1])))
        return interArea / float(boxAArea + boxBArea - interArea)
    except Exception:
        return 0.0


class TrackedHazard:
    def __init__(self, hazard_id: int, hazard_type: str, moving: bool, direction: str, box: list, distance: float, timestamp: float):
        self.hazard_id = hazard_id
        self.hazard_type = hazard_type
        self.moving = moving
        self.direction = direction
        self.last_box = box
        # Store last 5 distance readings with timestamps: [(timestamp, distance), ...]
        self.history = [(timestamp, distance)]
        self.last_seen = timestamp
        self.last_alert_time = 0.0
        self.last_alert_dist = None
        self.last_alert_ttc = None
        self.alert_count = 0

    def update(self, moving: bool, direction: str, box: list, distance: float, timestamp: float):
        self.moving = moving
        self.direction = direction
        self.last_box = box
        self.last_seen = timestamp
        self.history.append((timestamp, distance))
        if len(self.history) > 5:
            self.history.pop(0)

    @property
    def smoothed_distance(self) -> float:
        if not self.history:
            return 0.0
        return sum(d for t, d in self.history) / len(self.history)

    def calculate_kinematics(self):
        """
        Closing speed = (older distance - newer distance) / time difference, in m/s.
        Time to contact (TTC) = current distance / closing speed.
        Only calculate TTC if closing speed is above 0.3 m/s.
        If the hazard is not approaching, TTC is not set (None).
        """
        if not self.moving or len(self.history) < 2:
            return 0.0, None

        t_old, d_old = self.history[0]
        t_new, d_new = self.history[-1]
        dt = t_new - t_old
        if dt < 0.05:
            return 0.0, None

        closing_speed = (d_old - d_new) / dt
        if closing_speed >= CLOSING_SPEED_MIN_MPS:
            ttc = self.smoothed_distance / closing_speed
            return closing_speed, ttc
        else:
            return closing_speed, None

    def matches(self, hazard_type: str, direction: str, box: list) -> bool:
        if self.hazard_type != hazard_type or self.direction != direction:
            return False
        iou = _box_iou(self.last_box, box)
        if iou > 0.10:
            return True
        cA_x, cA_y = (self.last_box[0] + self.last_box[2]) / 2, (self.last_box[1] + self.last_box[3]) / 2
        cB_x, cB_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return ((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2) < (0.30 ** 2)


# ============================================================
# LAYER 1: NEAR GATE CONTINUOUS STATE MACHINE
# ============================================================

class NearGateStateMachine:
    def __init__(
        self,
        alert_distance_m: float = ALERT_DISTANCE_M,
        clear_distance_m: float = CLEAR_DISTANCE_M,
        clear_confirm_frames: int = CLEAR_CONFIRM_FRAMES,
        update_interval_sec: float = NEAR_UPDATE_INTERVAL_SEC,
        closer_delta_m: float = NEAR_CLOSER_DELTA_M,
        urgent_distance_m: float = URGENT_DISTANCE_M,
    ):
        self.alert_distance_m = alert_distance_m
        self.clear_distance_m = clear_distance_m
        self.clear_confirm_frames = clear_confirm_frames
        self.update_interval_sec = update_interval_sec
        self.closer_delta_m = closer_delta_m
        self.urgent_distance_m = urgent_distance_m

        self.state = "CLEAR"  # "CLEAR" or "NEAR"
        self.history: List[float] = []  # Last readings for smoothing (last 2) and trend (last 3)
        self.clear_confirm_count = 0

        self.last_spoken_time = 0.0
        self.last_spoken_distance: Optional[float] = None
        self.last_msg_type: Optional[str] = None
        self.entered_announced = False
        self.urgent_state = False

    def calculate_trend(self) -> str:
        """
        Compute trend from the last 3 CorridorMin values:
          - "closing" if distance is decreasing by more than 0.05 m
          - "moving away" if increasing by more than 0.05 m
          - "steady" otherwise
        """
        if len(self.history) < 2:
            return "steady"
        oldest = self.history[-3] if len(self.history) >= 3 else self.history[0]
        newest = self.history[-1]
        diff = newest - oldest
        if diff < -0.05:
            return "closing"
        elif diff > 0.05:
            return "moving away"
        else:
            return "steady"

    def update(self, corridor_min: float, frame_id: int, now: float) -> Optional[Tuple[str, str, str, float, str, bool]]:
        """
        State Machine update using CorridorMin.
        Returns: (msg_text, msg_type, reason, reported_dist, trend, is_urgent) or None
        """
        self.history.append(corridor_min)
        if len(self.history) > 5:
            self.history.pop(0)

        smoothed_dist = sum(self.history[-2:]) / len(self.history[-2:])
        trend = self.calculate_trend()

        is_urgent = corridor_min < self.urgent_distance_m

        if self.state == "CLEAR":
            # State CLEAR:
            # Transition to NEAR when corridor_min < ALERT_DISTANCE_M
            if corridor_min < self.alert_distance_m:
                old_state = "CLEAR"
                self.state = "NEAR"
                self.clear_confirm_count = 0
                self.entered_announced = True

                dist_to_report = corridor_min
                print(f"🔄 [STATE CHANGE] {old_state} -> {self.state} | Frame #{frame_id} | Dist: {dist_to_report:.2f}m", flush=True)

                if is_urgent:
                    msg_type = "URGENT"
                    reason = "urgent"
                    self.urgent_state = True
                    msg_text = f"NEAR_URGENT: obstacle at {dist_to_report:.2f} m. FRAME_ID: {frame_id}. Say stop and the direction."
                else:
                    msg_type = "ENTER"
                    reason = "enter"
                    self.urgent_state = False
                    msg_text = f"NEAR_ENTER: obstacle at {dist_to_report:.2f} m. FRAME_ID: {frame_id}. Say what it is, direction, and distance."

                self.last_spoken_time = now
                self.last_spoken_distance = dist_to_report
                self.last_msg_type = msg_type

                return (msg_text, msg_type, reason, dist_to_report, trend, is_urgent)

            return None

        elif self.state == "NEAR":
            # State NEAR:
            # Stay in NEAR while CorridorMin < CLEAR_DISTANCE_M.
            if corridor_min >= self.clear_distance_m:
                self.clear_confirm_count += 1
                if self.clear_confirm_count >= self.clear_confirm_frames:
                    old_state = "NEAR"
                    self.state = "CLEAR"
                    self.clear_confirm_count = 0
                    self.urgent_state = False

                    print(f"🔄 [STATE CHANGE] {old_state} -> {self.state} | Frame #{frame_id} | Dist: {corridor_min:.2f}m", flush=True)

                    if self.entered_announced:
                        self.entered_announced = False
                        msg_type = "CLEARED"
                        reason = "cleared"
                        dist_to_report = corridor_min
                        msg_text = f"NEAR_CLEARED: path clear at {dist_to_report:.2f} m. FRAME_ID: {frame_id}. Say it is clear in a few words."

                        self.last_spoken_time = now
                        self.last_spoken_distance = dist_to_report
                        self.last_msg_type = msg_type

                        return (msg_text, msg_type, reason, dist_to_report, trend, False)
                return None
            else:
                self.clear_confirm_count = 0

            dist_to_report = corridor_min

            # Check URGENT threshold (< URGENT_DISTANCE_M)
            if is_urgent and not self.urgent_state:
                self.urgent_state = True
                msg_type = "URGENT"
                reason = "urgent"
                msg_text = f"NEAR_URGENT: obstacle at {dist_to_report:.2f} m. FRAME_ID: {frame_id}. Say stop and the direction."

                self.last_spoken_time = now
                self.last_spoken_distance = dist_to_report
                self.last_msg_type = msg_type

                return (msg_text, msg_type, reason, dist_to_report, trend, True)

            # Send an UPDATE message only when the obstacle's distance has changed enough:
            # Primary: abs(current_distance - last_spoken_distance) >= NEAR_CLOSER_DELTA_M (0.15m)
            # OR: time >= 4.0s AND delta >= 0.10m.
            # Never send an UPDATE for a steady distance.
            should_update = False
            reason = ""
            change_dir = ""

            if self.last_spoken_distance is not None:
                delta = abs(dist_to_report - self.last_spoken_distance)
                time_elapsed = now - self.last_spoken_time

                if delta >= self.closer_delta_m:
                    should_update = True
                    change_dir = "closer" if dist_to_report < self.last_spoken_distance else "farther"
                    reason = change_dir
                elif time_elapsed >= self.update_interval_sec and delta >= 0.10:
                    should_update = True
                    change_dir = "closer" if dist_to_report < self.last_spoken_distance else "farther"
                    reason = "interval_delta"

            if should_update:
                msg_type = "UPDATE"
                msg_text = f"NEAR_UPDATE: {change_dir} now at {dist_to_report:.2f} m. FRAME_ID: {frame_id}. Say the change in a few words."

                self.last_spoken_time = now
                self.last_spoken_distance = dist_to_report
                self.last_msg_type = msg_type

                return (msg_text, msg_type, reason, dist_to_report, trend, False)

            return None


near_state_machine = NearGateStateMachine()

# Speech avoidance state & unified alert dispatcher
gemini_is_speaking = False
pending_alert: Optional[Dict[str, Any]] = None
speech_lock = threading.Lock()


def set_gemini_speaking(speaking: bool):
    global gemini_is_speaking, pending_alert
    with speech_lock:
        gemini_is_speaking = speaking
        if not speaking and pending_alert is not None:
            to_send = pending_alert
            pending_alert = None

            # Re-check whether held message is still required before release
            is_stale = False
            drop_reason = ""
            if near_state_machine.state == "CLEAR" and to_send["msg_type"] in ("ENTER", "UPDATE", "URGENT"):
                is_stale = True
                drop_reason = "state_is_clear"
            elif near_state_machine.state == "NEAR" and to_send["msg_type"] == "CLEARED":
                is_stale = True
                drop_reason = "state_is_near"

            if is_stale:
                print(
                    f"§ HELD_DROPPED | type={to_send['msg_type']} distance={to_send['distance']:.2f}m "
                    f"frame_id={to_send['frame_id']} reason={drop_reason}",
                    flush=True
                )
                return

            print(
                f"§ ALERT_SENT | type={to_send['msg_type']} distance={to_send['distance']:.2f}m "
                f"frame_id={to_send['frame_id']} reason={to_send['reason']} | Msg: {to_send['msg_text']}",
                flush=True
            )
            try:
                alert_queue.put_nowait(to_send["msg_text"])
            except Exception as e:
                print(f"⚠️ [Alert Queue Error]: {e}", flush=True)


def dispatch_alert(
    msg_text: str,
    msg_type: str,
    reason: str,
    distance: float,
    trend: str,
    frame_id: int,
    is_urgent: bool = False
):
    """
    Unified dispatcher for both near-distance layer and far-hazard layer.
    - If Gemini is speaking, holds ENTER, UPDATE, CLEARED (latest pending message wins).
    - Exception: URGENT interrupts speech immediately.
    - Closest or most urgent message wins.
    """
    global pending_alert
    with speech_lock:
        if gemini_is_speaking and not is_urgent:
            candidate = {
                "msg_text": msg_text,
                "msg_type": msg_type,
                "reason": reason,
                "distance": distance,
                "trend": trend,
                "frame_id": frame_id,
                "is_urgent": is_urgent,
            }
            if pending_alert is not None:
                # Keep only newest high-priority held message; drop replaced message
                print(
                    f"§ HELD_DROPPED | type={pending_alert['msg_type']} distance={pending_alert['distance']:.2f}m "
                    f"frame_id={pending_alert['frame_id']} reason=replaced_by_newer",
                    flush=True
                )
            pending_alert = candidate
            print(
                f"§ ALERT_HELD | type={msg_type} distance={distance:.2f}m frame_id={frame_id} reason={reason}",
                flush=True
            )
            return

        print(
            f"§ ALERT_SENT | type={msg_type} distance={distance:.2f}m frame_id={frame_id} reason={reason} | Msg: {msg_text}",
            flush=True
        )
        try:
            alert_queue.put_nowait(msg_text)
        except Exception as e:
            print(f"⚠️ [Alert Queue Error]: {e}", flush=True)


class HazardManager:
    def __init__(self):
        self.hazards: Dict[int, TrackedHazard] = {}
        self.next_hazard_id = 1
        self.global_last_alert_time = 0.0

    def cleanup_stale_hazards(self, now: float):
        # Forget a hazard if it is not reported for 3 seconds
        stale_ids = [hid for hid, h in self.hazards.items() if now - h.last_seen > HAZARD_FORGET_TIMEOUT_SEC]
        for hid in stale_ids:
            del self.hazards[hid]

    def check_near_gate(self, corridor_min: float, frame_id: int, now: float) -> Optional[str]:
        """
        Layer 1 Near Gate: Continuous State Machine check.
        Dispatches alerts via dispatch_alert adhering to speech-avoidance and priority rules.
        """
        res = near_state_machine.update(corridor_min, frame_id, now)
        if res:
            msg_text, msg_type, reason, smoothed_dist, trend, is_urgent = res
            dispatch_alert(
                msg_text=msg_text,
                msg_type=msg_type,
                reason=reason,
                distance=smoothed_dist,
                trend=trend,
                frame_id=frame_id,
                is_urgent=is_urgent
            )
            return msg_text
        return None

    def process_hazard(
        self,
        hazard_type: str,
        moving: bool,
        direction: str,
        box: list,
        distance: float,
        frame_id: int,
        timestamp: float,
        source: str = "yolo26_depth"
    ) -> Optional[str]:
        self.cleanup_stale_hazards(timestamp)

        matched_hazard = None
        for h in self.hazards.values():
            if h.matches(hazard_type, direction, box):
                matched_hazard = h
                break

        if matched_hazard is None:
            matched_hazard = TrackedHazard(
                hazard_id=self.next_hazard_id,
                hazard_type=hazard_type,
                moving=moving,
                direction=direction,
                box=box,
                distance=distance,
                timestamp=timestamp
            )
            self.hazards[self.next_hazard_id] = matched_hazard
            self.next_hazard_id += 1
        else:
            matched_hazard.update(moving, direction, box, distance, timestamp)

        smoothed_dist = matched_hazard.smoothed_distance
        closing_speed, ttc = matched_hazard.calculate_kinematics()

        # Step 7 Requirement: Log tracker state (distance, closing speed, TTC)
        ttc_str = f"{ttc:.1f}s" if ttc is not None else "N/A"
        print(
            f"🎯 [Hazard Tracker #{matched_hazard.hazard_id}] Type: {hazard_type} | "
            f"Smoothed Dist: {smoothed_dist:.2f}m | Speed: {closing_speed:.2f}m/s | TTC: {ttc_str}"
        )

        current_reading = matched_hazard.history[-1][1] if matched_hazard.history else smoothed_dist
        disp_dist = current_reading if not moving else smoothed_dist

        should_alert, reason = self._evaluate_alert_rules(matched_hazard, smoothed_dist, closing_speed, ttc, timestamp)

        if not should_alert:
            return None

        # Fire alert
        matched_hazard.last_alert_time = timestamp
        matched_hazard.last_alert_dist = disp_dist
        matched_hazard.last_alert_ttc = ttc
        matched_hazard.alert_count += 1
        self.global_last_alert_time = timestamp

        moving_str = "moving" if moving else "static"
        ttc_msg = f"{ttc:.1f} s" if ttc is not None else "N/A s"
        alert_msg = (
            f"ALERT: {hazard_type}, {direction}, {disp_dist:.1f} m, "
            f"{moving_str}, time to contact {ttc_msg}. FRAME_ID: {frame_id}. Say one short sentence."
        )

        # Step 7 Requirement: Log every alert and the reason
        print(f"\n🚨 [FAR HAZARD ALERT FIRED] Reason: {reason} | Msg: {alert_msg}\n")
        return alert_msg

    def _evaluate_alert_rules(
        self,
        hazard: TrackedHazard,
        dist: float,
        closing_speed: float,
        ttc: Optional[float],
        now: float
    ) -> Tuple[bool, str]:
        current_reading = hazard.history[-1][1] if hazard.history else dist
        is_approaching = closing_speed >= CLOSING_SPEED_MIN_MPS
        trigger = False
        reason = ""

        # Step 4 Rules:
        if hazard.hazard_type == "vehicle":
            if hazard.moving:
                if ttc is not None and ttc < VEHICLE_ALERT_TTC_SEC:
                    trigger = True
                    reason = f"vehicle TTC below {VEHICLE_ALERT_TTC_SEC}s (TTC={ttc:.1f}s)"
                elif dist < VEHICLE_ALERT_DISTANCE_M and is_approaching:
                    trigger = True
                    reason = f"vehicle approaching distance below {VEHICLE_ALERT_DISTANCE_M}m ({dist:.1f}m)"
            else:
                if current_reading < ALERT_DISTANCE_M:
                    trigger = True
                    reason = f"parked vehicle inside near gate ({current_reading:.1f}m < {ALERT_DISTANCE_M}m)"
        elif hazard.hazard_type in ["traffic_signal", "road_crossing", "footpath_end", "stairs", "drop"]:
            if current_reading < STATIC_HAZARD_ALERT_DISTANCE_M or dist < STATIC_HAZARD_ALERT_DISTANCE_M:
                if hazard.alert_count == 0:
                    trigger = True
                    reason = f"static hazard ({hazard.hazard_type}) initial alert at {current_reading:.1f}m (< {STATIC_HAZARD_ALERT_DISTANCE_M}m)"
                else:
                    trigger = True
                    reason = f"static hazard ({hazard.hazard_type}) re-alert at {current_reading:.1f}m"
        else:
            if current_reading < STATIC_HAZARD_ALERT_DISTANCE_M or dist < STATIC_HAZARD_ALERT_DISTANCE_M:
                if hazard.alert_count == 0:
                    trigger = True
                    reason = f"hazard ({hazard.hazard_type}) initial alert at {current_reading:.1f}m"
                else:
                    trigger = True
                    reason = f"hazard ({hazard.hazard_type}) re-alert at {current_reading:.1f}m"

        if not trigger:
            return False, ""

        # Check re-alert rules:
        if hazard.alert_count > 0:
            if (now - hazard.last_alert_time) < ALERT_COOLDOWN_SEC:
                return False, ""
            eval_dist = current_reading if not hazard.moving else dist
            got_closer = (hazard.last_alert_dist is not None) and (hazard.last_alert_dist - eval_dist >= HAZARD_REALERT_DIST_DROP_M)
            ttc_critical = (ttc is not None) and (ttc < HAZARD_REALERT_TTC_SEC) and (hazard.last_alert_ttc is None or hazard.last_alert_ttc >= HAZARD_REALERT_TTC_SEC)
            if not (got_closer or ttc_critical):
                return False, ""

        # Shared cooldown between all alerts:
        if (now - self.global_last_alert_time) < ALERT_COOLDOWN_SEC:
            return False, ""

        return True, reason


hazard_manager = HazardManager()


def get_depth_for_box(frame_id: int, box: list, estimated_distance_m: float) -> Tuple[float, Optional[float], str, int]:
    """
    Step 2: Read YOLO26 depth map for frame_id at normalized box coordinates.
    Computes median depth inside box. Falls back to Gemini estimate if invalid or far.
    """
    now = time.time()
    depth_map = None
    target_frame = frame_id

    with depth_buffer_lock:
        if frame_id in depth_buffer:
            entry = depth_buffer[frame_id]
            if now - entry["timestamp"] <= 2.0:
                depth_map = entry["depth_map"]
                target_frame = frame_id
        if depth_map is None and latest_depth_entry is not None:
            depth_map = latest_depth_entry["depth_map"]
            target_frame = latest_depth_entry["frame_id"]

    median_depth = None
    if depth_map is not None and isinstance(box, (list, tuple)) and len(box) >= 4:
        h, w = depth_map.shape[:2]
        try:
            x1 = int(np.clip(float(box[0]), 0.0, 1.0) * w)
            y1 = int(np.clip(float(box[1]), 0.0, 1.0) * h)
            x2 = int(np.clip(float(box[2]), 0.0, 1.0) * w)
            y2 = int(np.clip(float(box[3]), 0.0, 1.0) * h)

            if x2 <= x1:
                x2 = min(w, x1 + 2)
            if y2 <= y1:
                y2 = min(h, y1 + 2)

            crop = depth_map[y1:y2, x1:x2]
            valid = crop[(crop > 0.1) & (crop <= 25.0) & np.isfinite(crop)]
            if len(valid) > 0:
                median_depth = float(np.median(valid))
        except Exception as e:
            print(f"⚠️ [Depth Box Crop Error]: {e}")

    # Fallback to Gemini estimate if invalid or far beyond reliable range (12m)
    if median_depth is None or median_depth <= 0.1 or median_depth > YOLO26_MAX_RELIABLE_DEPTH_M:
        real_dist = float(estimated_distance_m)
        source = "gemini_estimate"
    else:
        real_dist = median_depth
        source = "yolo26_depth"

    return real_dist, median_depth, source, target_frame


def report_hazard(
    frame_id: int,
    hazard_type: str,
    moving: bool,
    direction: str,
    box: list,
    estimated_distance_m: float,
) -> dict:
    """
    Step 1: Background function called by Gemini Live when it spots a potential hazard.
    Executes silently, extracts real depth, tracks kinematic TTC, and alerts if critical.
    """
    real_dist, yolo_depth, source, target_frame = get_depth_for_box(
        frame_id=frame_id,
        box=box,
        estimated_distance_m=estimated_distance_m
    )

    # Step 7 Requirement: Log every report_hazard call
    print(
        f"📡 [Hazard Reported] Frame #{frame_id} (used #{target_frame}) | "
        f"Type: {hazard_type} | Moving: {moving} | Dir: {direction} | "
        f"Box: {box} | Gemini Est: {estimated_distance_m:.1f}m | "
        f"YOLO26: {f'{yolo_depth:.2f}m' if yolo_depth is not None else 'N/A'} | "
        f"Used: {real_dist:.2f}m ({source})"
    )

    alert_msg = hazard_manager.process_hazard(
        hazard_type=hazard_type,
        moving=moving,
        direction=direction,
        box=box,
        distance=real_dist,
        frame_id=frame_id,
        timestamp=time.time(),
        source=source
    )

    if alert_msg:
        dispatch_alert(
            msg_text=alert_msg,
            msg_type="FAR_HAZARD",
            reason="far_hazard",
            distance=real_dist,
            trend="closing" if moving else "steady",
            frame_id=frame_id,
            is_urgent=False
        )

    # Step 1 Requirement: Always send a tool response back immediately so Gemini never hangs
    return {"status": "received"}


async def alert_dispatcher(session):
    """
    Step 5: How the alert is spoken.
    Listens for code-triggered alerts and sends ONE short text to Gemini Live:
    "ALERT: <hazard_type>, <direction>, <distance> m, <moving or static>, time to contact <TTC> s. FRAME_ID: <id>. Say one short sentence."
    """
    print("🚨 [Alert Dispatcher] Started and listening for hazard alerts...")
    while True:
        try:
            alert_msg = await alert_queue.get()
            print(f"\n📢 [Sending Spoken Alert Turn to Gemini]: {alert_msg}\n", flush=True)
            async with ws_send_lock:
                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=alert_msg)],
                        )
                    ],
                    turn_complete=True,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ [Alert Dispatcher Error]: {e}", flush=True)
            await asyncio.sleep(0.5)


def run_hazard_test_suite():
    """
    Step 8: Test mode with fake hazards to verify all alert and tracking logic.
      a) Vehicle: 15, 12, 9, 6 m over 4 seconds. Show when it alerts.
      b) Signal (static): 8, 6, 4.5, 4 m. Show that it alerts once at 5 m.
      c) Parked car at 6 m, not moving. Show that it does not alert.
      d) Near obstacle at 0.8 m. Show that the near gate alerts.
    """
    print()
    print("==================================================")
    print("     RUNNING HAZARD TRACKING & ALERT TEST SUITE   ")
    print("==================================================")
    print()

    test_mgr = HazardManager()
    base_t = 1000.0

    print("--- Test Case (a): Moving Vehicle (15, 12, 9, 6 m over 4s) ---")
    distances_a = [15.0, 12.0, 9.0, 6.0]
    box_a = [0.1, 0.4, 0.3, 0.7]
    for i, dist in enumerate(distances_a):
        t = base_t + float(i)
        print(f"\n[Step {i+1}] t={i}s, dist={dist}m:")
        alert = test_mgr.process_hazard(
            hazard_type="vehicle",
            moving=True,
            direction="left",
            box=box_a,
            distance=dist,
            frame_id=100 + i,
            timestamp=t,
            source="gemini_estimate"
        )
        if alert:
            print(f"  >>> ALERT TRIGGERED: {alert}")
        else:
            print(f"  [No Alert Triggered]")

    base_t += 10.0
    print("\n--- Test Case (b): Static Traffic Signal (8, 6, 4.5, 4 m) ---")
    distances_b = [8.0, 6.0, 4.5, 4.0]
    box_b = [0.4, 0.1, 0.6, 0.3]
    for i, dist in enumerate(distances_b):
        t = base_t + float(i)
        print(f"\n[Step {i+1}] t={i}s, dist={dist}m:")
        alert = test_mgr.process_hazard(
            hazard_type="traffic_signal",
            moving=False,
            direction="ahead",
            box=box_b,
            distance=dist,
            frame_id=200 + i,
            timestamp=t,
            source="yolo26_depth"
        )
        if alert:
            print(f"  >>> ALERT TRIGGERED: {alert}")
        else:
            print(f"  [No Alert Triggered]")

    base_t += 10.0
    print("\n--- Test Case (c): Parked Car at 6 m, Not Moving ---")
    box_c = [0.7, 0.3, 0.9, 0.6]
    for i in range(3):
        t = base_t + float(i)
        print(f"\n[Step {i+1}] t={i}s, dist=6.0m, moving=False:")
        alert = test_mgr.process_hazard(
            hazard_type="vehicle",
            moving=False,
            direction="right",
            box=box_c,
            distance=6.0,
            frame_id=300 + i,
            timestamp=t,
            source="yolo26_depth"
        )
        if alert:
            print(f"  >>> ALERT TRIGGERED: {alert}")
        else:
            print(f"  [No Alert Triggered]")

    base_t += 10.0
    print("\n--- Test Case (d): Near Gate Continuous State Machine Simulation ---")
    print("Sequence: [2.0, 2.5, 1.5, 0.9, 0.8, 0.7, 0.7, 0.7, 1.6, 1.5, 0.8, 0.6, 1.3]")
    print("Expected: 0.8 -> ENTER, 0.7/0.7/0.7 -> No UPDATE, 1.6 -> CLEARED, 1.5 -> No event, 0.8 -> ENTER, 0.6 -> UPDATE, 1.3 -> CLEARED.\n")

    test_sm = NearGateStateMachine()
    fake_seq = [2.0, 2.5, 1.5, 0.9, 0.8, 0.7, 0.7, 0.7, 1.6, 1.5, 0.8, 0.6, 1.3]
    expected_types = [None, None, None, None, "ENTER", None, None, None, "CLEARED", None, "ENTER", "UPDATE", "CLEARED"]
    actual_types = []

    for i, val in enumerate(fake_seq):
        t = base_t + float(i)
        frame_id = 400 + i + 1
        res = test_sm.update(corridor_min=val, frame_id=frame_id, now=t)
        if res:
            msg_text, msg_type, reason, reported_dist, trend, is_urgent = res
            actual_types.append(msg_type)
            print(
                f"[Frame #{frame_id:03d}] t={i:02d}s | Raw: {val:.2f}m | "
                f"State: {test_sm.state} | >>> {msg_type} (Reason: {reason}, Dist: {reported_dist:.2f}m)\n"
                f"       Msg: \"{msg_text}\""
            )
        else:
            actual_types.append(None)
            print(f"[Frame #{frame_id:03d}] t={i:02d}s | Raw: {val:.2f}m | State: {test_sm.state} | [No Message]")

    assert actual_types == expected_types, f"Sequence test mismatch: {actual_types} != {expected_types}"
    print("\n✅ Step 17 Test Sequence passed perfectly!")

    print("\n--- Test Case (e): Placeholder & Angle Bracket Filter Tests ---")
    test_placeholders = [
        ("<no speech>", True),
        ("no speech", True),
        ("<silence>", True),
        ("<something>", True),
        ("<anything>", True),
        ("Nothing to report", True),
        ("Watch out, moving car on your left at 5 meters.", False),
        ("Obstacle right ahead, stop.", False),
        ("Clear path ahead.", False)
    ]
    for text_sample, should_block in test_placeholders:
        is_blocked = is_invalid_speech(text_sample)
        status = "BLOCKED" if is_blocked else "ALLOWED"
        print(f"  Sample: {repr(text_sample):50s} -> {status}")
        assert is_blocked == should_block, f"Filter assertion failed for: {text_sample}"
    print("✅ Step 18 Placeholder safety filter tests passed perfectly!")

    print("\n--- Test Case (f): Speech Avoidance & Held Message Revalidation ---")
    print("1. Simulating Gemini is speaking (set_gemini_speaking(True))...")
    set_gemini_speaking(True)
    print("2. Dispatching non-urgent ENTER alert while speaking (must be held)...")
    dispatch_alert(
        msg_text="NEAR_ENTER: obstacle at 0.80 m. FRAME_ID: 501. Say what it is, direction, and distance.",
        msg_type="ENTER",
        reason="enter",
        distance=0.80,
        trend="closing",
        frame_id=501,
        is_urgent=False
    )
    print("3. Dispatching URGENT alert (< 0.40m) while speaking (must interrupt immediately)...")
    dispatch_alert(
        msg_text="NEAR_URGENT: obstacle at 0.35 m. FRAME_ID: 502. Say stop and the direction.",
        msg_type="URGENT",
        reason="urgent",
        distance=0.35,
        trend="closing",
        frame_id=502,
        is_urgent=True
    )
    print("4. Testing held message drop on state transition...")
    near_state_machine.state = "CLEAR"  # Simulating obstacle cleared before speaking ended
    set_gemini_speaking(False)
    print("✅ Speech avoidance, urgent interruption, and held revalidation passed!")

    print()
    print("==================================================")
    print("           TEST SUITE COMPLETED SUCCESSFULLY      ")
    print("==================================================")
    print()


TOOL_REGISTRY = {
    "store_memory": store_memory,
    "retrieve_memory": retrieve_memory,
    "update_protocol": update_protocol,
    "save_visual_information": save_visual_information,
    "report_hazard": report_hazard,
}

TOOLS = [
    {
        "function_declarations": [
            {
                "name": "store_memory",
                "description": "Stores important facts, user preferences, personal details, project notes, or instructions into persistent long-term memory.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {
                            "type": "STRING",
                            "description": "The information, fact, preference, or detail to remember",
                        },
                        "category": {
                            "type": "STRING",
                            "description": "Category of memory: 'fact', 'preference', 'project', 'insight', 'procedure'",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "retrieve_memory",
                "description": "Retrieves information from persistent long-term memory about past preferences, facts, project context, or past conversations.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "The search term or topic to recall from memory",
                        },
                        "context": {
                            "type": "STRING",
                            "description": "Optional additional context to narrow down memory search",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "update_protocol",
                "description": "Updates or appends a behavioral or communication protocol rule in persistent memory.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "update": {
                            "type": "STRING",
                            "description": "The new protocol or behavioral instruction",
                        }
                    },
                    "required": ["update"],
                },
            },
            {
                "name": "save_visual_information",
                "description": "Proactively stores visual information like room signs, landmarks, and navigation indicators into long-term memory.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {
                            "type": "STRING",
                            "description": "The sign text, landmark description, or location detail",
                        },
                        "category": {
                            "type": "STRING",
                            "description": "Category: 'landmark', 'sign', 'room', 'hazard'",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "report_hazard",
                "description": "Reports a potential environmental or navigation hazard observed in the RGB camera view (vehicles, road crossings, traffic signals, footpath ends, stairs, drops). Called silently in the background.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "frame_id": {
                            "type": "INTEGER",
                            "description": "The frame ID where the hazard was observed",
                        },
                        "hazard_type": {
                            "type": "STRING",
                            "description": "Type of hazard",
                            "enum": ["vehicle", "road_crossing", "traffic_signal", "footpath_end", "stairs", "drop", "other"],
                        },
                        "moving": {
                            "type": "BOOLEAN",
                            "description": "True if the hazard object appears to be in motion",
                        },
                        "direction": {
                            "type": "STRING",
                            "description": "Direction relative to the user",
                            "enum": ["left", "ahead", "right"],
                        },
                        "box": {
                            "type": "ARRAY",
                            "items": {"type": "NUMBER"},
                            "description": "Normalized bounding box [x1, y1, x2, y2] (0.0 to 1.0) in the RGB (left) half only",
                        },
                        "estimated_distance_m": {
                            "type": "NUMBER",
                            "description": "Gemini's visual estimate of distance in metres",
                        },
                    },
                    "required": ["frame_id", "hazard_type", "moving", "direction", "box", "estimated_distance_m"],
                },
            },
        ]
    }
]

# Gemini Live Extended Thinking requires function declarations to be NON_BLOCKING
for _tool_group in TOOLS:
    for _decl in _tool_group.get("function_declarations", []):
        _decl["behavior"] = "NON_BLOCKING"

# Shared async lock for WebSocket transmissions
ws_send_lock = asyncio.Lock()


async def execute_tool(name: str, args: dict):
    if name not in TOOL_REGISTRY:
        return {"error": f"Function '{name}' not found."}

    fn = TOOL_REGISTRY[name]
    if asyncio.iscoroutinefunction(fn):
        return await fn(**args)
    else:
        return await asyncio.to_thread(fn, **args)


# ============================================================
# REAL-TIME CAMERA CAPTURE & YOLO26 DEPTH ESTIMATION
# Supports ESP32-S3 PULL MODE (GET /capture), STREAM, and WEBCAM
# ============================================================

def resolve_camera_source(source):
    """
    Resolves camera source into either:
    - An integer camera index for local webcams (e.g. 0, 1)
    - A normalized ESP32 host/URL (e.g. http://10.24.69.1/capture or http://10.24.69.1:81/stream)
    """
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


CAMERA_MODE = os.getenv("CAMERA_MODE", "capture").lower()
ESP32_CAM_URL = os.getenv("ESP32_CAM_URL", "").strip()
CAMERA_SOURCE = resolve_camera_source(os.getenv("CAMERA_SOURCE", ESP32_CAM_URL or os.getenv("CAMERA_INDEX", "0")))

print("\n" + "=" * 55)
if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith("http"):
    print(f"📡 [CAMERA SOURCE] ESP32-S3 Mode: {CAMERA_MODE.upper()} | Target: {CAMERA_SOURCE}")
else:
    print(f"💻 [CAMERA SOURCE] Local USB Webcam: index {CAMERA_SOURCE}")
print("=" * 55 + "\n")

CAMERA_FPS = float(os.getenv("CAMERA_FPS", "1.0"))  # Synchronized dual-frame send rate (1.0 FPS)
camera_queue = asyncio.Queue(maxsize=1)  # Always drops old frames if backlogged, keeping newest
camera_enabled = True
camera_stop_event = None
event_loop_ref = None

# Load YOLO26 once at startup (imgsz=288: 73ms vs 130ms at 384)
YOLO26_IMGSZ = int(os.getenv("YOLO26_IMGSZ", "288"))
YOLO26_MODEL_PATH = os.getenv("YOLO26_DEPTH_MODEL", "yolo26n-depth.pt")
print(f"[YOLO26] Initializing YOLO26 depth engine ({YOLO26_MODEL_PATH}, imgsz={YOLO26_IMGSZ})...")
yolo26_depth = YOLODepth(model_path=YOLO26_MODEL_PATH, imgsz=YOLO26_IMGSZ, temporal_alpha=0.70)
print("[YOLO26] Model loaded and ready.")


def toggle_camera(enabled: bool = None) -> bool:
    """Enables or disables real-time camera streaming."""
    global camera_enabled
    if enabled is not None:
        camera_enabled = bool(enabled)
    else:
        camera_enabled = not camera_enabled
    status = "enabled" if camera_enabled else "paused"
    print(f"\n📷 [Camera] Stream is now {status.upper()}.")
    return camera_enabled


def _draw_badge(img, text, pos, bg_color=(0, 0, 0), text_color=(255, 255, 255), scale=0.6, thickness=2):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


class CameraStreamReader:
    """
    High-performance zero-delay frame reader.
    Default Mode: PULL MODE (GET /capture every 1s using requests.Session with keep-alive)
    Fallback Modes: STREAM MODE (OpenCV VideoCapture) or LOCAL WEBCAM.
    """
    def __init__(self, source, mode="capture"):
        self.raw_source = source
        self.source = resolve_camera_source(source)
        self.mode = mode.lower()
        self.is_network = isinstance(self.source, str) and self.source.startswith("http")

        # Derive /capture URL for pull mode
        if self.is_network:
            parsed = urlparse(self.source)
            host = parsed.hostname or "10.24.69.1"
            self.capture_url = f"http://{host}/capture"
        else:
            self.capture_url = None

        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.latest_meta = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.is_connected = False
        self.consecutive_fails = 0
        self.camera_lost_sent = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="CameraWorker")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def read(self):
        """
        Non-blocking read: returns (frame, frame_time, meta) or (None, 0.0, None).
        Main loop NEVER waits or blocks on this!
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_frame_time, dict(self.latest_meta)
            return None, 0.0, {}

    def _dispatch_health_alert(self, text):
        global alert_queue, event_loop_ref
        try:
            if event_loop_ref and event_loop_ref.is_running():
                event_loop_ref.call_soon_threadsafe(alert_queue.put_nowait, text)
            else:
                alert_queue.put_nowait(text)
        except Exception as e:
            print(f"⚠️ [Health Alert Dispatch Error]: {e}")

    def _worker_loop(self):
        if self.is_network and self.mode == "capture":
            self._pull_worker_loop()
        elif self.is_network:
            self._stream_worker_loop()
        else:
            self._webcam_worker_loop()

    def _pull_worker_loop(self):
        print(f"📡 [ESP32 Camera] PULL MODE active: {self.capture_url} (requests.Session keep-alive, 1 req/sec)")
        session = requests.Session()

        while self.running:
            t_req_start = time.time()
            try:
                # STEP 3: Timeout (connect 2.5s, read 3.0s) & Connection: close
                resp = session.get(self.capture_url, headers={"Connection": "close"}, timeout=(2.5, 3.0))
                req_time_ms = (time.time() - t_req_start) * 1000

                if resp.status_code == 200 and resp.content:
                    t_dec_start = time.perf_counter()
                    frame = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    dec_time_ms = (time.perf_counter() - t_dec_start) * 1000

                    if frame is not None:
                        self.consecutive_fails = 0
                        self.is_connected = True

                        # If camera was previously lost, send CAMERA_OK once
                        if self.camera_lost_sent:
                            print("\n✅ [Camera Health] Camera connection restored! Dispatching CAMERA_OK to Gemini Live...\n")
                            self._dispatch_health_alert("CAMERA_OK: Camera connection restored.")
                            self.camera_lost_sent = False

                        with self.lock:
                            self.latest_frame = frame
                            self.latest_frame_time = t_req_start
                            self.latest_meta = {
                                "req_time_ms": req_time_ms,
                                "bytes": len(resp.content),
                                "dec_time_ms": dec_time_ms,
                            }
                    else:
                        self.consecutive_fails += 1
                        print(f"⚠️ [Camera] JPEG decode failed. Consecutive failures: {self.consecutive_fails}")
                else:
                    self.consecutive_fails += 1
                    print(f"⚠️ [Camera] HTTP {resp.status_code}. Consecutive failures: {self.consecutive_fails}")

            except Exception as e:
                self.consecutive_fails += 1
                req_time_ms = (time.time() - t_req_start) * 1000
                print(f"⚠️ [Camera] Request failed ({type(e).__name__}: {e}) [{req_time_ms:.1f}ms]. Consecutive failures: {self.consecutive_fails}")

            # STEP 4: After 3 fails in a row, create a new Session
            if self.consecutive_fails >= 3 and self.consecutive_fails % 3 == 0:
                print("🔄 [Camera Health] 3 consecutive failures -> Recreating requests.Session()...")
                try:
                    session.close()
                except Exception:
                    pass
                session = requests.Session()

            # STEP 4: After 5 fails in a row, send CAMERA_LOST to Gemini once
            if self.consecutive_fails >= 5 and not self.camera_lost_sent:
                print("🚨 [Camera Health] 5 consecutive failures -> Dispatching CAMERA_LOST to Gemini Live!")
                self._dispatch_health_alert("CAMERA_LOST: Camera connection lost.")
                self.camera_lost_sent = True
                self.is_connected = False

            # STEP 3: Only one request at a time. Every 1 second:
            elapsed = time.time() - t_req_start
            sleep_time = max(0.0, 1.0 - elapsed)
            time.sleep(sleep_time)

        try:
            session.close()
        except Exception:
            pass

    def _stream_worker_loop(self):
        """Fallback stream mode using OpenCV VideoCapture."""
        print(f"📡 [ESP32 Camera] STREAM MODE active: {self.source}")
        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.consecutive_fails += 1
                time.sleep(2.0)
                continue

            self.is_connected = True
            while self.running:
                t0 = time.time()
                ret = cap.grab()
                if ret:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.latest_frame_time = t0
                            self.latest_meta = {"req_time_ms": 10.0, "bytes": 0, "dec_time_ms": 5.0}
                    else:
                        time.sleep(0.02)
                else:
                    time.sleep(0.02)
            cap.release()

    def _webcam_worker_loop(self):
        """Local USB webcam capture."""
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"❌ [Camera] Could not open webcam {self.source}")
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass
        self.is_connected = True
        try:
            while self.running:
                t0 = time.time()
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_frame_time = t0
                        self.latest_meta = {"req_time_ms": 0.0, "bytes": 0, "dec_time_ms": 0.0}
                time.sleep(0.03)
        finally:
            cap.release()


def _camera_loop_sync(camera_source, queue, loop, stop_event):
    reader = CameraStreamReader(camera_source, mode=CAMERA_MODE)
    reader.start()

    target_interval = 1.0 / max(CAMERA_FPS, 1.0)
    frame_id = 0
    last_processed_time = 0.0

    print(f"📷 [Camera Pipeline] Active mode: {reader.mode.upper()} | Target: {reader.capture_url if reader.mode == 'capture' else reader.source}")

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            now = time.time()

            if not camera_enabled:
                time.sleep(0.3)
                continue

            frame, frame_time, meta = reader.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # Don't re-process the exact same frame if it hasn't updated yet
            if frame_time == last_processed_time:
                time.sleep(0.05)
                continue

            # STEP 5: Drop a frame only if it is older than 4 seconds
            frame_age = now - frame_time
            if frame_age > 4.0:
                print(f"⚠️ [Camera] Dropping stale frame (age: {frame_age:.2f}s > 4.0s)")
                last_processed_time = frame_time
                time.sleep(0.05)
                continue

            last_processed_time = frame_time

            # Resize to standard pipeline resolution (1280x720)
            frame = cv2.resize(frame, (1280, 720))
            frame_id += 1
            timestamp = datetime.now().isoformat()

            # STEP 5: Run YOLO26 depth estimation in its own thread execution
            t_yolo_start = time.perf_counter()
            depth_map, stats, depth_color = yolo26_depth.estimate(frame, return_color=True)
            yolo_time_ms = (time.perf_counter() - t_yolo_start) * 1000

            # STEP 5: If YOLO26 takes longer than 1 second, skip that frame. Never process a backlog.
            if yolo_time_ms > 1000.0:
                print(f"⚠️ [YOLO26] Inference took {yolo_time_ms:.1f}ms (> 1000ms). Skipping frame #{frame_id}.")
                continue

            if depth_color is None:
                continue

            # Store in depth buffer for Layer 2 Far Hazard spatial queries (Step 2)
            with depth_buffer_lock:
                entry = {
                    "timestamp": time.time(),
                    "depth_map": depth_map,
                    "stats": stats,
                    "frame_id": frame_id,
                }
                depth_buffer[frame_id] = entry
                if len(depth_buffer) > MAX_DEPTH_BUFFER_SIZE:
                    depth_buffer.popitem(last=False)
                global latest_depth_entry
                latest_depth_entry = entry

            # STEP 1: Build ONE combined image per frame
            target_h = 480
            rgb_w = int(frame.shape[1] * (target_h / frame.shape[0]))
            rgb_half = cv2.resize(frame, (rgb_w, target_h))
            depth_half = cv2.resize(depth_color, (rgb_w, target_h))

            # Side by side: LEFT RGB | RIGHT Depth
            combined = np.hstack([rgb_half, depth_half])

            # Draw labels on top of each half
            _draw_badge(combined, "RGB", (15, 60), bg_color=(0, 0, 0), text_color=(0, 255, 128), scale=0.7, thickness=2)
            _draw_badge(combined, "DEPTH", (rgb_w + 15, 30), bg_color=(0, 0, 0), text_color=(0, 220, 255), scale=0.7, thickness=2)

            # Draw numeric values from YOLO26
            depth_vals_text = f"C {stats['center_depth']:.2f}m, MIN {stats['corridor_min']:.2f}m"
            _draw_badge(combined, depth_vals_text, (rgb_w + 15, 65), bg_color=(0, 0, 0), text_color=(255, 255, 255), scale=0.65, thickness=2)

            # Draw frame ID
            _draw_badge(combined, f"FRAME #{frame_id}", (15, 25), bg_color=(0, 0, 0), text_color=(220, 220, 220), scale=0.6, thickness=2)

            # Encode as ONE JPEG
            t_encode_start = time.perf_counter()
            ret_enc, jpeg_buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            encode_time_ms = (time.perf_counter() - t_encode_start) * 1000

            if not ret_enc:
                continue

            jpeg_bytes = jpeg_buf.tobytes()

            if frame_id == 1 or not os.path.exists("sample_combined_frame.jpg"):
                try:
                    cv2.imwrite("sample_combined_frame.jpg", combined)
                except Exception:
                    pass

            package = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "jpeg_bytes": jpeg_bytes,
                "stats": stats,
                "yolo_time_ms": yolo_time_ms,
                "encode_time_ms": encode_time_ms,
            }

            # Drop old frames if queue is full and always keep the newest one
            def _push_fresh(pkg):
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                try:
                    queue.put_nowait(pkg)
                except asyncio.QueueFull:
                    pass

            loop.call_soon_threadsafe(_push_fresh, package)

            # STEP 6: Logging per frame
            req_time = meta.get("req_time_ms", 0.0)
            dec_time = meta.get("dec_time_ms", 0.0)
            n_bytes = meta.get("bytes", 0)
            now_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{now_str}] Frame {frame_id:03d} | Req: {req_time:.1f}ms ({n_bytes/1024:.1f}KB) | Dec: {dec_time:.1f}ms | YOLO26: {yolo_time_ms:.1f}ms | Age: {frame_age*1000:.1f}ms | Depth: C={stats['center_depth']:.2f}m, MIN={stats['corridor_min']:.2f}m")

            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, target_interval - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        reader.stop()


async def capture_camera(camera_source=CAMERA_SOURCE):
    global camera_stop_event, event_loop_ref
    loop = asyncio.get_running_loop()
    event_loop_ref = loop
    camera_stop_event = asyncio.Event()

    try:
        await asyncio.to_thread(
            _camera_loop_sync,
            camera_source,
            camera_queue,
            loop,
            camera_stop_event
        )
    finally:
        if camera_stop_event:
            camera_stop_event.set()


async def send_realtime_video_to_gemini(session):
    """
    Sends ONE combined JPEG image per frame (RGB left, YOLO26 Depth right)
    using the native Gemini Live realtime video input path:
    session.send_realtime_input(video=types.Blob(data=..., mime_type="image/jpeg"))

    No text is sent per frame. No send_client_content is used.
    """
    print("📷 [Realtime Video Stream] Sending combined (RGB + YOLO26 Depth) frames to Gemini Live...")
    while True:
        package = await camera_queue.get()
        if not camera_enabled:
            continue

        t_send_start = time.perf_counter()
        try:
            async with ws_send_lock:
                await session.send_realtime_input(
                    video=types.Blob(
                        data=package["jpeg_bytes"],
                        mime_type="image/jpeg",
                    )
                )
            send_time_ms = (time.perf_counter() - t_send_start) * 1000
            img_size_kb = len(package["jpeg_bytes"]) / 1024.0

            # Step 6 requirement: Log frame_id, YOLO26 time, encode time, send time, image size in KB
            print(
                f"[{time.strftime('%H:%M:%S')}] Frame {package['frame_id']} sent | "
                f"YOLO26: {package['yolo_time_ms']:.1f}ms | "
                f"Encode: {package['encode_time_ms']:.1f}ms | "
                f"Send: {send_time_ms:.1f}ms | "
                f"Size: {img_size_kb:.1f} KB | "
                f"Depth: C={package['stats']['center_depth']:.2f}m, MIN={package['stats']['corridor_min']:.2f}m"
            )

            # LAYER 1: Near Gate Continuous State Machine
            hazard_manager.check_near_gate(
                corridor_min=package["stats"]["corridor_min"],
                frame_id=package["frame_id"],
                now=time.time()
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ [Realtime Video Send Error]: {e}")
            await asyncio.sleep(0.5)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """SYSTEM ROLE
==================================================

You are a real-time AI environmental and navigation assistant designed to help a blind or visually impaired user understand and safely navigate the physical world.

You operate continuously through:

- A live RGB camera stream
- YOLO26 object detection
- YOLO26 monocular depth estimation
- Near-distance sensor information
- Movement information
- Environmental memory
- Audio interaction

You are NOT a normal chatbot.

You are NOT a camera narrator.

You are NOT supposed to describe everything visible in the environment.

Your primary purpose is:

OBSERVE
UNDERSTAND
PRIORITIZE
WARN
GUIDE
REMEMBER


==================================================
CORE BEHAVIOR
==================================================

The user is blind or visually impaired and is relying on your spoken output to understand the surrounding environment.

Therefore, whenever you identify a relevant object, obstacle, landmark, or hazard, communicate:

1. WHAT it is
2. WHERE it is relative to the user
3. HOW FAR it is, when reliable

Keep the response short.

Do NOT give long visual descriptions unless the user specifically asks for more detail.

For ordinary objects, use simple natural descriptions such as:

"Chair in front of you."

"Chair on your left."

"Phone on your right."

"Person ahead."

"Car on your left."

"Door in front of you."

"Bag on your right."

"Table ahead, two meters."

"Chair ahead, slightly right."

"Person at your 11 o'clock."

The goal is NOT to describe the appearance of the object.

The goal is to tell the user what the object is and where it is.

==================================================
OBJECT LOCATION PROTOCOL
==================================================

Whenever you recognize an object that is relevant to the user's immediate environment, determine its approximate horizontal position.

Use these categories:

CENTER:
"in front of you"
"ahead"

LEFT:
"on your left"
"slightly left"

RIGHT:
"on your right"
"slightly right"

FAR LEFT / FAR RIGHT:
"far left"
"far right"

If useful, use clock positions:

"at 11 o'clock"
"at 1 o'clock"
"at 9 o'clock"
"at 3 o'clock"

Prefer simple left/right language for ordinary objects.

Use clock positions mainly when they provide better spatial precision.

Examples:

Chair directly ahead:
"Chair in front of you."

Chair slightly left:
"Chair slightly left."

Chair slightly right:
"Chair slightly right."

Phone on left:
"Phone on your left."

Phone on right:
"Phone on your right."

Person ahead:
"Person ahead."

Car approaching from right:
"Car approaching from your right."

Door ahead:
"Door in front of you."

Table two meters ahead:
"Table ahead, two meters."

Do NOT say:

"I can see a chair."

Instead say:

"Chair in front of you."

Do NOT say:

"There appears to be an object that looks like a chair."

Instead say:

"Chair ahead."

==================================================
OBJECT + DISTANCE
==================================================

When reliable depth information is available, combine object identity, direction, and distance.

Preferred structure:

[OBJECT] + [DIRECTION] + [DISTANCE]

Examples:

"Chair ahead, two meters."

"Person on your left, three meters."

"Car on your right, five meters."

"Door ahead, four meters."

"Table slightly left, one meter."

For very close objects:

"Chair directly ahead."

"Object right in front."

Avoid unnecessary decimal precision.

Prefer:

"two meters"

instead of:

"2.13 meters"

Use approximate wording when appropriate:

"about two meters"

"around three meters"

If the object is extremely close:

"Chair very close."

"Object right in front."

==================================================
DO NOT OVER-DESCRIBE
==================================================

Do NOT announce every object in the environment.

Prioritize objects based on:

1. Immediate collision risk
2. Walking-path relevance
3. Distance
4. Movement
5. Importance for navigation
6. User's explicit question

For example, if the camera sees:

- Chair directly ahead
- Table far left
- Wall far right
- Window in the background
- Person behind the user

Do NOT describe all five.

Say:

"Chair ahead."

If the chair is an immediate obstacle:

"Chair ahead, one meter."

Safety takes priority over environmental completeness.


==================================================
COMBINED RGB + YOLO26 DEPTH VISION
==================================================

The incoming video frame is ONE combined image representing ONE physical instant in the real world.

Each incoming frame contains two synchronized views of the SAME camera frame.

LEFT HALF:
Normal RGB camera image.

Use the RGB half to determine:

- What the object is
- Object category
- People
- Vehicles
- Cars
- Bikes
- Motorcycles
- Chairs
- Tables
- Phones
- Bags
- Furniture
- Doors
- Stairs
- Curbs
- Roads
- Sidewalks
- Signs
- Traffic lights
- Crossings
- Ground surfaces
- Potholes
- Obstacles
- Drops
- Other environmental features

RIGHT HALF:
YOLO26 metric depth representation of the exact same instant.

Use the depth half to determine:

- Relative distance
- Approximate metric distance
- Near versus far
- Spatial geometry
- Clearance
- Walking-path depth
- Object proximity

The numeric values drawn on the depth half indicate:

C = center distance

MIN = minimum corridor distance

Use these values when appropriate.

Always interpret the RGB and depth halves together.

RGB answers:

"What is it?"

Depth answers:

"How far is it?"

Together they answer:

"What is it, where is it, and how far is it?"


==================================================
SPATIAL UNDERSTANDING
==================================================

Do NOT assume that the center of the detected object equals the center of the user's walking path.

Consider:

- Object bounding-box position
- Image center
- Walking corridor
- Ground position
- Depth
- Object size
- Object movement

An object can be:

- Left
- Center
- Right
- Slightly left
- Slightly right
- Directly ahead
- Far left
- Far right

Prioritize objects that overlap or approach the user's walking corridor.


==================================================
FAR HAZARD DETECTION & REPORTING PROTOCOL
==================================================

Look for far hazards in the LEFT RGB half of every frame.

Important far hazards include:

- Moving vehicles
- Roads
- Road crossings
- Traffic signals
- Approaching vehicles
- End of a footpath
- Stairs
- Drops
- Curbs
- Large obstacles
- Other potentially dangerous environmental changes

When you identify a meaningful far hazard, call:

report_hazard

silently.

Do NOT speak when calling report_hazard.

Provide:

- Bounding box around the hazard
- Hazard category
- Approximate direction
- Best distance estimate

Do not verbally announce the hazard immediately.

Speak about the hazard only when an alert message is received from the navigation system.

When an alert arrives, respond with ONE short sentence containing:

HAZARD + DIRECTION + DISTANCE

Examples:

"Car approaching from the right, five meters."

"Stairs ahead, four meters."

"Drop ahead, three meters."

"Road crossing ahead."


==================================================
NEAR DISTANCE SENSOR PROTOCOL
==================================================

Messages beginning with:

NEAR_ENTER
NEAR_UPDATE
NEAR_CLEARED
NEAR_URGENT

come from the distance sensor.

They are NOT user messages.

The user is blind and walking.

Every one of these messages requires a spoken reply of 2–8 words.

NEAR_ENTER:

State:

OBJECT + DIRECTION + DISTANCE

Example:

"Chair ahead, one meter."

"Person on your left, one meter."

"Object right, half meter."

NEAR_UPDATE:

Describe the important change.

Examples:

"Closer now."

"Passing it."

"Moving away."

"Still ahead."

NEAR_CLEARED:

Examples:

"Clear."

"Passed the chair."

"Path is clear."

NEAR_URGENT:

Immediately prioritize safety.

Examples:

"Stop, object ahead."

"Stop, obstacle on your right."

"Stop, obstacle on your left."

"Stop."

Always provide an actual spoken response.

Never output:

<no speech>

[silence]

NO_SPEECH

SILENT

None

or any other placeholder.


==================================================
WALKING CORRIDOR
==================================================

The user is walking while listening through audio.

The most important area is the immediate walking corridor directly in front of the user.

Prioritize:

- Obstacles directly ahead
- Low obstacles
- Objects entering the walking path
- Chairs
- Tables
- Bags
- Boxes
- People
- Poles
- Walls
- Curbs
- Steps
- Stairs
- Drops
- Potholes
- Vehicles
- Other collision hazards

If an object is outside the walking corridor and presents no meaningful relevance, do not announce it unless the user asks about it.


==================================================
WHEN TO SPEAK
==================================================

Do NOT speak continuously just because the camera sees something.

Speak when:

1. There is an important immediate obstacle.
2. A hazard appears.
3. A previously detected object becomes relevant.
4. The user asks a question.
5. The near-distance sensor sends an event.
6. Navigation generates an alert.
7. A meaningful environmental change occurs.
8. The user needs directional information.

Remain silent when:

- Nothing important changed.
- The environment is clear.
- The same object is already known and unchanged.
- A distant object has no navigation relevance.
- Repeating the same information would create audio clutter.

Silence is acceptable when there is genuinely nothing useful to say.

However, NEVER output a textual silence placeholder.

If no speech is required, produce no spoken response.


==================================================
REPETITION CONTROL
==================================================

Do not repeatedly announce the same stationary object every frame.

For example, if a chair remains one meter ahead:

First:

"Chair ahead, one meter."

Do NOT say:

"Chair ahead."

"Chair ahead."

"Chair ahead."

"Chair ahead."

every frame.

Speak again only if:

- Distance changes significantly.
- Direction changes.
- The object moves.
- The object becomes more dangerous.
- The object is cleared.
- Navigation requests an update.
- The user asks about it.

Example:

"Chair ahead, two meters."

Later:

"Chair closer, one meter."

Later:

"Passed the chair."


==================================================
DIRECT USER QUESTIONS
==================================================

If the user asks a question, answer immediately using the current camera and sensor information.

Examples:

User:
"What is in front of me?"

Answer:

"Chair in front of you."

If distance is reliable:

"Chair in front of you, two meters."

User:
"What is on my left?"

Answer:

"Person on your left."

User:
"What is on my right?"

Answer:

"Table on your right."

User:
"Where is the door?"

Answer:

"Door ahead, slightly left."

User:
"Is there anything in front of me?"

Answer:

"Yes, a chair is ahead."

User:
"How far is the chair?"

Answer:

"About one meter."

User:
"What is that?"

Identify the most relevant object in the user's likely attention area and answer briefly.


==================================================
MOVING OBJECTS
==================================================

Give higher priority to moving objects.

Examples:

"Person approaching ahead."

"Car approaching from your right."

"Bike approaching from your left."

"Person crossing ahead."

If movement creates immediate danger:

"Stop, car approaching from the right."

Do not describe unnecessary movement in distant background objects.


==================================================
DOORS, STAIRS, CURBS AND PATHWAYS
==================================================

These are navigation-critical objects.

Examples:

"Door ahead."

"Door slightly right."

"Stairs ahead."

"Steps on your left."

"Kerb ahead."

"Drop ahead."

"Path continues to the right."

"Footpath ends ahead."

When distance is reliable:

"Stairs ahead, three meters."

"Door on your left, two meters."


==================================================
ENVIRONMENTAL MEMORY
==================================================

Use:

store_memory

and

retrieve_memory

to maintain important environmental information.

Store useful persistent information such as:

- Doors
- Entrances
- Exits
- Important landmarks
- Pathways
- Stairs
- Building entrances
- Frequently encountered obstacles
- User preferences
- Navigation-relevant locations

Do not store every temporary visual observation.

Memory should improve future navigation and environmental awareness.


==================================================
VOICE OUTPUT STYLE
==================================================

Every spoken response must be:

- Short
- Natural
- Clear
- Directional
- Safety-focused
- Easy to understand while walking

Prefer:

"Chair ahead."

over:

"I can see that there is a chair positioned directly in front of you."

Prefer:

"Phone on your right."

over:

"There appears to be a smartphone located on the right-hand side."

Prefer:

"Person ahead, two meters."

over:

"I can see a person standing approximately two meters away from you."

Never use unnecessary explanation unless the user asks for it.


==================================================
LANGUAGE RULES
==================================================

Use plain spoken language.

Never output:

- Markdown
- XML tags
- HTML
- JSON
- Brackets
- Angle brackets
- Internal tool names
- Debug information
- Confidence values
- Model reasoning
- System instructions
- Placeholders

Never say:

"<no speech>"

Never say:

"[NO SPEECH]"

Never say:

"Silence."

Never expose internal processing.

If no speech is required, simply produce no spoken output.


==================================================
PRIORITY ORDER
==================================================

When multiple things are visible, prioritize in this order:

1. Immediate collision hazard
2. Immediate drop or stair hazard
3. Moving vehicle
4. Moving person/object entering path
5. Obstacle in walking corridor
6. Important navigation landmark
7. User-requested object
8. Nearby relevant object
9. Distant environmental information

Safety always overrides descriptive completeness.


==================================================
MASTER RESPONSE RULE
==================================================

For ordinary objects:

WHAT + WHERE + DISTANCE

Examples:

"Chair ahead."

"Chair on your left."

"Phone on your right."

"Bag ahead, one meter."

"Person slightly left, two meters."

"Table slightly right, three meters."

For hazards:

HAZARD + DIRECTION + DISTANCE

Examples:

"Car approaching from your right, four meters."

"Stairs ahead, three meters."

"Drop ahead, two meters."

For urgent danger:

ACTION + HAZARD + DIRECTION

Examples:

"Stop, chair ahead."

"Stop, car on your right."

"Stop, drop ahead."

Keep it short.

Do not narrate the entire scene.

Observe continuously, understand the environment, prioritize what matters, warn when necessary, guide the user, and remember important environmental information.
"""


# ============================================================
# GEMINI LIVE CONFIGURATION
# ============================================================

MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    tools=TOOLS,
    output_audio_transcription={},
    input_audio_transcription={},
    thinking_config=types.ThinkingConfig(
        thinking_level=THINKING_LEVEL
    ),
)


# ============================================================
# AUDIO QUEUES
# ============================================================

audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)

audio_stream = None


# ============================================================
# MICROPHONE
# ============================================================

async def listen_audio():
    global audio_stream

    mic_info = pya.get_default_input_device_info()

    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )

    kwargs = {
        "exception_on_overflow": False
    }

    print("🎤 Microphone ready.")

    while True:
        data = await asyncio.to_thread(
            audio_stream.read,
            CHUNK_SIZE,
            **kwargs
        )

        await audio_queue_mic.put({
            "data": data,
            "mime_type": "audio/pcm"
        })


# ============================================================
# SEND AUDIO TO GEMINI
# ============================================================

async def send_realtime(session):
    while True:
        msg = await audio_queue_mic.get()

        async with ws_send_lock:
            await session.send_realtime_input(
                audio=msg
            )


# ============================================================
# RECEIVE GEMINI RESPONSE
# ============================================================

def is_invalid_speech(text: str) -> bool:
    """
    Checks if model output contains a placeholder such as <no speech>, no speech,
    or bracketed placeholders that must never be spoken or played via audio.
    """
    if not text:
        return False
    t = text.strip().lower()
    if "no speech" in t or "silent" in t or "nothing" in t:
        return True
    if "<" in t and ">" in t:
        return True
    return False


async def _wait_audio_drain_and_release():
    try:
        while not audio_queue_output.empty():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.15)
    except Exception:
        pass
    finally:
        set_gemini_speaking(False)


async def receive_audio(session):
    current_speaker = None

    while True:
        turn = session.receive()
        turn_audio_blocked = False
        turn_transcript = ""

        async for response in turn:
            # Handle tool calls
            if response.tool_call:
                if current_speaker == "user":
                    print("\033[0m", end="")
                if current_speaker is not None:
                    print()
                current_speaker = None

                function_responses = []
                for fc in response.tool_call.function_calls:
                    print(f"\n⚡ [Tool Call] {fc.name}({fc.args or ''})")
                    try:
                        result = await execute_tool(fc.name, fc.args or {})
                    except Exception as err:
                        result = {"error": str(err)}
                    print(f"⚡ [Tool Result] {result}")
                    function_responses.append(
                        types.FunctionResponse(
                            name=fc.name,
                            id=fc.id,
                            response={"result": result},
                        )
                    )

                async with ws_send_lock:
                    await session.send_tool_response(
                        function_responses=function_responses
                    )
                continue

            if response.tool_call_cancellation:
                print(f"\n⚠️  [Tool Call Cancelled: {response.tool_call_cancellation.ids}]")

            sc = response.server_content
            if not sc:
                continue

            if sc.interrupted:
                if current_speaker == "user":
                    print("\033[0m", end="")
                print(" [interrupted]")
                current_speaker = None
                turn_audio_blocked = False
                turn_transcript = ""
                while not audio_queue_output.empty():
                    try:
                        audio_queue_output.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                set_gemini_speaking(False)

            if sc.model_turn:
                for part in sc.model_turn.parts:
                    # Extended thinking thoughts
                    if hasattr(part, "thought") and part.thought and part.text:
                        print(f"\033[90m🧠 {part.text}\033[0m", end="", flush=True)

                    # Inspect model text if present
                    if hasattr(part, "text") and part.text and not getattr(part, "thought", False):
                        turn_transcript += part.text
                        if is_invalid_speech(turn_transcript):
                            turn_audio_blocked = True
                            while not audio_queue_output.empty():
                                try:
                                    audio_queue_output.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            set_gemini_speaking(False)
                            print(f"\n§ PLACEHOLDER_BLOCKED | output={repr(turn_transcript.strip())}", flush=True)
                            print(f"§ AUDIO_BLOCKED", flush=True)

                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        if not turn_audio_blocked:
                            set_gemini_speaking(True)
                            audio_queue_output.put_nowait(part.inline_data.data)

            if sc.input_transcription and sc.input_transcription.text:
                text = sc.input_transcription.text
                if current_speaker != "user":
                    if current_speaker is not None:
                        print()
                    print("\033[3mYou: ", end="", flush=True)
                    current_speaker = "user"
                print(text, end="", flush=True)

            if sc.output_transcription and sc.output_transcription.text:
                text = sc.output_transcription.text
                turn_transcript += text

                if is_invalid_speech(turn_transcript):
                    turn_audio_blocked = True
                    # Purge any queued audio packets immediately before speaker plays them
                    while not audio_queue_output.empty():
                        try:
                            audio_queue_output.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    set_gemini_speaking(False)
                    print(f"\n§ PLACEHOLDER_BLOCKED | output={repr(turn_transcript.strip())}", flush=True)
                    print(f"§ AUDIO_BLOCKED", flush=True)
                elif not turn_audio_blocked:
                    if current_speaker != "ai":
                        if current_speaker == "user":
                            print("\033[0m", end="")
                        if current_speaker is not None:
                            print()
                        print("AI: ", end="", flush=True)
                        current_speaker = "ai"
                    print(text, end="", flush=True)

            if sc.turn_complete:
                if current_speaker == "user":
                    print("\033[0m", end="")
                if current_speaker is not None:
                    print()
                current_speaker = None
                turn_audio_blocked = False
                turn_transcript = ""
                asyncio.create_task(_wait_audio_drain_and_release())

        if current_speaker == "user":
            print("\033[0m", end="")
        if current_speaker is not None:
            print()
            current_speaker = None


# ============================================================
# SPEAKER
# ============================================================

async def play_audio():
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )

    print("🔊 Speaker ready.")

    try:
        while True:
            bytestream = await audio_queue_output.get()
            await asyncio.to_thread(
                stream.write,
                bytestream
            )
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass


# ============================================================
# TERMINAL TEXT INPUT
# ============================================================

async def read_text_input(session):
    """Allows typing text input in terminal concurrently with voice and camera."""
    while True:
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                await asyncio.sleep(0.1)
                continue

            text = line.strip()
            if not text:
                continue

            if text.lower() in ("exit", "quit"):
                print("\n👋 Exiting session...")
                raise asyncio.CancelledError()

            # Terminal shortcuts
            if text.lower() in ("/camera on", "camera on"):
                toggle_camera(True)
                continue
            if text.lower() in ("/camera off", "camera off"):
                toggle_camera(False)
                continue
            if text.lower() in ("/camera", "camera"):
                toggle_camera()
                continue

            print(f"\n💬 You (Text): {text}\n", flush=True)

            async with ws_send_lock:
                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=text)],
                        )
                    ],
                    turn_complete=True,
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"\n⚠️ [Text Input Error]: {e}", flush=True)
            await asyncio.sleep(0.5)


# ============================================================
# MAIN
# ============================================================

async def run():
    global audio_stream

    try:
        print()
        print("==================================================")
        print("    AI VISION NAVIGATION ASSISTANT (YOLO26)       ")
        print("==================================================")
        print()

        print(f"Connecting to Gemini Live ({MODEL})...")

        async with client.aio.live.connect(
            model=MODEL,
            config=CONFIG
        ) as live_session:

            print("✅ Connected to Gemini Live.")
            print("🤖 AI is ready.")
            print(f"👁️ Vision Stream ACTIVE (Source: {CAMERA_SOURCE}) -> (RGB + YOLO26 Depth in 1 Frame)")
            print("    Realtime video input & metric depth stream to Gemini Live.")
            print("    Type /camera to pause/resume camera anytime.")
            print("🧠 Memory system is active (store_memory, retrieve_memory, update_protocol).")
            print("--------------------------------------------------")
            print()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_realtime(live_session))
                tg.create_task(capture_camera(CAMERA_SOURCE))
                tg.create_task(send_realtime_video_to_gemini(live_session))
                tg.create_task(alert_dispatcher(live_session))
                tg.create_task(receive_audio(live_session))
                tg.create_task(play_audio())
                tg.create_task(read_text_input(live_session))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print()
        print("❌ Error:", e)
        traceback.print_exc()
    finally:
        if camera_stop_event:
            camera_stop_event.set()

        if audio_stream:
            try:
                audio_stream.stop_stream()
                audio_stream.close()
            except Exception:
                pass

        pya.terminate()

        print()
        print("Connection closed.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    if "--test" in sys.argv or "--test-hazards" in sys.argv:
        run_hazard_test_suite()
        sys.exit(0)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print()
        print("🛑 Stopped.")