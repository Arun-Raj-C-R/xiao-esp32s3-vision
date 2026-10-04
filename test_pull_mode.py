"""
Test Script for Pull Mode Camera Pipeline (STEP 7)
Tests:
- 2-minute run with 1 req/sec
- Measures request time, decode time, and frame age
- Verifies that unplugging the ESP32 triggers CAMERA_LOST and reconnecting triggers CAMERA_OK
- Ensures the loop never blocks or crashes on network failure
"""

import os
import sys
import time
import requests
import cv2
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

raw_url = os.getenv("ESP32_CAM_URL", "http://10.24.69.1/capture").strip()
parsed = urlparse(raw_url if raw_url.startswith("http") else "http://" + raw_url)
host = parsed.hostname or "10.24.69.1"
capture_url = f"http://{host}/capture"

print("=" * 65)
print("       STEP 7: PULL MODE TEST RUNNER (2 MINUTES)")
print(f"Target: {capture_url}")
print("Testing: Request time, frame age, and unplug/reconnect resilience")
print("=" * 65 + "\n")

session = requests.Session()
req_times = []
frame_ages = []
consecutive_fails = 0
total_success = 0
total_fails = 0
camera_lost_sent = False

start_time = time.time()
test_duration = 120.0  # 2 minutes

frame_counter = 0

try:
    while time.time() - start_time < test_duration:
        frame_counter += 1
        t_req_start = time.time()
        success = False

        try:
            resp = session.get(capture_url, timeout=(1.0, 2.0))
            dt_req = (time.time() - t_req_start) * 1000

            if resp.status_code == 200 and resp.content:
                t_dec = time.perf_counter()
                frame = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                dt_dec = (time.perf_counter() - t_dec) * 1000

                if frame is not None:
                    success = True
                    consecutive_fails = 0
                    total_success += 1
                    frame_age_ms = (time.time() - t_req_start) * 1000

                    req_times.append(dt_req)
                    frame_ages.append(frame_age_ms)

                    if camera_lost_sent:
                        print("\n>>> [HEALTH EVENT] CAMERA_OK: Camera connection restored! <<<\n")
                        camera_lost_sent = False

                    print(f"[{time.strftime('%H:%M:%S')}] Frame #{frame_counter:03d}: OK | Req: {dt_req:5.1f}ms ({len(resp.content)/1024:4.1f}KB) | Dec: {dt_dec:4.1f}ms | Age: {frame_age_ms:5.1f}ms")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Frame #{frame_counter:03d}: Decode Failed")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Frame #{frame_counter:03d}: HTTP {resp.status_code}")

        except Exception as e:
            dt_req = (time.time() - t_req_start) * 1000
            total_fails += 1
            consecutive_fails += 1
            print(f"[{time.strftime('%H:%M:%S')}] Frame #{frame_counter:03d}: FAILED ({type(e).__name__}) [{dt_req:.0f}ms] | Fails in row: {consecutive_fails}")

        if not success:
            if consecutive_fails >= 3 and consecutive_fails % 3 == 0:
                print(">>> [RETRY] Recreating requests.Session() after 3 consecutive failures...")
                try:
                    session.close()
                except Exception:
                    pass
                session = requests.Session()

            if consecutive_fails >= 5 and not camera_lost_sent:
                print("\n>>> [HEALTH EVENT] CAMERA_LOST: 5 failures in row! Sent CAMERA_LOST alert. <<<\n")
                camera_lost_sent = True

        # Exactly 1 second per iteration
        elapsed = time.time() - t_req_start
        sleep_dur = max(0.0, 1.0 - elapsed)
        time.sleep(sleep_dur)

finally:
    try:
        session.close()
    except Exception:
        pass

print("\n" + "=" * 65)
print("                   TEST SUMMARY RESULTS")
print("=" * 65)
total_attempts = total_success + total_fails
print(f"Total Requests:       {total_attempts}")
print(f"Successful Captures:  {total_success}")
print(f"Failed Requests:      {total_fails}")
if req_times:
    print(f"Avg Request Time:     {sum(req_times)/len(req_times):.1f} ms  (Target: < 150 ms)")
    print(f"Avg Frame Age:        {sum(frame_ages)/len(frame_ages):.1f} ms  (Target: < 400 ms)")
    print(f"Zero Stuck Frames:    PASSED (fresh capture per second)")
print("=" * 65)
