import os
import sys
import time
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

raw_url = os.getenv("ESP32_CAM_URL", "http://10.24.69.1/capture").strip()
parsed = urlparse(raw_url if raw_url.startswith("http") else "http://" + raw_url)
host = parsed.hostname or "10.24.69.1"
capture_url = f"http://{host}/capture"

print("=" * 60)
print(f"STEP 1 DIAGNOSTIC: Testing ESP32 Capture Endpoint")
print(f"Target URL: {capture_url}")
print("Testing 10 requests, 1 per second...")
print("=" * 60 + "\n")

session = requests.Session()
first_saved = False
last_img = None
success_count = 0
times = []

for i in range(1, 11):
    t0 = time.perf_counter()
    try:
        resp = session.get(capture_url, headers={"Connection": "close"}, timeout=(2.5, 3.0))
        dt = (time.perf_counter() - t0) * 1000
        code = resp.status_code
        nbytes = len(resp.content)
        times.append(dt)
        print(f"Request #{i:02d}: status={code} | bytes={nbytes} | time={dt:.1f} ms")
        if code == 200:
            success_count += 1
            if not first_saved:
                with open("first_capture.jpg", "wb") as f:
                    f.write(resp.content)
                first_saved = True
            last_img = resp.content
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        print(f"Request #{i:02d}: FAILED ({type(e).__name__}: {e}) | time={dt:.1f} ms")

    time.sleep(1.0)

print("\n" + "=" * 60)
print(f"Results: {success_count}/10 successful")
if times:
    print(f"Average time: {sum(times)/len(times):.1f} ms | Min: {min(times):.1f} ms | Max: {max(times):.1f} ms")
if last_img:
    with open("last_capture.jpg", "wb") as f:
        f.write(last_img)
    print("Saved first_capture.jpg and last_capture.jpg successfully.")
print("=" * 60)
