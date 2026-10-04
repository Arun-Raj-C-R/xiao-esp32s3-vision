@echo off
title YOLO26 Real-Time Depth Stream & Auto-Capture
cd /d "D:\Hackthon\YOLO"
echo ========================================================
echo Starting YOLO26 Real-Time Monocular Depth Estimation...
echo Web Viewer : http://localhost:5050
echo Saving to  : D:\Hackthon\YOLO Images
echo Close this window or press Ctrl+C to exit.
echo ========================================================
"C:\Users\arunr\python1\Scripts\python.exe" -u "D:\Hackthon\YOLO\live_depth_server.py"
pause
