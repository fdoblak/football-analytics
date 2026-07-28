@echo off
setlocal EnableExtensions
title R1 Blind GT Review
cd /d "%~dp0"
echo.
echo === R1 Blind GT Review ===
echo Local only. No internet. No secrets.
echo.

REM Start review server inside WSL on localhost:8765
wsl -e bash -lc "pkill -f 'r1_blind_gt_review_server.py' >/dev/null 2>&1 || true; cd /home/fdoblak/projects/football-analytics && nohup /home/fdoblak/miniconda3/envs/ai-dev/bin/python scripts/r1_blind_gt_review_server.py --host 127.0.0.1 --port 8765 --blind >/tmp/r1_gt_review_server.log 2>&1 & echo $!" > "%TEMP%\r1_gt_review_pid.txt"
timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:8765/"
start "" "%~dp0OPEN_R1_GT_REVIEW.html"

echo Browser opened at http://127.0.0.1:8765/
echo.
echo When finished, press any key here to stop the server...
pause >nul

wsl -e bash -lc "pkill -f 'r1_blind_gt_review_server.py' >/dev/null 2>&1 || true"
echo Server stopped.
pause
