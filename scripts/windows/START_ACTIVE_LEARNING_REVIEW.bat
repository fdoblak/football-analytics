@echo off
setlocal EnableExtensions
title R1 Active Learning Review
cd /d "%~dp0"

set "DISTRO=Ubuntu-22.04"
set "PORT=8768"
set "APP_URL=http://127.0.0.1:8768/"
set "WRAPPER=/home/fdoblak/projects/football-analytics/scripts/start_r1_active_learning_review.sh"
set "LINUX_LOG=/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning/server_wrapper.log"

echo R1 Active Learning + Blind Holdout review starting...
echo.

call :CHECK_HEALTH
if not errorlevel 1 (
  echo Server already ready.
  echo Browser opening...
  start "" "%APP_URL%"
  echo Do not close the existing server window during review.
  goto END_OK
)

start "R1-AL-Server" cmd /k wsl.exe -d %DISTRO% -- bash %WRAPPER%

set /a TRIES=0
:WAIT_HEALTH
set /a TRIES+=1
call :CHECK_HEALTH
if not errorlevel 1 goto READY
if %TRIES% GEQ 30 goto FAIL
timeout /t 1 /nobreak >nul
goto WAIT_HEALTH

:READY
echo Server ready.
echo Browser opening...
start "" "%APP_URL%"
echo Do not close the server window during review.
goto END_OK

:FAIL
echo.
echo ERROR: Server did not become healthy within 30 seconds.
echo Browser was NOT opened.
echo Open the R1-AL-Server window for stderr.
echo Linux log: %LINUX_LOG%
echo.
pause
exit /b 1

:END_OK
echo.
echo Review UI: %APP_URL%
echo Keep the R1-AL-Server window open until you finish.
pause
exit /b 0

:CHECK_HEALTH
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8768/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.service -eq 'r1_independent_gt_review' -and $r.active_learning -eq $true) { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %ERRORLEVEL%
