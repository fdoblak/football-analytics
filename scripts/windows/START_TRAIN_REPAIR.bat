@echo off
setlocal EnableExtensions
title R1 GT Train Repair Launcher
cd /d "%~dp0"

set "DISTRO=Ubuntu-22.04"
set "PORT=8766"
set "HEALTH_URL=http://127.0.0.1:8766/health"
set "APP_URL=http://127.0.0.1:8766/?repair=train-empty-complete"
set "WRAPPER=/home/fdoblak/projects/football-analytics/scripts/start_r1_gt_review.sh"
set "LINUX_LOG=/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4/server_wrapper.log"

echo R1 TRAIN REPAIR server starting...
echo.

call :CHECK_HEALTH
if not errorlevel 1 (
  echo Server already ready in repair mode.
  echo Browser opening...
  start "" "%APP_URL%"
  echo Do not close the existing server window during repair.
  goto END_OK
)

REM Visible persistent server window. Do NOT use ephemeral background WSL sessions.
start "R1-GT-Repair-Server" cmd /k wsl.exe -d %DISTRO% -- bash %WRAPPER% --repair train-empty-complete

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
echo Do not close the server window during repair.
goto END_OK

:FAIL
echo.
echo ERROR: Repair server did not become healthy within 30 seconds.
echo Browser was NOT opened.
echo Open the R1-GT-Repair-Server window for stderr.
echo If a normal GT server is already open, close it first.
echo Linux log: %LINUX_LOG%
echo.
pause
exit /b 1

:END_OK
echo.
echo Repair UI: %APP_URL%
echo Keep the R1-GT-Repair-Server window open until you finish.
pause
exit /b 0

:CHECK_HEALTH
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8766/health' -TimeoutSec 2; if ($r.status -eq 'ok' -and $r.service -eq 'r1_independent_gt_review' -and $r.source_id -eq 'own_video_97b298e4' -and $r.repair_mode -eq 'train-empty-complete') { exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %ERRORLEVEL%
