@echo off
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0") do set "RUNTIME=%%~fI"
echo Repairing Pigeon Score Scan runtime...
if exist "%RUNTIME%uv.exe" del /f /q "%RUNTIME%uv.exe" >nul 2>&1
if exist "%RUNTIME%uv.sha256" del /f /q "%RUNTIME%uv.sha256" >nul 2>&1
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%RUNTIME%uv-bootstrap.ps1"
if errorlevel 1 (
  echo.
  echo Repair failed. Check launcher.log and your network or antivirus settings.
  pause
  exit /b 1
)
echo.
echo Runtime repaired. You can run pigeon-score-scan.exe now.
pause
exit /b 0
