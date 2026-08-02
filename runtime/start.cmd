@echo off
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PYTHON_EXE=%ROOT%\runtime\python\python.exe"
set "SITE_PACKAGES=%ROOT%\runtime\site-packages"
set "RUNNER=%ROOT%\runtime\run_scorescan.py"
set "LOG_FILE=%ROOT%\runtime\launcher.log"
set "READY_FILE=%ROOT%\runtime\ready.txt"
set "FAILED_FILE=%ROOT%\runtime\start.failed"

if exist "%READY_FILE%" del /f /q "%READY_FILE%" >nul 2>&1
if exist "%FAILED_FILE%" del /f /q "%FAILED_FILE%" >nul 2>&1

if not exist "%PYTHON_EXE%" (
  set "EXIT_CODE=20"
  goto :failed
)
if not exist "%SITE_PACKAGES%\homr" (
  set "EXIT_CODE=21"
  goto :failed
)
if not exist "%RUNNER%" (
  set "EXIT_CODE=22"
  goto :failed
)

set "SCORESCAN_PORTABLE_ROOT=%ROOT%"
set "SCORESCAN_RUNTIME_PROFILE=cpu"
set "SCORESCAN_OFFLINE_RUNTIME=1"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "SCORESCAN_LAUNCHED_BY_EXE=1"
set "PYTHONPATH=%ROOT%\app\src;%SITE_PACKAGES%"
"%PYTHON_EXE%" -s "%RUNNER%" > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :failed
endlocal & exit /b 0

:failed
>"%FAILED_FILE%" echo %EXIT_CODE%
>>"%LOG_FILE%" echo Bundled offline runtime failed with exit code %EXIT_CODE%.
endlocal & exit /b %EXIT_CODE%
