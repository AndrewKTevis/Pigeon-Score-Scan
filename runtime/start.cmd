@echo off
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "UV_EXE=%ROOT%\runtime\uv.exe"
set "UV_HASH_FILE=%ROOT%\runtime\uv.sha256"
set "LOG_FILE=%ROOT%\runtime\launcher.log"
set "READY_FILE=%ROOT%\runtime\ready.txt"
set "FAILED_FILE=%ROOT%\runtime\start.failed"
set "PROJECT_ROOT=%ROOT%\app"
set "RUNTIME_PROFILE=cpu"
set "RUNTIME_ENVIRONMENT=%ROOT%\runtime\venv-cpu"

if exist "%READY_FILE%" del /f /q "%READY_FILE%" >nul 2>&1
if exist "%FAILED_FILE%" del /f /q "%FAILED_FILE%" >nul 2>&1

call :ensure_uv
if errorlevel 1 (
  echo [%DATE% %TIME%] uv bootstrap or validation failed>>"%LOG_FILE%"
  set "EXIT_CODE=20"
  goto :failed
)

set "SCORESCAN_PORTABLE_ROOT=%ROOT%"
set "SCORESCAN_RUNTIME_PROFILE=%RUNTIME_PROFILE%"
set "UV_PYTHON_INSTALL_DIR=%ROOT%\runtime\python"
set "UV_PROJECT_ENVIRONMENT=%RUNTIME_ENVIRONMENT%"
set "UV_CACHE_DIR=%ROOT%\runtime\uv-cache"
set "UV_LINK_MODE=copy"
set "UV_DEFAULT_INDEX=https://pypi.org/simple"
set "PYTHONUTF8=1"
set "SCORESCAN_LAUNCHED_BY_EXE=1"
set "PYTHONPATH=%ROOT%\app\src"
"%UV_EXE%" run --frozen --no-dev --project "%PROJECT_ROOT%" --python 3.12 --python-preference managed python -m scorescan > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :failed
endlocal & exit /b 0

:ensure_uv
call :validate_uv
if not errorlevel 1 exit /b 0

echo [%DATE% %TIME%] uv.exe missing, blocked, damaged, or incompatible; starting pinned bootstrap repair>>"%LOG_FILE%"
if exist "%UV_EXE%" del /f /q "%UV_EXE%" >nul 2>&1
if exist "%UV_HASH_FILE%" del /f /q "%UV_HASH_FILE%" >nul 2>&1
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%ROOT%\runtime\uv-bootstrap.ps1" >>"%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
call :validate_uv
exit /b %ERRORLEVEL%

:validate_uv
if not exist "%UV_EXE%" exit /b 1
if not exist "%UV_HASH_FILE%" exit /b 1
set "EXPECTED_UV="
set /p EXPECTED_UV=<"%UV_HASH_FILE%"
if not defined EXPECTED_UV exit /b 1
set "ACTUAL_UV="
for /f "usebackq delims=" %%H in (`powershell.exe -NoProfile -NonInteractive -Command "$p=$env:UV_EXE; if(Test-Path -LiteralPath $p){(Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant()}"`) do set "ACTUAL_UV=%%H"
if not defined ACTUAL_UV exit /b 1
if /I not "%ACTUAL_UV%"=="%EXPECTED_UV%" exit /b 1
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath $env:UV_EXE -ErrorAction SilentlyContinue" >nul 2>&1
"%UV_EXE%" --version >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:failed
>"%FAILED_FILE%" echo %EXIT_CODE%
endlocal & exit /b %EXIT_CODE%
