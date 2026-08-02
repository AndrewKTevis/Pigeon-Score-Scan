@echo off
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0") do set "RUNTIME=%%~fI"
>"%RUNTIME%show-window.signal" echo show
endlocal & exit /b 0
