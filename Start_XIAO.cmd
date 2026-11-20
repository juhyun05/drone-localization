@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0localization_app\tools\launch_xiao.ps1"
set "result=%ERRORLEVEL%"
if not "%result%"=="0" pause
exit /b %result%
