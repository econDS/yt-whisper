@echo off
setlocal
if not defined YTW_DATA_DIR set "YTW_DATA_DIR=E:\yt-whisper"
set "YTW_PYTHON=%YTW_DATA_DIR%\envs\py312\python.exe"
if not exist "%YTW_PYTHON%" (
  echo Environment missing: %YTW_PYTHON%
  echo Run scripts\setup-windows.ps1 first.
  pause
  exit /b 1
)
set "TEMP=%YTW_DATA_DIR%\tmp"
set "TMP=%TEMP%"
set "PYTHONDONTWRITEBYTECODE=1"
set "PATH=%YTW_DATA_DIR%\envs\py312;%YTW_DATA_DIR%\envs\py312\Scripts;%YTW_DATA_DIR%\envs\py312\Library\bin;%PATH%"
cd /d "%~dp0"
"%YTW_PYTHON%" -m yt_whisper.ui
if errorlevel 1 pause
endlocal
