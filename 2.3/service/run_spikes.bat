@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Сначала запустите install.bat & pause & exit /b 1)
.venv\Scripts\python spikes.py %*
pause
