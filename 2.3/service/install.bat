@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Установка: окружение Python и библиотеки ===
py -m venv .venv || (echo Python не найден. Установите Python 3.12 с python.org и запустите снова. & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo Не удалось установить библиотеки & pause & exit /b 1)
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama не найдена: доступен только режим classic. Для режима hybrid установите Ollama с ollama.com и выполните: ollama pull qwen3:8b
) else (
  ollama pull qwen3:8b
)
echo === Готово ===
pause
