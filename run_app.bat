@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo     ETHIOPIAN COFFEE AI ASSISTANT
echo     Computer Vision + RAG Chatbot + Amharic
echo ===================================================
echo.

:: Check for local virtual environment
if exist .\.venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (.venv)...
    call .\.venv\Scripts\activate.bat
) else if exist .\venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment (venv)...
    call .\venv\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)

echo [INFO] Launching Streamlit web application...
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
