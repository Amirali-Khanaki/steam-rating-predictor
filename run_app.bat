@echo off
REM Steam Rating Predictor - double-click to start. Needs Python (python.org) installed once.
cd /d "%~dp0"
echo Installing / checking the needed libraries (only slow the first time)...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
  echo.
  echo Could not install the libraries. Is Python installed and "Add python.exe to PATH" ticked?
  pause
  exit /b 1
)
echo.
echo Starting the site. Your browser opens in a few seconds: http://localhost:8501
echo Keep this window open while you use the site. Close it (or press Ctrl+C) to stop.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
python -m streamlit run app/app.py --server.port 8501
pause
