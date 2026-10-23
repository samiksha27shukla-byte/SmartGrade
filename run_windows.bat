@echo off
cd /d "%~dp0"
echo Installing requirements (first time only)...
python -m pip install -r requirements.txt
cd backend
echo.
echo Starting ASPIRE on http://127.0.0.1:5000  (first run trains the model, about 10 seconds)
python app.py
pause
