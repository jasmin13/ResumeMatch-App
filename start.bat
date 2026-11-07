@echo off
title ResumeMatch AI - Private Talent Screener
echo ========================================================
echo   Starting ResumeMatch AI Engine...
echo ========================================================
echo [1/2] Loading PyTorch & Local AI Model into memory...
echo [2/2] Browser will automatically open once server is READY.
echo.

:: Wait for server to respond on port 8082 before opening browser
start /min cmd /c "for /L %%i in (1,1,60) do (curl.exe -s http://localhost:8082/api/stats >nul 2>&1 && (start http://localhost:8082 & exit) || timeout /t 1 /nobreak >nul)"

python server.py
pause
