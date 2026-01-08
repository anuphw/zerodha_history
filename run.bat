@echo off
echo ============================================
echo   Zerodha History - Account Analyzer
echo ============================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv package manager...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Please restart this script after uv installation completes.
    pause
    exit /b
)

echo Installing dependencies...
uv sync

echo.
echo Installing browser (first run only)...
uv run playwright install chromium

echo.
echo ============================================
echo   Starting Zerodha History
echo ============================================
echo.
uv run python zerodha_history.py --fetch

echo.
echo ============================================
echo   Done! Check the tmp folder for your report.
echo ============================================
pause
