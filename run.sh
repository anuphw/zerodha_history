#!/bin/bash
echo "============================================"
echo "  Zerodha History - Account Analyzer"
echo "============================================"
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo
    echo "Please restart your terminal and run this script again."
    exit 1
fi

echo "Installing dependencies..."
uv sync

echo
echo "Installing browser (first run only)..."
uv run playwright install chromium

echo
echo "============================================"
echo "  Starting Zerodha History"
echo "============================================"
echo
uv run python zerodha_history.py --fetch

echo
echo "============================================"
echo "  Done! Check the tmp folder for your report."
echo "============================================"
