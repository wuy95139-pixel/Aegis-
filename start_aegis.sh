#!/bin/bash
echo "============================================================"
echo "  Aegis - 多智能体个人工作助理"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -f venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
pip install -r requirements.txt -q 2>/dev/null

echo ""
echo "Starting Aegis web server..."
echo ""
echo "  Open http://localhost:7860 in your browser"
echo "  Press Ctrl+C to stop"
echo ""

# Open browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:7860 &
elif command -v open &> /dev/null; then
    open http://localhost:7860 &
fi

# Start server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 7860 --reload
