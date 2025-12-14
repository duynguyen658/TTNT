#!/bin/bash
# Bash script để start Python Backend + ngrok tự động
# Usage: ./start-backend.sh

echo "========================================"
echo "🚀 Starting Plant Disease AI Backend"
echo "========================================"
echo ""

# Kiểm tra Python
echo "📦 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python not found! Please install Python first."
    exit 1
fi
echo "✅ $(python3 --version)"

# Kiểm tra ngrok
echo ""
echo "🌐 Checking ngrok..."
if ! command -v ngrok &> /dev/null; then
    echo "⚠️  ngrok not found!"
    echo "👉 Install ngrok: https://ngrok.com/download"
    echo ""
    read -p "Continue without ngrok? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Kiểm tra file api_server.py
if [ ! -f "api_server.py" ]; then
    echo "❌ api_server.py not found!"
    exit 1
fi

# Kiểm tra requirements
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt not found!"
else
    echo ""
    echo "📋 Checking dependencies..."
    echo "💡 Tip: Run 'pip install -r requirements.txt' if needed"
fi

# Start Python API
echo ""
echo "🚀 Starting Python API Server..."
echo "   API will run at: http://localhost:8000"
echo ""

# Start Python API trong background
python3 api_server.py &
PYTHON_PID=$!

# Đợi API khởi động
echo "⏳ Waiting for API to start..."
sleep 5

# Kiểm tra API health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running!"
else
    echo "⚠️  API might still be starting..."
fi

# Start ngrok nếu có
if command -v ngrok &> /dev/null; then
    echo ""
    echo "🌐 Starting ngrok tunnel..."
    echo "   ngrok dashboard: http://localhost:4040"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Copy the ngrok URL (e.g., https://abc123.ngrok-free.app)"
    echo "   2. Add to Vercel: PYTHON_API_URL=<ngrok-url>"
    echo "   3. Press Ctrl+C to stop both services"
    echo ""

    # Start ngrok
    ngrok http 8000

    # Cleanup khi ngrok dừng
    kill $PYTHON_PID
else
    echo ""
    echo "📋 Backend is running at: http://localhost:8000"
    echo "   Start ngrok manually: ngrok http 8000"
    echo ""
    echo "Press Ctrl+C to stop the API server"

    # Đợi user stop
    wait $PYTHON_PID
fi
