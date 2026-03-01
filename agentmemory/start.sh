#!/bin/bash

echo "=== Starting AgentMemory ==="

# The Endee binaries are: ndd-avx2, ndd-avx512, ndd-neon, ndd-sve2
# Hugging Face runs on x86_64 - use entrypoint.sh which auto-detects the right binary
# If entrypoint.sh doesn't exist, fall back to ndd-avx2

if [ -f "/usr/local/bin/entrypoint.sh" ]; then
    echo "Using Endee entrypoint.sh to auto-detect the correct binary..."
    # Run entrypoint.sh in background - it handles CPU detection automatically
    /usr/local/bin/entrypoint.sh &
else
    echo "Falling back to ndd-avx2 for x86_64..."
    /usr/local/bin/ndd-avx2 &
fi

echo "=== Waiting 5s for Endee to initialize on port 8080 ==="
sleep 5

# Verify Endee started
if curl -sf http://localhost:8080/api/v1/index/list > /dev/null 2>&1; then
    echo "=== Endee is running! ==="
else
    echo "=== WARNING: Endee may not have started. Attempting ndd-avx2 directly... ==="
    /usr/local/bin/ndd-avx2 &
    sleep 3
fi

echo "=== Starting FastAPI on port 7860 ==="
python -m uvicorn src.api:app --host 0.0.0.0 --port 7860
