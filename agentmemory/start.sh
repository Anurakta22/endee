#!/bin/bash

# Find the Endee binary dynamically since its path may vary
echo "Looking for Endee binary..."
ENDEE_BIN=$(find / -maxdepth 5 -type f -executable \( -name "endee*" -o -name "ndd*" \) 2>/dev/null | grep -v proc | head -1)

if [ -z "$ENDEE_BIN" ]; then
    echo "Endee binary not found by name. Checking entrypoints..."
    # Fallback: look for any binary in /usr/local/bin or /usr/bin that's not a system tool
    ENDEE_BIN=$(find /usr/local/bin /usr/bin -maxdepth 1 -type f -executable 2>/dev/null | grep -v -E "python|pip|curl|bash|sh|ls|cat|grep|find|awk|sed" | head -1)
fi

echo "Starting Endee Vector Database at: $ENDEE_BIN"
if [ -n "$ENDEE_BIN" ]; then
    $ENDEE_BIN &
else
    echo "WARNING: Could not find Endee binary. Attempting to run 'endee-server'..."
    endee-server &
fi

# Give Endee a few seconds to fully initialize
echo "Waiting for Endee to start..."
sleep 5

# Start the FastAPI Web Server on Hugging Face's required port (7860)
echo "Starting FastAPI server on port 7860..."
python -m uvicorn src.api:app --host 0.0.0.0 --port 7860
