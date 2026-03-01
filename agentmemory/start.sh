#!/bin/bash

echo "=== Starting AgentMemory ==="
echo "=== Searching for Endee binary ==="

# Print all executables to debug (helps read the HF logs)
echo "All executables found in /usr/local/bin:"
ls -la /usr/local/bin/ 2>/dev/null || echo "(empty)"

echo "All executables found in /usr/bin (non-standard):"
ls /usr/bin/ 2>/dev/null | grep -vE "^(python|pip|curl|bash|sh|ls|cat|grep|find|awk|sed|apt|dpkg|tar|gzip|cp|mv|rm|ln|chmod|chown|mkdir|touch|echo|test|true|false|env|head|tail|sort|wc|cut|tr|diff|du|df|pwd|which|type|set|export|unset|read|printf|date|id|groups|whoami|hostname|uname|ps|pgrep|kill|sleep|wait|exit|nohup|timeout)$" 2>/dev/null

echo "Files in root /:"
ls / 2>/dev/null | head -20

# Try starting Endee using common binary names
STARTED=false

for CMD in endee endee-server ndd ndd-server endeeio /endee /endee-server /ndd /usr/local/bin/endee-server; do
    if command -v "$CMD" &>/dev/null || [ -f "$CMD" ]; then
        echo "Found Endee at: $CMD — starting..."
        $CMD &
        STARTED=true
        break
    fi
done

if [ "$STARTED" = false ]; then
    echo "WARNING: Could not find Endee binary via name. Trying entrypoint from PATH..."
    # Last resort: find any unusual executable
    ENDEE_BIN=$(find /usr/local/bin /usr/bin / -maxdepth 2 -type f -executable 2>/dev/null \
        | grep -vE "(python|pip|curl|bash|sh|ls|cat|grep|find|awk|sed|apt|dpkg|tar|gzip|cp|mv|rm|ln|chmod|chown|mkdir|touch|echo|test|true|false|env|head|tail|sort|wc|cut|tr|diff|du|df|pwd|which|type|date|id|groups|whoami|hostname|uname|ps|pgrep|kill|sleep|nohup|timeout)" \
        | head -1)
    if [ -n "$ENDEE_BIN" ]; then
        echo "Trying binary: $ENDEE_BIN"
        $ENDEE_BIN &
    else
        echo "ERROR: No Endee binary found at all. API will start but Endee will be unavailable."
    fi
fi

echo "=== Waiting 5s for Endee to initialize ==="
sleep 5

echo "=== Starting FastAPI on port 7860 ==="
python -m uvicorn src.api:app --host 0.0.0.0 --port 7860
