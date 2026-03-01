#!/bin/bash

# Start the Endee Vector Database in the background
echo "Starting Endee Vector Database..."
/usr/local/bin/endee &

# Give Endee a few seconds to fully initialize
sleep 3

# Start the FastAPI Web Server on Hugging Face's required port (7860)
echo "Starting FastAPI server..."
uvicorn src.api:app --host 0.0.0.0 --port 7860
