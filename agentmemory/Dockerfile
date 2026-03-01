FROM endeeio/endee-server:latest AS endee_base

FROM python:3.10-slim

# Copy the Endee binary from the official image
COPY --from=endee_base /endee /usr/local/bin/endee

# Ensure Endee binary is executable
RUN chmod +x /usr/local/bin/endee

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy and setup the start script
COPY start.sh .
RUN chmod +x start.sh

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run the start script
CMD ["./start.sh"]
