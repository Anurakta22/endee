# Use the Endee server as the base image so we don't need to copy any binary
FROM endeeio/endee-server:latest

# Switch to root so we can install system packages
USER root

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' point to python3/pip3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Override the default Endee entrypoint and run our start script
ENTRYPOINT []
CMD ["/bin/bash", "start.sh"]
