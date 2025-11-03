# Use official Python image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

#TODO: Map instead of copyting
# Copy local files into container
# COPY . /app
COPY requirements.txt .

# Install dependencies if you have requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if your application uses a specific port)
EXPOSE 8000

CMD ["python3", "-m", "router_system.run_server", "--model_name", "TheBloke/Mistral-7B-Instruct-v0.1-AWQ", "--utilization", "0.5", "--dtype", "float16"]
# CMD ["uvicorn", "router_system.main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
