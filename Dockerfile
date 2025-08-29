# Use official Python image
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

#TODO: Map instead of copyting
# Copy local files into container
# COPY . /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Install dependencies if you have requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if your application uses a specific port)
EXPOSE 8000

# CMD ["python3", "-m", "router_system.run_server", "--model_name", "TheBloke/Mistral-7B-Instruct-v0.1-AWQ", "--utilization", "0.5", "--dtype", "float16"]
CMD ["uvicorn", "router_system.main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
