# Use official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

#TODO: Map instead of copyting
# Copy local files into container
# COPY . /app
COPY requirements.txt .
# Install dependencies if you have requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if your application uses a specific port)
EXPOSE 8000

# Default command
# CMD ["python3", "-m", "router_system.run_server", "--model_name", "TheBloke/Mistral-7B-Instruct-v0.1-AWQ", "--utilization", "0.5", "--dtype", "float16"]
CMD ["uvicorn", "router_system.main:app", "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
