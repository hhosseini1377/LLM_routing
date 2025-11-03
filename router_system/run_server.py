# run_server.py

import argparse
import uvicorn
from router_system.main import app
from router_system.engine_config import EngineConfig

def main():
    parser = argparse.ArgumentParser(description="Run the Router System API server with custom arguments.")
    
    parser.add_argument('--model_name', type=str, default='mistral', help='Name of the model to use.')
    parser.add_argument('--utilization', type=str, default='0.5', help='GPU memory utilization.')
    parser.add_argument('--dtype', type=str, default='8', help='Data type for model weights.')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on.')
    EngineConfig.model = parser.parse_args().model_name
    EngineConfig.memory_utilization = float(parser.parse_args().utilization)
    EngineConfig.dtype = parser.parse_args().dtype
    port = parser.parse_args().port
    print('Trying to run the server')
    # Launch the uvicorn server programmatically
    # The 'app' object is passed directly.
    uvicorn.run(app, host="0.0.0.0", port=port, lifespan="on")

if __name__ == "__main__":
    main()