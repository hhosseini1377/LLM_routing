from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uuid
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from router_system.async_prompt_handler import AsyncPromptHandler
from router_system.engine_config import EngineConfig
from huggingface_hub import login
import os 
# Define the FastAPI application
app = FastAPI()

prompt_queue = asyncio.Queue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A lifespan context manager to handle startup and shutdown events.
    """
    global background_task
    # --- Startup Logic (runs before the yield) ---
    # while True:
    #     print("Application startup complete. Starting background task...")
    #     time.sleep(2)
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    engine_args = AsyncEngineArgs(
        model=EngineConfig.model,
        quantization=EngineConfig.quantization,
        dtype=EngineConfig.dtype,
        gpu_memory_utilization=EngineConfig.memory_utilization,
        swap_space=EngineConfig.swap_space,
        enforce_eager=EngineConfig.enforce_eager,
        max_model_len=EngineConfig.max_model_len,
        kv_cache_dtype=EngineConfig.kv_cache_dtype,
        max_num_seqs=EngineConfig.max_num_seqs,
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print('salammmmmmmmm')
    sampling_params = SamplingParams(
        temperature=EngineConfig.temperature,
        top_p=EngineConfig.top_p,
        top_k=EngineConfig.top_k,
        max_tokens=EngineConfig.max_tokens_per_request,
    )

    prompt_handler = AsyncPromptHandler(
        llm_engine=llm_engine,
        sampling_params=sampling_params,
        promt_queue=prompt_queue,
        verbose=True
    )
    print('salam')
    background_task = asyncio.create_task(prompt_handler.start_async_engine())
    
    # The `yield` is the separator between startup and shutdown logic
    yield
    
    # --- Shutdown Logic (runs after the yield) ---
    print("Application is shutting down. Cancelling background task...")
    if background_task:
        background_task.cancel()
        # Wait for the task to finish its cancellation
        try:
            await background_task
        except asyncio.CancelledError:
            pass  # Expected exception
    print("Shutdown complete.")


# Tell FastAPI to use our lifespan event handler
app = FastAPI(lifespan=lifespan)
print('App initialized')
# --- Part 3: Request Handling Logic (API Endpoints) ---

# Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

@app.post("/submit_prompt")
async def submit_prompt(request: PromptRequest):
    """
    API endpoint to receive a prompt from a client.
    """
    request_id = str(uuid.uuid4())
    prompt_data = {"id": request_id, "prompt": request.prompt}
    
    # This is a non-blocking operation
    await prompt_queue.put(prompt_data)
    
    return {
        "message": "Prompt received and queued for processing.",
        "request_id": request_id,
    }

@app.get("/")
async def read_root():
    """
    A simple root endpoint to show the server is running.
    """
    return {"message": "Server is running."}

