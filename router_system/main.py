from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uuid
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from router_system.async_prompt_handler import AsyncPromptHandler
from router_system.engine_config import EngineConfig
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
    print("Application startup complete. Starting background task...")
    engine_config = EngineConfig()
    engine_args = AsyncEngineArgs(
        model=engine_config.model,
        dtype="half",
        gpu_memory_utilization=0.95,
        swap_space=3,
        enforce_eager=True,
        max_model_len=1024,
        kv_cache_dtype="fp8_e5m2",
        disable_log_requests=True
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=engine_config.temperature,
        top_p=engine_config.top_p,
        top_k=engine_config.top_k,
        max_tokens=512,
    )

    prompt_handler = AsyncPromptHandler(
        llm_engine=llm_engine,
        sampling_params=sampling_params,
        promt_queue=prompt_queue,
        verbose=True
    )

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

