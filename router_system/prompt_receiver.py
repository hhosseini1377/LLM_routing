from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
# Define the FastAPI application
app = FastAPI()

# --- Part 1: Other Logics (Background Task) ---
# Create an asynchronous queue to hold prompts for background processing
prompt_queue = asyncio.Queue()

# We need to store a reference to the background task to cancel it later
background_task = None

class BackgroundWorker:
    def __init__(self):
        self.args = AsyncEngineArgs(model="mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=1,gpu_memory_utilization=0.9)
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.args)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    async def run_single_prompt(self, prompt):
        """
        Process a single prompt using the vLLM engine.
        """
        print(f"Processing prompt: {prompt}")
        result_generator = self.llm_engine.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            request_id=str(uuid.uuid4()),  # Unique ID for tracking
        )
        final_output = None
        async for result in result_generator:
            final_output = result.outputs[0].text
        print(f"Generated response: {final_output}")

    async def start_background(self):
        """
        A long-running background task that processes prompts from a queue.
        """
        print("Background worker has started...")
        while True:
            try:
                request = await prompt_queue.get()
                print(f"Background worker processing prompt (ID: {request['id']})...")

                asyncio.create_task(self.run_single_prompt(request['text']))
                print('Prompt added to processing tasks. Waiting for next prompt...')
            
            except asyncio.CancelledError:
                print("Background worker task was cancelled.")
                break
            except Exception as e:
                print(f"An error occurred in the background worker: {e}")

# --- Part 2: The New Lifespan Event Handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A lifespan context manager to handle startup and shutdown events.
    """
    global background_task
    # --- Startup Logic (runs before the yield) ---
    print("Application startup complete. Starting background task...")

    backgournd_worker = BackgroundWorker()
    background_task = asyncio.create_task(backgournd_worker.start_background())
    
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
    prompt_data = {"id": request_id, "text": request.prompt}
    
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