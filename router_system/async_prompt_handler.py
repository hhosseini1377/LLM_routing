from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs


# We need to store a reference to the background task to cancel it later
background_task = None

class AsyncPromptHandler:
    def __init__(self, 
                 llm_engine, 
                 sampling_params, 
                 promt_queue,
                 verbose = True,
                 outputs= None):
        self.llm_engine = llm_engine
        self.sampling_params = sampling_params
        self.prompt_queue = promt_queue
        self.verbose = verbose
        self.outputs = outputs if outputs is not None else {}

    async def run_async_prompt(self, request):
        """
        Process a single prompt using the vLLM engine.
        """

        result_generator = self.llm_engine.generate(
            prompt=request['prompt'],
            sampling_params=self.sampling_params,
            request_id=request['id'],  # Unique ID for tracking
        )
        final_output = None

        async for result in result_generator:
            final_output = result.outputs[0].text
        
        self.outputs[request['id']] = final_output
        if self.verbose:
            print("Outout generated")

    async def start_async_engine(self):
        """
        A long-running background task that processes prompts from a queue.
        """
        print("Background worker has started...")
        while True:
            try:
                request = await self.prompt_queue.get()
                print(f"Background worker processing prompt (ID: {request['id']})...")

                asyncio.create_task(self.run_async_prompt(request))
                print('Prompt added to processing tasks. Waiting for next prompt...')
            
            except asyncio.CancelledError:
                print("Background worker task was cancelled.")
                break
            except Exception as e:
                print(f"An error occurred in the background worker: {e}")

    def get_output(self, request_id):
        """
        Retrieve the output for a specific request ID.
        """
        try:
            return self.outputs.get(request_id, None)
        except KeyError:
            print(f"No output found for request ID: {request_id}")
            return None
    
    @property
    def generated_outputs_count(self):
        return len(self.outputs)