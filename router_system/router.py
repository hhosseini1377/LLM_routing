import queue
import threading
import time
from vllm import LLM, SamplingParams

# --- 1. Prompt Queue ---
prompt_queue = queue.Queue()

# --- 2. vLLM Consumer (Worker) ---
class VLLMWorker(threading.Thread):
    def __init__(self, queue_in, llm_model, sampling_params):
        super().__init__()
        self.queue_in = queue_in
        self.llm = llm_model
        self.sampling_params = sampling_params
        self.running = True

    def run(self):
        print("vLLM worker thread started.")
        while self.running:
            try:
                # Dequeue a prompt with a timeout to avoid blocking forever
                item = self.queue_in.get(timeout=1)
                if item is None:  # A special sentinel to stop the worker
                    print("Received stop signal. Stopping worker.")
                    self.running = False
                    break

                prompt, result_callback = item
                print(f"Processing prompt: '{prompt}'")

                # Generate the response using vLLM
                outputs = self.llm.generate([prompt], self.sampling_params)

                # Get the generated text
                generated_text = outputs[0].outputs[0].text
                print(f"Generated text: '{generated_text}'")

                # Call the callback with the result
                result_callback(generated_text)

                # Mark the task as done
                self.queue_in.task_done()

            except queue.Empty:
                # Queue is empty, just continue and check again
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                # It's good practice to handle errors and mark the task as done
                # even on failure to prevent the queue from getting stuck.
                self.queue_in.task_done()

    def stop(self):
        self.running = False

# --- 3. Prompt Producer ---
def add_prompt_to_queue(prompt, callback_func):
    """
    Adds a new prompt to the queue for processing.
    """
    print(f"Adding prompt to queue: '{prompt}'")
    prompt_queue.put((prompt, callback_func))

# --- 4. Main Application Logic ---
def main():
    # Load the vLLM model once at the start
    print("Loading vLLM model...")
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
    print("vLLM model loaded.")

    # Start the vLLM worker thread
    vllm_worker = VLLMWorker(prompt_queue, llm, sampling_params)
    vllm_worker.start()

    # Define a callback function to handle the results
    def result_handler(text_result):
        print(f"\n--- Result received ---")
        print(f"Result: {text_result}")
        print(f"----------------------\n")

    # Simulate adding prompts to the queue
    prompts = [
        "What is the capital of France?",
        "Tell me a short story about a brave knight.",
        "What are the benefits of using a queue in software design?"
    ]

    for p in prompts:
        add_prompt_to_queue(p, result_handler)
        time.sleep(2) # Simulate some delay between requests

    # Wait for all prompts to be processed
    print("Waiting for queue to be empty...")
    prompt_queue.join()

    # Signal the worker thread to stop
    prompt_queue.put(None)
    vllm_worker.join()

    print("All tasks are done. Program finished.")

if __name__ == "__main__":
    main()