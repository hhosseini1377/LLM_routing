# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import random

app = FastAPI()

# ======= Config =======
MODEL_A_URL = "http://localhost:8000/model_a"  # Dummy small model
MODEL_B_URL = "http://localhost:8000/model_b"  # Dummy large model

# ======= Router Model (Simple Heuristic for Demo) =======
def route_prompt(prompt: str) -> str:
    if len(prompt.split()) < 20:
        return "A"  # Use small model
    return "B"  # Use big model

# ======= Request Schema =======
class QueryRequest(BaseModel):
    prompt: str

# ======= Main Endpoint =======
@app.post("/query")
async def query_router(req: QueryRequest):
    model_choice = route_prompt(req.prompt)
    model_url = MODEL_A_URL if model_choice == "A" else MODEL_B_URL

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(model_url, json={"prompt": req.prompt})
            output = response.json()
        except Exception as e:
            return {"error": str(e)}

    return {
        "model_used": model_choice,
        "output": output
    }

# ======= Dummy Model Endpoints for Testing =======
@app.post("/model_a")
async def model_a(req: QueryRequest):
    return {"response": f"[Model A] Processed: {req.prompt}"}

@app.post("/model_b")
async def model_b(req: QueryRequest):
    return {"response": f"[Model B] Processed: {req.prompt}"}
