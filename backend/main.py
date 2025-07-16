from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, os, requests

# === CONFIG ===
MEMORY_PATH = "memory.json"
HF_API_KEY = os.getenv("HF_API_KEY")  # Set this on Render dashboard
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# === INIT MEMORY ===
if not os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, "w") as f:
        json.dump([], f)

with open(MEMORY_PATH, "r", encoding="utf-8") as f:
    memory = json.load(f)

# === APP INIT ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELS ===
class Message(BaseModel):
    message: str

class Feedback(BaseModel):
    question: str

class Correction(BaseModel):
    question: str
    answer: str

# === CHAT LOGIC ===
@app.post("/chat")
def chat(msg: Message):
    user_q = msg.message.strip()
    if not user_q:
        return {"reply": "Please say something.", "bot_name": "Thursday"}

    if not memory:
        return {"reply": "I don't know. Can you teach me the right answer?", "bot_name": "Thursday"}

    # Very basic semantic matching (no embeddings, memory-light)
    matches = [m for m in memory if user_q.lower() in m["question"].lower()]
    if matches:
        return {"reply": matches[0]["answer"], "bot_name": "Thursday"}

    # Merge top 3 answers for context (naively)
    context = " ".join([m["answer"] for m in memory[-3:]])
    prompt = f"question: {user_q} context: {context}"

    payload = {"inputs": prompt}
    try:
        res = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=20)
        res.raise_for_status()
        result = res.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return {"reply": result[0]["generated_text"], "bot_name": "Thursday"}
        else:
            return {"reply": "Hmm, I couldn’t think of a good answer.", "bot_name": "Thursday"}
    except Exception as e:
        print("Error:", e)
        return {"reply": "Something went wrong with my brain 😓", "bot_name": "Thursday"}

# === LIKE ===
@app.post("/like")
def like(feedback: Feedback):
    print(f"[LIKE] {feedback.question}")
    return {"status": "liked"}

# === CORRECTION ===
@app.post("/correct")
def correct(c: Correction):
    memory.append({"question": c.question.strip(), "answer": c.answer.strip()})
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    print(f"[CORRECTED] Q: {c.question} => A: {c.answer}")
    return {"status": "correction saved"}
