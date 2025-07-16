from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, os, requests

# === CONFIG ===
MEMORY_PATH = "memory.json"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in Render
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

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

    # Simple fuzzy match
    matches = [m for m in memory if user_q.lower() in m["question"].lower()]
    if matches:
        return {"reply": matches[0]["answer"], "bot_name": "Thursday"}

    # Merge last 3 answers
    context = " ".join([m["answer"] for m in memory[-3:]])
    prompt = f"question: {user_q}\ncontext: {context}"

    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return {"reply": reply, "bot_name": "Thursday"}
    except Exception as e:
        print("OpenRouter error:", e)
        return {"reply": "Sorry, I'm having trouble thinking right now 🧠", "bot_name": "Thursday"}
