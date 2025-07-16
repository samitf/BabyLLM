from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer

MEMORY_PATH = "memory.json"

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
generator = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, "w") as f:
        json.dump([], f)

with open(MEMORY_PATH, "r", encoding="utf-8") as f:
    memory = json.load(f)

def update_embeddings():
    questions = [m["question"] for m in memory]
    return embedder.encode(questions, convert_to_tensor=True) if questions else None

question_embeddings = update_embeddings()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

class Feedback(BaseModel):
    question: str

class Correction(BaseModel):
    question: str
    answer: str

@app.post("/chat")
def chat(msg: Message):
    user_q = msg.message.strip()
    if not user_q:
        return {"reply": "Please say something.", "bot_name": "Thursday"}

    if not memory:
        return {"reply": "I don't know. Can you teach me the right answer?", "bot_name": "Thursday"}

    input_embed = embedder.encode(user_q, convert_to_tensor=True)
    scores = util.cos_sim(input_embed, question_embeddings)[0]
    top_idx = int(scores.argmax())
    top_score = float(scores[top_idx])

    top_answer = memory[top_idx]["answer"]
    if top_score >= 0.9:
        return {"reply": top_answer, "bot_name": "Thursday"}

    # Merge top 3 for T5 generation
    top_indices = scores.argsort(descending=True)[:3]
    context = " ".join([memory[int(i)]["answer"] for i in top_indices])
    prompt = f"question: {user_q} context: {context}"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output = generator.generate(input_ids, max_length=64)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"reply": reply, "bot_name": "Thursday"}

@app.post("/like")
def like(feedback: Feedback):
    # You can log or increment like count if needed
    print(f"[LIKE] {feedback.question}")
    return {"status": "liked"}

@app.post("/correct")
def correct(c: Correction):
    # Append corrected Q-A to memory
    memory.append({"question": c.question.strip(), "answer": c.answer.strip()})
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

    # Recalculate embeddings
    global question_embeddings
    question_embeddings = update_embeddings()

    print(f"[CORRECTED] Q: {c.question} => A: {c.answer}")
    return {"status": "correction saved"}