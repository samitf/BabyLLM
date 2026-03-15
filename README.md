# 🧒 BabyLLM — Community-Built Self-Learning Knowledge Agent

BabyLLM is a minimal, low-cost AI assistant that **learns from its community**. Instead of relying on massive pre-trained knowledge, BabyLLM starts nearly blank and grows smarter as people teach it facts, correct its mistakes, and reinforce good answers.

**Built for small organizations** who don't need a billion-dollar model — just a smart assistant trained on *their own data*.

## How It Works

```
Community teaches facts → Stored as vector embeddings (FAISS)
User asks a question   → Relevant memories retrieved → LLM generates answer
User gives feedback    → Corrections/reinforcements stored → Model improves
```

**Key Concepts:**
- **Memory-only answering**: BabyLLM only answers from what it's been taught. No hallucination from general knowledge.
- **Community learning**: Every user can teach, correct, and verify — the model improves collectively.
- **Self-reinforcement**: Good answers get stored back as verified knowledge, strengthening retrieval.
- **Low cost**: Uses Groq's free API tier (Llama 3) + local embeddings. No GPU required.

## Quick Start

### 1. Get a Groq API Key (free)
Go to [console.groq.com](https://console.groq.com) → Create account → Generate API key.

### 2. Configure
Open `.env` and replace the placeholder:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Launch
```bash
docker compose up --build
```

### 4. Open
Visit **http://localhost:8000** in your browser.

That's it! Start teaching BabyLLM in the **Teach** tab, then ask questions in the **Chat** tab.

## Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat** | Ask questions — BabyLLM answers from its learned memory |
| 📝 **Teach** | Teach individual facts or bulk-upload knowledge files |
| 📂 **File Upload** | Drop a `.txt` file to teach entire documents at once |
| 👍 **Feedback Loop** | Rate answers as good (reinforces) or correct mistakes |
| 🧠 **Memory Browser** | View everything BabyLLM has learned |
| 📊 **Stats** | Track memory growth and model configuration |

## Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS)                 │
│  Served by FastAPI static files         │
├─────────────────────────────────────────┤
│  FastAPI Backend                        │
│  ├── /api/teach      → Add knowledge   │
│  ├── /api/ask        → Query + LLM     │
│  ├── /api/feedback   → Learn from users │
│  ├── /api/upload     → File ingestion   │
│  ├── /api/bulk-teach → Batch learning   │
│  ├── /api/memories   → Browse memory   │
│  └── /api/stats      → Analytics       │
├─────────────────────────────────────────┤
│  FAISS Vector Store (persistent)        │
│  Sentence Transformers (all-MiniLM-L6)  │
├─────────────────────────────────────────┤
│  Groq API (Llama 3 8B)                 │
└─────────────────────────────────────────┘
```

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *required* | Your Groq API key |
| `MODEL_NAME` | `llama3-8b-8192` | Groq LLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K` | `5` | Number of memories to retrieve per query |
| `TEMPERATURE` | `0.4` | LLM creativity (lower = more factual) |
| `SIMILARITY_THRESHOLD` | `0.35` | Memory relevance cutoff |

## Running Without Docker

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## API Reference

All endpoints accept/return JSON.

**POST /api/teach** — Teach a single fact
```json
{ "text": "Our CEO is Jane Doe", "source": "hr", "category": "company" }
```

**POST /api/ask** — Ask a question
```json
{ "question": "Who is the CEO?" }
```

**POST /api/feedback** — Correct or reinforce
```json
{ "question": "...", "original_answer": "...", "rating": "correct", "correct_answer": "..." }
```

**POST /api/bulk-teach** — Teach multiple facts
```json
{ "facts": ["Fact 1", "Fact 2", "Fact 3"] }
```

**POST /api/upload** — Upload a text file (multipart form)

**GET /api/stats** — Memory statistics

**GET /api/memories** — List all learned memories

## Roadmap

- [ ] Multi-tenant support (separate knowledge bases per org)
- [ ] Authentication & role-based access
- [ ] Export/import knowledge bases
- [ ] Scheduled self-reflection (model reviews its own memories)
- [ ] Fine-tuning pipeline (when enough data is collected, train a custom model)
- [ ] Web scraping module (teach from URLs)
- [ ] Conversation memory (multi-turn context)

## License

MIT — Use it, modify it, build on it.
