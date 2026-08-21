# Voice RAG

Speak a question, get a grounded answer. `audio -> Deepgram STT -> engineered
chunking + vector retrieval -> guardrail check -> Gemini generation`, wired
end to end with a retry/timeout harness and structured I/O at every stage.

## Endpoints

| Endpoint          | In            | Out                          | Use when |
|--------------------|---------------|-------------------------------|----------|
| `POST /api/query`      | JSON `{query}` | JSON answer                   | Debugging/testing without audio |
| `POST /api/voice-query`| audio file     | JSON (transcript + text answer) | Chat UI that shows transcript + text |
| `POST /api/voice-chat` | audio file     | **audio/wav** (spoken answer) | **The actual "speak a question, hear an answer" loop** — metadata (transcript, answer text, guardrail verdict, per-stage latency) rides along in response headers (`X-Transcript`, `X-Answer`, etc., percent-encoded since HTTP headers are ASCII-only) since the body is binary audio, not JSON |

```
audio file
   │  DeepgramService.transcribe_audio()      [retried, timed out, structured]
   ▼
transcript
   │  Retriever.retrieve()                    [embed query -> vector search]
   ▼
retrieved chunks + scores
   │  guardrails.check()                      [local, no network — runs first]
   ├── BLOCK_* ──────────────────► refusal message, LLM never called
   ▼ ALLOW
context
   │  LLMService.generate()                   [retried, timed out, falls back
   ▼                                            to raw context on failure]
answer text
   │  CartesiaService.synthesize()            [retried, timed out — /voice-chat only]
   ▼
spoken answer (audio/wav) + sources + per-stage latency
```

Every stage returns a Pydantic model (`backend/schemas.py`), not a bare
dict — `TranscriptionResult`, `RetrievalResult`, `GuardrailResult`,
`GenerationResult`, rolled up into `RAGResponse` / `VoiceRAGResponse`. That's
what "runs inside a real harness" means here: typed boundaries between
stages, not just a chain of function calls.

## Engineered chunking (`backend/ingestion/chunkers.py`)

Three strategies, chosen per-ingest via `--strategy`:

| Strategy    | How it splits                                             | Good for |
|-------------|------------------------------------------------------------|----------|
| `fixed`     | Sliding window over words, fixed overlap                   | Cheap baseline, unstructured text |
| `sentence`  | Packs whole sentences up to a word budget, never splits one | Text where a sentence = a fact |
| `recursive` | Paragraph-first, recurses into sentence-packing when a paragraph busts the budget | Structured docs — default |

Chunk metadata records which strategy produced it (`chunk_strategy`), so
retrieval results can be inspected/compared across strategies later.

## Guardrails (`backend/guardrails.py`)

Runs **before** the LLM is called, not after, so a query that shouldn't be
answered never pays the network-latency cost of generation:

1. `BLOCK_EMPTY_QUERY` — query is empty or under `min_query_words`.
2. `BLOCK_NO_CONTEXT` — vector store returned zero chunks (empty index).
3. `BLOCK_LOW_RELEVANCE` — requires **both** signals to pass:
   - cosine similarity clears `min_relevance_score` (default `0.35`), **and**
   - at least `min_lexical_overlap` (default `1`) shared non-stopword
     tokens between the query and the top-scoring chunk.
   Cosine similarity alone is noisy: general-purpose embeddings give
   unrelated English sentences a nonzero baseline similarity from shared
   grammar/structure alone. In testing, "the stale smell of old beer
   lingers" scored *higher* against a tech corpus than "what is the
   capital of France?" despite having zero real overlap with the corpus.
   The lexical-overlap check catches that case; cosine alone can't.
4. `ALLOW` — proceeds to generation.

All thresholds live in `backend/config.py` / `.env`.

## Resilience (`backend/services/stt.py`, `llm.py`)

- Both external calls (Deepgram, Gemini) are wrapped in `tenacity` retries
  with exponential backoff (`max_retries`, `retry_backoff_seconds`) and a
  hard `asyncio.wait_for` timeout.
- STT failure after all retries raises a typed `TranscriptionError` ->
  HTTP 502, not a stack trace.
- LLM failure after all retries **degrades instead of hard-failing**: the
  pipeline returns the retrieved context directly with `fallback_used=True`
  rather than a 500, since the retrieval half of the job still succeeded.

## Latency: an honest number, not a lucky one

**"Under 200ms" is scoped to the local leg of the pipeline** — query
embedding + vector search + guardrail check — because that's the part
that's actually in-process. The STT and LLM legs are third-party network
calls (Deepgram, Gemini); no amount of engineering makes a round trip to
an external API reliably land under 200ms, and reporting a number that
pretends otherwise would just be a lucky-run benchmark on a fast day.

`benchmark.py` reports both, honestly, over multiple passes of a real
query set (not one query run once):

```bash
python benchmark.py --runs 20              # local + end-to-end (needs API keys)
python benchmark.py --runs 20 --local-only # local leg only, no network/keys needed
```

It prints and saves P50 / P70 / P100 (and mean) for:
- `local` — embed + retrieve + guardrail (target: comfortably under 200ms)
- `generation` — the Gemini call alone
- `end_to_end` — the full text-query pipeline

The query set intentionally includes one out-of-corpus question ("What is
the capital of France?") so the benchmark also reports how often the
guardrail correctly refuses rather than every query being a clean hit.

Numbers move with hardware, network conditions, and time of day — run it
yourself rather than trusting a number in a README.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DEEPGRAM_API_KEY, GEMINI_API_KEY

# 1. Ingest the sample corpus (downloads the embedding model on first run)
python -m backend.ingestion.pipeline --strategy recursive

# 2. Serve
uvicorn backend.main:app --reload

# 3. Query
curl -X POST localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI used for?"}'

# 4. Or speak it — POST audio to /voice-query for a text answer back...
curl -X POST localhost:8000/api/voice-query \
  -F "file=@data/raw/harvard.wav"

# ...or POST audio to /voice-chat for an actual spoken answer back
curl -X POST localhost:8000/api/voice-chat \
  -F "file=@data/raw/harvard.wav" \
  -D - -o answer.wav
# -D - prints response headers (X-Transcript, X-Answer, latencies);
# -o answer.wav saves the spoken answer — play it to hear the response.
```

## Tests

```bash
pytest tests/ -v
```

The embedding model and both third-party APIs need network access and
keys that won't exist in every environment (CI, sandboxes), so
`tests/conftest.py` provides deterministic offline fakes for the
embedding model and the LLM. `test_pipeline_integration.py` exercises the
full orchestration — guardrail short-circuiting, generation happy path,
empty-index refusal, per-stage latency population — against those fakes,
so the harness itself is tested independent of third-party availability.
`test_chunkers.py`, `test_vector_store.py`, and `test_guardrails.py` cover
each unit directly.

## Known gaps / next steps

- Vector store is a flat JSON file with an in-memory matrix — fine for a
  demo corpus, would move to a proper ANN index (FAISS/Qdrant) past a few
  thousand chunks.
- Only `.txt` ingestion currently; `loaders.py` is the place to add
  PDF/markdown support.
- `/voice-chat` synthesizes the *whole* answer before returning audio
  (no streaming TTS yet) — fine for short answers, adds latency for long
  ones. Cartesia supports SSE/websocket streaming (`tts.sse`,
  `tts.websocket`) if that becomes a problem.

---
#RAGInGoa
