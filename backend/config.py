from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Voice RAG"
    environment: str = "development"

    # --- provider keys ---
    deepgram_api_key: str = ""
    groq_api_key: str = ""
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Cartesia default demo voice
    cartesia_model_id: str = "sonic-2"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    # --- retrieval ---
    vector_store_path: str = "vector_data/store.json"
    default_top_k: int = 5
    default_chunk_strategy: str = "recursive"

    # --- guardrails ---
    # Cosine-similarity floor a top result must clear before we let the
    # LLM answer from it at all.
    min_relevance_score: float = 0.35
    # Minimum number of chunks that must clear min_relevance_score.
    min_supporting_chunks: int = 1
    # Minimum number of shared non-stopword tokens between the query and
    # the top-scoring chunk, required alongside min_relevance_score. Cosine
    # similarity alone lets an off-topic query score higher than a
    # relevant one by chance (see backend/guardrails.py docstring); this
    # second signal catches that case.
    min_lexical_overlap: int = 1
    # Below this word count a query is treated as too short/ambiguous to
    # safely answer.
    min_query_words: int = 2

    # --- resilience / harness ---
    stt_timeout_seconds: float = 10.0
    llm_timeout_seconds: float = 8.0
    tts_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5

    # --- latency budget (used by the benchmark harness, not enforced at
    # request time since it depends on network-bound third-party calls) ---
    target_pipeline_ms: float = 200.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
