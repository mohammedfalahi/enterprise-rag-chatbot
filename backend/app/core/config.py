from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Knowledge Intelligence Assistant"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:54322/postgres"
    
    # Knowledge Vault / RAG Configuration
    ollama_embedding_model: Literal["qwen3-embedding:0.6b"] = "qwen3-embedding:0.6b"
    ollama_embedding_dimensions: Literal[1024] = 1024
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 120.0
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_rerank_score: float = -8.5
    
    # Generative Model Configuration
    query_rewrite_model: str = "gemini-3.5-flash-lite"
    chat_llm_model: str = "gemini-3.5-flash-lite"
    GEMINI_API_KEY: str = ""

    # Alternative LLM Provider Configuration (Groq - OpenAI-compatible)
    USE_GROQ_FOR_QUERY_REWRITE: bool = False
    USE_GROQ_FOR_GENERATION: bool = False
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    groq_query_rewrite_model: str = "openai/gpt-oss-120b"
    groq_chat_llm_model: str = "openai/gpt-oss-120b"

    # Multi-Turn Conversation Memory
    conversation_history_window: int = 9
    
    # Supabase Settings
    SUPABASE_URL: str = "http://127.0.0.1:54321"
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_INVITE_REDIRECT_URL: str = "http://localhost:3000/accept-invite"
    SUPABASE_JWT_SECRET: str = ""
    SUPABASE_JWKS_URL: str | None = None
    SUPABASE_JWT_AUDIENCE: str = "authenticated"
    SUPABASE_JWT_ISSUER: str | None = None

    @property
    def jwks_url(self) -> str:
        if self.SUPABASE_JWKS_URL and self.SUPABASE_JWKS_URL.strip():
            return self.SUPABASE_JWKS_URL.strip()
        base = (
            self.SUPABASE_URL.rstrip("/")
            if self.SUPABASE_URL and self.SUPABASE_URL.strip()
            else "http://127.0.0.1:54321"
        )
        return f"{base}/auth/v1/.well-known/jwks.json"

    @property
    def jwt_issuer(self) -> str:
        if self.SUPABASE_JWT_ISSUER and self.SUPABASE_JWT_ISSUER.strip():
            return self.SUPABASE_JWT_ISSUER.strip()
        base = (
            self.SUPABASE_URL.rstrip("/")
            if self.SUPABASE_URL and self.SUPABASE_URL.strip()
            else "http://127.0.0.1:54321"
        )
        return f"{base}/auth/v1"
    
    # Observability & Langfuse Configuration (Reused from DEV_AI)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_capture_content: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
