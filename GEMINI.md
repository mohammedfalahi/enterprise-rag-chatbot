# Gemini CLI — Project Instructions

You are working on the Enterprise RAG Chatbot project.

## Read These Files First

Before starting any task, read:

1. `architecture.md` — system architecture and technical decisions.
2. `learning.md` — implementation history and completed tasks.
3. `build_plan.md` — implementation roadmap, if available.

Do not assume a feature is implemented just because it appears in the architecture.

## Development Rules

1. Work on **one small task at a time**.
2. Inspect the relevant code before proposing changes.
3. Explain the current implementation and identify the exact problem.
4. Propose the smallest reasonable solution.
5. List the files you intend to create, modify, or delete.
6. Explain risks, dependencies, and required tests.
7. **Wait for my explicit approval before modifying code.**
8. Implement only the approved task.
9. Do not silently expand the task or start the next task automatically.
10. Preserve existing working functionality and architectural decisions.

## Testing and Documentation

After implementation:

- Run relevant tests.
- Report actual test results. Never claim tests passed if they were not run.
- List every file created, modified, or deleted.
- Explain important technical decisions.
- Record known limitations.
- Append a task entry to `learning.md`.

Never overwrite or delete previous `learning.md` entries.

If implementation changes the system architecture, propose the corresponding `architecture.md` update and wait for approval.

## Security Rules

- Never expose `.env` values, API keys, JWTs, or credentials.
- Never trust tenant IDs or roles supplied by the frontend.
- Never bypass backend authorization.
- Never allow the LLM to make access-control decisions.
- Apply tenant and role restrictions to both dense and sparse retrieval.
- Do not perform destructive database operations or migrations without approval.
- Do not modify the original evaluation dataset.


## Important Technical Decisions

- Backend: FastAPI and Python.
- Frontend: Next.js.
- Database: Supabase PostgreSQL with pgvector.
- Current embedding model: Gemini Embedding 001.
- Current embedding dimension: 768.
- Retrieval: Dense vector search + PostgreSQL Full-Text Search + RRF.
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2.
- Generation: Gemini.

Do not change these technologies or their configurations without approval.

## Communication Style

Be precise and practical. Explain technical decisions clearly.

Do not implement an entire feature set in one operation. Work incrementally:

**Inspect → Explain → Propose → Wait for approval → Implement → Test → Document → Stop.**