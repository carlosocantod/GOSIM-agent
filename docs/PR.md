## TopicFlow — Documentation & Environment Update

### Summary

- Rewrites and expands `README.md` with accurate cost/performance figures, a current-state section, known limitations, updated TODO list, and proper demo links (online demo + YouTube placeholder).
- Updates `.env.sample` to expose Langfuse observability variables and clarify existing settings.

### What changed and why

**README.md**
- Added **Performance & Cost** section: ~$0.014 per run, ~3 min 30 s wall-clock time (progressive UI feedback makes this feel close to real-time), cost-control strategy via mini LLMs.
- Added **Observability** section explaining Langfuse integration: traces every LangGraph node, LLM-as-a-judge search-quality evaluations are production-ready but prompt-tuning is still needed for precision. Langfuse adds some latency; a production deployment would use the async SDK path.
- Added **Known limitations / current state** section documenting:
  - OpenAlex as the current bottleneck; semantic search alternatives (including OpenAlex beta) not yet evaluated.
  - Context truncation on LLM calls to control token spend.
  - Result diversity/aggregation assessment still in progress.
  - UI is an exploitable MVP; richer output formatting is planned.
- Updated **TODO** list to reflect actual priorities: query-result caching, output polish, aggregation diversity, source expansion, LLM-as-a-judge prompt tuning.
- Added prominent **online demo** link and a **YouTube demo placeholder** (to be filled in).

**.env.sample**
- Added `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (all optional, for observability).
- Clarified `TOPIC_MODEL_MAX_DOC_CHARS` default and added comment guidance.

### Test plan

- [ ] Verify the HF Spaces demo link is reachable.
- [ ] Check that `.env.sample` variables match what the running app reads (`os.getenv` calls in `src/`).
- [ ] Review rendered README in GitHub preview for formatting.
- [ ] Replace YouTube placeholder URL once demo video is published.
