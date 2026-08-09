# Models used in FACM (Paper 4)

Scope: **Generate → Validate → Transfer**. See [PAPER.md](PAPER.md).

## LLM (mutation + semantic judge)

| Role | Model ID | Serving |
|------|----------|---------|
| Generator / judge | `openai.gpt-oss-120b-1:0` | Bedrock OpenAI-compat (`FACM_INFERENCE_URL`) |

Env: `FACM_INFERENCE_URL`, `FACM_INFERENCE_API_KEY`, `FACM_INFERENCE_MODEL`  
See `example_inference.md`.

## Retrievers

### Construction / validation (RQ2) — may select accepted mutations

| Name | Model / method |
|------|----------------|
| `bm25` | BM25Okapi |
| `bge_base` | `BAAI/bge-base-en-v1.5` |
| `hybrid` | BM25 + BGE RRF |

### Held-out (RQ3) — never used for acceptance

| Name | Model |
|------|-------|
| `e5_large` | `intfloat/e5-large-v2` (`query:` / `passage:` prefixes) |
| `gte_large` | `thenlper/gte-large` |

Local SentenceTransformers; embeddings cached under `{output_dir}/_embed_cache/`.

## Main run scale

- `configs/main.yaml` — ~531 merged unique seeds + Random HN baseline  
- `configs/pilot.yaml` — 200-seed archive  
- 1000 seeds requires more RAGFailBench clean-seed generation (Paper 2)
