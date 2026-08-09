# Human semantic audit (RQ1)

Manual check that mutations look like valid corpus edits for their failure type. This is **not** behavioral effectiveness (RQ2).

## Sample

| File | Contents |
|------|----------|
| `semantic_audit_sample.jsonl` | ~50 rows / failure (seed=0), from main semantic mutations |
| `semantic_audit_summary.json` | Aggregated human labels (written after audit) |

### Labeled results (n=250)

| Failure | ok rate | Notes |
|---------|---------|--------|
| missing | 50/50 | — |
| conflict | 50/50 | — |
| noise | 50/50 | — |
| hard_negative | 26/50 | LLM HN 26/26 ok; Random HN 0/24 (`topical_fail`) |
| boundary | 0/50 | all `answer_leak` (a split piece still holds full gold) |

Overall human ok: **176/250 (70.4%)**. Among auto `semantic_passed`: 171/234 (73.1%).

Regenerate:

```bash
python scripts/export_semantic_audit.py --per-failure 50
```

## How to label

For each JSONL row, fill:

| Field | Values |
|-------|--------|
| `auditor_label` | `ok` \| `topical_fail` \| `answer_leak` \| `unnatural` \| `other` |
| `auditor_notes` | short free text (optional; required for non-`ok`) |

Use the question, `gold_answer`, and `added_texts` (chunks injected by the mutation). Missing has empty `added_texts` (gold removed).

### Labels

| Label | Meaning |
|-------|---------|
| `ok` | Edit matches failure intent; Wikipedia-like; no gold answer leak |
| `topical_fail` | Off-topic or wrong entity/relation for the failure |
| `answer_leak` | Added text directly gives / restates the gold answer when it should not |
| `unnatural` | Not plausible corpus text (gibberish, prompt residue, broken split) |
| `other` | Invalid for another reason (note it) |

### Failure intent (what “ok” means)

| Failure | Expect |
|---------|--------|
| `missing` | Gold removed; no substitute answer chunk required |
| `hard_negative` | Same entity/relation, **wrong** plausible fact; topical, does not correctly answer |
| `conflict` | Contradicts gold / competing claim on the same question |
| `boundary` | Split gold into incomplete pieces A/B; together cover gold, each alone incomplete |
| `noise` | Distractors; may be loosely related; should not be a clean answer substitute |

Do **not** reject because retrieval failed or ranks dropped — that is RQ2.

## Suggested workflow

1. Open `semantic_audit_sample.jsonl` (one object per line).
2. Label all rows (or at least 30–50 per failure for the paper).
3. Keep labels in this file (or copy to `semantic_audit_labeled.jsonl` if you want a clean original).
4. Report agreement / error rates in RQ1 (e.g. % `ok`, breakdown by failure and error type).

Optional tally after labeling:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
path = Path("reports/audit/semantic_audit_sample.jsonl")
# or: semantic_audit_labeled.jsonl
labels, by_fail = Counter(), Counter()
for line in path.open():
    r = json.loads(line)
    lab = (r.get("auditor_label") or "").strip() or "(unlabeled)"
    labels[lab] += 1
    by_fail[(r["failure"], lab)] += 1
print("overall", dict(labels))
for (f, lab), n in sorted(by_fail.items()):
    print(f"{f:16} {lab:14} {n}")
PY
```
