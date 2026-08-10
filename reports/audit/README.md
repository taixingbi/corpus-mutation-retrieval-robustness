# Human semantic audit (RQ1)

Manual check that mutations look like valid corpus edits for their failure type. This is **not** behavioral effectiveness (RQ2).

## Sample

After a main run:

```bash
python scripts/export_semantic_audit.py --per-failure 50
```

| File | Contents |
|------|----------|
| `semantic_audit_sample.jsonl` | ~50 rows / failure (seed=0) |
| `semantic_audit_summary.json` | Optional aggregate after labeling |

Re-export after regenerating mutations (e.g. full `facm -c configs/main.yaml run`).

## How to label

For each JSONL row, fill:

| Field | Values |
|-------|--------|
| `auditor_label` | `ok` \| `topical_fail` \| `answer_leak` \| `unnatural` \| `other` |
| `auditor_notes` | short free text (optional; required for non-`ok`) |

Use the question, `gold_answer`, and `added_texts`. Missing has empty `added_texts` (gold removed).

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
| `boundary` | Split **through** gold answer → A/B; neither piece alone contains full gold; union covers it |
| `noise` | Distractors; may be loosely related; should not be a clean answer substitute |

Do **not** reject because retrieval failed or ranks dropped — that is RQ2.

## Workflow

1. Export sample after main semantic stage.
2. Label ≥30–50 / failure.
3. Tally for RQ1 (ok rate, error types; split LLM vs Random HN if needed).

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
path = Path("reports/audit/semantic_audit_sample.jsonl")
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
