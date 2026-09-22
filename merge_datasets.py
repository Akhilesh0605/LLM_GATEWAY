# merge_datasets.py
"""
Merges:
  1. Root 'corpus_labeled.jsonl' (keys: 'text', 'final_score' / 'labeled_score')
  2. 'data/corpus_labeled.jsonl' (keys: 'query', 'complexity_score')
Into 'data/merged_corpus.jsonl' with unified schema {'id', 'text', 'score'}.
Tier definitions:
  - SIMPLE:  score <= 3.0
  - MEDIUM:  3.0 < score < 7.0
  - COMPLEX: score >= 7.0
"""

import os
import json


def extract_record(item: dict, fallback_id: str):
    if not isinstance(item, dict):
        return None

    # 1. Resolve Text (handles 'query', 'text', 'prompt')
    text = item.get("text") or item.get("query") or item.get("prompt")

    # 2. Resolve Score with explicit key priority
    score = None
    for key in ["complexity_score", "final_score", "score", "labeled_score"]:
        if key in item and item[key] is not None:
            try:
                score = float(item[key])
                break
            except (ValueError, TypeError):
                continue

    if text and score is not None:
        return {
            "id": str(item.get("id", fallback_id)),
            "text": str(text).strip(),
            "score": round(score, 2),
        }
    return None


def load_file(path: str):
    if not os.path.exists(path):
        print(f"Warning: File not found at '{path}'")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Case A: Standard JSON Array `[...]`
    if content.startswith("["):
        try:
            data = json.loads(content)
            for idx, item in enumerate(data, 1):
                rec = extract_record(item, fallback_id=f"arr_{idx}")
                if rec:
                    records.append(rec)
            print(f"Loaded {len(records):,} records from JSON array: {path}")
            return records
        except json.JSONDecodeError:
            pass

    # Case B: Dict-of-dicts `{"id": {"text": ...}}`
    if content.startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                first_val = next(iter(data.values()), None)
                if isinstance(first_val, dict):
                    for uid, item in data.items():
                        rec = extract_record(item, fallback_id=str(uid))
                        if rec:
                            records.append(rec)
                    print(f"Loaded {len(records):,} records from Dict-of-dicts: {path}")
                    return records
        except json.JSONDecodeError:
            pass

    # Case C: JSONL (One JSON object per line)
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            rec = extract_record(item, fallback_id=f"line_{line_num}")
            if rec:
                records.append(rec)
        except json.JSONDecodeError:
            continue

    print(f"Loaded {len(records):,} records from JSONL: {path}")
    return records


def main():
    root_file = "corpus_labeled.jsonl"
    data_file = os.path.join("data", "corpus_labeled.jsonl")
    output_file = os.path.join("data", "merged_corpus.jsonl")

    os.makedirs("data", exist_ok=True)

    print("--- [1/3] Loading Both Dataset Sources ---")
    records_root = load_file(root_file)
    records_data = load_file(data_file)

    all_records = records_root + records_data
    if not all_records:
        print("Error: No valid records loaded from either file.")
        return

    print(f"\nTotal raw combined count: {len(all_records):,}")

    print("--- [2/3] Deduplicating by Query Content ---")
    seen_texts = set()
    deduped = []
    for r in all_records:
        norm_text = r["text"].lower()
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            deduped.append(r)

    print(f"Total deduplicated count: {len(deduped):,}")

    print("--- [3/3] Saving Unified Dataset ---")
    with open(output_file, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    # Tier statistics with score >= 7.0 for COMPLEX
    scores = [r["score"] for r in deduped]
    simple = sum(1 for s in scores if s <= 3.0)
    medium = sum(1 for s in scores if 3.0 < s < 7.0)
    complex_ = sum(1 for s in scores if s >= 7.0)

    print("\n" + "=" * 50)
    print("        FINAL MERGED DATASET BREAKDOWN           ")
    print("=" * 50)
    print(f"  Total Valid Records : {len(deduped):,}")
    print(f"  Score Range         : [{min(scores):.2f} - {max(scores):.2f}]")
    print(f"  SIMPLE  (score <= 3.0) : {simple:>5} ({simple/len(deduped)*100:.1f}%)")
    print(f"  MEDIUM  (3.0 < s < 7.0): {medium:>5} ({medium/len(deduped)*100:.1f}%)")
    print(f"  COMPLEX (score >= 7.0) : {complex_:>5} ({complex_/len(deduped)*100:.1f}%)")
    print("=" * 50)
    print(f"✓ Successfully written to: {output_file}\n")


if __name__ == "__main__":
    main()