"""
STEP 1 & 2: Corpus Building + LLM-as-Judge Labeling Pipeline

Loads multi-source queries, batches them for LLM labeling via TokenRouter,
handles rate limits, checkpoints progress, and blends with dataset priors.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import numpy as np

# Try to import datasets; graceful fallback to synthetic
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠ 'datasets' library not installed. Using synthetic corpus only.")

# Try OpenAI-compatible client
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠ 'openai' library not installed. Labeling will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
CORPUS_OUTPUT_FILE = "corpus_raw.jsonl"
LABELED_FILE = "corpus_labeled.jsonl"
LABEL_FAILURES_FILE = "label_failures.jsonl"
CHECKPOINT_FILE = "label_checkpoint.json"

BATCH_SIZE = 20
RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

# TokenRouter config
TOKENROUTER_API_KEY = os.getenv("TOKENROUTER_API_KEY", "sk-kzs0pRteikatZWOOpHpddFQM6bhU9jx1nsUiuynl1n1fr3jq")
TOKENROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
LABEL_MODEL = os.getenv("LABEL_MODEL", "z-ai/glm-5.3-free")

# Tier thresholds
TIER_THRESHOLDS = {
    "simple": (1.0, 3.5),
    "medium": (3.5, 6.5),
    "complex": (6.5, 10.0),
}

# ============================================================================
# LABELING RUBRIC (System Prompt)
# ============================================================================
LABRIC_SYSTEM_PROMPT = """You are an expert LLM capability evaluator. Score queries 1-10 on required reasoning depth.

RUBRIC:
- 1-2: Factual lookup, definition, list. No reasoning. (e.g., "What is recursion?")
- 3-4: Simple explanation, single-step logic. (e.g., "Explain how binary search works")
- 5-6: Multi-step procedure, code generation, comparison. (e.g., "Write a function to...")
- 7-8: Complex reasoning, algorithm design, trade-off analysis. (e.g., "Design a cache system")
- 9-10: Advanced: distributed systems, formal proofs, novel architecture. (e.g., "Implement Byzantine consensus")

CRITERIA:
- Reasoning steps (1→0, 5→2, 9→5)
- Domain depth (generic→simple, domain-specific→complex)
- Ambiguity (clear→lower, underspecified→higher)
- Output effort (summary→simple, implementation→complex)

Return ONLY a valid JSON array.

For every query, return exactly:
{"id": "<query id>", "score": <number from 1 to 10>}

Do NOT return a reason.
Do NOT return markdown.
Do NOT add explanations.
Do NOT omit any query."""

ANCHOR_EXAMPLES = [
    {"query": "What is Python?", "score": 1, "tier": "simple"},
    {"query": "List the advantages of Docker", "score": 2, "tier": "simple"},
    {"query": "Explain how quicksort works", "score": 4, "tier": "medium"},
    {"query": "Write a REST API in FastAPI that connects to PostgreSQL", "score": 5, "tier": "medium"},
    {"query": "Design a distributed cache with consistency guarantees", "score": 8, "tier": "complex"},
    {"query": "Implement a Byzantine fault tolerant consensus algorithm", "score": 10, "tier": "complex"},
]

# ============================================================================
# CORPUS BUILDING
# ============================================================================

def hash_query(source: str, text: str) -> str:
    """Stable ID for deduplication."""
    normalized = f"{source}:{text.lower().strip()}"
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def load_simple_queries() -> List[Tuple[str, str, Optional[float]]]:
    """Simple queries: factual, short. Source: Alpaca + synthetic."""
    queries = []
    
    if HAS_DATASETS:
        try:
            alpaca = load_dataset("tatsu-lab/alpaca", split="train[:3000]")
            logger.info("✓ Alpaca dataset loaded for simple tier")
            for item in alpaca:
                text = item["instruction"].strip()
                if len(text.split()) <= 15 and any(w in text.lower() for w in 
                    ["what", "define", "explain", "list", "who", "when", "meaning"]):
                    qid = hash_query("alpaca_simple", text)
                    queries.append((qid, text, None))
                if len(queries) >= 1200:
                    break
        except Exception as e:
            logger.warning(f"✗ Alpaca load failed: {e}")
    
    # Synthetic fallback
    if len(queries) < 1200:
        synthetic_simple = [
            "What is machine learning?",
            "Define recursion",
            "Explain what an API is",
            "List the benefits of Docker",
            "Who invented Python?",
            "What does REST stand for?",
            "Name three advantages of cloud computing",
            "What is a hash table?",
            "Explain what caching means",
            "Define Big O notation",
        ]
        for q in synthetic_simple:
            qid = hash_query("synthetic", q)
            queries.append((qid, q, None))
    
    logger.info(f"✓ Loaded {len(queries)} simple queries")
    return queries


def load_medium_queries() -> List[Tuple[str, str, Optional[float]]]:
    """Medium queries: code, multi-step instructions."""
    queries = []
    
    if HAS_DATASETS:
        try:
            alpaca = load_dataset("tatsu-lab/alpaca", split="train[3000:7000]")
            logger.info("✓ Alpaca dataset loaded for medium tier")
            for item in alpaca:
                text = item["instruction"].strip()
                if len(text.split()) > 15 or any(w in text.lower() for w in
                    ["write", "code", "implement", "explain how", "compare"]):
                    qid = hash_query("alpaca_medium", text)
                    queries.append((qid, text, None))
                if len(queries) >= 1200:
                    break
        except Exception as e:
            logger.warning(f"✗ Alpaca medium load failed: {e}")
    
    # Synthetic fallback
    if len(queries) < 1200:
        synthetic_medium = [
            "Write a Python function to implement binary search",
            "How would you implement an LRU cache?",
            "Compare REST and GraphQL",
            "Write code to reverse a linked list",
            "Explain the difference between SQL and NoSQL",
            "Implement a simple web scraper in Python",
            "How do you optimize a database query?",
            "Write a class to represent a graph",
            "Explain async/await in JavaScript",
            "Design a rate limiter for an API",
        ]
        for q in synthetic_medium:
            qid = hash_query("synthetic", q)
            queries.append((qid, q, None))
    
    logger.info(f"✓ Loaded {len(queries)} medium queries")
    return queries


def load_complex_queries() -> List[Tuple[str, str, Optional[float]]]:
    """Complex queries: math, reasoning, architecture. Source: GSM8K, synthetic."""
    queries = []
    
    if HAS_DATASETS:
        try:
            gsm8k = load_dataset("openai/gsm8k", "main", split="train[:1500]")
            logger.info("✓ GSM8K dataset loaded for complex tier")
            for item in gsm8k:
                text = item["question"].strip()
                # Extract step count as prior
                solution = item["answer"]
                step_count = solution.count("\n")
                prior_score = min(7.0 + step_count * 0.2, 10.0)  # Map steps → [7, 10]
                
                qid = hash_query("gsm8k", text)
                queries.append((qid, text, prior_score))
                if len(queries) >= 1200:
                    break
        except Exception as e:
            logger.warning(f"✗ GSM8K load failed: {e}")
    
    # Synthetic fallback
    if len(queries) < 1200:
        synthetic_complex = [
            "Design and implement a distributed consensus protocol with Byzantine fault tolerance",
            "How would you build a distributed cache system with multi-region replication?",
            "Explain how to optimize a machine learning model for production serving",
            "Design a microservices architecture for a large-scale e-commerce platform",
            "How would you implement a real-time streaming pipeline for analytics?",
            "Design an algorithm to detect anomalies in time-series data at scale",
        ]
        for q in synthetic_complex:
            qid = hash_query("synthetic", q)
            queries.append((qid, q, 7.5))  # Prior score
    
    logger.info(f"✓ Loaded {len(queries)} complex queries")
    return queries


def build_corpus() -> List[Dict]:
    """Assemble corpus from all sources, deduplicate, balance tiers."""
    seen_text = set()
    corpus = []
    
    logger.info("\n=== CORPUS BUILDING PHASE ===")
    
    # Load each tier
    simple = load_simple_queries()
    medium = load_medium_queries()
    complex_q = load_complex_queries()
    
    # Deduplicate & merge
    for qid, text, prior in simple + medium + complex_q:
        norm_text = text.lower().strip()
        if norm_text not in seen_text:
            seen_text.add(norm_text)
            corpus.append({
                "id": qid,
                "text": text,
                "prior_score": prior,
                "labeled_score": None,
            })
    
    logger.info(f"\n✓ Total unique queries: {len(corpus)}")
    logger.info(f"  Simple: {len([q for q in corpus if q['prior_score'] is None or q['prior_score'] < 4])}")
    logger.info(f"  Medium: {len([q for q in corpus if q['prior_score'] and 4 <= q['prior_score'] < 7])}")
    logger.info(f"  Complex: {len([q for q in corpus if q['prior_score'] and q['prior_score'] >= 7])}")
    
    return corpus


# ============================================================================
# LLM LABELING WITH BATCHING & RATE LIMITING
# ============================================================================

def create_labeling_client() -> Optional[OpenAI]:
    """Initialize TokenRouter OpenAI-compatible client."""
    if not HAS_OPENAI:
        logger.warning("OpenAI library not available; skipping LLM labeling")
        return None
    
    if not TOKENROUTER_API_KEY:
        logger.warning("TOKENROUTER_API_KEY not set; skipping LLM labeling")
        return None
    
    return OpenAI(
        base_url=TOKENROUTER_BASE_URL,
        api_key=TOKENROUTER_API_KEY,
    )


def format_batch_prompt(queries: List[Dict]) -> str:
    """Format batch of queries as JSON for LLM."""
    batch_json = json.dumps([
        {"id": q["id"], "query": q["text"]}
        for q in queries
    ])
    
    examples_str = "\n".join([
        f"  - '{ex['query']}' → {ex['score']} ({ex['tier']})"
        for ex in ANCHOR_EXAMPLES
    ])
    
    prompt = f"""Label these queries with complexity scores [1-10]:

EXAMPLES:
{examples_str}

BATCH (respond with JSON array ONLY):
{batch_json}"""
    
    return prompt


def parse_batch_response(response_text: str, batch_ids: List[str]) -> Tuple[Dict, List[str]]:
    """Parse LLM response, extract scores. Returns (labeled_dict, failed_ids)."""
    labeled = {}
    failed = []
    
    try:
        # Extract JSON array
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            results = json.loads(json_str)
            
            for result in results:
                qid = result.get("id")
                score = result.get("score")
                if qid and isinstance(score, (int, float)):
                    labeled[qid] = {
                        "score": float(np.clip(score, 1.0, 10.0)),
                        "reason": result.get("reason", ""),
                    }
            
            # Identify missing IDs
            for qid in batch_ids:
                if qid not in labeled:
                    failed.append(qid)
        else:
            failed = batch_ids
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        failed = batch_ids
    
    return labeled, failed


def label_corpus_batch(corpus: List[Dict], client: Optional[OpenAI]) -> List[Dict]:
    """Label corpus in batches with rate limiting & checkpointing."""
    if not client:
        logger.warning("Skipping LLM labeling (no client)")
        return corpus
    
    # Load checkpoint
    checkpoint = {}
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
        logger.info(f"✓ Loaded checkpoint: {len(checkpoint)} labeled")
    
    # Track failures
    failures = []
    
    logger.info("\n=== LLM LABELING PHASE ===")
    logger.info(f"Batch size: {BATCH_SIZE}, delay: {RATE_LIMIT_DELAY}s")
    
    # Process in batches
    for batch_idx in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[batch_idx : batch_idx + BATCH_SIZE]
        batch_ids = [q["id"] for q in batch]
        
        # Skip already labeled
        batch = [q for q in batch if q["id"] not in checkpoint]
        if not batch:
            logger.info(f"Batch {batch_idx // BATCH_SIZE + 1}: skipped (already labeled)")
            continue
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        # Call LLM with retries
        labeled = {}
        for attempt in range(MAX_RETRIES):
            try:
                prompt = format_batch_prompt(batch)
                response = client.chat.completions.create(
                    model=LABEL_MODEL,
                    messages=[
                        {"role": "system", "content": LABRIC_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=3000,
                )
                
                response_text = response.choices[0].message.content
                labeled, failed = parse_batch_response(response_text, batch_ids)
                
                # Checkpoint
                checkpoint.update(labeled)
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(checkpoint, f)
                
                logger.info(f"Batch {batch_idx // BATCH_SIZE + 1}: {len(labeled)} labeled, {len(failed)} failed")
                
                # Log failures
                if failed:
                    for fid in failed:
                        failures.append({"id": fid, "error": "missing_in_response"})
                
                break
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY * (BACKOFF_FACTOR ** attempt))
                else:
                    for q in batch:
                        failures.append({"id": q["id"], "error": str(e)})
    
    # Write failures
    if failures:
        with open(LABEL_FAILURES_FILE, "w") as f:
            for failure in failures:
                f.write(json.dumps(failure) + "\n")
        logger.warning(f"✗ {len(failures)} labeling failures written to {LABEL_FAILURES_FILE}")
    
    # Merge labeled scores into corpus
    for q in corpus:
        if q["id"] in checkpoint:
            q["labeled_score"] = checkpoint[q["id"]]["score"]
    
    return corpus


def blend_scores(q: Dict) -> float:
    """Blend LLM score with dataset prior. Returns final score [1, 10]."""
    llm_score = q.get("labeled_score")
    prior_score = q.get("prior_score")
    
    if llm_score is None and prior_score is None:
        return 5.0  # Default
    elif llm_score is None:
        return prior_score
    elif prior_score is None:
        return llm_score
    else:
        # Blend: 70% LLM, 30% prior
        blended = 0.7 * llm_score + 0.3 * prior_score
        
        # Flag if discrepancy > 3
        if abs(llm_score - prior_score) > 3.0:
            logger.warning(f"Large discrepancy for {q['id']}: LLM={llm_score:.1f}, Prior={prior_score:.1f}")
        
        return np.clip(blended, 1.0, 10.0)


def save_labeled_corpus(corpus: List[Dict]) -> None:
    """Save labeled corpus to JSONL."""
    with open(LABELED_FILE, "w") as f:
        for q in corpus:
            final_score = blend_scores(q)
            q["final_score"] = final_score
            f.write(json.dumps(q) + "\n")
    
    logger.info(f"\n✓ Saved {len(corpus)} labeled queries to {LABELED_FILE}")


def main():
    """Main pipeline: build corpus → label → save."""
    logger.info("="*60)
    logger.info("COMPLEXITY CLASSIFIER - CORPUS LABELING PIPELINE")
    logger.info("="*60)
    
    # Build corpus
    corpus = build_corpus()
    
    # Label with LLM
    client = create_labeling_client()
    if client:
        corpus = label_corpus_batch(corpus, client)
    
    # Save
    save_labeled_corpus(corpus)
    
    logger.info("\n" + "="*60)
    logger.info(f"Corpus ready at: {LABELED_FILE}")
    logger.info("Next: python train_classifier.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
