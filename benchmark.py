# benchmark.py
"""
Self-Healing Gateway Benchmarking & Shadow Evaluation Suite.
Dynamically fetches active provider models, measures latency P50/P95,
and calculates real-world cost savings.
"""

import os
import sys
import time
import httpx
import numpy as np

BASE_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ----------------------------------------------------------------------------
# 1. VALIDATE API KEY UPFRONT
# ----------------------------------------------------------------------------
if not GROQ_KEY or len(GROQ_KEY) < 10:
    print("\n\033[91m[ERROR] GROQ_API_KEY environment variable is missing or empty!\033[0m")
    print("Please set your real Groq key in PowerShell first:")
    print('  $env:GROQ_API_KEY="gsk_your_actual_key_here"')
    print("  python benchmark.py\n")
    sys.exit(1)

# Benchmark queries across all tiers + repeated queries for cache testing
BENCHMARK_QUERIES = [
    # SIMPLE (Score 1-3)
    "What is 15 * 24?",
    "What is the capital of Japan?",
    "Explain the word 'serendipity' in one sentence.",
    "Write a regex to match an email address.",
    
    # MEDIUM (Score 4-6)
    "Write a Python function to find the longest palindromic substring.",
    "Compare PostgreSQL B-Tree vs GIN indexes for JSONB queries.",
    "Explain the difference between optimistic and pessimistic locking in databases.",
    "Write a Dockerfile for a multi-stage Python FastAPI build.",
    
    # COMPLEX (Score 7-10)
    "Architect a distributed rate limiter for 100k RPS using Redis sliding window and token bucket algorithms.",
    "Explain Paxos consensus algorithm vs Raft for multi-region leader election under network partitions.",
    "How would you execute a zero-downtime schema migration for a 10TB Postgres database under high write load?",
    "Derive the time and space complexity of Dijkstra's algorithm using a Fibonacci heap vs Binary heap.",
    
    # REPEATED QUERIES (Tests Semantic Cache HITs)
    "What is 15 * 24?",
    "What is the capital of Japan?",
    "Write a regex to match an email address.",
    "Architect a distributed rate limiter for 100k RPS using Redis sliding window and token bucket algorithms.",
]


def run_benchmark():
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    print("\n" + "=" * 70)
    print("      🚀 LAUNCHING LLM GATEWAY BENCHMARK & EVALUATION SUITE      ")
    print("=" * 70)

    # 1. Setup User & Key
    user_email = f"benchmark_{int(time.time())}@example.com"
    reg_resp = client.post("/v1/auth/register", json={"name": "Benchmark Runner", "email": user_email})
    if reg_resp.status_code != 201:
        print(f"\033[91mRegistration failed: {reg_resp.text}\033[0m")
        sys.exit(1)
        
    gw_key = reg_resp.json()["gateway_api_key"]
    headers = {"X-Gateway-Key": gw_key, "Content-Type": "application/json"}

    # 2. Add Provider Key
    key_resp = client.post("/v1/providers/add", json={"provider": "groq", "api_key": GROQ_KEY, "label": "Bench Key"}, headers=headers)
    if key_resp.status_code != 200:
        print(f"\033[91mFailed to add provider key: {key_resp.text}\033[0m")
        sys.exit(1)

    # 3. Dynamically Fetch Available Groq Models
    models_resp = client.get("/v1/providers/groq/models")
    available_models = models_resp.json()
    
    simple_model = available_models[0]
    medium_model = available_models[1] if len(available_models) > 1 else available_models[0]
    complex_model = available_models[-1]

    # 4. Configure Tiers with Active Models
    client.put("/v1/tiers/config", json={
        "simple_provider": "groq",
        "simple_model": simple_model,
        "medium_provider": "groq",
        "medium_model": medium_model,
        "complex_provider": "groq",
        "complex_model": complex_model
    }, headers=headers)

    print(f"✓ Registered tenant: {user_email}")
    print(f"✓ Model Routing: Simple -> {simple_model} | Medium -> {medium_model} | Complex -> {complex_model}\n")
    print(f"{'#':<3} | {'QUERY':<42} | {'TIER':<8} | {'SCORE':<5} | {'CACHE':<6} | {'LATENCY':<8}")
    print("-" * 84)

    miss_latencies = []
    hit_latencies = []
    total_cost = 0.0

    for idx, query in enumerate(BENCHMARK_QUERIES, 1):
        start = time.time()
        resp = client.post("/query", json={"query": query}, headers=headers)
        elapsed_ms = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            is_hit = data.get("cache_hit", False)
            score = data.get("complexity_score", 0)
            tier = data.get("tier", "").upper()
            cost = data.get("cost_usd", 0.0)
            total_cost += cost

            if is_hit:
                hit_latencies.append(elapsed_ms)
                cache_status = "\033[92mHIT\033[0m"
            else:
                miss_latencies.append(elapsed_ms)
                cache_status = "\033[93mMISS\033[0m"

            trunc_q = query[:39] + "..." if len(query) > 39 else query
            print(f"{idx:<3} | {trunc_q:<42} | {tier:<8} | {score:<5} | {cache_status:<15} | {elapsed_ms:>6.1f}ms")
        else:
            print(f"{idx:<3} | \033[91mFAILED: {resp.text}\033[0m")

    # Wait for background shadow evaluation tasks in Postgres
    print("\n⏳ Settle background Shadow Evaluation tasks...")
    time.sleep(2.0)

    # 5. Fetch Aggregated Analytics
    analytics_resp = client.get("/analytics", headers=headers)
    analytics = analytics_resp.json()

    # 6. Performance Summary
    p50_miss = np.percentile(miss_latencies, 50) if miss_latencies else 0
    p95_miss = np.percentile(miss_latencies, 95) if miss_latencies else 0
    avg_hit = np.mean(hit_latencies) if hit_latencies else 0
    speedup = (p50_miss / avg_hit) if avg_hit > 0 else 0

    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK REPORT                      ")
    print("=" * 70)
    print(f"  Total Queries Processed : {len(BENCHMARK_QUERIES)}")
    print(f"  Semantic Cache Hits     : {len(hit_latencies)} ({analytics.get('cache_hit_rate')}%)")
    print(f"  LLM Cache Misses        : {len(miss_latencies)}")
    print("-" * 70)
    print(f"  Cache MISS P50 Latency  : {p50_miss:.1f} ms")
    print(f"  Cache MISS P95 Latency  : {p95_miss:.1f} ms")
    print(f"  Cache HIT Avg Latency   : {avg_hit:.1f} ms")
    print(f"  ⚡ Latency Speedup       : {speedup:.1f}x FASTER on Cache Hit")
    print("-" * 70)
    print(f"  Total Spend (Actual)    : ${analytics.get('total_cost_usd'):.6f}")
    print(f"  Total Cost Saved        : ${analytics.get('cost_saved_usd'):.6f} ({analytics.get('cost_saved_percent')}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()