import time
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings

settings = get_settings()

COST_PER_MILLION_TOKENS = {
    settings.MODEL_SIMPLE: 0.05,
    settings.MODEL_MEDIUM: 0.59,
    settings.MODEL_COMPLEX: 2.00,
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4)
)
async def _call_groq(model: str, query: str, api_key: str):
    client = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=1000,
    )
    return response


async def call_llm(
    model: str,
    query: str,
    fallback_chain: list[str],
    groq_api_key: str | None = None,
) -> tuple[str, int, float, float]:
    """
    Calls Groq LLM using user's decrypted API key (or global fallback).
    Returns: (response_text, tokens_used, cost_usd, latency_ms)
    """
    api_key = groq_api_key or settings.GROQ_API_KEY
    if not api_key:
        raise ValueError("No Groq API key available for completion request.")

    models_to_try = [model] + fallback_chain

    for current_model in models_to_try:
        try:
            start = time.perf_counter()
            response = await _call_groq(current_model, query, api_key)
            latency_ms = (time.perf_counter() - start) * 1000

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            cost_usd = (tokens_used / 1_000_000) * COST_PER_MILLION_TOKENS.get(current_model, 0.0)

            return response_text, tokens_used, cost_usd, latency_ms

        except Exception:
            continue

    raise RuntimeError("All models in fallback chain failed")# app/llm_client.py
"""
Universal Multi-Provider Async LLM Client.
Supports Groq, OpenAI, Gemini, Anthropic, Together AI.
Uses async httpx + tenacity retries. No LangChain.

To add a new provider: add one entry to PROVIDER_REGISTRY.
"""

import time
import logging
import httpx
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ============================================================================
# PROVIDER REGISTRY (model catalogs for dropdown + API config)
# ============================================================================
PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "format": "openai",
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gpt-oss-20b",
            "gpt-oss-120b",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "format": "gemini",
        "models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "format": "openai",
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "o1-mini",
            "o1-preview",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "format": "anthropic",
        "models": [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-haiku-4.5",
            "claude-sonnet-5",
            "claude-fable-5.1",
        ],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "format": "openai",
        "models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
        ],
    },
}

# Unified pricing map containing verified current and legacy rates (Blended Input/Output tracking)
COST_PER_1M_TOKENS: Dict[str, float] = {
    # Groq (including new MoE lines)
    "llama-3.1-8b-instant": 0.05,
    "llama-3.3-70b-versatile": 0.59,
    "llama3-8b-8192": 0.05,
    "llama3-70b-8192": 0.59,
    "mixtral-8x7b-32768": 0.24,
    "gemma2-9b-it": 0.20,
    "gpt-oss-20b": 0.187,   # Blended average ($0.075 in / $0.30 out)
    "gpt-oss-120b": 0.375,  # Blended average ($0.15 in / $0.60 out)
    
    # OpenAI (including new Frontier tier)
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
    "o1-mini": 1.10,
    "o1-preview": 7.50,
    "gpt-5.6-sol": 17.50,   # Blended average ($5.00 in / $30.00 out)
    "gpt-5.6-terra": 8.75,  # Blended average ($2.50 in / $15.00 out)
    
    # Gemini (standard free tier parameters)
    "gemini-2.0-flash": 0.00,
    "gemini-2.0-flash-lite": 0.00,
    "gemini-2.5-flash-preview-04-17": 0.00,
    "gemini-2.5-pro-preview-05-06": 0.00,
    "gemini-1.5-flash": 0.00,
    "gemini-1.5-pro": 0.00,
    "gemini-3.5-flash": 0.00,
    "gemini-3.8-flash": 0.00,
    
    # Anthropic Core Lineup
    "claude-3-haiku-20240307": 0.25,
    "claude-3-5-haiku-20241022": 0.80,
    "claude-3-5-sonnet-20241022": 3.00,
    "claude-3-opus-20240229": 15.00,
    "claude-haiku-4.5": 3.00,   # Blended average ($1.00 in / $5.00 out)
    "claude-sonnet-5": 6.00,   # Blended average ($2.00 in / $10.00 out)
    "claude-fable-5.1": 30.00, # Blended average ($10.00 in / $50.00 out)
    
    # Together AI Engine
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.18,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 0.88,
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": 3.50,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.60,
    "Qwen/Qwen2.5-72B-Instruct-Turbo": 1.20,
    "deepseek-ai/DeepSeek-R1": 5.00, # Blended average ($3.00 in / $7.00 out)
}


# ============================================================================
# PUBLIC HELPERS (used by main.py dropdown endpoint)
# ============================================================================

def get_provider_models(provider: str) -> List[str]:
    """Returns available model list for a provider (for frontend dropdown)."""
    provider = provider.lower()
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")
    return PROVIDER_REGISTRY[provider]["models"]


def get_all_providers() -> List[str]:
    """Returns list of all supported provider names."""
    return list(PROVIDER_REGISTRY.keys())


# ============================================================================
# UNIVERSAL LLM CLIENT
# ============================================================================

class UniversalLLMClient:

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def chat(
        self,
        provider: str,
        model: str,
        api_key: str,
        messages: list,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Universal chat completion. Dispatches to correct API format.

        Args:
            provider: "groq", "openai", "gemini", "anthropic", "together"
            model: Model string from provider's catalog
            api_key: Decrypted provider API key (in-memory only)
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            {"response": str, "tokens_used": int, "cost_usd": float, "latency_ms": int}
        """
        provider = provider.lower()
        if provider not in PROVIDER_REGISTRY:
            raise ValueError(f"Unsupported provider: '{provider}'")

        config = PROVIDER_REGISTRY[provider]
        start_time = time.time()

        if config["format"] == "openai":
            text, tokens = await self._call_openai_compatible(
                config["base_url"], api_key, model, messages, max_tokens, temperature
            )
        elif config["format"] == "anthropic":
            text, tokens = await self._call_anthropic(
                config["base_url"], api_key, model, messages, max_tokens, temperature
            )
        elif config["format"] == "gemini":
            text, tokens = await self._call_gemini(
                config["base_url"], api_key, model, messages, max_tokens, temperature
            )
        else:
            raise ValueError(f"Unknown format: {config['format']}")

        latency_ms = int((time.time() - start_time) * 1000)
        rate = COST_PER_1M_TOKENS.get(model, 0.0)
        cost_usd = (tokens / 1_000_000) * rate

        return {
            "response": text,
            "tokens_used": tokens,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": latency_ms,
        }

    # ── OpenAI-Compatible (Groq, OpenAI, Together) ──

    async def _call_openai_compatible(
        self, base_url: str, api_key: str, model: str,
        messages: list, max_tokens: int, temperature: float
    ) -> tuple[str, int]:
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return text, tokens

    # ── Anthropic Native ──

    async def _call_anthropic(
        self, base_url: str, api_key: str, model: str,
        messages: list, max_tokens: int, temperature: float
    ) -> tuple[str, int]:
        url = f"{base_url}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if system_msg:
            payload["system"] = system_msg

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["content"][0]["text"]
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
        return text, tokens

    # ── Google Gemini Native ──

    async def _call_gemini(
        self, base_url: str, api_key: str, model: str,
        messages: list, max_tokens: int, temperature: float
    ) -> tuple[str, int]:
        url = f"{base_url}/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        contents = []
        system_instruction = None
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        tokens = usage.get("totalTokenCount", 0)
        return text, tokens


# Singleton instance
llm_client = UniversalLLMClient()