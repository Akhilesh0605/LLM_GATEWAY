# app/llm_client.py
"""
Universal Multi-Provider Async LLM Client.
Supports Groq, OpenAI, Gemini, Anthropic, Together AI.
Uses async httpx + tenacity retries. No LangChain.
"""

import time
import logging
import httpx
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ============================================================================
# PROVIDER REGISTRY (model catalogs for dropdowns & base URLs)
# ============================================================================

PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "format": "openai",
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
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

# Pricing map for budget tracking (per 1 Million tokens)
COST_PER_1M_TOKENS: Dict[str, float] = {
    # Groq (current catalog; legacy IDs removed)
    "openai/gpt-oss-120b": 0.375,
    "openai/gpt-oss-20b": 0.187,
    "qwen/qwen3.8-27b": 0.20,
    "allam-2-7b": 0.10,
    # OpenAI
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "o1-mini": 1.10,
    "o1-preview": 7.50,
    "gpt-5.6-sol": 17.50,
    "gpt-5.6-terra": 8.75,
    # Gemini
    "gemini-2.0-flash": 0.00,
    "gemini-2.0-flash-lite": 0.00,
    "gemini-1.5-flash": 0.00,
    "gemini-1.5-pro": 0.00,
    "gemini-3.5-flash": 0.00,
    "gemini-3.8-flash": 0.00,
    # Anthropic
    "claude-3-5-haiku-20241022": 0.80,
    "claude-3-5-sonnet-20241022": 3.00,
    "claude-haiku-4.5": 3.00,
    "claude-sonnet-5": 6.00,
    "claude-fable-5.1": 30.00,
    # Together AI
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.18,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": 0.88,
    "Qwen/Qwen2.5-72B-Instruct-Turbo": 1.20,
    "deepseek-ai/DeepSeek-R1": 5.00,
}


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def get_provider_models(provider: str) -> List[str]:
    """Returns available model list for a provider (used for frontend dropdown)."""
    provider = provider.lower()
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: '{provider}'. Available: {list(PROVIDER_REGISTRY.keys())}")
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