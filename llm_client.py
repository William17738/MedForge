"""
MedForge LLM Client

Unified interface for multiple LLM providers with smart routing and failover.
Supports Google Gemini, Anthropic Claude, and OpenAI GPT models.
"""

import os
import time
import threading
import logging
from typing import Optional
from abc import ABC, abstractmethod

from config import (
    DEFAULT_MODEL, FALLBACK_MODELS, RETRIES_PER_MODEL,
    QUOTA_KEYWORDS, ROUTING_CONFIG, MAX_RETRIES, RETRY_DELAY,
    LOG_DIR
)

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("medforge.llm")
logger.setLevel(logging.INFO)

if not logger.handlers:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "llm_client.log", encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# =============================================================================
# Exceptions
# =============================================================================

class QuotaExhausted(Exception):
    """Raised when API quota is exhausted for a model."""
    pass


class ModelUnavailable(Exception):
    """Raised when a model is temporarily unavailable."""
    pass


# =============================================================================
# LLM Provider Implementations
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, prompt: str, model: str) -> str:
        """Make an API call to the LLM provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        pass


def _normalize_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is None:
        return None
    api_key = api_key.strip()
    return api_key or None


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _normalize_api_key(api_key) or _normalize_api_key(os.environ.get("GEMINI_API_KEY"))
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai package not installed")
        return self._client

    def call(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
        client = self._get_client()
        model_instance = client.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _normalize_api_key(api_key) or _normalize_api_key(os.environ.get("ANTHROPIC_API_KEY"))
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client

    def call(self, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _normalize_api_key(api_key) or _normalize_api_key(os.environ.get("OPENAI_API_KEY"))
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client

    def call(self, prompt: str, model: str = "gpt-4o") -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# =============================================================================
# Provider Registry
# =============================================================================

PROVIDERS = {
    "google": GeminiProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(provider_name: str, api_key: Optional[str] = None) -> LLMProvider:
    """Get an LLM provider instance by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    return PROVIDERS[provider_name](api_key=api_key)


# =============================================================================
# Smart Router
# =============================================================================

class ModelRouter:
    """
    Manages intelligent model switching and routing.

    Features:
    - Automatic failover to backup models
    - Exponential backoff for failed models
    - Periodic retry of primary model
    - Thread-safe state management
    """

    def __init__(self):
        self.current_model = DEFAULT_MODEL
        self.fallback_start_time: Optional[float] = None
        self.last_primary_attempt: Optional[float] = None
        self.primary_fail_count = 0
        self.requests_since_fallback = 0
        self._lock = threading.Lock()

    def should_retry_primary(self) -> bool:
        """Check if we should attempt to route back to primary model."""
        with self._lock:
            if self.current_model == DEFAULT_MODEL:
                return False

            now = time.time()

            # Exponential backoff check
            if self.last_primary_attempt:
                cooldown = min(
                    ROUTING_CONFIG["max_cooldown"],
                    ROUTING_CONFIG["min_cooldown"] * (2 ** min(self.primary_fail_count, 5))
                )
                if now - self.last_primary_attempt < cooldown:
                    return False

            # Request count based retry
            if self.requests_since_fallback >= ROUTING_CONFIG["requests_per_retry"]:
                return True

            # Time-based retry while on fallback
            if self.fallback_start_time:
                if now - self.fallback_start_time > ROUTING_CONFIG["fallback_retry_delay"]:
                    return True

            return False

    def mark_primary_success(self):
        """Called when primary model succeeds."""
        with self._lock:
            self.current_model = DEFAULT_MODEL
            self.fallback_start_time = None
            self.last_primary_attempt = None
            self.primary_fail_count = 0
            self.requests_since_fallback = 0

    def mark_primary_failed(self):
        """Called when primary model fails."""
        with self._lock:
            self.last_primary_attempt = time.time()
            self.primary_fail_count += 1
            self.requests_since_fallback = 0

    def switch_to_fallback(self, model: str):
        """Switch to a fallback model."""
        with self._lock:
            if self.current_model != model:
                self.current_model = model
                if model != DEFAULT_MODEL and not self.fallback_start_time:
                    self.fallback_start_time = time.time()
            self.requests_since_fallback += 1

    def get_current_model(self) -> str:
        """Thread-safe getter for current model."""
        with self._lock:
            return self.current_model


# Global router instance
model_router = ModelRouter()


# =============================================================================
# Main API Functions
# =============================================================================

def call_llm(
    prompt: str,
    model: str = None,
    provider: str = None,
    max_retries: int = MAX_RETRIES,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Call an LLM with automatic retry logic.

    Args:
        prompt: The prompt to send to the model
        model: Model name (defaults to config default)
        provider: Provider name (auto-detected from model if not specified)
        max_retries: Maximum retry attempts
        api_key: Optional API key override (preferred over environment variables)

    Returns:
        Model response text, or None if all retries failed

    Raises:
        QuotaExhausted: If API quota is exhausted
    """
    model = model or DEFAULT_MODEL

    # Find provider from model config if not specified
    if not provider:
        for fm in FALLBACK_MODELS:
            if fm["name"] == model:
                provider = fm["provider"]
                break
        if not provider:
            provider = "google"  # Default to Gemini

    llm_provider = get_provider(provider, api_key=api_key)

    if not llm_provider.is_available():
        raise ModelUnavailable(f"Provider {provider} is not configured")

    start_time = time.time()
    last_error = None

    for attempt in range(max_retries):
        try:
            result = llm_provider.call(prompt, model)

            duration = time.time() - start_time
            if duration > 10:
                logger.info(f"LLM call completed: model={model}, duration={duration:.1f}s")

            return result

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Check for quota exhaustion
            if any(kw in error_msg for kw in QUOTA_KEYWORDS):
                raise QuotaExhausted(f"Quota exhausted for {model}: {e}")

            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model}: {e}")

            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    logger.error(f"All {max_retries} attempts failed for {model}: {last_error}")
    return None


def call_llm_with_smart_routing(
    prompt: str,
    request_id: str = "unknown",
    api_key: str = None,
    *,
    debug_id: str = None,
) -> Optional[str]:
    """
    Call LLM with smart routing and automatic failover.

    This function implements intelligent model switching:
    - Uses the current best available model
    - Automatically fails over to backup models
    - Periodically retries the primary model

    Args:
        prompt: The prompt to send
        request_id: Identifier for logging/debugging
        api_key: Optional API key override
        debug_id: Alias for request_id (kept for backwards compatibility)

    Returns:
        Model response text

    Raises:
        QuotaExhausted: If all models are exhausted
    """
    global model_router
    if debug_id:
        request_id = debug_id

    # Try routing back to primary if conditions are met
    if model_router.should_retry_primary():
        logger.info(f"[{request_id}] Attempting to restore primary model: {DEFAULT_MODEL}")
        try:
            result = call_llm(prompt, model=DEFAULT_MODEL, max_retries=2, api_key=api_key)
            if result:
                logger.info(f"[{request_id}] Primary model restored successfully")
                model_router.mark_primary_success()
                return result
        except QuotaExhausted:
            logger.info(f"[{request_id}] Primary model still quota-limited")
            model_router.mark_primary_failed()
        except Exception as e:
            logger.warning(f"[{request_id}] Primary model check failed: {e}")
            model_router.mark_primary_failed()

    # Build candidate model list
    current_model = model_router.get_current_model()
    fallback_names = [m["name"] for m in FALLBACK_MODELS]
    candidate_models = [current_model] + [m for m in fallback_names if m != current_model]

    # Try each model in order
    for model in candidate_models:
        try:
            # Find provider for this model
            provider = None
            for fm in FALLBACK_MODELS:
                if fm["name"] == model:
                    provider = fm["provider"]
                    break

            result = call_llm(prompt, model=model, provider=provider, max_retries=RETRIES_PER_MODEL, api_key=api_key)

            if result:
                model_router.switch_to_fallback(model)
                if model == DEFAULT_MODEL:
                    model_router.mark_primary_success()
                return result

        except QuotaExhausted:
            logger.info(f"[{request_id}] Quota exhausted for {model}, trying next...")
            continue

        except Exception as e:
            logger.warning(f"[{request_id}] Model {model} failed: {e}")
            continue

    logger.error(f"[{request_id}] All models failed")
    raise QuotaExhausted("All available models failed or are out of quota")
