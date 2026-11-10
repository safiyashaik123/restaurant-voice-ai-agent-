"""Local LLM Service for Restaurant Voice AI Agent.

Communicates with local Ollama daemon via the /api/chat endpoint using the qwen2.5:1.5b model.
Uses Python standard library (urllib.request) to avoid third-party dependencies and cloud APIs.
Includes robust timeout, connection failure handling, and voice-optimized system prompts.
"""

from dataclasses import dataclass, field
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from src.core.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class OllamaError(Exception):
    """Base exception for Ollama LLM service errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached or connection is refused."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected HTTP status code or malformed JSON."""
    pass


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from the local Ollama LLM."""
    content: str
    model: str
    total_duration_ms: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0
    raw_response: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Restaurant Voice Assistant System Prompt
# ---------------------------------------------------------------------------

DEFAULT_RESTAURANT_SYSTEM_PROMPT = """You are the friendly, professional voice assistant for our restaurant.
Your role is to assist guests by phone or voice interface.

CORE RESPONSIBILITIES:
1. Answer questions about our menu items, descriptions, and prices.
2. Provide verified ingredient and allergen information.
3. Check table availability and assist with reservations.
4. Welcome guests with warm, courteous hospitality.

VOICE-FIRST COMMUNICATION RULES:
- Keep responses concise, clear, and conversational (1 to 3 sentences maximum).
- Speak naturally. Never use markdown formatting (no asterisks, bullet points, headers, or tables).
- Never use emojis or special characters, as they sound awkward when read aloud by text-to-speech.
- Present numbers and prices conversationally (e.g., "nine dollars and fifty cents" or "$9.50").

STRICT SAFETY AND GROUNDING INVARIANTS:
- NEVER invent, assume, or extrapolate menu dishes, recipe ingredients, or allergen statuses.
- You must rely ONLY on the verified factual context provided to you from the restaurant database.
- If a dish, ingredient, or allergen is unverified, missing, or unknown, explicitly state that you do not have verified information and advise the guest to speak directly with restaurant staff.
- Never confirm a table reservation unless verified availability and confirmation details are provided in your context.
- If a guest's request is unclear, politely ask for clarification.
"""


# ---------------------------------------------------------------------------
# Core Service Class
# ---------------------------------------------------------------------------

class OllamaService:
    """Service client for local Ollama chat API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> None:
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.model = model or config.ollama_model
        self.timeout = timeout if timeout is not None else config.ollama_timeout_seconds

    def check_health(self) -> bool:
        """Check if local Ollama daemon is reachable and responding."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(
                tags_url,
                headers={"Accept": "application/json"},
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> LLMResponse:
        """Send chat messages to Ollama /api/chat endpoint and return structured response.

        Args:
            messages: List of message dictionaries with 'role' ('user', 'assistant', 'system') and 'content'.
            system_prompt: Optional system prompt to prepend if not already present.
            temperature: Sampling temperature (0.0 - 1.0; low value 0.2 recommended for factual grounding).
            model: Override target model (defaults to configured model, e.g., 'qwen2.5:1.5b').
            timeout: Override request timeout in seconds.

        Returns:
            LLMResponse containing assistant text and inference metrics.

        Raises:
            OllamaConnectionError: If server cannot be reached.
            OllamaTimeoutError: If inference exceeds timeout limit.
            OllamaResponseError: If server returns HTTP error or invalid response.
        """
        target_model = model or self.model
        req_timeout = timeout if timeout is not None else self.timeout
        chat_url = f"{self.base_url}/api/chat"

        # Prepare payload
        payload_messages = []
        
        # Prepend system prompt if provided and not already in messages
        has_system = any(m.get("role") == "system" for m in messages)
        if system_prompt and not has_system:
            payload_messages.append({"role": "system", "content": system_prompt})
            
        payload_messages.extend(messages)

        request_body = {
            "model": target_model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        json_data = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            chat_url,
            data=json_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                status_code = resp.status
                if status_code != 200:
                    raise OllamaResponseError(
                        f"Ollama returned unexpected HTTP status {status_code}"
                    )
                response_text = resp.read().decode("utf-8")
                
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise OllamaResponseError(
                f"Ollama HTTP error {e.code}: {e.reason}. Details: {error_body}"
            ) from e
            
        except urllib.error.URLError as e:
            # Check for socket timeout vs connection refused
            reason_str = str(e.reason).lower()
            if "timed out" in reason_str:
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {req_timeout}s: {e.reason}"
                ) from e
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {chat_url}: {e.reason}"
            ) from e
            
        except TimeoutError as e:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {req_timeout}s"
            ) from e
            
        except Exception as e:
            raise OllamaError(f"Unexpected error communicating with Ollama: {e}") from e

        # Parse JSON response
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise OllamaResponseError(
                f"Malformed JSON response from Ollama: {e}"
            ) from e

        # 1. Verify that the Ollama response is a JSON object (dictionary)
        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Expected Ollama response to be a JSON object, got {type(data).__name__}"
            )

        # 2. Verify that 'message' is present and is a dictionary
        if "message" not in data or not isinstance(data["message"], dict):
            raise OllamaResponseError(
                "Ollama response missing required 'message' dictionary"
            )

        message_obj = data["message"]

        # 3. Verify that 'content' is present and is a non-empty string
        content_val = message_obj.get("content")
        if not isinstance(content_val, str) or not content_val.strip():
            raise OllamaResponseError(
                "Ollama response missing required non-empty 'content' string in message"
            )

        content = content_val.strip()

        # Extract timing metrics (durations are in nanoseconds in Ollama)
        total_duration_ms = data.get("total_duration", 0) / 1_000_000.0 if isinstance(data.get("total_duration"), (int, float)) else 0.0
        eval_count = data.get("eval_count", 0) if isinstance(data.get("eval_count"), int) else 0
        prompt_eval_count = data.get("prompt_eval_count", 0) if isinstance(data.get("prompt_eval_count"), int) else 0

        return LLMResponse(
            content=content,
            model=data.get("model", target_model),
            total_duration_ms=total_duration_ms,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
            raw_response=data
        )

    def generate_restaurant_reply(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_facts: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> LLMResponse:
        """Generate a restaurant assistant reply grounded in retrieved context facts.

        Args:
            user_message: Latest speech/text from customer.
            conversation_history: Optional prior messages in the conversation.
            context_facts: Authoritative factual text retrieved from SQLite database.
            system_prompt: Optional custom system prompt (defaults to DEFAULT_RESTAURANT_SYSTEM_PROMPT).
            timeout: Optional timeout override.
        """
        sys_prompt = system_prompt or DEFAULT_RESTAURANT_SYSTEM_PROMPT

        if context_facts:
            sys_prompt += (
                f"\n\nAUTHORITATIVE DATABASE CONTEXT (ONLY RELY ON THESE VERIFIED FACTS):\n"
                f"{context_facts.strip()}\n"
            )

        messages: List[Dict[str, str]] = []
        if conversation_history:
            messages.extend(conversation_history)
            
        messages.append({"role": "user", "content": user_message.strip()})

        return self.chat(
            messages=messages,
            system_prompt=sys_prompt,
            timeout=timeout
        )


# Singleton instance
default_llm_service = OllamaService()
