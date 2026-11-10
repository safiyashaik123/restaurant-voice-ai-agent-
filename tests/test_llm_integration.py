"""Integration test connecting to live local Ollama server with qwen2.5:1.5b.

Requires:
- Ollama service running on http://localhost:11434
- qwen2.5:1.5b pulled locally
"""

import time
import unittest

from src.core.config import config
from src.services.llm_service import (
    OllamaConnectionError,
    OllamaError,
    OllamaService,
    OllamaTimeoutError,
)


class TestOllamaIntegration(unittest.TestCase):
    """Integration test suite against live local Ollama daemon."""

    @classmethod
    def setUpClass(cls):
        """Initialize service against local Ollama."""
        cls.service = OllamaService(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=45.0  # Allow sufficient time for CPU inference on 8 GB laptop
        )
        # Check if Ollama is accessible
        if not cls.service.check_health():
            raise unittest.SkipTest(
                f"Local Ollama is not accessible at {config.ollama_base_url}. Skipping integration test."
            )

    def test_live_health_check(self):
        """Verify live Ollama health check succeeds."""
        is_healthy = self.service.check_health()
        self.assertTrue(is_healthy, "Ollama health check failed on live server.")

    def test_live_restaurant_conversation(self):
        """Test a live restaurant voice assistant interaction."""
        user_query = "Hello, what are your opening hours today?"
        context_facts = "Restaurant Operating Hours: 11:00 AM to 10:00 PM daily. Dinner and lunch service."

        start_time = time.perf_counter()
        response = self.service.generate_restaurant_reply(
            user_message=user_query,
            context_facts=context_facts
        )
        elapsed_sec = time.perf_counter() - start_time

        print(f"\n[Live Ollama Response in {elapsed_sec:.2f}s]:")
        print(f"Assistant: {response.content}")
        print(f"Model: {response.model} | Tokens: {response.eval_count} | Duration: {response.total_duration_ms:.1f}ms")

        self.assertIsNotNone(response.content)
        self.assertTrue(len(response.content) > 0)
        self.assertIn("qwen2.5:1.5b", response.model)
        # Verify response mentions hours based on context
        lower_resp = response.content.lower()
        self.assertTrue(
            "11" in lower_resp or "10" in lower_resp or "open" in lower_resp or "hour" in lower_resp,
            f"Expected hours mentioned in response: {response.content}"
        )

    def test_live_grounded_menu_query(self):
        """Test grounded dish inquiry relying strictly on database facts."""
        user_query = "Does the Caprese Salad contain any dairy?"
        context_facts = (
            "Item: Caprese Salad\n"
            "Category: Appetizers\n"
            "Allergen Status: verified\n"
            "Verified Allergens: Dairy (contains fresh Buffalo mozzarella)\n"
            "Ingredients: Buffalo mozzarella, Heirloom tomatoes, Fresh basil, Balsamic vinegar, Extra virgin olive oil, Sea salt, Black pepper"
        )

        start_time = time.perf_counter()
        response = self.service.generate_restaurant_reply(
            user_message=user_query,
            context_facts=context_facts
        )
        elapsed_sec = time.perf_counter() - start_time

        print(f"\n[Grounded Allergen Response in {elapsed_sec:.2f}s]:")
        print(f"Assistant: {response.content}")

        self.assertIsNotNone(response.content)
        self.assertTrue(len(response.content) > 0)
        # The assistant must accurately report dairy
        self.assertIn("dairy", response.content.lower())

    def test_live_connection_failure_handling(self):
        """Verify that attempting to connect to an invalid port raises OllamaConnectionError."""
        invalid_service = OllamaService(
            base_url="http://localhost:59999",  # Port with no listening service
            model="qwen2.5:1.5b",
            timeout=2.0
        )
        with self.assertRaises((OllamaConnectionError, OllamaTimeoutError)):
            invalid_service.chat([{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
