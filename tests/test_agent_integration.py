"""Integration tests for AgentService against live local Ollama and persistent SQLite database.

Verifies end-to-end conversation flow:
1. Live menu inquiry.
2. Live allergen inquiry with verified database facts.
3. Live unverified allergen safety warning fallback.
4. Live table availability check.
"""

from datetime import date, timedelta
import unittest

from src.core.reservation_policy import restaurant_today
from src.core.config import config
from src.services.agent_service import AgentService, Intent, _format_menu_price
from src.services.llm_service import OllamaService


class TestAgentIntegration(unittest.TestCase):
    """End-to-end integration tests using live Ollama qwen2.5:1.5b and restaurant.db."""

    @classmethod
    def setUpClass(cls):
        cls.llm = OllamaService(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=30.0
        )
        if not cls.llm.check_health():
            raise unittest.SkipTest("Local Ollama is not accessible. Skipping live integration test.")

        cls.agent = AgentService(
            db_path=str(config.db_path),
            llm_service=cls.llm
        )

    def test_live_menu_query(self):
        """Live inquiry for a specific dish on the menu."""
        res = self.agent.handle_message("How much is the Margherita Pizza?")
        self.assertEqual(res.intent, Intent.MENU_INQUIRY)
        self.assertFalse(res.is_fallback)
        self.assertIsNotNone(res.data)
        self.assertIn(_format_menu_price(res.data["price"]), res.text)
        print(f"\n[Live Agent Menu Query]: {res.text}")

    def test_live_verified_allergen_query(self):
        """Live inquiry verifying allergens on a dish."""
        for allergen in ("dairy", "gluten"):
            with self.subTest(allergen=allergen):
                res = self.agent.handle_message(f"Does the Margherita Pizza contain {allergen}?")
                self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
                self.assertFalse(res.is_fallback)
                self.assertIn(f"contains {allergen}", res.text.lower())
                self.assertIn("cross-contact", res.text.lower())
                print(f"\n[Live Agent Allergen Query]: {res.text}")

    def test_live_unverified_allergen_query_triggers_safe_fallback(self):
        """Live inquiry for unverified dish must trigger staff referral fallback."""
        res = self.agent.handle_message("Does the Chef's Daily Rotating Special contain nuts?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertTrue(res.is_fallback)
        self.assertIn("not been formally verified", res.text.lower())
        print(f"\n[Live Agent Unverified Allergen Query]: {res.text}")

    def test_live_availability_check(self):
        """Live table availability check."""
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")
        res = self.agent.handle_message(f"Do you have a table for 2 on {tomorrow_str} at 19:00?")
        self.assertEqual(res.intent, Intent.AVAILABILITY_INQUIRY)
        self.assertFalse(res.needs_clarification)
        self.assertIn("table available", res.text.lower())
        print(f"\n[Live Agent Availability Query]: {res.text}")


if __name__ == "__main__":
    unittest.main()
