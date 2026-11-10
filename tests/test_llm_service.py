"""Focused unit tests for OllamaService with mocked HTTP responses.

Does NOT require a running Ollama daemon or model.
Tests response parsing, error handling (connection, timeout, response errors),
and grounded context construction.
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from src.services.llm_service import (
    DEFAULT_RESTAURANT_SYSTEM_PROMPT,
    LLMResponse,
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
)


class TestOllamaServiceUnit(unittest.TestCase):
    """Unit tests for OllamaService using mocked HTTP interactions."""

    def setUp(self):
        self.service = OllamaService(
            base_url="http://localhost:11434",
            model="qwen2.5:1.5b",
            timeout=10.0
        )

    def _mock_http_response(self, status=200, json_payload=None):
        """Helper to create a mocked urllib HTTP response."""
        payload_bytes = json.dumps(json_payload or {}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = payload_bytes
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    # -----------------------------------------------------------------------
    # 1. Successful Chat Completion Tests
    # -----------------------------------------------------------------------
    @patch("urllib.request.urlopen")
    def test_successful_chat_completion(self, mock_urlopen):
        """Verify proper extraction of message content and metrics."""
        mock_payload = {
            "model": "qwen2.5:1.5b",
            "message": {
                "role": "assistant",
                "content": "Welcome to our restaurant! How can I help you today?"
            },
            "done": True,
            "total_duration": 450_000_000,  # 450 ms in nanoseconds
            "prompt_eval_count": 25,
            "eval_count": 14
        }
        mock_urlopen.return_value = self._mock_http_response(200, mock_payload)

        messages = [{"role": "user", "content": "Hello"}]
        response = self.service.chat(messages=messages)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(
            response.content,
            "Welcome to our restaurant! How can I help you today?"
        )
        self.assertEqual(response.model, "qwen2.5:1.5b")
        self.assertEqual(response.eval_count, 14)
        self.assertEqual(response.prompt_eval_count, 25)
        self.assertAlmostEqual(response.total_duration_ms, 450.0)

    @patch("urllib.request.urlopen")
    def test_system_prompt_prepended_properly(self, mock_urlopen):
        """Verify that system prompt is placed in the payload messages."""
        mock_payload = {
            "model": "qwen2.5:1.5b",
            "message": {"role": "assistant", "content": "Understood."},
            "done": True
        }
        mock_urlopen.return_value = self._mock_http_response(200, mock_payload)

        messages = [{"role": "user", "content": "What is the soup today?"}]
        custom_system = "You are an assistant. Answer briefly."

        self.service.chat(messages=messages, system_prompt=custom_system)

        # Inspect request payload sent to urlopen
        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))

        self.assertEqual(sent_body["messages"][0]["role"], "system")
        self.assertEqual(sent_body["messages"][0]["content"], custom_system)
        self.assertEqual(sent_body["messages"][1]["role"], "user")
        self.assertEqual(sent_body["messages"][1]["content"], "What is the soup today?")

    @patch("urllib.request.urlopen")
    def test_grounded_reply_includes_database_facts(self, mock_urlopen):
        """Verify generate_restaurant_reply injects database context into system prompt."""
        mock_payload = {
            "model": "qwen2.5:1.5b",
            "message": {"role": "assistant", "content": "The Margherita Pizza is $16.00."},
            "done": True
        }
        mock_urlopen.return_value = self._mock_http_response(200, mock_payload)

        context_facts = "Margherita Pizza: Price $16.00. Allergens: Gluten, Dairy."
        self.service.generate_restaurant_reply(
            user_message="How much is the Margherita Pizza?",
            context_facts=context_facts
        )

        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))
        sent_sys_prompt = sent_body["messages"][0]["content"]

        self.assertIn("AUTHORITATIVE DATABASE CONTEXT", sent_sys_prompt)
        self.assertIn(context_facts, sent_sys_prompt)

    # -----------------------------------------------------------------------
    # 2. Connection, Timeout & HTTP Error Handling Tests
    # -----------------------------------------------------------------------
    @patch("urllib.request.urlopen")
    def test_connection_error_handling(self, mock_urlopen):
        """Verify connection refused raises OllamaConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("Failed to connect to Ollama", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_timeout_error_handling(self, mock_urlopen):
        """Verify request timeout raises OllamaTimeoutError."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")

        with self.assertRaises(OllamaTimeoutError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("timed out", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Verify non-200 HTTP response raises OllamaResponseError."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b'{"error": "model crashed"}')
        )

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("HTTP error 500", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_malformed_json_handling(self, mock_urlopen):
        """Verify malformed JSON raises OllamaResponseError."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"<html>Not JSON</html>"
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("Malformed JSON", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_response_not_a_json_object(self, mock_urlopen):
        """Verify top-level JSON array raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(200, ["an", "array"])

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("Expected Ollama response to be a JSON object", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_missing_message_key(self, mock_urlopen):
        """Verify missing 'message' key raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(200, {"model": "qwen2.5:1.5b", "done": True})

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("missing required 'message' dictionary", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_invalid_message_structure_not_dict(self, mock_urlopen):
        """Verify 'message' value that is not a dictionary raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(
            200,
            {"model": "qwen2.5:1.5b", "message": "invalid string instead of dict"}
        )

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("missing required 'message' dictionary", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_missing_content_key(self, mock_urlopen):
        """Verify 'message' dict missing 'content' raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(
            200,
            {"model": "qwen2.5:1.5b", "message": {"role": "assistant"}}
        )

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("missing required non-empty 'content' string", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_empty_content_string(self, mock_urlopen):
        """Verify empty 'content' string raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(
            200,
            {"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": ""}}
        )

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("missing required non-empty 'content' string", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_whitespace_only_content_string(self, mock_urlopen):
        """Verify whitespace-only 'content' string raises OllamaResponseError."""
        mock_urlopen.return_value = self._mock_http_response(
            200,
            {"model": "qwen2.5:1.5b", "message": {"role": "assistant", "content": "   \n\t  "}}
        )

        with self.assertRaises(OllamaResponseError) as ctx:
            self.service.chat([{"role": "user", "content": "Hi"}])
        self.assertIn("missing required non-empty 'content' string", str(ctx.exception))

    # -----------------------------------------------------------------------
    # 3. Health Check Tests
    # -----------------------------------------------------------------------
    @patch("urllib.request.urlopen")
    def test_check_health_success(self, mock_urlopen):
        """Verify health check returns True when daemon responds."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(self.service.check_health())

    @patch("urllib.request.urlopen")
    def test_check_health_failure(self, mock_urlopen):
        """Verify health check returns False when daemon is down."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        self.assertFalse(self.service.check_health())


if __name__ == "__main__":
    unittest.main()
