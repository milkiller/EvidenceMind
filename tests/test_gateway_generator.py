import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request

from app.gateway_generator import (
    GatewayChatCompletionGenerator,
    GatewayRequestError,
    GatewaySettings,
    _post_json,
    load_gateway_settings,
)
from app.retriever import SearchResult


class RecordingExecutor:
    """Fake gateway transport that records a request without network access."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[tuple[Request, float]] = []

    def __call__(self, request: Request, timeout: float) -> dict[str, object]:
        self.requests.append((request, timeout))
        return self.response


class GatewayGeneratorTests(unittest.TestCase):
    def test_load_settings_resolves_environment_variable_names(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(
                """
model:
  name: test-model
  api_style: chat_completions
  api_base_env: GATEWAY_URL
  api_key_env: GATEWAY_KEY
  timeout_seconds: 12
answering:
  max_evidence_chars: 99
""".strip(),
                encoding="utf-8",
            )

            settings = load_gateway_settings(
                config_path,
                environ={"GATEWAY_URL": "https://gateway.example/v1", "GATEWAY_KEY": "key"},
            )

        self.assertEqual(
            settings,
            GatewaySettings("test-model", "https://gateway.example/v1", "key", 12, 99),
        )

    def test_generate_sends_bounded_traceable_evidence(self) -> None:
        executor = RecordingExecutor(
            {"choices": [{"message": {"content": "基于证据的回答【第2页】"}}]}
        )
        generator = GatewayChatCompletionGenerator(
            GatewaySettings("test-model", "https://gateway.example/v1", "secret", 8, 200),
            request_executor=executor,
        )
        evidence = (
            SearchResult("chunk-2", 2, "第二页证据", 0.8),
            SearchResult("chunk-7", 7, "第七页证据", 0.7),
        )

        answer = generator.generate("请总结", evidence)

        self.assertEqual(answer, "基于证据的回答【第2页】")
        self.assertEqual(len(executor.requests), 1)
        request, timeout = executor.requests[0]
        self.assertEqual(request.full_url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(timeout, 8)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")
        self.assertIn("问题：请总结", payload["messages"][1]["content"])
        self.assertIn("页码 2 | Chunk chunk-2", payload["messages"][1]["content"])
        self.assertIn("页码 7 | Chunk chunk-7", payload["messages"][1]["content"])

    def test_generate_rejects_gateway_response_without_text(self) -> None:
        generator = GatewayChatCompletionGenerator(
            GatewaySettings("test-model", "https://gateway.example/v1", "secret", 8, 200),
            request_executor=RecordingExecutor({"choices": []}),
        )

        with self.assertRaisesRegex(GatewayRequestError, "no assistant message"):
            generator.generate("请总结", (SearchResult("chunk", 1, "证据", 0.8),))

    def test_transport_timeout_has_clear_error(self) -> None:
        request = Request("https://gateway.example/v1/chat/completions")
        with patch("app.gateway_generator.urlopen", side_effect=TimeoutError("slow")):
            with self.assertRaisesRegex(GatewayRequestError, "timed out"):
                _post_json(request, 1)


if __name__ == "__main__":
    unittest.main()
