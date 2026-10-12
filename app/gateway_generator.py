"""OpenAI-compatible gateway adapter for evidence-constrained answers."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from dotenv import load_dotenv

from .retriever import SearchResult


class GatewayRequestError(RuntimeError):
    """Raised when an OpenAI-compatible gateway cannot return answer text."""


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """Non-secret request settings resolved from config and environment."""

    model_name: str
    api_base_url: str
    api_key: str
    timeout_seconds: int
    max_evidence_chars: int


RequestExecutor = Callable[[Request, float], Mapping[str, Any]]


def load_gateway_settings(
    config_path: str | Path = "config.yaml",
    *,
    environ: Mapping[str, str] | None = None,
) -> GatewaySettings:
    """Load non-secret config and resolve gateway credentials from the environment."""
    path = Path(config_path).expanduser().resolve()
    if environ is None:
        load_dotenv(path.parent / ".env")
        environment: Mapping[str, str] = os.environ
    else:
        environment = environ

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config.yaml must contain a mapping")
    model_config = config.get("model")
    answering_config = config.get("answering")
    if not isinstance(model_config, Mapping) or not isinstance(answering_config, Mapping):
        raise ValueError("config.yaml must contain model and answering mappings")
    if model_config.get("api_style") != "chat_completions":
        raise ValueError("only chat_completions gateway style is supported")

    model_name = _required_string(model_config, "name")
    api_base_env = _required_string(model_config, "api_base_env")
    api_key_env = _required_string(model_config, "api_key_env")
    api_base_url = environment.get(api_base_env, "").strip()
    api_key = environment.get(api_key_env, "").strip()
    if not api_base_url:
        raise ValueError(f"missing gateway URL environment variable: {api_base_env}")
    if not api_key:
        raise ValueError(f"missing API key environment variable: {api_key_env}")

    timeout_seconds = model_config.get("timeout_seconds")
    max_evidence_chars = answering_config.get("max_evidence_chars")
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("model.timeout_seconds must be a positive integer")
    if not isinstance(max_evidence_chars, int) or max_evidence_chars <= 0:
        raise ValueError("answering.max_evidence_chars must be a positive integer")

    return GatewaySettings(
        model_name=model_name,
        api_base_url=api_base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_evidence_chars=max_evidence_chars,
    )


class GatewayChatCompletionGenerator:
    """Generate an answer from retrieved evidence through one gateway request."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        request_executor: RequestExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._request_executor = request_executor or _post_json

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.yaml",
    ) -> "GatewayChatCompletionGenerator":
        """Create a generator using local config and ignored environment values."""
        return cls(load_gateway_settings(config_path))

    def generate(self, question: str, evidence: Sequence[SearchResult]) -> str:
        """Send a bounded evidence prompt and return the gateway answer text."""
        if not question.strip():
            raise ValueError("question must be non-blank")
        if not evidence:
            raise ValueError("evidence must be non-empty")

        payload = {
            "model": self._settings.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是文档问答助手。只能依据给出的证据回答；"
                        "证据不足时明确说明。回答简洁，并在相关结论后标注证据来源和位置。"
                        "不同论文版本有冲突时，以标记为当前版本的证据为准，旧版只用于解释修改历程。"
                    ),
                },
                {
                    "role": "user",
                    "content": _build_evidence_prompt(
                        question,
                        evidence,
                        max_evidence_chars=self._settings.max_evidence_chars,
                    ),
                },
            ],
        }
        response = send_chat_completion(self._settings, payload, self._request_executor)
        return extract_answer_text(response)


def send_chat_completion(
    settings: GatewaySettings,
    payload: Mapping[str, Any],
    request_executor: RequestExecutor | None = None,
) -> Mapping[str, Any]:
    """Send one Chat Completions request for either answering or Agent decisions."""
    request = Request(
        _chat_completions_endpoint(settings.api_base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return (request_executor or _post_json)(request, settings.timeout_seconds)


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model.{key} must be a non-blank string")
    return value.strip()


def _build_evidence_prompt(
    question: str,
    evidence: Sequence[SearchResult],
    *,
    max_evidence_chars: int,
) -> str:
    blocks = tuple(
        "\n".join(
            (
                (
                    f"[证据 {rank} | 来源 {getattr(result, 'source_title')} | "
                    f"位置 {getattr(result, 'locator')} | "
                    f"版本 {getattr(result, 'version', None) or '无'} | "
                    f"Chunk {result.chunk_id}]"
                    if getattr(result, "source_title", None) and getattr(result, "locator", None) else
                    f"[证据 {rank} | 页码 {result.page_number} | Chunk {result.chunk_id}]"
                ),
                result.text.strip(),
            )
        )
        for rank, result in enumerate(evidence, start=1)
    )
    bounded_evidence = "\n\n".join(blocks)[:max_evidence_chars]
    return f"问题：{question.strip()}\n\n证据：\n{bounded_evidence}"


def _chat_completions_endpoint(api_base_url: str) -> str:
    base_url = api_base_url.rstrip("/")
    if not base_url:
        raise ValueError("api_base_url must be non-blank")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _post_json(request: Request, timeout_seconds: float) -> Mapping[str, Any]:
    """Execute one POST request without logging credentials or document evidence."""
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise GatewayRequestError(f"gateway returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise GatewayRequestError("gateway is unreachable") from exc
    except TimeoutError as exc:
        raise GatewayRequestError("gateway request timed out") from exc
    if not isinstance(payload, Mapping):
        raise GatewayRequestError("gateway response must be a JSON object")
    return payload


def extract_answer_text(response: Mapping[str, Any]) -> str:
    """Extract the first assistant text from a Chat Completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GatewayRequestError("gateway response has no assistant message") from exc
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, Mapping)
        )
    if not isinstance(content, str) or not content.strip():
        raise GatewayRequestError("gateway returned empty assistant text")
    return content.strip()
