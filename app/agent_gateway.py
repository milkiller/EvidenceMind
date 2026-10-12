"""Translate between the hand-written Agent and Chat Completions tool calls."""

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .agent_core import AgentAction, AgentMessage, AgentToolCall, ToolSpec
from .gateway_generator import (
    GatewayRequestError, GatewaySettings, RequestExecutor, extract_answer_text,
    send_chat_completion,
)


_SYSTEM_PROMPT = (
    "你是 EvidenceMind 研究助手。需要事实证据时先调用 search_knowledge 工具；"
    "需要计算时可调用 calculator。只依据工具返回的证据陈述文档事实，"
    "在相关结论后引用来源、位置，并用【Chunk:实际ID】引用工具返回的 Chunk ID。"
    "不得编造或引用本轮工具未返回的 ID。证据不足时只回复："
    "证据不足，无法根据当前知识库回答。"
    "默认以当前论文版本为准；旧版仅用于明确要求比较版本的查询。"
    "论文正文可能是英文；中文问题检索不到方法时，改用英文技术术语查询。"
    "用户明确点名 Zotero 文献时，search_knowledge 的 source_title 填该文献标题或独特子串；"
    "没有明确文献标题时填空字符串，不要猜测标题。"
    "拿到足以回答的工具结果后，下一轮立即给出最终回答。"
    "仅在缺少关键证据时再检索，不要重复相同或近似查询。"
)


def parse_agent_message(message: Mapping[str, object]) -> AgentAction:
    """Convert one model message into a single validated Agent decision."""
    calls = message.get("tool_calls")
    if calls:
        if not isinstance(calls, list):
            raise ValueError("工具调用列表格式无效")
        parsed_calls: list[AgentToolCall] = []
        for call in calls:
            if not isinstance(call, Mapping):
                raise ValueError("工具调用格式无效")
            identifier = call.get("id")
            function = call.get("function")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("工具调用缺少有效 ID")
            if not isinstance(function, Mapping):
                raise ValueError("工具调用缺少 function")
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(name, str) or not name.strip() or not isinstance(raw_arguments, str):
                raise ValueError("工具名称或参数格式无效")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("工具参数不是有效的 JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是字典")
            parsed_calls.append(AgentToolCall(name, arguments, identifier))
        if len({call.identifier for call in parsed_calls}) != len(parsed_calls):
            raise ValueError("工具调用 ID 不能重复")
        first = parsed_calls[0]
        return AgentAction(
            kind="tool", tool_name=first.name, arguments=first.arguments,
            tool_identifier=first.identifier, tool_calls=tuple(parsed_calls),
        )

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型没有返回有效回答")
    if "<tool_call>" in content or "<function=" in content:
        raise ValueError("模型把工具调用写成了普通文本")
    return AgentAction(kind="final", answer=content.strip())


class GatewayLLMClient:
    """Ask the configured model for one Agent action at a time."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        request_executor: RequestExecutor | None = None,
        max_knowledge_searches: int = 2,
    ) -> None:
        if max_knowledge_searches <= 0:
            raise ValueError("max_knowledge_searches must be positive")
        self._settings = settings
        self._request_executor = request_executor
        self._max_knowledge_searches = max_knowledge_searches

    def decide(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentAction:
        completed_searches = sum(
            message.role == "tool" and message.tool_name == "search_knowledge"
            for message in messages
        )
        if completed_searches >= self._max_knowledge_searches:
            return self._answer_from_observations(messages)
        payload: dict[str, Any] = {
            "model": self._settings.model_name,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                *(_serialize_message(message) for message in messages),
            ],
        }
        if tools:
            payload["tools"] = [_serialize_tool(spec) for spec in tools]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        response = send_chat_completion(self._settings, payload, self._request_executor)
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayRequestError("gateway response has no assistant message") from exc
        if not isinstance(message, Mapping):
            raise GatewayRequestError("gateway assistant message is invalid")
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        if finish_reason == "length":
            raise GatewayRequestError("gateway response was cut off at max_tokens")
        try:
            action = parse_agent_message(message)
        except ValueError as exc:
            fields = ", ".join(sorted(str(key) for key in message))
            raise GatewayRequestError(
                f"gateway assistant action is invalid: {exc} "
                f"(finish_reason={finish_reason}, fields={fields})"
            ) from exc
        if action.kind == "tool":
            if len(action.tool_calls) > 4:
                raise GatewayRequestError("gateway requested too many tools at once")
            remaining = self._max_knowledge_searches - completed_searches
            selected: list[AgentToolCall] = []
            for call in action.tool_calls:
                if call.name == "search_knowledge":
                    if remaining <= 0:
                        continue
                    remaining -= 1
                selected.append(call)
            if len(selected) != len(action.tool_calls):
                first = selected[0]
                return AgentAction(
                    kind="tool", tool_name=first.name, arguments=first.arguments,
                    tool_identifier=first.identifier, tool_calls=tuple(selected),
                )
        return action

    def _answer_from_observations(self, messages: Sequence[AgentMessage]) -> AgentAction:
        """Finish from bounded retrieved evidence when the search budget is spent."""
        question = next(message.content for message in messages if message.role == "user")
        observations = [
            message.content for message in messages
            if message.role == "tool" and message.tool_name == "search_knowledge"
            and message.content.startswith("[来源：")
        ]
        if not observations:
            return AgentAction(kind="final", answer="证据不足，无法根据当前知识库回答。")
        evidence = "\n\n".join(reversed(observations))[:self._settings.max_evidence_chars]
        correction = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            question,
        )
        correction_text = "" if correction == question else f"\n\n修正要求：{correction}"
        payload = {
            "model": self._settings.model_name,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": (
                    "你是文档问答助手。只能依据给出的证据回答；证据不足要明确说明。"
                    "最多用两句话回答，并在关键结论后引用来源、位置，"
                    "使用【Chunk:实际ID】引用给出证据中的 Chunk ID。"
                    "若证据不足只回复：证据不足，无法根据当前知识库回答。"
                    "不得输出工具调用文本或请求新的检索。"
                )},
                {"role": "user", "content": f"问题：{question}\n\n证据：\n{evidence}{correction_text}"},
            ],
        }
        response = send_chat_completion(self._settings, payload, self._request_executor)
        try:
            finish_reason = response["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayRequestError("gateway response has no completion choice") from exc
        if finish_reason == "length":
            raise GatewayRequestError("gateway final answer was cut off at max_tokens")
        answer = extract_answer_text(response)
        if "<tool_call>" in answer or "<function=" in answer:
            raise GatewayRequestError("gateway returned a tool call instead of an answer")
        return AgentAction(kind="final", answer=answer)


def _serialize_tool(spec: ToolSpec) -> dict[str, object]:
    """Expose only registered names and simple required argument schemas."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": kind} for name, kind in spec.parameters.items()
                },
                "required": list(spec.parameters),
                "additionalProperties": False,
            },
        },
    }


def _serialize_message(message: AgentMessage) -> dict[str, object]:
    if message.role == "user":
        return {"role": "user", "content": message.content}
    if message.role == "tool":
        if not message.tool_identifier:
            raise ValueError("tool observation is missing its call ID")
        return {
            "role": "tool", "tool_call_id": message.tool_identifier,
            "content": message.content,
        }
    if message.tool_calls or message.tool_name:
        calls = message.tool_calls or (
            AgentToolCall(message.tool_name or "", message.arguments, message.tool_identifier),
        )
        if any(not call.identifier for call in calls):
            raise ValueError("assistant tool call is missing its call ID")
        return {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": call.identifier,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            } for call in calls],
        }
    return {"role": "assistant", "content": message.content}
