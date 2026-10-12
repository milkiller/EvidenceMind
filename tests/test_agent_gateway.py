import json
import unittest
from urllib.request import Request

from app.agent_core import AgentMessage, ToolOutput, ToolRegistry, ToolSpec, run_agent
from app.agent_gateway import GatewayLLMClient, parse_agent_message
from app.agent_grounding import validate_research_answer
from app.agent_tools import register_calculator
from app.gateway_generator import GatewayRequestError, GatewaySettings


class AgentGatewayTests(unittest.TestCase):
    def test_exhausted_search_budget_keeps_correction_feedback(self) -> None:
        requests: list[Request] = []
        replies = iter((
            {"choices": [{"message": {"tool_calls": [{
                "id": "call_search", "function": {
                    "name": "search_knowledge", "arguments": '{"query":"AE32"}',
                },
            }]}}]},
            {"choices": [{"message": {"content": "没有引用的结论"}}]},
            {"choices": [{"message": {"content": "有引用的结论【Chunk:real】"}}]},
        ))

        def execute(request: Request, timeout: float) -> dict[str, object]:
            requests.append(request)
            return next(replies)

        registry = ToolRegistry()
        registry.register(
            ToolSpec("search_knowledge", "Search", {"query": "string"}),
            lambda arguments: ToolOutput("[来源：论文 | Chunk：real]\n证据", ("real",)),
        )
        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)

        result = run_agent(
            "AE32 是什么？",
            GatewayLLMClient(settings, request_executor=execute, max_knowledge_searches=1),
            registry, final_validator=validate_research_answer,
        )

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.steps, 3)
        third = json.loads(requests[2].data.decode("utf-8"))
        self.assertIn("修正要求", third["messages"][1]["content"])
        self.assertIn("可用 ID：real", third["messages"][1]["content"])

    def test_plain_answer_becomes_final_action(self) -> None:
        # 安排：模拟模型返回的消息
        message = {"content": "结果是 14", "tool_calls": None}

        # 执行：调用我们自己的解析函数
        action = parse_agent_message(message)

        # 检查：转换后的动作是否正确
        self.assertEqual(action.kind, "final")
        self.assertEqual(action.answer, "结果是 14")
    def test_tool_call_becomes_tool_action(self) -> None:
        message = {
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "2 + 3 * 4"}',
                },
            }],
        }

        action = parse_agent_message(message)

        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool_name, "calculator")
        self.assertEqual(action.arguments, {"expression": "2 + 3 * 4"})   
        self.assertEqual(action.tool_identifier, "call_123")

    def test_real_client_replays_call_id_with_tool_observation(self) -> None:
        requests: list[Request] = []
        replies = iter((
            {"choices": [{"message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_123", "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"2+3*4"}'},
                }],
            }}]},
            {"choices": [{"message": {"content": "结果是 14"}}]},
        ))

        def execute(request: Request, timeout: float) -> dict[str, object]:
            requests.append(request)
            return next(replies)

        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)
        registry = ToolRegistry()
        register_calculator(registry)
        result = run_agent("计算 2+3*4", GatewayLLMClient(settings, request_executor=execute), registry)

        self.assertEqual(result.answer, "结果是 14")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(requests), 2)
        first = json.loads(requests[0].data.decode("utf-8"))
        second = json.loads(requests[1].data.decode("utf-8"))
        self.assertEqual(first["tools"][0]["function"]["name"], "calculator")
        self.assertEqual(first["tools"][0]["function"]["parameters"]["required"], ["expression"])
        self.assertEqual(second["messages"][2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(second["messages"][3]["tool_call_id"], "call_123")
        self.assertEqual(second["messages"][3]["content"], "14")

    def test_multiple_tool_calls_keep_distinct_ids_and_results(self) -> None:
        requests: list[Request] = []

        def execute(request: Request, timeout: float) -> dict[str, object]:
            requests.append(request)
            if len(requests) == 1:
                return {"choices": [{"message": {"tool_calls": [
                    {"id": "call_a", "function": {
                        "name": "calculator", "arguments": '{"expression":"1+1"}',
                    }},
                    {"id": "call_b", "function": {
                        "name": "calculator", "arguments": '{"expression":"2+2"}',
                    }},
                ]}}]}
            return {"choices": [{"message": {"content": "2 和 4"}}]}

        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)
        registry = ToolRegistry()
        register_calculator(registry)
        result = run_agent("算两题", GatewayLLMClient(settings, request_executor=execute), registry)

        self.assertEqual(result.answer, "2 和 4")
        self.assertEqual(result.steps, 2)
        second = json.loads(requests[1].data.decode("utf-8"))
        self.assertEqual(
            [call["id"] for call in second["messages"][2]["tool_calls"]],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            [(item["tool_call_id"], item["content"]) for item in second["messages"][3:]],
            [("call_a", "2"), ("call_b", "4")],
        )

    def test_duplicate_tool_call_ids_are_rejected(self) -> None:
        call = {"id": "call_123", "function": {"name": "calculator", "arguments": "{}"}}
        with self.assertRaisesRegex(ValueError, "不能重复"):
            parse_agent_message({"tool_calls": [call, call]})

    def test_one_model_decision_cannot_exceed_two_document_searches(self) -> None:
        requests: list[Request] = []

        def execute(request: Request, timeout: float) -> dict[str, object]:
            requests.append(request)
            if len(requests) == 1:
                return {"choices": [{"message": {"tool_calls": [
                    {"id": f"call_{number}", "function": {
                        "name": "search_knowledge", "arguments": '{"query":"method"}',
                    }} for number in (1, 2, 3)
                ]}}]}
            return {"choices": [{"message": {"content": "基于两次检索的回答"}}]}

        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)
        registry = ToolRegistry()
        searched: list[object] = []
        registry.register(
            ToolSpec("search_knowledge", "Search notes", {"query": "string"}),
            lambda arguments: searched.append(arguments) or "[来源：Paper]\nevidence",
        )
        result = run_agent("method?", GatewayLLMClient(settings, request_executor=execute), registry)

        self.assertEqual(result.answer, "基于两次检索的回答")
        self.assertEqual(len(searched), 2)
        self.assertEqual([item.tool_identifier for item in result.messages if item.role == "tool"],
                         ["call_1", "call_2"])

    def test_document_search_budget_forces_a_final_decision(self) -> None:
        requests: list[Request] = []
        replies = iter((
            {"choices": [{"message": {"tool_calls": [{
                "id": f"call_{number}", "function": {
                    "name": "search_knowledge", "arguments": '{"query":"method"}',
                },
            }]}}]}
            for number in (1, 2)
        ))

        def execute(request: Request, timeout: float) -> dict[str, object]:
            requests.append(request)
            if len(requests) <= 2:
                return next(replies)
            return {"choices": [{"message": {"content": "根据证据回答"}}]}

        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)
        registry = ToolRegistry()
        registry.register(ToolSpec("search_knowledge", "Search notes", {"query": "string"}),
                          lambda arguments: "[来源：Paper | 位置：段落 1 | Chunk：one]\nsource and evidence")
        result = run_agent(
            "method?", GatewayLLMClient(settings, request_executor=execute),
            registry, max_steps=4,
        )

        self.assertEqual(result.answer, "根据证据回答")
        self.assertEqual(result.steps, 3)
        third = json.loads(requests[2].data.decode("utf-8"))
        self.assertNotIn("tools", third)
        self.assertIn("source and evidence", third["messages"][1]["content"])
        self.assertIn("method?", third["messages"][1]["content"])

    def test_tool_markup_is_not_accepted_as_an_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "普通文本"):
            parse_agent_message({"content": "<tool_call><function=search_knowledge></tool_call>"})

    def test_cut_off_evidence_answer_is_rejected(self) -> None:
        settings = GatewaySettings("test", "https://example.test/v1", "secret", 9, 500)
        client = GatewayLLMClient(
            settings,
            request_executor=lambda request, timeout: {
                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
            },
        )
        messages = (
            AgentMessage(role="user", content="method?"),
            AgentMessage(role="tool", content="[来源：Paper]\nevidence", tool_name="search_knowledge"),
            AgentMessage(role="tool", content="[来源：Paper]\nevidence", tool_name="search_knowledge"),
        )
        with self.assertRaisesRegex(GatewayRequestError, "cut off"):
            client.decide(messages, ())
