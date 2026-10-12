"""Offline behavior tests for the hand-written agent loop."""

from collections.abc import Sequence
import unittest

from app.agent_core import AgentAction, AgentMessage, ToolRegistry, ToolSpec, run_agent
from app.agent_tools import register_calculator


class ScriptedClient:
    def __init__(self, actions: Sequence[AgentAction]) -> None:
        self.actions = list(actions)
        self.calls: list[tuple[tuple[AgentMessage, ...], tuple[ToolSpec, ...]]] = []

    def decide(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentAction:
        self.calls.append((tuple(messages), tuple(tools)))
        return self.actions.pop(0)


class AgentCoreTests(unittest.TestCase):
    def test_direct_answer_never_calls_a_tool(self) -> None:
        client = ScriptedClient([AgentAction(kind="final", answer="直接回答")])
        registry = ToolRegistry()
        register_calculator(registry)

        result = run_agent("你好", client, registry, max_steps=1)

        self.assertEqual(result.answer, "直接回答")
        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.steps, 1)
        self.assertEqual([message.role for message in result.messages], ["user", "assistant"])

    def test_calculator_observation_reaches_next_model_decision(self) -> None:
        client = ScriptedClient(
            [
                AgentAction(kind="tool", tool_name="calculator", arguments={"expression": "2 + 3 * 4"}),
                AgentAction(kind="final", answer="结果是 14"),
            ]
        )
        registry = ToolRegistry()
        register_calculator(registry)

        result = run_agent("计算 2 + 3 * 4", client, registry)

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.steps, 2)
        self.assertEqual(client.calls[1][0][-1].role, "tool")
        self.assertEqual(client.calls[1][0][-1].content, "14")
        self.assertEqual(client.calls[1][0][-1].tool_name, "calculator")

    def test_fake_search_result_becomes_an_observation(self) -> None:
        client = ScriptedClient(
            [
                AgentAction(kind="tool", tool_name="search_web", arguments={"query": "ReAct"}),
                AgentAction(kind="final", answer="找到论文"),
            ]
        )
        registry = ToolRegistry()
        registry.register(
            ToolSpec("search_web", "Search a web page.", {"query": "string"}),
            lambda arguments: f"假搜索结果：{arguments['query']}",
        )

        result = run_agent("找 ReAct", client, registry)

        self.assertEqual(result.answer, "找到论文")
        self.assertEqual(client.calls[1][0][-1].content, "假搜索结果：ReAct")
        self.assertEqual(client.calls[0][1][0].name, "search_web")

    def test_unknown_tool_stops_without_executing_any_handler(self) -> None:
        client = ScriptedClient(
            [AgentAction(kind="tool", tool_name="missing", arguments={})]
        )

        result = run_agent("调用未知工具", client, ToolRegistry())

        self.assertEqual(result.stop_reason, "unknown_tool")
        self.assertIsNone(result.answer)
        self.assertEqual(result.steps, 1)

    def test_wrong_tool_arguments_stop_before_handler(self) -> None:
        calls: list[object] = []
        registry = ToolRegistry()
        registry.register(
            ToolSpec("search_web", "Search a web page.", {"query": "string"}),
            lambda arguments: calls.append(arguments) or "should not execute",
        )
        client = ScriptedClient(
            [AgentAction(kind="tool", tool_name="search_web", arguments={"query": 42})]
        )

        result = run_agent("搜索", client, registry)

        self.assertEqual(result.stop_reason, "invalid_arguments")
        self.assertEqual(calls, [])

    def test_calculator_rejects_python_code(self) -> None:
        registry = ToolRegistry()
        register_calculator(registry)
        client = ScriptedClient(
            [
                AgentAction(
                    kind="tool", tool_name="calculator", arguments={"expression": "__import__('os')"}
                )
            ]
        )

        result = run_agent("执行代码", client, registry)

        self.assertEqual(result.stop_reason, "invalid_arguments")

    def test_max_steps_stops_before_another_model_or_tool_call(self) -> None:
        client = ScriptedClient(
            [AgentAction(kind="tool", tool_name="calculator", arguments={"expression": "1 + 1"})]
        )
        registry = ToolRegistry()
        register_calculator(registry)

        result = run_agent("计算并继续", client, registry, max_steps=1)

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertIsNone(result.answer)
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.messages[-1].content, "2")


if __name__ == "__main__":
    unittest.main()
