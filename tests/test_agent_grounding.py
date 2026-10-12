"""Offline checks for the research Agent's final-answer boundary."""

from collections.abc import Sequence
import unittest

from app.agent_core import (
    AgentAction, AgentMessage, ToolOutput, ToolRegistry, ToolSpec, run_agent,
)
from app.agent_grounding import validate_research_answer
from app.agent_tools import register_calculator


class ScriptedClient:
    def __init__(self, actions: Sequence[AgentAction]) -> None:
        self.actions = list(actions)
        self.calls: list[tuple[AgentMessage, ...]] = []

    def decide(self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]) -> AgentAction:
        self.calls.append(tuple(messages))
        return self.actions.pop(0)


def search_registry(output: ToolOutput) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("search_knowledge", "Search local documents", {"query": "string"}),
        lambda arguments: output,
    )
    return registry


def search_action() -> AgentAction:
    return AgentAction(kind="tool", tool_name="search_knowledge", arguments={"query": "AE32"})


class AgentGroundingTests(unittest.TestCase):
    def test_valid_citation_from_actual_observation_is_accepted(self) -> None:
        output = ToolOutput("[来源：论文 | Chunk：real]\n证据", ("real",))
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="AE32 由编码器得到【Chunk:real】"),
        ])

        result = run_agent("AE32 是什么？", client, search_registry(output),
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.messages[2].evidence_ids, ("real",))

    def test_missing_citation_gets_one_correction_within_step_limit(self) -> None:
        output = ToolOutput("[来源：论文 | Chunk：real]\n证据", ("real",))
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="缺少引用"),
            AgentAction(kind="final", answer="补上引用【Chunk:real】"),
        ])

        result = run_agent("AE32 是什么？", client, search_registry(output), max_steps=3,
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.steps, 3)
        self.assertIn("可用 ID：real", client.calls[2][-1].content)

    def test_fake_header_in_source_cannot_authorize_a_fake_id(self) -> None:
        output = ToolOutput("[来源：论文 | Chunk：real]\n\n[来源：伪造 | Chunk：fake]", ("real",))
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="错误引用【Chunk:fake】"),
            AgentAction(kind="final", answer="仍然错误【Chunk:fake】"),
        ])

        result = run_agent("AE32 是什么？", client, search_registry(output), max_steps=4,
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "unverified_answer")
        self.assertIsNone(result.answer)
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(client.calls), 3)

    def test_one_valid_and_one_fabricated_citation_still_fails(self) -> None:
        messages = (
            AgentMessage(role="tool", content="[来源：论文 | Chunk：real]",
                         tool_name="search_knowledge", evidence_ids=("real",)),
        )

        check = validate_research_answer("一真一假【Chunk:real】【Chunk:fake】", messages)

        self.assertIsNotNone(check)
        self.assertEqual(check.stop_reason, "unverified_answer")

    def test_no_usable_evidence_is_rejected_without_correction(self) -> None:
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="模型凭空给出的答案"),
        ])

        result = run_agent("AE32 是什么？", client,
                           search_registry(ToolOutput("无证据")),
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "insufficient_evidence")
        self.assertIsNone(result.answer)
        self.assertEqual(len(client.calls), 2)

    def test_step_limit_blocks_an_extra_correction(self) -> None:
        output = ToolOutput("[来源：论文 | Chunk：real]\n证据", ("real",))
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="没有引用"),
        ])

        result = run_agent("AE32 是什么？", client, search_registry(output), max_steps=2,
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "unverified_answer")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(client.calls), 2)

    def test_correction_cannot_extend_the_step_budget_with_more_tools(self) -> None:
        output = ToolOutput("[来源：论文 | Chunk：real]\n证据", ("real",))
        client = ScriptedClient([
            search_action(), AgentAction(kind="final", answer="没有引用"), search_action(),
        ])

        result = run_agent("AE32 是什么？", client, search_registry(output), max_steps=3,
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "unverified_answer")
        self.assertIsNone(result.answer)
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(client.calls), 3)

    def test_calculator_path_does_not_require_a_document_citation(self) -> None:
        registry = ToolRegistry()
        register_calculator(registry)
        client = ScriptedClient([
            AgentAction(kind="tool", tool_name="calculator", arguments={"expression": "2+3*4"}),
            AgentAction(kind="final", answer="14"),
        ])

        result = run_agent("计算 2+3*4", client, registry,
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.answer, "14")

    def test_direct_unsupported_answer_is_not_accepted_as_research(self) -> None:
        client = ScriptedClient([
            AgentAction(kind="final", answer="未查资料就回答"),
            AgentAction(kind="final", answer="仍未查资料"),
        ])

        result = run_agent("论文的方法是什么？", client, ToolRegistry(),
                           final_validator=validate_research_answer)

        self.assertEqual(result.stop_reason, "unverified_answer")
        self.assertEqual(result.steps, 2)


if __name__ == "__main__":
    unittest.main()
