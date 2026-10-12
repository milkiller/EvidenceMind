import json
import unittest
from urllib.request import Request

from app.answer_cli import format_answer
from app.answering import answer_question
from app.gateway_generator import GatewayChatCompletionGenerator, GatewaySettings
from app.kb_index import build_knowledge_index, search_knowledge_index
from app.kb_sources import KnowledgeSource, TextSection


class FakeEmbedding:
    def embed(self, texts):
        return tuple((1.0, 0.0) for _ in texts)


class KnowledgeAnswerFlowTests(unittest.TestCase):
    def test_prompt_and_citation_name_source_without_fake_page(self) -> None:
        source = KnowledgeSource(
            "paper-5.1", "paper", "当前论文", "current.docx", "5.1", True,
            (TextSection("段落 17-21", "odor conditioning"),),
        )
        index = build_knowledge_index((source,), FakeEmbedding(), model_name="fake")
        evidence = search_knowledge_index("问题", index, FakeEmbedding(), top_k=1)
        captured: list[dict] = []

        def executor(request: Request, timeout: float):
            captured.append(json.loads(request.data.decode("utf-8")))
            return {"choices": [{"message": {"content": "模型回答"}}]}

        generator = GatewayChatCompletionGenerator(
            GatewaySettings("test", "https://example.invalid/v1", "secret", 5, 2000),
            request_executor=executor,
        )
        answer = answer_question("问题", evidence, generator)
        rendered = format_answer(answer)
        prompt = captured[0]["messages"][1]["content"]
        self.assertIn("来源 当前论文", prompt)
        self.assertIn("位置 段落 17-21", prompt)
        self.assertIn("当前论文（版本 5.1） | 段落 17-21", rendered)
        self.assertNotIn("第 0 页", rendered)


if __name__ == "__main__":
    unittest.main()
