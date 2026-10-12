from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.kb_cli import main
from app.kb_index import KnowledgeSearchResult


class KnowledgeCliTests(unittest.TestCase):
    def test_retrieval_only_never_calls_gateway(self) -> None:
        result = KnowledgeSearchResult(
            "chunk", 0, "private evidence", 0.8,
            "Paper 5.1", "paper", "段落 1-2", "5.1",
        )
        output = StringIO()
        with (
            patch("app.kb_cli.load_knowledge_index", return_value=SimpleNamespace(model_name="fake")),
            patch("app.kb_cli.SentenceTransformerEmbedding", return_value=object()),
            patch("app.kb_cli.search_knowledge_index", return_value=(result,)) as search,
            patch("app.kb_cli.GatewayChatCompletionGenerator.from_config") as gateway,
            redirect_stdout(output),
        ):
            exit_code = main([
                "ask", "private.json", "question", "--retrieval-only", "--scope", "paper"
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn("Paper 5.1", output.getvalue())
        self.assertNotIn("private evidence", output.getvalue())
        self.assertEqual(search.call_args.kwargs["scope"], "paper")
        gateway.assert_not_called()


if __name__ == "__main__":
    unittest.main()
