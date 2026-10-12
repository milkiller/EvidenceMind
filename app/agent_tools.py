"""Small local tools for the hand-written agent."""

import ast
from collections.abc import Mapping
import operator

from .agent_core import ToolOutput, ToolRegistry, ToolSpec
from .embedding import EmbeddingModel
from .kb_index import KnowledgeIndex, search_knowledge_index
from .reranker import PassageReranker, rerank_candidates


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_expression(node: ast.AST) -> int | float:
    """Evaluate only arithmetic syntax; never execute Python names or calls."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        try:
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("division by zero") from exc
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate_expression(node.operand))
    else:
        raise ValueError("unsupported calculator expression")
    if abs(value) > 1_000_000_000_000:
        raise ValueError("calculator result is too large")
    return value


def calculate(arguments: Mapping[str, object]) -> str:
    """Return the result of a bounded arithmetic expression as text."""
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 100:
        raise ValueError("expression must be a non-blank string of at most 100 characters")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid calculator expression") from exc
    return str(_evaluate_expression(parsed.body))


def register_calculator(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters={"expression": "string"},
        ),
        calculate,
    )


def resolve_zotero_pdf_title(index: KnowledgeIndex, requested_title: str) -> str:
    """Resolve an explicitly named PDF to one indexed title, without guessing."""
    if not requested_title.strip():
        raise ValueError("source_title must be non-blank")
    matches = {
        chunk.title for chunk in index.chunks
        if chunk.kind == "zotero_pdf"
        and requested_title.casefold().strip() in chunk.title.casefold()
    }
    if len(matches) != 1:
        raise ValueError("source_title must uniquely identify one Zotero PDF")
    return matches.pop()


def register_knowledge_search(
    registry: ToolRegistry,
    index: KnowledgeIndex,
    embedding_model: EmbeddingModel,
    *,
    scope: str = "research",
    include_history: bool = False,
    top_k: int = 3,
    min_similarity: float = 0.5,
    max_evidence_chars: int = 2400,
    user_question: str | None = None,
    reranker: PassageReranker | None = None,
    candidate_k: int = 20,
) -> None:
    """Register bounded local retrieval as an Agent tool."""
    if top_k <= 0 or not -1 <= min_similarity <= 1 or max_evidence_chars <= 0:
        raise ValueError("invalid knowledge search limits")
    if reranker is not None and candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    def search(arguments: Mapping[str, object]) -> ToolOutput:
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-blank string")
        requested_title = arguments["source_title"]
        if not isinstance(requested_title, str):
            raise ValueError("source_title must be a string")
        source_title: str | None = None
        if requested_title.strip() and scope != "paper":
            if (
                user_question is not None
                and requested_title.casefold().strip() not in user_question.casefold()
            ):
                raise ValueError("source_title must appear in the user question")
            source_title = resolve_zotero_pdf_title(index, requested_title)
        results = search_knowledge_index(
            query, index, embedding_model,
            top_k=candidate_k if reranker is not None else top_k,
            scope=scope, include_history=include_history, source_title=source_title,
        )
        eligible = [result for result in results if result.similarity >= min_similarity]
        evidence = (
            rerank_candidates(query, eligible, reranker, top_k=top_k)
            if reranker is not None else eligible
        )
        if not evidence:
            return ToolOutput("没有达到相似度阈值的证据；请说明证据不足。")
        blocks: list[str] = []
        included_ids: list[str] = []
        used = 0
        for result in evidence:
            header = (
                f"[来源：{result.source_title} | 位置：{result.locator} | "
                f"版本：{result.version or '无'} | Chunk：{result.chunk_id} | "
                f"相似度：{result.similarity:.4f}]\n"
            )
            room = max_evidence_chars - used - len(header) - (2 if blocks else 0)
            if room <= 0:
                break
            block = header + result.text.strip()[:room]
            used += len(block) + (2 if blocks else 0)
            blocks.append(block)
            included_ids.append(result.chunk_id)
        if not blocks:
            return ToolOutput("证据预算不足；请说明证据不足。")
        return ToolOutput("\n\n".join(blocks), tuple(included_ids))

    registry.register(
        ToolSpec(
            name="search_knowledge",
            description=(
                "Search the local paper, Obsidian and Zotero knowledge base. "
                "For an English paper, search using English technical terms even if the "
                "user asks in Chinese. When the user explicitly names a Zotero PDF, "
                "set source_title to that title or its distinctive substring; otherwise "
                "set source_title to an empty string. Cite source, location and Chunk ID."
            ),
            parameters={"query": "string", "source_title": "string"},
        ),
        search,
    )
