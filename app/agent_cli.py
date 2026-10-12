"""Run the real model through the hand-written Agent and local tools."""

import argparse
from collections.abc import Sequence
from dataclasses import replace

from .agent_core import ToolRegistry, run_agent
from .agent_gateway import GatewayLLMClient
from .agent_grounding import REFUSAL_MESSAGE, validate_research_answer
from .agent_tools import register_calculator, register_knowledge_search
from .answer_cli import ensure_utf8_stdout
from .embedding import SentenceTransformerEmbedding
from .gateway_generator import GatewayRequestError, load_gateway_settings
from .kb_index import load_knowledge_index
from .retrieval_cli import parse_min_similarity
from .reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实模型自主调用本地工具并回答")
    parser.add_argument("index_path", help="由 app.kb_cli build 生成的本地索引")
    parser.add_argument("question", help="要提给 Agent 的问题")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", help="仅本次覆盖 config.yaml 中的模型名")
    parser.add_argument("--scope", choices=("research", "paper", "literature", "literature_pdf", "all"),
                        default="research")
    parser.add_argument("--history", action="store_true", help="检索旧论文版本")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=parse_min_similarity, default=0.5)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="仅从本地加载嵌入模型")
    parser.add_argument("--rerank", action="store_true", help="对候选片段重排后取最终 Top-K")
    parser.add_argument("--candidate-k", type=int, default=20, help="重排前的 Dense 候选数")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load trusted tools, run the Agent, and show its answer or stop reason."""
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        settings = load_gateway_settings(args.config)
        if args.model:
            settings = replace(settings, model_name=args.model)
        index = load_knowledge_index(args.index_path)
        embedding_model = SentenceTransformerEmbedding(
            model_name=index.model_name, local_files_only=args.offline, device="cpu",
        )
        reranker = CrossEncoderReranker(args.reranker_model, offline=args.offline) if args.rerank else None
        registry = ToolRegistry()
        register_calculator(registry)
        register_knowledge_search(
            registry, index, embedding_model,
            scope=args.scope, include_history=args.history, top_k=args.top_k,
            min_similarity=args.min_similarity,
            max_evidence_chars=settings.max_evidence_chars,
            user_question=args.question,
            reranker=reranker, candidate_k=args.candidate_k,
        )
        result = run_agent(
            args.question, GatewayLLMClient(settings), registry,
            max_steps=args.max_steps,
            final_validator=validate_research_answer,
        )
    except (FileNotFoundError, ValueError, RuntimeError, GatewayRequestError) as exc:
        raise SystemExit(str(exc)) from exc

    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                print(f"工具调用：{call.name} {dict(call.arguments)}")
    if result.answer is not None:
        if any(message.evidence_ids for message in result.messages if message.role == "tool"):
            print("引用校验：通过（仅核对 Chunk ID 来自本轮证据）")
        print(result.answer)
        return 0
    if result.stop_reason == "insufficient_evidence":
        print(REFUSAL_MESSAGE)
    elif result.stop_reason == "unverified_answer":
        print("引用未核实，回答已拦截。")
    print(f"Agent 停止：{result.stop_reason}（模型决策 {result.steps} 次）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
