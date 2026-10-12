"""Check that research answers cite IDs returned by trusted local tools."""

import re
from collections.abc import Sequence

from .agent_core import AgentMessage, FinalAnswerCheck


REFUSAL_MESSAGE = "证据不足，无法根据当前知识库回答。"
_CITATION = re.compile(r"【Chunk:([^】\r\n]+)】")


def validate_research_answer(
    answer: str, messages: Sequence[AgentMessage]
) -> FinalAnswerCheck | None:
    """Verify cited Chunk IDs, not whether the cited text supports each claim."""
    searches = [
        message for message in messages
        if message.role == "tool" and message.tool_name == "search_knowledge"
    ]
    if not searches:
        if any(message.role == "tool" and message.tool_name == "calculator" for message in messages):
            return None
        return FinalAnswerCheck(
            "unverified_answer",
            "这是研究问答命令。请先调用 search_knowledge 获取证据；"
            "若只需算术，请调用 calculator。不要直接陈述未检索的文档事实。",
        )

    allowed_ids = {chunk_id for message in searches for chunk_id in message.evidence_ids}
    if not allowed_ids or answer.strip() == REFUSAL_MESSAGE:
        return FinalAnswerCheck("insufficient_evidence")

    cited_ids = _CITATION.findall(answer)
    if cited_ids and all(chunk_id in allowed_ids for chunk_id in cited_ids):
        return None

    listed_ids = "、".join(sorted(allowed_ids))
    return FinalAnswerCheck(
        "unverified_answer",
        "引用校验失败。请只依据已返回的证据重新回答，每个文档结论后"
        "使用精确格式【Chunk:实际ID】；可用 ID：" + listed_ids + "。"
        "若这些片段不能支持答案，请只回复：" + REFUSAL_MESSAGE,
    )
