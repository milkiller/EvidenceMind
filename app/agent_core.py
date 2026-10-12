"""A small, framework-free agent loop with explicit tool boundaries."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


ActionKind = Literal["tool", "final"]
StopReason = Literal[
    "final_answer", "max_steps", "unknown_tool", "invalid_arguments", "tool_error",
    "insufficient_evidence", "unverified_answer",
]
ParameterType = Literal["string", "number", "integer", "boolean"]


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One tool request and the provider ID needed for its result."""

    name: str
    arguments: Mapping[str, object]
    identifier: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not isinstance(self.arguments, Mapping):
            raise ValueError("tool call needs a name and argument mapping")
        if self.identifier is not None and not self.identifier.strip():
            raise ValueError("tool call identifier must be non-blank")


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One structured model decision: call a tool or give a final answer."""

    kind: ActionKind
    tool_name: str | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)
    answer: str | None = None
    tool_identifier: str | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "tool":
            if not self.tool_name or not self.tool_name.strip() or self.answer is not None:
                raise ValueError("tool action needs a tool name and no final answer")
            if not isinstance(self.arguments, Mapping):
                raise ValueError("tool arguments must be a mapping")
            if self.tool_identifier is not None and not self.tool_identifier.strip():
                raise ValueError("tool call identifier must be non-blank")
            if self.tool_calls and (
                self.tool_calls[0].name != self.tool_name
                or self.tool_calls[0].arguments != self.arguments
                or self.tool_calls[0].identifier != self.tool_identifier
            ):
                raise ValueError("first tool call must match action fields")
        elif self.kind == "final":
            if not self.answer or not self.answer.strip():
                raise ValueError("final action needs a non-blank answer")
            if self.tool_name is not None or self.arguments or self.tool_identifier is not None or self.tool_calls:
                raise ValueError("final action cannot contain a tool call")
        else:
            raise ValueError(f"unsupported action kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A user request, assistant decision, or tool observation."""

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_identifier: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)
    tool_calls: tuple[AgentToolCall, ...] = ()
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The public name and required argument types shown to a model."""

    name: str
    description: str
    parameters: Mapping[str, ParameterType]


class LLMClient(Protocol):
    def decide(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentAction:
        """Choose one structured action using the conversation so far."""
        ...


class UnknownToolError(ValueError):
    """The model requested a tool that was not registered."""


class InvalidToolArgumentsError(ValueError):
    """The model supplied missing, extra, or invalid tool arguments."""


class ToolExecutionError(RuntimeError):
    """A registered tool failed for a reason other than invalid input."""


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Tool text plus trusted evidence IDs that are not model-supplied."""

    content: str
    evidence_ids: tuple[str, ...] = ()


ToolHandler = Callable[[Mapping[str, object]], str | ToolOutput]


class ToolRegistry:
    """Keep tool descriptions separate from trusted Python handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if not spec.name.strip() or not spec.description.strip():
            raise ValueError("tool name and description must be non-blank")
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        if not all(kind in ("string", "number", "integer", "boolean") for kind in spec.parameters.values()):
            raise ValueError("unsupported tool parameter type")
        self._tools[spec.name] = (spec, handler)

    def definitions(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._tools.values())

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        """Preserve the original text-only interface for direct callers."""
        return self.execute_result(name, arguments).content

    def execute_result(self, name: str, arguments: Mapping[str, object]) -> ToolOutput:
        """Validate a requested call before invoking the registered handler."""
        if name not in self._tools:
            raise UnknownToolError(f"unknown tool: {name}")
        spec, handler = self._tools[name]
        if set(arguments) != set(spec.parameters):
            raise InvalidToolArgumentsError(f"invalid arguments for tool: {name}")
        for key, kind in spec.parameters.items():
            value = arguments[key]
            valid = {
                "string": lambda: isinstance(value, str),
                "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
                "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
                "boolean": lambda: isinstance(value, bool),
            }[kind]()
            if not valid:
                raise InvalidToolArgumentsError(f"invalid type for {name}.{key}")
        try:
            observation = handler(arguments)
        except ValueError as exc:
            raise InvalidToolArgumentsError(f"invalid arguments for tool: {name}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"tool failed: {name}") from exc
        if isinstance(observation, str):
            return ToolOutput(observation)
        if not isinstance(observation, ToolOutput) or not isinstance(observation.content, str):
            raise ToolExecutionError(f"tool must return text: {name}")
        if any(not isinstance(chunk_id, str) or not chunk_id.strip()
               for chunk_id in observation.evidence_ids):
            raise ToolExecutionError(f"tool returned invalid evidence IDs: {name}")
        return observation


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    answer: str | None
    stop_reason: StopReason
    messages: tuple[AgentMessage, ...]
    steps: int


@dataclass(frozen=True, slots=True)
class FinalAnswerCheck:
    """A rejected answer can be corrected once if the decision budget permits."""

    stop_reason: Literal["insufficient_evidence", "unverified_answer"]
    feedback: str = ""


FinalValidator = Callable[[str, Sequence[AgentMessage]], FinalAnswerCheck | None]


def run_agent(
    question: str,
    client: LLMClient,
    registry: ToolRegistry,
    *,
    max_steps: int = 4,
    final_validator: FinalValidator | None = None,
) -> AgentRunResult:
    """Repeat model decisions and tool observations until an explicit stop."""
    if not question.strip():
        raise ValueError("question must be non-blank")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")

    messages = [AgentMessage(role="user", content=question)]
    steps = 0
    corrections = 0
    while True:
        if steps >= max_steps:
            reason: StopReason = "unverified_answer" if corrections else "max_steps"
            return AgentRunResult(None, reason, tuple(messages), steps)

        action = client.decide(tuple(messages), registry.definitions())
        steps += 1
        if action.kind == "final":
            check = final_validator(action.answer or "", tuple(messages)) if final_validator else None
            if check is not None:
                if (check.stop_reason == "unverified_answer" and check.feedback
                        and corrections == 0 and steps < max_steps):
                    messages.append(AgentMessage(role="assistant", content=action.answer or ""))
                    messages.append(AgentMessage(role="user", content=check.feedback))
                    corrections += 1
                    continue
                return AgentRunResult(None, check.stop_reason, tuple(messages), steps)
            messages.append(AgentMessage(role="assistant", content=action.answer or ""))
            return AgentRunResult(action.answer, "final_answer", tuple(messages), steps)

        name = action.tool_name or ""
        calls = action.tool_calls or (
            AgentToolCall(name, action.arguments, action.tool_identifier),
        )
        messages.append(
            AgentMessage(
                role="assistant",
                content="",
                tool_name=name,
                arguments=dict(action.arguments),
                tool_identifier=action.tool_identifier,
                tool_calls=calls,
            )
        )
        for call in calls:
            try:
                observation = registry.execute_result(call.name, call.arguments)
            except UnknownToolError:
                return AgentRunResult(None, "unknown_tool", tuple(messages), steps)
            except InvalidToolArgumentsError:
                return AgentRunResult(None, "invalid_arguments", tuple(messages), steps)
            except ToolExecutionError:
                return AgentRunResult(None, "tool_error", tuple(messages), steps)
            messages.append(
                AgentMessage(
                    role="tool", content=observation.content, tool_name=call.name,
                    tool_identifier=call.identifier, evidence_ids=observation.evidence_ids,
                )
            )
