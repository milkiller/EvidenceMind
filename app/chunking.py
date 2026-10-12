"""Chunking contracts and practice skeletons for page-level PDF text."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .pdf_reader import PDFDocument


class ChunkStrategy(StrEnum):
    """Chunk strategies included in the first comparison experiment."""

    FIXED_CHAR = "fixed_char"
    FIXED_TOKEN = "fixed_token"
    CHAR_OVERLAP = "char_overlap"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Parameters shared by the chunking strategies."""

    strategy: ChunkStrategy
    chunk_size: int
    overlap: int = 0


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable text unit that preserves its source page."""

    chunk_id: str
    text: str
    page_number: int
    strategy: ChunkStrategy
    position: int


TokenEncoder = Callable[[str], Sequence[int]]
TokenDecoder = Callable[[Sequence[int]], str]


def chunk_document(
    document: PDFDocument,
    config: ChunkConfig,
    *,
    encode: TokenEncoder | None = None,
    decode: TokenDecoder | None = None,
) -> tuple[Chunk, ...]:
    """Route every page to one strategy and return one ordered chunk sequence.

    TODO:
    1. Validate chunk_size and overlap before processing pages.
    2. Select the matching page-level function from config.strategy.
    3. Convert each returned text into Chunk and create stable chunk IDs.
    4. Skip empty results without changing the order of non-empty chunks.

    Token splitting should require encode and decode. Other strategies should
    not depend on a tokenizer.
    """
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be more than 0")
    
    chunks = []
    for page in document.pages:
        if not page.text.strip():
            continue  # Skip empty pages
        if config.strategy == ChunkStrategy.FIXED_CHAR:
            page_chunks = split_fixed_chars(page.text, config.chunk_size)
        elif config.strategy == ChunkStrategy.CHAR_OVERLAP:
            if config.overlap < 0:
                raise ValueError("overlap must be a non-negative integer")
            if config.overlap >= config.chunk_size:
                raise ValueError("overlap must be smaller than chunk_size")
            page_chunks = split_chars_with_overlap(page.text, config.chunk_size, config.overlap)
        elif config.strategy == ChunkStrategy.FIXED_TOKEN:
            if encode is None or decode is None:
                raise ValueError("encode and decode functions must be provided for FIXED_TOKEN strategy")
            page_chunks = split_fixed_tokens(page.text, config.chunk_size, encode=encode, decode=decode)
        elif config.strategy == ChunkStrategy.PARAGRAPH:
            page_chunks = split_paragraphs(page.text, config.chunk_size)
        else:
            raise ValueError(f"Unsupported chunk strategy: {config.strategy}")
        for position, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                continue  # Skip empty chunks
            chunk_id = _build_chunk_id(document.filename, page.page_number, position)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page.page_number,
                    strategy=config.strategy,
                    position=position,
                )
            )

    return tuple(chunks)
    raise NotImplementedError("TODO: implement the chunking router")


def split_fixed_chars(text: str, chunk_size: int) -> tuple[str, ...]:
    """Split text into non-overlapping windows measured in characters.
    
    Example: text="abcdef", chunk_size=2 -> ("ab", "cd", "ef")
    """

    if chunk_size <=0:
        raise ValueError("chunk_size must be more than 0")
    
    chunks = []
    for start in range(0, len(text),chunk_size):
        chunks.append(text[start:start+chunk_size])
    return tuple(chunks)
    raise NotImplementedError("TODO: implement fixed-character splitting")


def split_fixed_tokens(
    text: str,
    chunk_size: int,
    *,
    encode: TokenEncoder,
    decode: TokenDecoder,
) -> tuple[str, ...]:
    """Split text into non-overlapping windows measured by a tokenizer.

    Keep the tokenizer outside this module. This makes the algorithm testable
    with a small fake tokenizer and lets the experiment choose a real model.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be more than 0")
    
    token_ids = encode(text)
    chunks = []
    for start in range(0, len(token_ids), chunk_size):
        chunk_token_ids = token_ids[start:start+chunk_size]
        chunk_text = decode(chunk_token_ids)
        chunks.append(chunk_text)
    return tuple(chunks)
    raise NotImplementedError("TODO: implement fixed-token splitting")


def split_chars_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
) -> tuple[str, ...]:
    """Split character windows while repeating overlap characters.

    Think about the step size first. Reject configurations that cannot move
    the window forward.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be more than 0")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if chunk_size > 0 and overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start+chunk_size])
        end = start + chunk_size
        if end >= len(text):
            break
    return tuple(chunks)

    # raise NotImplementedError("TODO: implement character overlap splitting")


def split_paragraphs(text: str, chunk_size: int) -> tuple[str, ...]:
    """Group adjacent paragraphs without exceeding the character budget.

    Decide and document how to handle one paragraph longer than chunk_size.
    Empty lines should separate paragraphs and should not become chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be more than 0")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        separator_length = 2 if current_chunk else 0
        if len(current_chunk) + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(paragraph) > chunk_size:
                chunks.append(paragraph)
                current_chunk = ""
            else:
                current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return tuple(chunks)
            
    raise NotImplementedError("TODO: implement paragraph-aware splitting")


def _build_chunk_id(filename: str, page_number: int, position: int) -> str:
    """Build a deterministic ID that can be traced to the source page."""
    if page_number < 1:
        raise ValueError("page_number must be a positive integer")
    if position < 0:
        raise ValueError("position must be a non-negative integer")
    if filename is None or filename.strip() == "":
        raise ValueError("filename must be a non-empty string")
    if filename is None or filename == "":
        raise ValueError("filename must be a non-empty string")
    return f"{filename}-page{page_number}-pos{position}"
    # raise NotImplementedError("TODO: choose and implement a stable ID format")
