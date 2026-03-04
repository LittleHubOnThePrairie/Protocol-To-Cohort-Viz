"""Side-by-side original vs retemplated protocol diff view (PTCV-79).

Pure-Python helpers for building diff inputs. The Streamlit rendering
(``st_code_diff``) is called from ``app.py`` to keep this module
testable without a Streamlit runtime.
"""

from __future__ import annotations

from typing import Sequence


def build_original_text(
    text_block_dicts: Sequence[dict],
) -> str:
    """Concatenate sorted extraction text blocks into a single string.

    Args:
        text_block_dicts: List of dicts with ``page_number`` and
            ``text`` keys, already sorted by (page_number, block_index).

    Returns:
        Concatenated text with page break markers.
    """
    if not text_block_dicts:
        return ""

    lines: list[str] = []
    current_page: int | None = None

    for block in text_block_dicts:
        page = block.get("page_number", 0)
        text = str(block.get("text", "")).strip()
        if not text:
            continue

        if current_page is not None and page != current_page:
            lines.append(f"\n--- Page {page} ---\n")
        current_page = page
        lines.append(text)

    return "\n\n".join(lines)


def build_diff_label(
    registry_id: str,
    retemplating_method: str = "",
) -> tuple[str, str]:
    """Return (left_label, right_label) for the diff panels.

    Args:
        registry_id: Protocol registry ID (e.g. NCT00112827).
        retemplating_method: Method used (llm, rag, ich_parser).

    Returns:
        Tuple of (original_label, retemplated_label).
    """
    left = f"Original — {registry_id}"
    method_tag = ""
    if retemplating_method == "llm":
        method_tag = " (LLM)"
    elif retemplating_method == "rag":
        method_tag = " (RAG)"
    right = f"Retemplated ICH E6(R3){method_tag}"
    return left, right
