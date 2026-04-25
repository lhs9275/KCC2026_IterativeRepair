"""
Patch generation wrapper.

Wraps the LLM backend and code extraction logic from the ICSE pipeline.
extract_correct_block_with_reason is imported from the ICSE codebase
since it has deep dependencies (~6500 lines) that are impractical to copy.
"""

import sys
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

# Add icse_lib path for extract_correct_block_with_reason and its helpers
_ICSE_LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "icse_lib"))
if _ICSE_LIB_DIR not in sys.path:
    sys.path.insert(0, _ICSE_LIB_DIR)

# Import from the local copy of llm_backend
_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from llm_backend import LLMBackend, create_backend

logger = logging.getLogger(__name__)


def _import_extract_fn():
    """Lazy import of extract_correct_block_with_reason from ICSE."""
    try:
        # The module name has dots so we use importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "token_patch_gen",
            os.path.join(_ICSE_LIB_DIR, "5.TokenPatchGeneratorAgent.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.extract_correct_block_with_reason
    except Exception as e:
        logger.warning("Failed to import extract_correct_block_with_reason: %s", e)
        return None

# Try importing at module load time
_extract_fn = None


def _get_extract_fn():
    global _extract_fn
    if _extract_fn is None:
        _extract_fn = _import_extract_fn()
    return _extract_fn


def extract_code_from_output(
    text: str,
    language: str = "java",
    base_code: str = "",
    expected_name: str = "",
    repair_branch: str = "java_base",
) -> Tuple[str, str]:
    """
    Extract code from LLM output text.

    Returns:
        (extracted_code, failure_reason)
        If extraction succeeds, failure_reason is empty.
    """
    fn = _get_extract_fn()
    if fn is not None:
        code, reason, _meta = fn(
            text,
            language=language,
            base_code=base_code,
            expected_name=expected_name,
            repair_branch=repair_branch,
        )
        if code:
            return code, reason
        # Fallback for small models that output body-only inside
        # @@BEGIN_JAVA_CODE@@...@@END_JAVA_CODE@@ markers.
        if language == "java":
            body_wrapped = _wrap_body_with_header(text, base_code)
            if body_wrapped:
                return body_wrapped, ""
        return code, reason

    # Fallback: simple extraction if ICSE import fails
    return _simple_extract(text, language)


def _wrap_body_with_header(text: str, base_code: str) -> str:
    """
    Fragment-aware merge for small-LLM outputs.

    Small models (e.g., Qwen 7B) frequently emit only a CONTIGUOUS fragment
    of the target method — the part they actually modified — instead of the
    full method. This function:
      1. Extracts the fragment from @@BEGIN_JAVA_CODE@@...@@END_JAVA_CODE@@
         (falling back to markdown code fences).
      2. If the fragment already looks like a full method, returns it as-is.
      3. Otherwise, uses difflib.SequenceMatcher to align the fragment
         against base_code and splices it in, preserving the surrounding
         method context.
    """
    import re as _re
    from difflib import SequenceMatcher

    if not text or not base_code:
        return ""

    # --- 1. Extract fragment from sentinel markers or code fence ---
    m = _re.search(
        r"@@BEGIN_JAVA_CODE@@\s*([\s\S]*?)\s*@@END_JAVA_CODE@@",
        text,
        flags=_re.IGNORECASE,
    )
    if m:
        fragment = m.group(1).strip()
    else:
        mfence = _re.search(r"```(?:\w+)?\s*\n([\s\S]*?)```", text)
        fragment = mfence.group(1).strip() if mfence else ""
    if not fragment:
        return ""

    # Strip any stray markdown fences inside the fragment
    fragment = _re.sub(r"^```\w*\s*\n?", "", fragment)
    fragment = _re.sub(r"\n?```\s*$", "", fragment).strip()
    if not fragment:
        return ""

    # --- 2. If fragment already has a method signature, return as-is ---
    first_meaningful = next(
        (ln.strip() for ln in fragment.splitlines() if ln.strip() and not ln.strip().startswith("@")),
        "",
    )
    signature_pat = _re.compile(
        r"^(?:public|private|protected|static|final|abstract|synchronized|native|\s)+.*\("
    )
    if signature_pat.match(first_meaningful):
        return fragment

    # --- 3. Align fragment into base_code via strong anchors ---
    base_lines = base_code.splitlines()
    frag_lines = fragment.splitlines()
    while frag_lines and not frag_lines[0].strip():
        frag_lines.pop(0)
    while frag_lines and not frag_lines[-1].strip():
        frag_lines.pop()
    if not frag_lines:
        return ""

    # A "strong anchor" is a line whose stripped text is unlikely to appear
    # spuriously elsewhere in the method (i.e. not just "{", "}", "else {",
    # "break;", etc.). We use strong anchors to align fragment ↔ base_code.
    _WEAK = {"{", "}", "};", "else", "else {", "} else {", "} else",
             "break;", "continue;", "return;", "try {", "} catch", "};",
             "} finally {", "default:", "case :"}
    def _is_strong(ln: str) -> bool:
        s = ln.strip()
        if not s or len(s) <= 2:
            return False
        if s in _WEAK:
            return False
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            return False
        return True

    frag_strong = [(i, ln.strip()) for i, ln in enumerate(frag_lines) if _is_strong(ln)]
    if not frag_strong:
        return ""  # no way to align

    # Find first strong anchor in base_code
    first_frag_idx, first_frag_text = frag_strong[0]
    base_start = None
    for bi, bln in enumerate(base_lines):
        if bln.strip() == first_frag_text:
            base_start = bi
            break
    if base_start is None:
        return ""

    # Find last strong anchor in base_code (must be AFTER base_start)
    last_frag_idx, last_frag_text = frag_strong[-1]
    base_end = None
    for bi in range(len(base_lines) - 1, base_start - 1, -1):
        if base_lines[bi].strip() == last_frag_text:
            base_end = bi
            break
    if base_end is None:
        base_end = base_start

    # start_i: base index where the fragment's first line should land.
    # The fragment may have a few lines (like a new `if (c != null) {`)
    # before its first strong anchor that are being ADDED, not replacing.
    start_i = base_start - first_frag_idx
    start_i = max(0, start_i)

    # end_i: must be chosen so that the net brace balance after replacement
    # is preserved. The idea: after replacement, the number of unmatched
    # "{" minus "}" in the replaced region must equal the fragment's.
    #
    # Algorithm: walk forward from base_end, absorbing lines until the
    # fragment's (open-count - close-count) == base region's
    # (open-count - close-count). This keeps the outer method's brace
    # balance intact.
    def _brace_delta(lines):
        opens = closes = 0
        for ln in lines:
            in_string = in_char = False
            i = 0
            while i < len(ln):
                c = ln[i]
                if in_string:
                    if c == "\\":
                        i += 2
                        continue
                    if c == '"':
                        in_string = False
                elif in_char:
                    if c == "\\":
                        i += 2
                        continue
                    if c == "'":
                        in_char = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == "'":
                        in_char = True
                    elif c == "/" and i + 1 < len(ln) and ln[i + 1] == "/":
                        break
                    elif c == "{":
                        opens += 1
                    elif c == "}":
                        closes += 1
                i += 1
        return opens - closes

    frag_delta = _brace_delta(frag_lines)

    end_i = base_end + 1
    while end_i <= len(base_lines):
        region = base_lines[start_i:end_i]
        if _brace_delta(region) == frag_delta:
            # Also require that the last absorbed line is a closing-brace
            # line (to avoid cutting mid-statement). Walk one more if not.
            break
        if end_i == len(base_lines):
            break
        end_i += 1

    # Safety: if we blew past the method, fall back to base_end + 1
    if _brace_delta(base_lines[start_i:end_i]) != frag_delta:
        end_i = base_end + 1

    if end_i <= start_i:
        end_i = start_i + 1

    # Preserve the indentation of the replaced region: the first replaced
    # line's indent becomes the anchor for the fragment's top-level indent.
    anchor_indent = 0
    for i in range(start_i, end_i):
        stripped = base_lines[i].lstrip(" \t")
        if stripped:
            anchor_indent = len(base_lines[i]) - len(stripped)
            break

    frag_first_indent = 0
    for ln in frag_lines:
        stripped = ln.lstrip(" \t")
        if stripped:
            frag_first_indent = len(ln) - len(stripped)
            break
    delta = anchor_indent - frag_first_indent

    def _reindent(line: str) -> str:
        if not line.strip():
            return line
        if delta > 0:
            return " " * delta + line
        if delta < 0:
            strip_n = min(-delta, len(line) - len(line.lstrip(" \t")))
            return line[strip_n:]
        return line

    reindented = [_reindent(ln) for ln in frag_lines]
    merged_lines = base_lines[:start_i] + reindented + base_lines[end_i:]
    return "\n".join(merged_lines)


def _simple_extract(text: str, language: str) -> Tuple[str, str]:
    """Minimal fallback: extract code between ##correct markers or code fences."""
    import re
    # Try ##correct marker
    m = re.search(r"^[ \t]*##correct[ \t]*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        code = text[m.end():].strip()
        # Remove trailing code fences
        code = re.sub(r"```\s*$", "", code).strip()
        if code:
            return code, ""

    # Try code fence
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), ""

    # Return raw text as last resort
    stripped = text.strip()
    if stripped:
        return stripped, ""
    return "", "extraction_failed_empty_after_trim"


def generate_candidates(
    backend: LLMBackend,
    prompt: str,
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    seed: int = 42,
    n: int = 5,
    language: str = "java",
    base_code: str = "",
    expected_name: str = "",
    repair_branch: str = "java_base",
    stop: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate patch candidates via LLM and extract code.

    Returns list of dicts:
        [{"code": str, "reason": str, "raw_output": str}, ...]
    """
    records = backend.generate_records(
        [prompt],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        stop=stop,
        n=n,
    )

    candidates = []
    seen_codes = set()
    for rec in records:
        code, reason = extract_code_from_output(
            rec.text,
            language=language,
            base_code=base_code,
            expected_name=expected_name,
            repair_branch=repair_branch,
        )
        # Dedup
        if code and code in seen_codes:
            reason = "duplicate"
            code = ""
        if code:
            seen_codes.add(code)

        candidates.append({
            "code": code,
            "reason": reason,
            "raw_output": rec.text,
            "tokens_in": rec.tokens_in,
            "tokens_out": rec.tokens_out,
        })

    return candidates
