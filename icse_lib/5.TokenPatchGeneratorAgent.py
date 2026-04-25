# =============================================
# File: 5.TokenPatchGeneratorAgent_vllm_greedy1_diverse9.MULTIRUN.py
# Desc: Patch generator (prompt-only, diversified sampling, vLLM backend)
# I/O:  prom  = ./Results/4/*.PlanAgent.json
#       out   = ./Results/5/5.PatchesResults_vllm.json
# =============================================
import re
import ast
import glob
import json
import textwrap
import random
import copy
import logging
import hashlib
import time
import os
import subprocess
import sys
from functools import cmp_to_key
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import difflib
from llm_backend import (
    BACKEND_CHOICES,
    HF_DEVICE_CHOICES,
    HF_DTYPE_CHOICES,
    create_backend,
)

# NOTE: Some environments have torch/torchvision mismatch.
# For this text-only pipeline, hide torchvision to avoid import-time failures.
if os.environ.get("ESWA_DISABLE_TORCHVISION", "1") == "1":
    import importlib.util as _importlib_util

    _orig_find_spec = _importlib_util.find_spec

    def _find_spec(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    _importlib_util.find_spec = _find_spec

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    LLM = None  # type: ignore
    SamplingParams = Any  # type: ignore
    _VLLM_IMPORT_ERR = _e

try:
    import torch
except Exception:
    torch = None  # type: ignore

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ===== Default Paths =====
RESULTS_ROOT = Path(os.environ.get("PIPELINE_RESULTS_ROOT", "./Results"))
DEFAULT_PROMPTS_PATH = RESULTS_ROOT / "4"
DEFAULT_OUT_PATH     = RESULTS_ROOT / "5/5.PatchesResults_vllm.json"
DEFAULT_MODEL_NAME   = "../models/Qwen2.5-Coder-7B-Instruct"
REPAIR_BRANCH_CHOICES = ("auto", "python_base", "java_base", "java_semantic", "java_v2")  # Phase1-E: java_v2 추가
JAVA_REPAIR_BRANCHES = {"java_base", "java_semantic", "java_v2"}  # Phase1-E: java_v2 추가
JAVA_SURVIVOR_BACKOFF_CHOICES = ("off", "hybrid", "fill")
JAVA_SEMANTIC_RERANK_FEATURES = (
    "exact_signature_preservation",
    "throws_clause_preservation",
    "original_identifier_reuse",
    "new_helper_call_count",
    "new_type_reference_count",
    "new_constant_reference_count",
    "new_identifier_reference_count",
    "api_signature_drift",
    "plan_ok",
    "low_violation_count",
    "near_bug_region",
    "markdown_or_explanation_tail",
    "model_artifact",
    "invented_helper_declaration",
    "extra_nested_declaration",
    "plan_fill_selected",
    "high_violation_count",
    "broad_rewrite",
    "broad_rewrite_severe",
    "penalty_only_reason_count",
    "broad_rewrite_combo",
    "risk_combo_penalty",
    "change_fraction_risk_penalty",
    "context_supported_reuse_bonus",
    "risk_tier",
)

# ===== Regex =====
CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```", re.IGNORECASE)
DEFCLASS_HEAD_RE = re.compile(
    r"^\s*(?:@[^\n]+\n\s*)*(?:async\s+def|def|class)\s",
    re.MULTILINE,
)
JAVA_DECL_HEAD_RE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*"
    r"(?:"
    r"(?:class|interface|enum|record)\s+[A-Za-z_]\w*"
    r"|(?:<[^>{}]+>\s*)?[A-Za-z_][\w<>\[\],.?&\s]*\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\s*\{"
    r"|(?!(?:if|for|while|switch|catch|return|throw|new|synchronized|do|else|try)\b)"
    r"[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\s*\{"
    r")",
    re.MULTILINE,
)
JAVA_NAME_RE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*"
    r"(?:"
    r"(?:class|interface|enum|record)\s+([A-Za-z_]\w*)"
    r"|(?:<[^>{}]+>\s*)?[A-Za-z_][\w<>\[\],.?&\s]*\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\s*\{"
    r"|(?!(?:if|for|while|switch|catch|return|throw|new|synchronized|do|else|try)\b)"
    r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\s*\{"
    r")",
    re.MULTILINE,
)
MODEL_ARTIFACT_COMMENT_HINT_RE = re.compile(
    r"(?:<---\s*buggy line|corrected\b|fixed(?:\s+line)?\b|patched(?:\s+line)?\b)",
    re.IGNORECASE,
)
MODEL_ARTIFACT_TEXT_RE = re.compile(
    r"(?:this is the corrected version|changed line|replace if|buggy line|rest of the code|rest of the function|rest of the method|todo\b)",
    re.IGNORECASE,
)
TRUNCATION_ARTIFACT_RE = re.compile(
    r"(?:rest of the code|rest of the function|rest of the method|omitted for brevity)"
    r"|^\s*(?://|#|\*)\s*\.\.\.\s*(?:\(|$)"
    r"|^\s*\.\.\.\s*(?:\(|$)",
    re.IGNORECASE | re.MULTILINE,
)
JAVA_OLD_COMPAT_HARD_RE = re.compile(r"<>")
JAVA_STRING_OR_CHAR_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
LANGUAGE_LABELS = {"java", "python"}
JAVA_PLAN_TARGET_SLACK = 2
JAVA_PLAN_CHANGED_LINES_SLACK = 3
JAVA_CODE_BEGIN_MARKER = "@@BEGIN_JAVA_CODE@@"
JAVA_CODE_END_MARKER = "@@END_JAVA_CODE@@"
JAVA_BODY_BEGIN_MARKER = "@@BEGIN_JAVA_BODY@@"    # Phase1-E: java_v2 body-only 마커
JAVA_BODY_END_MARKER = "@@END_JAVA_BODY@@"        # Phase1-E: java_v2 body-only 마커
JAVA_VALIDITY_REASON_KEYS = (
    "unbalanced_delimiters",
    "decl_kind_mismatch",
    "signature_mismatch",
    "wrapper_mismatch",
    "model_artifact",
    "broad_rewrite",
    "new_helper_call",
    "new_type_reference",
    "new_constant_reference",
    "new_identifier_reference",
    "api_signature_drift",
    "compile_fail",
)
JAVA_VALIDITY_RERANK_BONUS = 60.0

DROP_REASON_KEYS = (
    "empty_code",
    "extraction_failed",
    "extraction_failed_empty_after_trim",
    "extraction_failed_no_code_like",
    "extraction_failed_model_artifact",
    "extraction_failed_too_short",
    "extraction_failed_postprocess",
    "introduced_model_artifact",
    "introduced_non_ascii",
    "introduced_java_incompatible_syntax",
    "syntax_fail",
    "no_def_or_class",
    "name_mismatch",
    "java_compile_invalid",
    "no_change",
    "plan_violation",
    "duplicate",
    "other",
)
PLAN_VIOLATION_TYPE_KEYS = (
    "edit_type",
    "out_of_target",
    "changed_lines_exceed",
    "other",
)

# ===== Banned patterns =====
BANNED_PATTERNS = [
    r"\.\.\.",
    r"\braise\s+NotImplementedError\b",
    r"\bprint\s*\(",
]
BANNED_TOKENS = [re.compile(p) for p in BANNED_PATTERNS]
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false",
    "null",
}
JAVA_COMMON_TYPE_NAMES = {
    "Object", "String", "Class", "Throwable", "Exception", "RuntimeException",
    "IllegalArgumentException", "IllegalStateException", "UnsupportedOperationException",
    "Boolean", "Byte", "Short", "Integer", "Long", "Float", "Double", "Number",
    "Character", "Math", "System", "Arrays", "Collections", "Collection", "List",
    "ArrayList", "LinkedList", "Set", "HashSet", "Map", "HashMap", "TreeMap",
    "Iterator", "Iterable", "Comparator", "StringBuilder", "StringBuffer",
    "Pattern", "Matcher", "IOException",
}
JAVA_COMMON_CALL_NAMES = {
    "equals", "hashCode", "toString", "valueOf", "ordinal", "name", "length",
}
JAVA_SOFT_CONTEXT_REASON_KEYS = {
    "broad_rewrite",
    "new_identifier_reference",
    "new_type_reference",
    "new_constant_reference",
    "new_helper_call",              # Phase1-B: 추가 — hard에서 soft로 이동
}
JAVA_NEW_REFERENCE_REASON_KEYS = {
    "new_identifier_reference",
    "new_type_reference",
    "new_constant_reference",
}
JAVA_REJECT_POLICY_VERSION = "round4"
JAVA_RESCUE_POLICY_VERSION = "round3"
JAVA_RESCUE_ALLOWED_REASON_KEYS = {
    "new_identifier_reference",
    "new_type_reference",
    "new_constant_reference",
    "new_helper_call",              # Phase1-B: 추가 — 새 헬퍼 호출도 구출 허용
}
JAVA_RESCUE_ALLOWED_SUPPORT_SOURCES = {
    "helper_calls",
    "types",
    "instance_fields",
    "method_signature",
    "context_summary",
}
JAVA_RESCUE_MAX_SCAN = 5                # Phase1-B: 2→5 더 많은 후보 검토
JAVA_RESCUE_MAX_SURVIVORS_PER_BUG = 3   # Phase1-B: 1→3 구출 가능 후보 확대
JAVA_RESCUE_MAX_REASON_ITEMS = 4        # Phase1-B: 2→4 새 식별자 허용 범위 확대
JAVA_RESCUE_MAX_CHANGE_FRACTION = 0.40  # Phase1-B: 0.25→0.40 변경 비율 허용 확대
JAVA_RISK_TIE_MARGIN = 10.0            # Phase1-B: 6.0→10.0 동점 마진 확대

# ===== Common utils =====
def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    cur = data
    for k in path:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def normalize_candidate_code_for_hash(code: str) -> str:
    text = str(code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def compute_normalized_candidate_hash(code: str) -> str:
    normalized = normalize_candidate_code_for_hash(code)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

def _strip_leading_language_label(text: str) -> str:
    lines = (text or "").splitlines()
    while lines and lines[0].strip().lower() in LANGUAGE_LABELS:
        lines = lines[1:]
    return "\n".join(lines).strip()

def _contains_model_artifact(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    lines = s.splitlines()
    if lines and lines[0].strip().lower() in LANGUAGE_LABELS:
        return True
    if "```" in s or re.search(r"^[ \t]*##correct\b", s, flags=re.IGNORECASE | re.MULTILINE):
        return True
    return bool(TRUNCATION_ARTIFACT_RE.search(s))

def _strip_inline_artifact_comments(text: str, language: str) -> str:
    cleaned: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if language == "java":
            if re.match(r"^\s*(?://+|\*+|/\*+)\s*", stripped) and MODEL_ARTIFACT_COMMENT_HINT_RE.search(stripped):
                continue
            line = re.sub(
                r"\s*//\s*(?:<---\s*buggy line|corrected\b.*|fixed(?:\s+line)?\b.*|patched(?:\s+line)?\b.*)$",
                "",
                line,
                flags=re.IGNORECASE,
            )
            line = re.sub(
                r"\s*/\*\s*(?:<---\s*buggy line|corrected\b.*|fixed(?:\s+line)?\b.*|patched(?:\s+line)?\b.*)\*/\s*$",
                "",
                line,
                flags=re.IGNORECASE,
            )
        else:
            if stripped.startswith("#") and MODEL_ARTIFACT_COMMENT_HINT_RE.search(stripped):
                continue
            line = re.sub(
                r"\s*#\s*(?:<---\s*buggy line|corrected\b.*|fixed(?:\s+line)?\b.*|patched(?:\s+line)?\b.*)$",
                "",
                line,
                flags=re.IGNORECASE,
            )
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()

def _code_like(code: str, language: str) -> bool:
    s = (code or "").strip()
    if not s:
        return False
    if language == "java":
        return bool(JAVA_DECL_HEAD_RE.search(s))
    return bool(DEFCLASS_HEAD_RE.search(s))


def _strip_java_literals(text: str) -> str:
    return JAVA_STRING_OR_CHAR_RE.sub('""', text or "")


def _changed_new_text_chunks(base_code: str, code: str) -> List[str]:
    summary = compute_edit_summary(base_code or "", code or "")
    chunks: List[str] = []
    for edit in summary.get("edits") or []:
        new_text = str(edit.get("new_text", "") or "")
        if new_text.strip():
            chunks.append(new_text)
    return chunks


def _detect_java_candidate_issue(base_code: str, code: str) -> str:
    # Only reject content newly introduced by the candidate.
    changed_chunks = _changed_new_text_chunks(base_code, code)
    if not changed_chunks:
        return ""
    changed_text = "\n".join(changed_chunks)
    stripped = _strip_java_literals(changed_text)
    if MODEL_ARTIFACT_TEXT_RE.search(stripped) or MODEL_ARTIFACT_COMMENT_HINT_RE.search(stripped):
        return "introduced_model_artifact"
    if TRUNCATION_ARTIFACT_RE.search(stripped):
        return "introduced_model_artifact"
    if any(ord(ch) > 127 for ch in stripped):
        return "introduced_non_ascii"
    if JAVA_OLD_COMPAT_HARD_RE.search(stripped):
        return "introduced_java_incompatible_syntax"
    return ""


# ===== Diff/patch output sanitization =====
DIFF_HEADER_RE = re.compile(
    r"^(diff --git|index\s|---\s|\+\+\+\s|@@\s|new file mode|deleted file mode|similarity index|rename from|rename to)\b"
)

def _sanitize_diff_like(text: str, language: str) -> str:
    """Best-effort salvage when the model outputs unified diff or +/- prefixed lines.

    - Removes diff headers (diff --git, ---/+++, @@ hunks, etc.)
    - Strips leading diff markers (+/-/space) from remaining lines
    - Handles the common failure case: '+def foo(...)' -> 'def foo(...)'
    """
    s = (text or "").replace("\r\n", "\n").replace("\x00", "")
    if not s.strip():
        return s

    lines = s.splitlines()
    head = lines[:40]
    diffish = any(DIFF_HEADER_RE.match(l) for l in head) or any(l.startswith("@@") for l in head)

    if language == "java":
        plus_decl = any(
            re.match(r"^[+-]\s*", l) and JAVA_DECL_HEAD_RE.search(l[1:])
            for l in head
        )
    else:
        plus_decl = any(
            re.match(r"^[+-]\s*", l)
            and re.match(r"^\s*(?:@|async\s+def\b|def\b|class\b)", l[1:])
            for l in head
        )

    if not diffish and not plus_decl:
        return s

    cleaned: List[str] = []
    if diffish:
        for l in lines:
            if DIFF_HEADER_RE.match(l) or l.startswith("@@"):
                continue
            # unified diff line prefix: ' ', '+', '-'
            if l[:1] in "+- ":
                cleaned.append(l[1:])
            else:
                cleaned.append(l)
    else:
        # Diff-like output without explicit headers: strip unified diff prefixes from all lines.
        for l in lines:
            if l[:1] in "+- ":
                cleaned.append(l[1:])
            else:
                cleaned.append(l)

    return "\n".join(cleaned).strip()


def _init_extract_meta(language: str) -> Dict[str, Any]:
    return {
        "java_sentinel_extract_hit": False,
        "java_decl_salvage_used": False,
        "java_explanation_tail_stripped": False,
        "java_decl_recovery_used": False,
        "java_decl_recovery_mode": "",
        "java_v2_body_mode": False,             # Phase1-E: body-only 추출 여부
        "notes": [],
    } if language == "java" else {}


def _merge_extract_meta(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not extra:
        return base
    out = dict(base or {})
    for key in (
        "java_sentinel_extract_hit",
        "java_decl_salvage_used",
        "java_explanation_tail_stripped",
        "java_decl_recovery_used",
    ):
        out[key] = bool(out.get(key, False) or extra.get(key, False))
    if not out.get("java_decl_recovery_mode") and extra.get("java_decl_recovery_mode"):
        out["java_decl_recovery_mode"] = str(extra.get("java_decl_recovery_mode") or "")
    notes = list(out.get("notes") or [])
    for note in list(extra.get("notes") or []):
        note_s = str(note or "")
        if note_s and note_s not in notes:
            notes.append(note_s)
    if notes:
        out["notes"] = notes
    return out


def _strip_java_output_markers(text: str) -> str:
    lines: List[str] = []
    marker_set = {JAVA_CODE_BEGIN_MARKER, JAVA_CODE_END_MARKER,
                  JAVA_BODY_BEGIN_MARKER, JAVA_BODY_END_MARKER}  # Phase1-E: body 마커 추가
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if stripped in marker_set:
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def _extract_java_sentinel_block(text: str) -> str:
    source = (text or "").replace("\x00", "")
    if not source.strip():
        return ""
    pat = re.compile(
        re.escape(JAVA_CODE_BEGIN_MARKER) + r"\s*([\s\S]*?)\s*" + re.escape(JAVA_CODE_END_MARKER),
        re.IGNORECASE,
    )
    matches = [m.group(1).strip() for m in pat.finditer(source) if m.group(1).strip()]
    if not matches:
        return ""
    return max(matches, key=len)


def _extract_java_body_sentinel_block(text: str) -> str:
    """Phase1-E: java_v2 body-only 마커(@@BEGIN_JAVA_BODY@@...@@END_JAVA_BODY@@) 추출"""
    source = (text or "").replace("\x00", "")
    if not source.strip():
        return ""
    pat = re.compile(
        re.escape(JAVA_BODY_BEGIN_MARKER) + r"\s*([\s\S]*?)\s*" + re.escape(JAVA_BODY_END_MARKER),
        re.IGNORECASE,
    )
    matches = [m.group(1).strip() for m in pat.finditer(source) if m.group(1).strip()]
    if not matches:
        return ""
    return max(matches, key=len)


def _first_nonempty_java_line(text: str) -> str:
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if stripped:
            return stripped
    return ""


def _looks_like_java_decl_opening(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    while lines and lines[0].startswith("@"):
        lines = lines[1:]
    if not lines:
        return False
    first = lines[0]
    if re.search(r"\b(?:class|interface|enum|record)\b", first):
        return True
    return "(" in first


def _trim_to_java_decl_boundary(text: str) -> str:
    lines = (text or "").splitlines()
    for idx in range(len(lines)):
        candidate = "\n".join(lines[idx:]).strip()
        if not candidate:
            continue
        match = JAVA_DECL_HEAD_RE.search(candidate)
        if not match or match.start() != 0:
            continue
        if _looks_like_java_decl_opening(candidate):
            return candidate
    return str(text or "").strip()


def _find_java_balanced_block(source: str, start_idx: int) -> Tuple[str, int]:
    src = str(source or "")
    if not src or start_idx < 0 or start_idx >= len(src):
        return "", -1
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False
    open_idx = -1
    depth = 0
    i = int(start_idx)
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_char = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue
        if ch == ";" and open_idx < 0:
            return "", -1
        if ch == "{":
            if open_idx < 0:
                open_idx = i
                depth = 1
            else:
                depth += 1
        elif ch == "}" and open_idx >= 0:
            depth -= 1
            if depth == 0:
                return src[start_idx:i + 1], i + 1
        i += 1
    return "", -1


def _collect_java_decl_blocks(source: str) -> List[Dict[str, Any]]:
    src = str(source or "")
    decls: List[Dict[str, Any]] = []
    seen_starts: set[int] = set()
    for match in JAVA_DECL_HEAD_RE.finditer(src):
        start = int(match.start())
        if start in seen_starts:
            continue
        block, end = _find_java_balanced_block(src, start)
        if not block:
            continue
        cleaned = _strip_inline_artifact_comments(_strip_java_output_markers(block), language="java").strip()
        cleaned = _trim_to_java_decl_boundary(cleaned)
        if not cleaned:
            continue
        decls.append({
            "start": start,
            "end": int(end),
            "block": cleaned,
            "name": extract_decl_name(cleaned, language="java"),
            "contract": _extract_java_decl_contract(cleaned),
        })
        seen_starts.add(start)
    return decls


def _select_java_decl_block(source: str, expected_name: str = "") -> Optional[Dict[str, Any]]:
    decls = _collect_java_decl_blocks(source)
    if not decls:
        return None
    if expected_name:
        exact = [d for d in decls if str(d.get("name") or "") == str(expected_name)]
        if exact:
            return exact[0]
    return decls[0]


def _extract_java_decl_body(block: str) -> Optional[str]:
    src = str(block or "")
    if not src.strip():
        return None
    decl_match = JAVA_DECL_HEAD_RE.search(src)
    if not decl_match:
        return None
    full_block, _end = _find_java_balanced_block(src, decl_match.start())
    if not full_block:
        return None
    open_idx = full_block.find("{")
    if open_idx < 0 or not full_block.endswith("}"):
        return None
    return full_block[open_idx + 1:-1].strip("\n")


def _compose_java_decl_with_base_header(base_code: str, body: str) -> str:
    header = _extract_java_decl_header(base_code)
    if not header:
        return ""
    body_text = str(body or "").strip("\n")
    if body_text:
        return f"{header} {{\n{body_text}\n}}"
    return f"{header} {{\n}}"


def _maybe_recover_java_decl_wrapper(
    *,
    base_code: str,
    candidate_source: str,
    expected_name: str,
) -> Tuple[str, Dict[str, Any]]:
    meta = _init_extract_meta("java")
    if not str(base_code or "").strip():
        return "", meta
    if not str(expected_name or "").strip():
        return "", meta
    decls = _collect_java_decl_blocks(candidate_source)
    if len(decls) != 1:
        return "", meta
    decl = decls[0]
    block = str(decl.get("block") or "")
    if not block or not _java_delimiters_balanced(block):
        return "", meta
    body = _extract_java_decl_body(block)
    if body is None:
        return "", meta
    base_contract = _extract_java_decl_contract(base_code)
    cand_contract = _extract_java_decl_contract(block)
    base_kind = str(base_contract.get("kind") or "")
    cand_kind = str(cand_contract.get("kind") or "")
    need_recovery = False
    if str(cand_contract.get("name") or "") != str(expected_name):
        need_recovery = True
    if base_kind and cand_kind and base_kind != cand_kind:
        need_recovery = True
    if not need_recovery:
        return "", meta
    recovered = _compose_java_decl_with_base_header(base_code, body)
    if not recovered or not _java_delimiters_balanced(recovered):
        return "", meta
    meta["java_decl_recovery_used"] = True
    meta["java_decl_recovery_mode"] = "reuse_base_declaration"
    return recovered, meta


def _postprocess_java_candidate(
    *,
    candidate: str,
    source_text: str,
    base_code: str,
    expected_name: str,
) -> Tuple[str, Dict[str, Any]]:
    meta = _init_extract_meta("java")
    working = str(candidate or "")
    working = _strip_java_output_markers(working)
    working = _sanitize_diff_like(working, language="java")
    working = _strip_leading_language_label(working)
    working = _strip_inline_artifact_comments(working, language="java")
    working = textwrap.dedent(working).strip()

    source = str(source_text or "")
    source = _strip_java_output_markers(source)
    source = _sanitize_diff_like(source, language="java")
    source = _strip_leading_language_label(source)
    source = _strip_inline_artifact_comments(source, language="java")
    source = textwrap.dedent(source).strip()

    selected_source = working
    selected = _select_java_decl_block(working, expected_name=expected_name)
    if selected is None and source and source != working:
        selected = _select_java_decl_block(source, expected_name=expected_name)
        if selected is not None:
            selected_source = source
            meta["java_decl_salvage_used"] = True

    if selected is not None:
        selected_block = str(selected.get("block") or "").strip()
        selected_start = int(selected.get("start") or 0)
        selected_end = int(selected.get("end") or 0)
        source_for_selected = selected_source
        prefix = source_for_selected[:selected_start].strip() if source_for_selected else ""
        suffix = source_for_selected[selected_end:].strip() if source_for_selected and selected_end <= len(source_for_selected) else ""
        if prefix or suffix:
            meta["java_explanation_tail_stripped"] = True
        if selected_block and selected_block != working:
            meta["java_decl_salvage_used"] = True
        working = selected_block

    recovered, recovery_meta = _maybe_recover_java_decl_wrapper(
        base_code=base_code,
        candidate_source=working,
        expected_name=expected_name,
    )
    if recovered:
        working = recovered
        meta = _merge_extract_meta(meta, recovery_meta)

    return working.strip(), meta

def _fallback_extract_code_block(text: str, language: str) -> str:
    source = (text or "").replace("\x00", "")
    source = _sanitize_diff_like(source, language=language)
    source = _strip_leading_language_label(source)
    if not source.strip():
        return ""
    if language == "java":
        pat = JAVA_DECL_HEAD_RE
    else:
        pat = DEFCLASS_HEAD_RE
    starts = list(pat.finditer(source))
    if not starts:
        return ""

    best = ""
    for i, m in enumerate(starts):
        sidx = m.start()
        if i + 1 < len(starts):
            eidx = starts[i + 1].start()
            cand = source[sidx:eidx].strip()
        else:
            cand = source[sidx:].strip()
        cand = _strip_leading_language_label(cand)
        if _contains_model_artifact(cand):
            continue
        cand = _strip_inline_artifact_comments(cand, language=language)
        if len(cand) > len(best):
            best = cand
    return best


def extract_correct_block_with_reason(
    text: str,
    language: str = "python",
    min_chars: int = 30,
    *,
    base_code: str = "",
    expected_name: str = "",
    repair_branch: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    extract_meta = _init_extract_meta(language)
    if not text:
        return "", "extraction_failed_empty_after_trim", extract_meta
    artifact_reason = ""
    try:
        m = re.search(r"^[ \t]*##correct[ \t]*$", text, flags=re.IGNORECASE | re.MULTILINE)
        sub = text[m.end():].strip() if m else text.strip()
        candidate = ""
        # Phase1-E: java_v2 body-only 추출 우선 시도
        java_body_extracted = False
        if language == "java" and str(repair_branch) == "java_v2":
            body_sentinel = _extract_java_body_sentinel_block(sub)
            if body_sentinel:
                # body-only: base_code의 헤더와 결합하여 전체 메서드 재구성
                header = _extract_java_decl_header(base_code)
                if header:
                    candidate = f"{header} {{\n{body_sentinel}\n}}"
                    extract_meta["java_sentinel_extract_hit"] = True
                    extract_meta["java_v2_body_mode"] = True
                    java_body_extracted = True
        if language == "java" and not candidate:
            sentinel = _extract_java_sentinel_block(sub)
            if sentinel:
                candidate = sentinel.strip()
                extract_meta["java_sentinel_extract_hit"] = True
        if not candidate:
            m2 = CODE_FENCE_RE.search(sub)
            candidate = (m2.group(1) if m2 else sub).strip()
        candidate = re.split(r"^###\s", candidate, flags=re.MULTILINE)[0].strip()
        candidate = candidate.replace("\x00", "").replace("```", "")
        if _contains_model_artifact(candidate):
            artifact_reason = "extraction_failed_model_artifact"
            candidate = ""
        elif language == "java":
            candidate, java_meta = _postprocess_java_candidate(
                candidate=candidate,
                source_text=sub,
                base_code=base_code,
                expected_name=expected_name,
            )
            extract_meta = _merge_extract_meta(extract_meta, java_meta)
        else:
            candidate = _sanitize_diff_like(candidate, language=language)
            candidate = _strip_leading_language_label(candidate)
            m3 = DEFCLASS_HEAD_RE.search(candidate)
            if m3:
                candidate = candidate[m3.start():].strip()
            candidate = _strip_inline_artifact_comments(candidate, language=language)
        if candidate:
            try:
                candidate = textwrap.dedent(candidate).strip()
            except IndentationError:
                candidate = candidate.strip()
    except Exception:
        candidate = ""
        # fallback still attempted below

    def _validate(code_text: str) -> Tuple[str, str]:
        c = (code_text or "").strip()
        if not c:
            return "", "extraction_failed_empty_after_trim"
        if len(c) < int(min_chars):
            return "", "extraction_failed_too_short"
        if not _code_like(c, language):
            return "", "extraction_failed_no_code_like"
        return c, ""

    code, reason = _validate(candidate)
    if code:
        return code, "", extract_meta

    fallback = _fallback_extract_code_block(sub if "sub" in locals() else text, language=language)
    if language == "java" and fallback:
        fallback, fallback_meta = _postprocess_java_candidate(
            candidate=fallback,
            source_text=sub if "sub" in locals() else text,
            base_code=base_code,
            expected_name=expected_name,
        )
        extract_meta = _merge_extract_meta(extract_meta, fallback_meta)
    code2, reason2 = _validate(fallback)
    if code2:
        return code2, "", extract_meta
    if artifact_reason:
        return "", artifact_reason, extract_meta
    if reason2:
        return "", reason2, extract_meta
    if reason:
        return "", reason, extract_meta
    return "", "extraction_failed_postprocess", extract_meta


def extract_correct_block(text: str, language: str = "python") -> str:
    code, _reason, _meta = extract_correct_block_with_reason(text, language=language)
    return code

def ast_ok(py_text: str, language: str = "python") -> Tuple[bool, Optional[str]]:
    if language != "python":
        # No built-in Java parser in this pipeline.
        return True, None
    try:
        ast.parse(py_text)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def canon(code: str) -> str:
    s = re.sub(r"#.*", "", code)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_edit_summary(base_code: str, code: str) -> Dict[str, Any]:
    base_lines = (base_code or "").splitlines()
    new_lines = (code or "").splitlines()
    matcher = difflib.SequenceMatcher(None, base_lines, new_lines)
    edits: List[Dict[str, Any]] = []
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":
            continue
        start = int(alo + 1)
        end = int(ahi)
        if tag == "insert":
            end = int(start)
        edits.append(
            {
                "op": tag,
                "start": start,
                "end": end,
                "new_text": "\n".join(new_lines[blo:bhi]).strip(),
            }
        )
    return {"edits": edits, "num_edits": len(edits)}


def build_patch_signature(base_code: str, code: str) -> Tuple[str, Dict[str, Any]]:
    edit_summary = compute_edit_summary(base_code, code)
    edits = edit_summary.get("edits") if isinstance(edit_summary, dict) else []
    if edits:
        compact = []
        for e in edits:
            new_txt = str(e.get("new_text", ""))
            compact.append(
                (
                    str(e.get("op", "")),
                    int(e.get("start", 0)),
                    int(e.get("end", 0)),
                    hashlib.sha1(new_txt.encode("utf-8")).hexdigest(),
                )
            )
        payload = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        sig = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return sig, edit_summary
    fallback = hashlib.sha1(canon(code).encode("utf-8")).hexdigest()
    return fallback, {"edits": [], "num_edits": 0}


def make_drop_counter() -> Counter:
    return Counter({k: 0 for k in DROP_REASON_KEYS})


def make_plan_violation_counter() -> Counter:
    return Counter({k: 0 for k in PLAN_VIOLATION_TYPE_KEYS})


def norm_drop_reason(reason: str) -> str:
    key = (reason or "").strip()
    return key if key in DROP_REASON_KEYS else "other"


def bump_drop(counter: Counter, reason: str) -> None:
    counter[norm_drop_reason(reason)] += 1


def merge_counter_dicts(dst: Dict[str, int], src: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, int]:
    out = {k: int(dst.get(k, 0)) for k in keys}
    if not isinstance(src, dict):
        return out
    for key in keys:
        out[key] = int(out.get(key, 0)) + int(src.get(key, 0) or 0)
    return out


def make_java_validity_reason_counter() -> Counter:
    return Counter({k: 0 for k in JAVA_VALIDITY_REASON_KEYS})


def _argv_has_flag(args: Optional[Namespace], *flags: str) -> bool:
    argv = list(getattr(args, "_argv", sys.argv) or sys.argv)
    for token in argv:
        token_s = str(token or "")
        for flag in flags:
            if token_s == flag or token_s.startswith(f"{flag}="):
                return True
    return False


def normalize_repair_branch(repair_branch: str) -> str:
    value = str(repair_branch or "auto").strip().lower()
    if value in REPAIR_BRANCH_CHOICES:
        return value
    return "auto"


def language_branch_for_language(language: str) -> str:
    return "java_branch" if str(language or "").strip().lower() == "java" else "python_branch"


def normalize_java_survivor_backoff(mode: str) -> str:
    value = str(mode or "off").strip().lower()
    if value in JAVA_SURVIVOR_BACKOFF_CHOICES:
        return value
    return "off"


def resolve_repair_branch_metadata(
    args: Optional[Namespace],
    row: Optional[Dict[str, Any]],
    prom_row: Optional[Dict[str, Any]],
    language: str,
) -> Dict[str, str]:
    requested = normalize_repair_branch(getattr(args, "repair_branch", "auto"))
    lang = str(language or "").strip().lower()
    cli_explicit = _argv_has_flag(args, "--repair_branch")

    if lang == "java" and cli_explicit and requested in JAVA_REPAIR_BRANCHES:
        return {"requested": str(requested), "effective": str(requested), "source": "cli"}

    valid_branches = JAVA_REPAIR_BRANCHES if lang == "java" else {"python_base"}
    for source_name, source in (("row", row or {}), ("prom_row", prom_row or {})):
        candidate = normalize_repair_branch(source.get("repair_branch"))
        if candidate in valid_branches:
            return {"requested": str(requested), "effective": str(candidate), "source": source_name}

    if requested in valid_branches:
        return {
            "requested": str(requested),
            "effective": str(requested),
            "source": "cli" if cli_explicit else "auto",
        }

    default_branch = "java_base" if lang == "java" else "python_base"
    return {"requested": str(requested), "effective": str(default_branch), "source": "auto"}


def resolve_effective_repair_branch(
    args: Optional[Namespace],
    row: Optional[Dict[str, Any]],
    prom_row: Optional[Dict[str, Any]],
    language: str,
) -> str:
    metadata = resolve_repair_branch_metadata(args, row, prom_row, language)
    return str(metadata.get("effective") or "python_base")


def resolve_effective_plan_enforcement(args: Namespace, repair_branch: str) -> str:
    requested = str(getattr(args, "plan_enforcement", "filter_then_fill") or "filter_then_fill").strip().lower()
    if requested not in {"off", "penalty", "filter", "filter_then_fill"}:
        requested = "filter_then_fill"
    if repair_branch in JAVA_REPAIR_BRANCHES and not _argv_has_flag(args, "--plan_enforcement"):
        return "filter"
    return requested


def resolve_effective_java_survivor_backoff(
    args: Namespace,
    repair_branch: str,
    plan_enforcement: Optional[str] = None,
) -> str:
    if repair_branch not in JAVA_REPAIR_BRANCHES:
        return "off"
    requested = normalize_java_survivor_backoff(getattr(args, "java_survivor_backoff", "off"))
    if _argv_has_flag(args, "--java_survivor_backoff"):
        return requested
    effective_plan = str(plan_enforcement or resolve_effective_plan_enforcement(args, repair_branch) or "filter").strip().lower()
    if _argv_has_flag(args, "--plan_enforcement"):
        if effective_plan == "filter_then_fill":
            return "fill"
        return "off"
    return "hybrid"


def java_semantic_rerank_enabled(repair_branch: str, language: str) -> bool:
    return str(language or "").strip().lower() == "java" and repair_branch == "java_semantic"


def java_compile_feedback_requested(args: Namespace, language: str, repair_branch: str) -> bool:
    return bool(
        str(language or "").strip().lower() == "java"
        and repair_branch in JAVA_REPAIR_BRANCHES
        and getattr(args, "java_compile_feedback_once", False)
    )


def resolve_java_compile_feedback_available(args: Optional[Namespace], language: str) -> bool:
    dataset_tag = str(getattr(args, "_dataset_tag_current", "") or "").strip().lower()
    if str(language or "").strip().lower() != "java":
        return False
    if dataset_tag != "defects4j":
        return False
    return True  # Phase2-A: defects4j Java에 대해 compile feedback 활성화


def resolve_java_compile_validation_mode(args: Namespace) -> str:
    if not (
        bool(getattr(args, "java_filter_compile_valid", False))
        or bool(getattr(args, "java_rerank_compile_valid", False))
    ):
        return "none"
    return "proxy"


def init_java_validity_stats(args: Namespace, language: str) -> Dict[str, Any]:
    mode = resolve_java_compile_validation_mode(args) if language == "java" else "none"
    branch_meta = resolve_repair_branch_metadata(args, None, None, language)
    requested_branch = str(branch_meta.get("effective") or resolve_effective_repair_branch(args, None, None, language))
    plan_enforcement = resolve_effective_plan_enforcement(args, requested_branch)
    java_survivor_backoff_mode = resolve_effective_java_survivor_backoff(args, requested_branch, plan_enforcement)
    compile_feedback_once = java_compile_feedback_requested(args, language, requested_branch)
    return {
        "java_filter_compile_valid": bool(language == "java" and getattr(args, "java_filter_compile_valid", False)),
        "java_rerank_compile_valid": bool(language == "java" and getattr(args, "java_rerank_compile_valid", False)),
        "java_candidate_compile_check": bool(getattr(args, "java_candidate_compile_check", False)),
        "java_compile_validation_mode": str(mode),
        "java_semantic_rerank_enabled": bool(java_semantic_rerank_enabled(requested_branch, language)),
        "java_semantic_rerank_features": list(JAVA_SEMANTIC_RERANK_FEATURES),
        "java_survivor_backoff_mode": str(java_survivor_backoff_mode),
        "java_branch_no_fill": bool(
            language == "java"
            and plan_enforcement not in {"filter_then_fill"}
            and java_survivor_backoff_mode == "off"
        ),
        "java_compile_feedback_once": bool(compile_feedback_once),
        "java_compile_feedback_available": bool(resolve_java_compile_feedback_available(args, language)),
        "java_compile_feedback_attempted_n": 0,
        "java_compile_feedback_applied_n": 0,
        "java_validity_checked_n": 0,
        "java_validity_valid_n": 0,
        "java_validity_invalid_n": 0,
        "java_validity_rejected_n": 0,
        "java_validity_reranked_n": 0,
        "java_survivor_backoff_used": False,
        "java_survivor_backoff_added_count": 0,
        "zero_candidate_after_filter": False,
        "java_validity_reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
    }


def update_java_validity_stats(
    stats: Dict[str, Any],
    validation_record: Dict[str, Any],
    *,
    rejected: bool = False,
    reranked: bool = False,
) -> None:
    if not isinstance(stats, dict) or not validation_record.get("checked"):
        return
    stats["java_validity_checked_n"] = int(stats.get("java_validity_checked_n", 0)) + 1
    if validation_record.get("valid") is True:
        stats["java_validity_valid_n"] = int(stats.get("java_validity_valid_n", 0)) + 1
    else:
        stats["java_validity_invalid_n"] = int(stats.get("java_validity_invalid_n", 0)) + 1
    stats["java_validity_reason_counts"] = merge_counter_dicts(
        stats.get("java_validity_reason_counts") or {},
        validation_record.get("reason_counts") or {},
        JAVA_VALIDITY_REASON_KEYS,
    )
    if rejected:
        stats["java_validity_rejected_n"] = int(stats.get("java_validity_rejected_n", 0)) + 1
    if reranked:
        stats["java_validity_reranked_n"] = int(stats.get("java_validity_reranked_n", 0)) + 1


def merge_java_validity_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    if not isinstance(dst, dict):
        return
    src = src or {}
    dst["java_filter_compile_valid"] = bool(dst.get("java_filter_compile_valid", False) or src.get("java_filter_compile_valid", False))
    dst["java_rerank_compile_valid"] = bool(dst.get("java_rerank_compile_valid", False) or src.get("java_rerank_compile_valid", False))
    dst["java_candidate_compile_check"] = bool(dst.get("java_candidate_compile_check", False) or src.get("java_candidate_compile_check", False))
    dst["java_semantic_rerank_enabled"] = bool(dst.get("java_semantic_rerank_enabled", False) or src.get("java_semantic_rerank_enabled", False))
    dst["java_semantic_rerank_features"] = list(
        dst.get("java_semantic_rerank_features")
        or src.get("java_semantic_rerank_features")
        or list(JAVA_SEMANTIC_RERANK_FEATURES)
    )
    dst["java_branch_no_fill"] = bool(dst.get("java_branch_no_fill", False) or src.get("java_branch_no_fill", False))
    dst["java_compile_feedback_once"] = bool(dst.get("java_compile_feedback_once", False) or src.get("java_compile_feedback_once", False))
    dst["java_compile_feedback_available"] = bool(dst.get("java_compile_feedback_available", False) or src.get("java_compile_feedback_available", False))
    dst["java_compile_validation_mode"] = str(
        src.get("java_compile_validation_mode")
        or dst.get("java_compile_validation_mode")
        or "none"
    )
    for key in (
        "java_validity_checked_n",
        "java_validity_valid_n",
        "java_validity_invalid_n",
        "java_validity_rejected_n",
        "java_validity_reranked_n",
        "java_compile_feedback_attempted_n",
        "java_compile_feedback_applied_n",
    ):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0) or 0)
    dst["java_validity_reason_counts"] = merge_counter_dicts(
        dst.get("java_validity_reason_counts") or {},
        src.get("java_validity_reason_counts") or {},
        JAVA_VALIDITY_REASON_KEYS,
    )


def _strip_java_strings_and_comments(text: str) -> str:
    stripped = JAVA_STRING_OR_CHAR_RE.sub('""', text or "")
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"//.*?$", "", stripped, flags=re.MULTILINE)
    return stripped


def _java_delimiters_balanced(text: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: List[str] = []
    for ch in _strip_java_strings_and_comments(text):
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _extract_java_decl_header(code: str) -> str:
    src = str(code or "")
    m = JAVA_DECL_HEAD_RE.search(src)
    if not m:
        return ""
    decl = src[m.start():]
    depth_paren = 0
    depth_angle = 0
    depth_bracket = 0
    for idx, ch in enumerate(decl):
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle = max(0, depth_angle - 1)
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == '{' and depth_paren == 0 and depth_angle == 0 and depth_bracket == 0:
            return decl[:idx].strip()
    return decl.strip()


def _split_java_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth_paren = 0
    depth_angle = 0
    depth_bracket = 0
    for ch in text or "":
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle = max(0, depth_angle - 1)
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket = max(0, depth_bracket - 1)
        if ch == ',' and depth_paren == 0 and depth_angle == 0 and depth_bracket == 0:
            token = "".join(cur).strip()
            if token:
                parts.append(token)
            cur = []
            continue
        cur.append(ch)
    token = "".join(cur).strip()
    if token:
        parts.append(token)
    return parts


def _extract_java_param_list(header: str) -> str:
    start = header.find('(')
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(header)):
        ch = header[idx]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return header[start + 1:idx]
    return ""


def _count_java_params(header: str) -> Optional[int]:
    params = _extract_java_param_list(header)
    if params == "":
        return 0 if "(" in header and ")" in header else None
    return len(_split_java_top_level_commas(params))


def _normalize_java_throws_clause(header: str) -> str:
    m = re.search(r"\bthrows\b\s+([^{}]+)", header or "")
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1).strip())


def _normalize_java_header(header: str) -> str:
    return re.sub(r"\s+", " ", str(header or "").strip())


def _extract_java_decl_contract(code: str) -> Dict[str, Any]:
    header = _extract_java_decl_header(code)
    name = extract_decl_name(code, language="java")
    kind = "unknown"
    for decl_kind in ("class", "interface", "enum", "record"):
        if re.search(rf"\b{decl_kind}\b", header):
            kind = decl_kind
            break
    if kind == "unknown" and '(' in header and ')' in header:
        prefix = header.split('(', 1)[0].strip()
        tokens = prefix.split()
        if tokens and name and tokens[-1] == name:
            if len(tokens) == 1:
                kind = "constructor"
            elif len(tokens) >= 2 and tokens[-2] in {
                "public", "protected", "private", "static", "final", "abstract",
                "synchronized", "native", "strictfp", "default",
            }:
                kind = "constructor"
            else:
                kind = "method"
        else:
            kind = "method"
    return {
        "header": header,
        "name": name,
        "kind": kind,
        "param_count": _count_java_params(header) if kind in {"method", "constructor"} else None,
        "has_throws": bool(re.search(r"\bthrows\b", header)),
        "throws_clause": _normalize_java_throws_clause(header),
    }


def _collect_java_identifiers(text: str) -> set:
    return {
        tok for tok in re.findall(r"\b[A-Za-z_]\w*\b", text or "")
        if tok not in JAVA_KEYWORDS
    }


def _count_java_decl_like(code: str) -> int:
    return len(list(JAVA_DECL_HEAD_RE.finditer(code or "")))


def _java_text_chunks_for_diff(base_code: str, code: str) -> Tuple[str, str]:
    base_lines = (base_code or "").splitlines()
    new_lines = (code or "").splitlines()
    matcher = difflib.SequenceMatcher(None, base_lines, new_lines)
    base_chunks: List[str] = []
    new_chunks: List[str] = []
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":
            continue
        base_text = "\n".join(base_lines[alo:ahi]).strip()
        new_text = "\n".join(new_lines[blo:bhi]).strip()
        if base_text:
            base_chunks.append(base_text)
        if new_text:
            new_chunks.append(new_text)
    return "\n".join(base_chunks), "\n".join(new_chunks)


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_java_symbol_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\s*this\.", "", text)
    text = re.sub(r"\(\s*\)$", "", text)
    text = text.strip()
    if "." in text and not text.startswith("@"):
        text = text.split(".")[-1].strip()
    return text


def _normalize_java_symbol_items(values: Any) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for raw in _as_string_list(values):
        norm = _normalize_java_symbol_name(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _collect_java_prompt_context_text(prom_row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("prompt", "plan_prompt", "input", "instruction", "text", "user_prompt"):
        val = prom_row.get(key)
        if isinstance(val, str) and val.strip():
            chunks.append(val)
    messages = prom_row.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content)
    return "\n".join(chunks)


def _extract_java_context_labeled_items(text: str, prefixes: Tuple[str, ...]) -> List[str]:
    items: List[str] = []
    for line in str(text or "").splitlines():
        clean = re.sub(r"^\s*-\s*", "", line).strip()
        if not clean:
            continue
        lower = clean.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                raw = clean[len(prefix):].strip(" :")
                parts = _split_java_top_level_commas(raw)
                for part in parts:
                    norm = _normalize_java_symbol_name(part)
                    if norm:
                        items.append(norm)
                break
    return _normalize_java_symbol_items(items)


def _extract_java_signature_param_names(text: str) -> set:
    header = str(text or "").strip()
    if not header:
        return set()
    if "(" not in header or ")" not in header:
        header = _extract_java_decl_header(header)
    params = _extract_java_param_list(header)
    if not params:
        return set()
    names: set = set()
    for token in _split_java_top_level_commas(params):
        param_name = re.findall(r"\b([a-z_][A-Za-z0-9_$]*)\b", token)
        if param_name:
            names.add(param_name[-1])
    return {name for name in names if name not in JAVA_KEYWORDS}


def _extract_java_call_names(text: str) -> set:
    stripped = _strip_java_strings_and_comments(text)
    calls = set(re.findall(r"(?<![\w$.])([a-z_][A-Za-z0-9_$]*)\s*\(", stripped))
    calls.update(re.findall(r"\bthis\.([a-z_][A-Za-z0-9_$]*)\s*\(", stripped))
    return {name for name in calls if name not in JAVA_KEYWORDS}


def _extract_java_upper_types(text: str) -> set:
    stripped = _strip_java_strings_and_comments(text)
    return set(re.findall(r"\b([A-Z][A-Za-z0-9_$]*)\b", stripped))


def _extract_java_constant_names(text: str) -> set:
    stripped = _strip_java_strings_and_comments(text)
    return set(re.findall(r"\b([A-Z][A-Z0-9_]{1,})\b", stripped))


def _extract_java_lower_identifiers(text: str) -> set:
    stripped = _strip_java_strings_and_comments(text)
    return {
        tok
        for tok in re.findall(r"\b([a-z_][A-Za-z0-9_$]*)\b", stripped)
        if tok not in JAVA_KEYWORDS and len(tok) >= 2
    }


def _extract_java_declared_names(text: str) -> set:
    stripped = _strip_java_strings_and_comments(text)
    declared: set = set()
    header = _extract_java_decl_header(stripped)
    if header:
        params = _extract_java_param_list(header)
        for part in _split_java_top_level_commas(params):
            token = str(part or "").strip()
            if not token:
                continue
            param_name = re.findall(r"\b([a-z_][A-Za-z0-9_$]*)\b", token)
            if param_name:
                declared.add(param_name[-1])
    type_pattern = (
        r"(?:"
        r"[A-Z][\w$<>\[\],.?&\s]*"
        r"|String|Object|Class|Exception|RuntimeException|Throwable"
        r"|boolean|byte|short|int|long|float|double|char"
        r")"
    )
    patterns = [
        rf"\b(?:final\s+)?{type_pattern}\s+([a-z_][A-Za-z0-9_$]*)\s*(?=[=;,\)])",
        rf"\bfor\s*\(\s*(?:final\s+)?{type_pattern}\s+([a-z_][A-Za-z0-9_$]*)\s*:",
        rf"\bfor\s*\(\s*(?:final\s+)?{type_pattern}\s+([a-z_][A-Za-z0-9_$]*)\s*=",
        r"\bcatch\s*\(\s*[A-Z][\w<>\[\],.?&\s|]*\s+([a-z_][A-Za-z0-9_$]*)\s*\)",
    ]
    for pattern in patterns:
        declared.update(re.findall(pattern, stripped))
    for params in re.findall(r"\(\s*([a-z_][A-Za-z0-9_$]*(?:\s*,\s*[a-z_][A-Za-z0-9_$]*)*)\s*\)\s*->", stripped):
        for name in re.findall(r"\b([a-z_][A-Za-z0-9_$]*)\b", params):
            declared.add(name)
    declared.update(re.findall(r"\b([a-z_][A-Za-z0-9_$]*)\s*->", stripped))
    return {name for name in declared if name not in JAVA_KEYWORDS}


def _extract_java_changed_identifiers(base_code: str, code: str) -> Dict[str, Any]:
    base_changed, code_changed = _java_text_chunks_for_diff(base_code, code)
    declared_names = _extract_java_declared_names(code_changed)
    return {
        "base_changed_text": base_changed,
        "code_changed_text": code_changed,
        "base_call_names": _extract_java_call_names(base_changed),
        "code_call_names": _extract_java_call_names(code_changed),
        "base_type_names": _extract_java_upper_types(base_changed),
        "code_type_names": _extract_java_upper_types(code_changed),
        "base_constant_names": _extract_java_constant_names(base_changed),
        "code_constant_names": _extract_java_constant_names(code_changed),
        "base_identifier_names": _extract_java_lower_identifiers(base_changed),
        "code_identifier_names": _extract_java_lower_identifiers(code_changed),
        "declared_names": declared_names,
    }


def _update_java_symbol_sources(
    target: Dict[str, set],
    values: Any,
    source: str,
) -> None:
    for item in _normalize_java_symbol_items(values):
        target.setdefault(item, set()).add(str(source))


def _merge_java_symbol_sources(
    target: Dict[str, set],
    *source_maps: Dict[str, set],
) -> None:
    for source_map in source_maps:
        for key, values in (source_map or {}).items():
            target.setdefault(str(key), set()).update(str(v) for v in list(values or []) if str(v))


def _freeze_java_symbol_sources(source_map: Dict[str, set]) -> Dict[str, List[str]]:
    return {str(key): sorted(str(v) for v in values if str(v)) for key, values in (source_map or {}).items() if values}


def build_java_context_allowlist(base_code: str, prom_row: Dict[str, Any], *, rescue_mode: bool = False) -> Dict[str, Any]:
    summary_text = str(prom_row.get("java_context_summary") or "")
    signature_text = str(prom_row.get("java_method_signature") or "")
    prompt_text = _collect_java_prompt_context_text(prom_row)

    summary_fields = _extract_java_context_labeled_items(summary_text, ("referenced instance fields",))
    summary_calls = _extract_java_context_labeled_items(summary_text, ("referenced helper methods in this method",))
    summary_types = _extract_java_context_labeled_items(
        summary_text,
        ("types referenced in this method", "referenced types in this method"),
    )
    prompt_fields = _extract_java_context_labeled_items(prompt_text, ("referenced instance fields",))
    prompt_calls = _extract_java_context_labeled_items(prompt_text, ("referenced helper methods in this method",))
    prompt_types = _extract_java_context_labeled_items(
        prompt_text,
        ("types referenced in this method", "referenced types in this method"),
    )

    method_call_sources: Dict[str, set] = {}
    type_sources: Dict[str, set] = {}
    field_sources: Dict[str, set] = {}
    identifier_sources: Dict[str, set] = {}
    constant_sources: Dict[str, set] = {}

    _update_java_symbol_sources(method_call_sources, _extract_java_call_names(base_code), "helper_calls")
    _update_java_symbol_sources(method_call_sources, prom_row.get("java_helper_calls"), "helper_calls")
    _update_java_symbol_sources(method_call_sources, summary_calls, "context_summary")
    if rescue_mode:
        _update_java_symbol_sources(method_call_sources, prompt_calls, "helper_calls")

    _update_java_symbol_sources(type_sources, _extract_java_upper_types(base_code), "types")
    _update_java_symbol_sources(type_sources, prom_row.get("java_types"), "types")
    _update_java_symbol_sources(type_sources, _extract_java_upper_types(signature_text), "method_signature")
    _update_java_symbol_sources(type_sources, summary_types, "context_summary")
    if rescue_mode:
        _update_java_symbol_sources(type_sources, prompt_types, "types")

    _update_java_symbol_sources(
        field_sources,
        re.findall(r"\bthis\.([A-Za-z_$][A-Za-z0-9_$]*)\b", base_code or ""),
        "instance_fields",
    )
    _update_java_symbol_sources(field_sources, prom_row.get("java_instance_fields"), "instance_fields")
    _update_java_symbol_sources(field_sources, summary_fields, "context_summary")
    if rescue_mode:
        _update_java_symbol_sources(field_sources, prompt_fields, "instance_fields")

    signature_param_names = _normalize_java_symbol_items(_extract_java_signature_param_names(signature_text))
    _update_java_symbol_sources(identifier_sources, signature_param_names, "method_signature")
    _update_java_symbol_sources(identifier_sources, _extract_java_lower_identifiers(summary_text), "context_summary")

    allowed_method_calls = set(method_call_sources.keys())

    allowed_types = set(type_sources.keys())
    allowed_types.update(JAVA_COMMON_TYPE_NAMES)

    allowed_fields = set(field_sources.keys())

    allowed_identifiers = set(_normalize_java_symbol_items(_extract_java_lower_identifiers(base_code)))
    allowed_identifiers.update(_normalize_java_symbol_items(_extract_java_declared_names(base_code)))
    allowed_identifiers.update(_normalize_java_symbol_items(_extract_java_signature_param_names(base_code)))
    allowed_identifiers.update(signature_param_names)
    allowed_identifiers.update(_normalize_java_symbol_items(_extract_java_lower_identifiers(summary_text)))
    allowed_identifiers.update(_normalize_java_symbol_items(_extract_java_lower_identifiers(signature_text)))

    allowed_constants = set(_normalize_java_symbol_items(_extract_java_constant_names(base_code)))
    allowed_constants.update(_normalize_java_symbol_items(_extract_java_constant_names(summary_text)))
    _update_java_symbol_sources(constant_sources, _extract_java_constant_names(summary_text), "context_summary")
    if rescue_mode:
        _update_java_symbol_sources(constant_sources, _extract_java_constant_names(prompt_text), "context_summary")
        allowed_constants.update(_normalize_java_symbol_items(_extract_java_constant_names(prompt_text)))

    allowed_identifiers.update(allowed_method_calls)
    allowed_identifiers.update(allowed_fields)
    allowed_identifiers.update(allowed_types)
    allowed_identifiers.update(allowed_constants)

    _merge_java_symbol_sources(identifier_sources, method_call_sources, type_sources, field_sources, constant_sources)

    return {
        "allowed_method_calls": allowed_method_calls,
        "allowed_types": allowed_types,
        "allowed_fields": allowed_fields,
        "allowed_identifiers": allowed_identifiers,
        "allowed_constants": allowed_constants,
        "method_call_sources": _freeze_java_symbol_sources(method_call_sources),
        "type_sources": _freeze_java_symbol_sources(type_sources),
        "field_sources": _freeze_java_symbol_sources(field_sources),
        "identifier_sources": _freeze_java_symbol_sources(identifier_sources),
        "constant_sources": _freeze_java_symbol_sources(constant_sources),
        "summary": {
            "allowed_method_calls_count": int(len(allowed_method_calls)),
            "allowed_method_calls_sample": sorted(allowed_method_calls)[:12],
            "allowed_types_count": int(len(allowed_types)),
            "allowed_types_sample": sorted(allowed_types)[:12],
            "allowed_fields_count": int(len(allowed_fields)),
            "allowed_fields_sample": sorted(allowed_fields)[:12],
            "allowed_identifiers_count": int(len(allowed_identifiers)),
            "allowed_identifiers_sample": sorted(allowed_identifiers)[:16],
            "rescue_mode": bool(rescue_mode),
        },
    }


def _collect_java_support_sources(
    names: set,
    *source_maps: Dict[str, Any],
) -> Dict[str, List[str]]:
    support: Dict[str, List[str]] = {}
    for name in sorted(str(item) for item in names if str(item)):
        merged: set = set()
        for source_map in source_maps:
            merged.update(str(v) for v in list((source_map or {}).get(name) or []) if str(v))
        if merged:
            support[name] = sorted(merged)
    return support


def analyze_java_context_drift(base_code: str, code: str, prom_row: Dict[str, Any], *, rescue_mode: bool = False) -> Dict[str, Any]:
    changed = _extract_java_changed_identifiers(base_code, code)
    expected_name = str(
        prom_row.get("function_name")
        or safe_get(prom_row, ["function", "function_name"])
        or extract_decl_name(base_code, language="java")
        or ""
    ).strip()

    allowlist = build_java_context_allowlist(base_code, prom_row or {}, rescue_mode=rescue_mode)
    allowed_helper_calls = set(allowlist.get("allowed_method_calls") or set())
    allowed_types = set(allowlist.get("allowed_types") or set())
    allowed_fields = set(allowlist.get("allowed_fields") or set())
    allowed_identifiers = set(allowlist.get("allowed_identifiers") or set())
    allowed_constants = set(allowlist.get("allowed_constants") or set())
    method_call_sources = dict(allowlist.get("method_call_sources") or {})
    type_sources = dict(allowlist.get("type_sources") or {})
    field_sources = dict(allowlist.get("field_sources") or {})
    identifier_sources = dict(allowlist.get("identifier_sources") or {})
    constant_sources = dict(allowlist.get("constant_sources") or {})
    if expected_name:
        allowed_identifiers.add(expected_name)

    declared_names = {_normalize_java_symbol_name(name) for name in (changed.get("declared_names") or set()) if _normalize_java_symbol_name(name)}
    changed_call_names = {_normalize_java_symbol_name(name) for name in (changed.get("code_call_names") or set()) if _normalize_java_symbol_name(name)}
    base_call_names = {_normalize_java_symbol_name(name) for name in (changed.get("base_call_names") or set()) if _normalize_java_symbol_name(name)}
    changed_type_names = {_normalize_java_symbol_name(name) for name in (changed.get("code_type_names") or set()) if _normalize_java_symbol_name(name)}
    base_type_names = {_normalize_java_symbol_name(name) for name in (changed.get("base_type_names") or set()) if _normalize_java_symbol_name(name)}
    changed_constant_names = {_normalize_java_symbol_name(name) for name in (changed.get("code_constant_names") or set()) if _normalize_java_symbol_name(name)}
    base_constant_names = {_normalize_java_symbol_name(name) for name in (changed.get("base_constant_names") or set()) if _normalize_java_symbol_name(name)}
    changed_identifier_names = {_normalize_java_symbol_name(name) for name in (changed.get("code_identifier_names") or set()) if _normalize_java_symbol_name(name)}
    base_identifier_names = {_normalize_java_symbol_name(name) for name in (changed.get("base_identifier_names") or set()) if _normalize_java_symbol_name(name)}

    supported_new_type_names = {
        name
        for name in (changed_type_names - base_type_names)
        if name in allowed_types
    }
    supported_new_constant_names = {
        name
        for name in (changed_constant_names - base_constant_names)
        if name in allowed_constants or name in allowed_types
    }
    supported_new_identifier_names = {
        name
        for name in (changed_identifier_names - base_identifier_names)
        if name in allowed_identifiers or name in allowed_fields
    }
    supported_new_type_sources = _collect_java_support_sources(
        supported_new_type_names,
        type_sources,
        identifier_sources,
    )
    supported_new_constant_sources = _collect_java_support_sources(
        supported_new_constant_names,
        constant_sources,
        type_sources,
        identifier_sources,
    )
    supported_new_identifier_sources = _collect_java_support_sources(
        supported_new_identifier_names,
        identifier_sources,
        field_sources,
        method_call_sources,
        type_sources,
        constant_sources,
    )
    supported_new_types = sorted(supported_new_type_names)
    supported_new_constants = sorted(supported_new_constant_names)
    supported_new_identifiers = sorted(supported_new_identifier_names)
    new_helper_calls = sorted(
        name
        for name in changed_call_names - base_call_names
        if name not in allowed_helper_calls
        and name not in declared_names
        and name not in JAVA_COMMON_CALL_NAMES
        and name != expected_name
    )
    new_type_references = sorted(
        name
        for name in changed_type_names - base_type_names
        if name not in allowed_types
    )
    new_constant_references = sorted(
        name
        for name in changed_constant_names - base_constant_names
        if name not in allowed_constants and name not in allowed_types
    )
    new_identifier_references = sorted(
        name
        for name in changed_identifier_names - base_identifier_names
        if name not in allowed_identifiers
        and name not in declared_names
        and name not in new_helper_calls
        and name not in allowed_fields
        and name != expected_name
    )

    base_decl = _extract_java_decl_contract(base_code)
    cand_decl = _extract_java_decl_contract(code)
    api_signature_drift = bool(
        _normalize_java_header(base_decl.get("header", "")) != ""
        and _normalize_java_header(cand_decl.get("header", "")) != ""
        and _normalize_java_header(base_decl.get("header", "")) != _normalize_java_header(cand_decl.get("header", ""))
    )

    summary_parts: List[str] = []
    if new_helper_calls:
        summary_parts.append("new helper calls: " + ", ".join(new_helper_calls[:5]))
    if new_type_references:
        summary_parts.append("new type references: " + ", ".join(new_type_references[:5]))
    if new_constant_references:
        summary_parts.append("new constants: " + ", ".join(new_constant_references[:5]))
    if new_identifier_references:
        summary_parts.append("new identifiers: " + ", ".join(new_identifier_references[:5]))
    if api_signature_drift:
        summary_parts.append("api signature drift")

    return {
        "new_helper_calls": new_helper_calls,
        "new_type_references": new_type_references,
        "new_constant_references": new_constant_references,
        "new_identifier_references": new_identifier_references,
        "api_signature_drift": api_signature_drift,
        "context_allowlist_summary": dict(allowlist.get("summary") or {}),
        "declared_names_in_patch": sorted(declared_names),
        "context_supported_new_identifiers": supported_new_identifiers,
        "context_supported_new_types": supported_new_types,
        "context_supported_new_constants": supported_new_constants,
        "context_supported_new_identifier_sources": supported_new_identifier_sources,
        "context_supported_new_type_sources": supported_new_type_sources,
        "context_supported_new_constant_sources": supported_new_constant_sources,
        "summary": "; ".join(summary_parts) if summary_parts else "none",
    }


def _compute_local_bug_line(prom_row: Dict[str, Any], base_code: str) -> Optional[int]:
    buggy_loc = _safe_int(safe_get(prom_row, ["buggy_line_location"]))
    func_start = (
        _safe_int(safe_get(prom_row, ["function", "function_before_start_line"]))
        or _safe_int(safe_get(prom_row, ["function", "function_after_start_line"]))
    )
    n_lines = max(1, len((base_code or "").splitlines()))
    if buggy_loc is None:
        return None
    if func_start is not None:
        local = buggy_loc - func_start + 1
        if 1 <= local <= n_lines:
            return local
    if 1 <= buggy_loc <= n_lines:
        return buggy_loc
    return None


def _mean_edit_distance_to_line(edit_summary: Dict[str, Any], local_bug_line: Optional[int]) -> Optional[float]:
    if local_bug_line is None:
        return None
    distances: List[float] = []
    for edit in list((edit_summary or {}).get("edits") or []):
        if not isinstance(edit, dict):
            continue
        start = _safe_int(edit.get("start"))
        end = _safe_int(edit.get("end"))
        if start is None and end is None:
            continue
        start = local_bug_line if start is None else start
        end = start if end is None else end
        anchor = (float(start) + float(end)) / 2.0
        distances.append(abs(anchor - float(local_bug_line)))
    if not distances:
        return None
    return sum(distances) / float(len(distances))


def maybe_apply_java_compile_feedback_once(
    *,
    args: Namespace,
    language: str,
    repair_branch: str,
    code: str,
) -> Tuple[str, Dict[str, Any]]:
    enabled = java_compile_feedback_requested(args, language, repair_branch)
    available = resolve_java_compile_feedback_available(args, language)
    record = {
        "enabled": bool(enabled),
        "available": bool(available),
        "attempted": 0,
        "applied": False,
        "status": "disabled",
    }
    if not enabled:
        return code, record
    if not available:
        record["status"] = "unavailable"
        return code, record
    record["status"] = "not_implemented"
    return code, record


def safe_local_java_compile_check(
    *,
    base_code: str,
    code: str,
    expected_name: str,
) -> Optional[Dict[str, Any]]:
    """Phase2-A: 구조적 Java 유효성 검사 — javac 없이 가능한 범위의 사전 검증.
    
    검사 항목:
    1. 중괄호/괄호/대괄호 균형
    2. 빈 코드 또는 너무 짧은 코드
    3. Java 선언부가 존재하는지
    4. 기본적인 구조 패턴 (return path, 세미콜론 등)
    """
    if not (code or "").strip():
        return {"ok": False, "reason": "empty_code"}
    
    stripped = (code or "").strip()
    
    # 1. 중괄호 균형 검사
    if not _java_delimiters_balanced(stripped):
        return {"ok": False, "reason": "unbalanced_delimiters"}
    
    # 2. Java 선언부 존재 검사
    if not JAVA_DECL_HEAD_RE.search(stripped):
        return {"ok": False, "reason": "no_java_declaration"}
    
    # 3. 너무 짧은 코드 (base_code 대비)
    base_lines = len((base_code or "").strip().splitlines())
    code_lines = len(stripped.splitlines())
    if base_lines > 5 and code_lines < max(3, base_lines // 4):
        return {"ok": False, "reason": "suspiciously_short"}
    
    # 4. 중복 메서드 선언 (모델이 같은 메서드를 두 번 출력하는 경우)
    decl_count = _count_java_decl_like(stripped)
    base_decl_count = _count_java_decl_like(base_code or "")
    if decl_count > base_decl_count + 2:
        return {"ok": False, "reason": "too_many_declarations"}
    
    # 5. 문자열/주석 외부에서 명백한 비정상 패턴 검사
    stripped_no_strings = _strip_java_strings_and_comments(stripped)
    # 이중 세미콜론
    if ";;" in stripped_no_strings.replace(" ", ""):
        lines_with_double_semi = [
            ln for ln in stripped_no_strings.splitlines()
            if ";;" in ln.replace(" ", "") and "for" not in ln
        ]
        if lines_with_double_semi:
            return {"ok": False, "reason": "double_semicolon"}
    
    # 기본 검사 통과
    return {"ok": True, "reason": ""}


def _compute_java_compile_risk_combo(
    *,
    broad_rewrite_severe: bool,
    broad_rewrite_moderate: bool,
    plan_ok: bool,
    drift: Dict[str, Any],
) -> Dict[str, Any]:
    unsupported_reason_items = {
        "new_helper_call": list(drift.get("new_helper_calls") or []),
        "new_type_reference": list(drift.get("new_type_references") or []),
        "new_constant_reference": list(drift.get("new_constant_references") or []),
        "new_identifier_reference": list(drift.get("new_identifier_references") or []),
    }
    nonempty_categories = [
        reason for reason, items in unsupported_reason_items.items() if items
    ]
    unsupported_total = sum(len(items) for items in unsupported_reason_items.values())
    details: List[str] = []
    promoted_hard_reasons: List[str] = []

    # Phase1-B: 임계값 완화
    if broad_rewrite_severe and len(nonempty_categories) >= 2:  # Phase1-B: >=1 → >=2
        details.append("broad_rewrite_severe+new_refs")
        promoted_hard_reasons.append("broad_rewrite")
        promoted_hard_reasons.extend(nonempty_categories)
    if broad_rewrite_moderate and len(nonempty_categories) >= 3:  # Phase1-B: >=2 → >=3
        details.append("broad_rewrite_moderate+new_ref_categories_ge_3")
        promoted_hard_reasons.append("broad_rewrite")
        promoted_hard_reasons.extend(nonempty_categories)
    if unsupported_total >= 4:  # Phase1-B: >=2 → >=4
        details.append("new_ref_total_ge_4")
        promoted_hard_reasons.extend(nonempty_categories)
    # Phase1-B: type+identifier 콤보는 유지하되 plan_ok이면 면제
    if (
        unsupported_reason_items["new_type_reference"]
        and unsupported_reason_items["new_identifier_reference"]
        and not bool(plan_ok)
    ):
        details.append("type_plus_identifier")
        promoted_hard_reasons.extend(["new_type_reference", "new_identifier_reference"])
    if (
        unsupported_reason_items["new_constant_reference"]
        and unsupported_reason_items["new_identifier_reference"]
        and not bool(plan_ok)
    ):
        details.append("constant_plus_identifier_without_plan_ok")
        promoted_hard_reasons.extend(["new_constant_reference", "new_identifier_reference"])

    return {
        "triggered": bool(details),
        "details": list(dict.fromkeys(details)),
        "promoted_hard_reasons": list(dict.fromkeys(promoted_hard_reasons)),
    }


def assess_java_compile_validity(
    *,
    base_code: str,
    code: str,
    expected_name: str,
    requested_mode: str,
    prom_row: Optional[Dict[str, Any]] = None,
    plan_ok: bool = True,
    filter_enabled: bool = False,
    rescue_mode: bool = False,
    hard_reject_compile_risk_combos: bool = False,
) -> Dict[str, Any]:
    record = {
        "checked": False,
        "mode": "none",
        "valid": None,
        "reasons": [],
        "hard_reasons": [],
        "context_drift_summary": "none",
        "context_allowlist_summary": {},
        "declared_names_in_patch": [],
        "context_supported_new_identifiers": [],
        "context_supported_new_types": [],
        "context_supported_new_constants": [],
        "context_supported_new_identifier_sources": {},
        "context_supported_new_type_sources": {},
        "context_supported_new_constant_sources": {},
        "new_helper_calls": [],
        "new_type_references": [],
        "new_constant_references": [],
        "new_identifier_references": [],
        "broad_rewrite_fraction": 0.0,
        "risk_combo_triggered": False,
        "risk_combo_details": [],
        "reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
    }
    if requested_mode == "none":
        return record

    reasons: List[str] = []
    hard_reasons: List[str] = []
    if _contains_model_artifact(code) or _detect_java_candidate_issue(base_code, code):
        reasons.append("model_artifact")
        hard_reasons.append("model_artifact")
    if not _java_delimiters_balanced(code):
        reasons.append("unbalanced_delimiters")
        hard_reasons.append("unbalanced_delimiters")

    base_decl = _extract_java_decl_contract(base_code)
    cand_decl = _extract_java_decl_contract(code)
    if (
        base_decl.get("kind") not in {"", "unknown"}
        and cand_decl.get("kind") not in {"", "unknown"}
        and base_decl.get("kind") != cand_decl.get("kind")
    ):
        reasons.append("decl_kind_mismatch")
        hard_reasons.append("decl_kind_mismatch")
    if (
        base_decl.get("kind") in {"method", "constructor"}
        and cand_decl.get("kind") in {"method", "constructor"}
        and base_decl.get("param_count") is not None
        and cand_decl.get("param_count") is not None
        and int(base_decl.get("param_count")) != int(cand_decl.get("param_count"))
    ):
        reasons.append("signature_mismatch")
        hard_reasons.append("signature_mismatch")
    if expected_name and cand_decl.get("name") and cand_decl.get("name") != expected_name:
        reasons.append("signature_mismatch")
        hard_reasons.append("signature_mismatch")

    decl_match = JAVA_DECL_HEAD_RE.search(code or "")
    prefix = (code or "")[: decl_match.start()].strip() if decl_match else (code or "").strip()
    if prefix and any(tok in prefix for tok in ("package ", "import ")):
        reasons.append("wrapper_mismatch")
        hard_reasons.append("wrapper_mismatch")

    rewrite_frac = float(change_fraction(base_code, code))
    broad_rewrite_severe = rewrite_frac > 0.60                         # Phase1-C: 0.50→0.60
    broad_rewrite_moderate = rewrite_frac > 0.45 and not bool(plan_ok) # Phase1-C: 0.35→0.45
    if broad_rewrite_severe or broad_rewrite_moderate:
        reasons.append("broad_rewrite")

    drift = analyze_java_context_drift(base_code, code, prom_row or {}, rescue_mode=bool(rescue_mode))
    if drift.get("new_helper_calls"):
        reasons.append("new_helper_call")
        # Phase1-B: new_helper_call을 hard reject에서 제거 — soft penalty만 부여
        # 기존: if filter_enabled and not hard_reject_compile_risk_combos: hard_reasons.append("new_helper_call")
    if drift.get("new_type_references"):
        reasons.append("new_type_reference")
    if drift.get("new_constant_references"):
        reasons.append("new_constant_reference")
    if drift.get("new_identifier_references"):
        reasons.append("new_identifier_reference")
    if drift.get("api_signature_drift"):
        reasons.append("api_signature_drift")
        if filter_enabled:
            hard_reasons.append("api_signature_drift")

    risk_combo = {
        "triggered": False,
        "details": [],
        "promoted_hard_reasons": [],
    }
    if hard_reject_compile_risk_combos:
        risk_combo = _compute_java_compile_risk_combo(
            broad_rewrite_severe=bool(broad_rewrite_severe),
            broad_rewrite_moderate=bool(broad_rewrite_moderate),
            plan_ok=bool(plan_ok),
            drift=drift,
        )
        if risk_combo.get("triggered"):
            hard_reasons.extend(list(risk_combo.get("promoted_hard_reasons") or []))

    compile_result = safe_local_java_compile_check(
        base_code=base_code,
        code=code,
        expected_name=expected_name,
    )
    mode = str(requested_mode)
    if compile_result is not None:
        mode = "compile"
        if not bool(compile_result.get("ok", False)):
            reasons.append("compile_fail")
            hard_reasons.append("compile_fail")

    reason_counts = make_java_validity_reason_counter()
    for reason in dict.fromkeys(reasons):
        if reason in reason_counts:
            reason_counts[reason] += 1

    record.update({
        "checked": True,
        "mode": mode,
        "valid": len(dict.fromkeys(hard_reasons)) == 0,
        "reasons": list(dict.fromkeys(reasons)),
        "hard_reasons": list(dict.fromkeys(hard_reasons)),
        "context_drift_summary": str(drift.get("summary") or "none"),
        "context_allowlist_summary": dict(drift.get("context_allowlist_summary") or {}),
        "declared_names_in_patch": list(drift.get("declared_names_in_patch") or []),
        "context_supported_new_identifiers": list(drift.get("context_supported_new_identifiers") or []),
        "context_supported_new_types": list(drift.get("context_supported_new_types") or []),
        "context_supported_new_constants": list(drift.get("context_supported_new_constants") or []),
        "context_supported_new_identifier_sources": dict(drift.get("context_supported_new_identifier_sources") or {}),
        "context_supported_new_type_sources": dict(drift.get("context_supported_new_type_sources") or {}),
        "context_supported_new_constant_sources": dict(drift.get("context_supported_new_constant_sources") or {}),
        "new_helper_calls": list(drift.get("new_helper_calls") or []),
        "new_type_references": list(drift.get("new_type_references") or []),
        "new_constant_references": list(drift.get("new_constant_references") or []),
        "new_identifier_references": list(drift.get("new_identifier_references") or []),
        "broad_rewrite_fraction": float(rewrite_frac),
        "risk_combo_triggered": bool(risk_combo.get("triggered", False)),
        "risk_combo_details": list(risk_combo.get("details") or []),
        "reason_counts": {k: int(reason_counts.get(k, 0)) for k in JAVA_VALIDITY_REASON_KEYS},
    })
    return record


def apply_java_validity_rerank(
    score: float,
    validation_record: Dict[str, Any],
    *,
    rerank_enabled: bool,
) -> Tuple[float, bool]:
    if not rerank_enabled or not validation_record.get("checked"):
        return float(score), False
    if validation_record.get("valid") is True:
        if validation_record.get("reasons"):
            return float(score), True
        return float(score) + float(JAVA_VALIDITY_RERANK_BONUS), True
    return float(score) - float(JAVA_VALIDITY_RERANK_BONUS), True


def apply_java_semantic_rerank(
    *,
    score: float,
    context_row: Dict[str, Any],
    prom_row: Dict[str, Any],
    base_code: str,
    code: str,
    repair_branch: str,
    language: str,
    edit_summary: Dict[str, Any],
    plan_ok: bool,
    plan_violations: List[str],
    plan_fill_selected: bool = False,
    rescue_mode: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    feature_info = {
        "enabled": False,
        "score_adjustment": 0.0,
        "features": {},
    }
    if not java_semantic_rerank_enabled(repair_branch, language):
        return float(score), feature_info

    base_decl = _extract_java_decl_contract(base_code)
    cand_decl = _extract_java_decl_contract(code)
    drift = analyze_java_context_drift(base_code, code, prom_row or {}, rescue_mode=bool(rescue_mode))
    adjustment = 0.0
    features: Dict[str, Any] = {}

    exact_signature = bool(
        _normalize_java_header(base_decl.get("header", "")) != ""
        and _normalize_java_header(base_decl.get("header", "")) == _normalize_java_header(cand_decl.get("header", ""))
    )
    features["exact_signature_preservation"] = exact_signature
    if exact_signature:
        adjustment += 20.0

    throws_preserved = bool(base_decl.get("throws_clause", "") == cand_decl.get("throws_clause", ""))
    features["throws_clause_preservation"] = throws_preserved
    if throws_preserved and base_decl.get("throws_clause", ""):
        adjustment += 8.0

    base_ids = _collect_java_identifiers(base_code)
    cand_ids = _collect_java_identifiers(code)
    reuse_ratio = 0.0
    if cand_ids:
        reuse_ratio = float(len(base_ids & cand_ids)) / float(len(cand_ids))
    features["original_identifier_reuse"] = round(reuse_ratio, 4)
    if reuse_ratio >= 0.9:
        adjustment += 12.0
    elif reuse_ratio >= 0.75:
        adjustment += 8.0
    elif reuse_ratio < 0.45:
        adjustment -= 8.0

    new_helper_call_count = len(list(drift.get("new_helper_calls") or []))
    features["new_helper_call_count"] = int(new_helper_call_count)
    if new_helper_call_count:
        adjustment -= min(24.0, float(new_helper_call_count) * 8.0)   # Phase1-C: 54/18→24/8

    rewrite_frac = float(change_fraction(base_code, code))
    broad_rewrite_flag = bool(rewrite_frac > 0.50 or (rewrite_frac > 0.35 and not bool(plan_ok)))

    new_type_reference_count = len(list(drift.get("new_type_references") or []))
    features["new_type_reference_count"] = int(new_type_reference_count)
    if new_type_reference_count:
        adjustment -= min(18.0, float(new_type_reference_count) * 6.0) # Phase1-C: 42/14→18/6

    new_constant_reference_count = len(list(drift.get("new_constant_references") or []))
    features["new_constant_reference_count"] = int(new_constant_reference_count)
    if new_constant_reference_count:
        adjustment -= min(12.0, float(new_constant_reference_count) * 4.0) # Phase1-C: 18/9→12/4

    new_identifier_reference_count = len(list(drift.get("new_identifier_references") or []))
    features["new_identifier_reference_count"] = int(new_identifier_reference_count)
    if new_identifier_reference_count:
        adjustment -= min(12.0, float(new_identifier_reference_count) * 3.0) # Phase1-C: 24/6→12/3

    api_signature_drift = bool(drift.get("api_signature_drift"))
    features["api_signature_drift"] = api_signature_drift
    if api_signature_drift:
        adjustment -= 20.0                                             # Phase1-C: 30→20

    penalty_only_reasons = set()
    if broad_rewrite_flag:
        penalty_only_reasons.add("broad_rewrite")
    if new_type_reference_count:
        penalty_only_reasons.add("new_type_reference")
    if new_constant_reference_count:
        penalty_only_reasons.add("new_constant_reference")
    if new_identifier_reference_count:
        penalty_only_reasons.add("new_identifier_reference")
    penalty_only_reason_count = int(len(penalty_only_reasons))
    broad_rewrite_combo = bool(
        broad_rewrite_flag and any(reason in penalty_only_reasons for reason in JAVA_NEW_REFERENCE_REASON_KEYS)
    )
    features["penalty_only_reason_count"] = penalty_only_reason_count
    features["broad_rewrite_combo"] = bool(broad_rewrite_combo)

    combo_penalty = 0.0
    if broad_rewrite_combo:
        combo_penalty += 8.0
    if penalty_only_reason_count >= 2:
        combo_penalty += 6.0
    if penalty_only_reason_count >= 3:
        combo_penalty += 8.0
    features["risk_combo_penalty"] = float(combo_penalty)
    if combo_penalty:
        adjustment -= combo_penalty

    change_fraction_risk_penalty = 0.0
    if penalty_only_reason_count > 0 and rewrite_frac > 0.18:
        change_fraction_risk_penalty += 2.0
    if penalty_only_reason_count > 0 and rewrite_frac > 0.28:
        change_fraction_risk_penalty += 3.0
    features["change_fraction_risk_penalty"] = float(change_fraction_risk_penalty)
    if change_fraction_risk_penalty:
        adjustment -= change_fraction_risk_penalty

    context_supported_reuse = (
        len(list(drift.get("context_supported_new_identifiers") or []))
        + len(list(drift.get("context_supported_new_types") or []))
        + len(list(drift.get("context_supported_new_constants") or []))
    )
    features["context_supported_reuse"] = int(context_supported_reuse)
    context_supported_reuse_bonus = 0.0
    # Phase1-C: 조건 완화 — broad_rewrite가 아니면 항상 보너스 부여
    if context_supported_reuse and not broad_rewrite_flag and penalty_only_reason_count < 3:
        context_supported_reuse_bonus = min(6.0, float(context_supported_reuse) * 2.0)  # Phase1-C: max 2→6, per 1→2
    features["context_supported_reuse_bonus"] = float(context_supported_reuse_bonus)
    if context_supported_reuse_bonus:
        adjustment += context_supported_reuse_bonus

    features["plan_ok"] = bool(plan_ok)
    if plan_ok:
        adjustment += 15.0                                             # Phase1-C: 10→15

    violation_count = len(list(plan_violations or []))
    features["low_violation_count"] = int(violation_count)
    if violation_count == 0:
        adjustment += 8.0                                              # Phase1-C: 6→8
    features["high_violation_count"] = int(violation_count)
    if violation_count > 0:
        adjustment -= min(16.0, float(violation_count) * 3.0)         # Phase1-C: 20/4→16/3

    mean_dist = _mean_edit_distance_to_line(edit_summary, _compute_local_bug_line(prom_row, base_code))
    features["near_bug_region"] = mean_dist
    if mean_dist is not None:
        if mean_dist <= 2.0:
            adjustment += 12.0                                         # Phase1-C: 8→12
        elif mean_dist <= 5.0:
            adjustment += 6.0                                          # Phase1-C: 4→6
        elif mean_dist >= 12.0:
            adjustment -= 4.0                                          # Phase1-C: -6→-4

    has_artifact = bool(_contains_model_artifact(code) or MODEL_ARTIFACT_TEXT_RE.search(code or ""))
    features["model_artifact"] = has_artifact
    if has_artifact:
        adjustment -= 40.0

    markdown_tail = bool("```" in (code or "") or re.search(r"(?i)\b(explanation|analysis|reasoning)\b", code or ""))
    features["markdown_or_explanation_tail"] = markdown_tail
    if markdown_tail:
        adjustment -= 25.0

    base_decl_count = _count_java_decl_like(base_code)
    cand_decl_count = _count_java_decl_like(code)
    extra_decl = cand_decl_count > base_decl_count
    features["invented_helper_declaration"] = extra_decl
    features["extra_nested_declaration"] = int(max(0, cand_decl_count - base_decl_count))
    if extra_decl:
        adjustment -= 12.0                                             # Phase1-C: 20→12

    features["plan_fill_selected"] = bool(plan_fill_selected)
    if plan_fill_selected:
        adjustment -= 8.0                                              # Phase1-C: 12→8

    features["broad_rewrite"] = round(rewrite_frac, 4)
    features["broad_rewrite_severe"] = bool(rewrite_frac > 0.50)
    if rewrite_frac > 0.50:
        adjustment -= 20.0                                             # Phase1-C: 35→20
    elif rewrite_frac > 0.35:
        adjustment -= 10.0                                             # Phase1-C: 16→10
    elif rewrite_frac > 0.25:
        adjustment -= 4.0                                              # Phase1-C: 8→4

    features["risk_tier"] = int(
        3 if api_signature_drift else (0 if penalty_only_reason_count <= 0 else (1 if penalty_only_reason_count <= 1 else 2))
        # Phase1-C: new_helper_call_count 제거 — tier 3 조건에서 제외
    )

    feature_info["enabled"] = True
    feature_info["score_adjustment"] = float(adjustment)
    feature_info["features"] = features
    return float(score) + float(adjustment), feature_info


def build_java_candidate_metadata(
    *,
    cand: Optional[Dict[str, Any]],
    validation_record: Dict[str, Any],
    semantic_record: Dict[str, Any],
    java_semantic_enabled: bool,
) -> Dict[str, Any]:
    cand = cand or {}
    return {
        "java_compile_validation_mode": str(validation_record.get("mode") or cand.get("java_compile_validation_mode") or "none"),
        "java_compile_valid": validation_record.get("valid", cand.get("java_compile_valid")),
        "java_compile_validation_reasons": list(validation_record.get("reasons") or cand.get("java_compile_validation_reasons") or []),
        "java_compile_validation_hard_reasons": list(validation_record.get("hard_reasons") or cand.get("java_compile_validation_hard_reasons") or []),
        "java_compile_risk_combo_triggered": bool(
            validation_record.get("risk_combo_triggered", cand.get("java_compile_risk_combo_triggered", False))
        ),
        "java_compile_risk_combo_details": list(
            validation_record.get("risk_combo_details") or cand.get("java_compile_risk_combo_details") or []
        ),
        "java_context_drift_summary": str(validation_record.get("context_drift_summary") or cand.get("java_context_drift_summary") or "none"),
        "java_context_allowlist_summary": dict(validation_record.get("context_allowlist_summary") or cand.get("java_context_allowlist_summary") or {}),
        "java_declared_names_in_patch": list(validation_record.get("declared_names_in_patch") or cand.get("java_declared_names_in_patch") or []),
        "java_context_supported_new_identifiers": list(validation_record.get("context_supported_new_identifiers") or cand.get("java_context_supported_new_identifiers") or []),
        "java_context_supported_new_types": list(validation_record.get("context_supported_new_types") or cand.get("java_context_supported_new_types") or []),
        "java_context_supported_new_constants": list(validation_record.get("context_supported_new_constants") or cand.get("java_context_supported_new_constants") or []),
        "java_context_supported_new_identifier_sources": dict(validation_record.get("context_supported_new_identifier_sources") or cand.get("java_context_supported_new_identifier_sources") or {}),
        "java_context_supported_new_type_sources": dict(validation_record.get("context_supported_new_type_sources") or cand.get("java_context_supported_new_type_sources") or {}),
        "java_context_supported_new_constant_sources": dict(validation_record.get("context_supported_new_constant_sources") or cand.get("java_context_supported_new_constant_sources") or {}),
        "java_new_helper_calls": list(validation_record.get("new_helper_calls") or cand.get("java_new_helper_calls") or []),
        "java_new_type_references": list(validation_record.get("new_type_references") or cand.get("java_new_type_references") or []),
        "java_new_constant_references": list(validation_record.get("new_constant_references") or cand.get("java_new_constant_references") or []),
        "java_new_identifier_references": list(validation_record.get("new_identifier_references") or cand.get("java_new_identifier_references") or []),
        "java_broad_rewrite_fraction": float(validation_record.get("broad_rewrite_fraction", cand.get("java_broad_rewrite_fraction", 0.0)) or 0.0),
        "java_semantic_rerank_enabled": bool(cand.get("java_semantic_rerank_enabled", java_semantic_enabled)),
        "java_semantic_score_adjustment": float(semantic_record.get("score_adjustment", 0.0) or 0.0),
        "java_semantic_features": dict(semantic_record.get("features") or cand.get("java_semantic_features") or {}),
        "java_rescue_survivor": bool(cand.get("java_rescue_survivor", False)),
        "java_reject_policy_version": str(cand.get("java_reject_policy_version") or JAVA_REJECT_POLICY_VERSION),
    }


def _java_candidate_only_penalty_reasons(candidate: Dict[str, Any]) -> bool:
    reasons = {str(item) for item in list(candidate.get("java_compile_validation_reasons") or []) if str(item)}
    return bool(reasons) and reasons.issubset(JAVA_SOFT_CONTEXT_REASON_KEYS)


def _java_candidate_penalty_reasons(candidate: Dict[str, Any]) -> List[str]:
    reasons = []
    seen = set()
    for raw_reason in list(candidate.get("java_compile_validation_reasons") or []):
        reason = str(raw_reason or "").strip()
        if not reason or reason not in JAVA_SOFT_CONTEXT_REASON_KEYS or reason in seen:
            continue
        seen.add(reason)
        reasons.append(reason)
    return reasons


def _java_candidate_penalty_reason_count(candidate: Dict[str, Any]) -> int:
    return int(len(_java_candidate_penalty_reasons(candidate)))


def _java_candidate_risk_tier(candidate: Dict[str, Any]) -> int:
    if bool(list(candidate.get("java_compile_validation_hard_reasons") or [])):
        return 3
    penalty_reason_count = _java_candidate_penalty_reason_count(candidate)
    if penalty_reason_count <= 0:
        return 0
    if penalty_reason_count == 1:
        return 1
    return 2


def _java_candidate_eligible_for_safety_tiebreak(candidate: Dict[str, Any]) -> bool:
    if str(candidate.get("language_branch") or "") != "java_branch":
        return False
    if bool(candidate.get("java_rescue_survivor", False)):
        return False
    return True


def _candidate_priority_compare(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    score_key: str,
) -> int:
    left_score = _safe_float(left.get(score_key), -10**9)
    right_score = _safe_float(right.get(score_key), -10**9)

    left_safe = _java_candidate_eligible_for_safety_tiebreak(left)
    right_safe = _java_candidate_eligible_for_safety_tiebreak(right)
    if left_safe and right_safe:
        left_tier = _java_candidate_risk_tier(left)
        right_tier = _java_candidate_risk_tier(right)
        if left_tier != right_tier and abs(left_score - right_score) <= JAVA_RISK_TIE_MARGIN:
            return -1 if left_tier < right_tier else 1
        if left_tier == right_tier and abs(left_score - right_score) <= 1.0:
            left_change = _safe_float(left.get("java_broad_rewrite_fraction"), 0.0)
            right_change = _safe_float(right.get("java_broad_rewrite_fraction"), 0.0)
            if left_change != right_change:
                return -1 if left_change < right_change else 1

    if left_score > right_score:
        return -1
    if left_score < right_score:
        return 1

    left_change = _safe_float(left.get("java_broad_rewrite_fraction"), 0.0)
    right_change = _safe_float(right.get("java_broad_rewrite_fraction"), 0.0)
    if left_change != right_change:
        return -1 if left_change < right_change else 1

    left_text = str(left.get("patch_signature") or left.get("candidate_code_hash") or "")
    right_text = str(right.get("patch_signature") or right.get("candidate_code_hash") or "")
    if left_text < right_text:
        return -1
    if left_text > right_text:
        return 1
    return 0


def _sorted_candidates_by_priority(
    items: List[Dict[str, Any]],
    *,
    score_key: str,
) -> List[Dict[str, Any]]:
    return sorted(
        list(items or []),
        key=cmp_to_key(lambda left, right: _candidate_priority_compare(left, right, score_key=score_key)),
    )


def _java_candidate_reason_items(candidate: Dict[str, Any], reason: str) -> List[str]:
    reason = str(reason or "").strip()
    if reason == "new_identifier_reference":
        return [str(item) for item in list(candidate.get("java_new_identifier_references") or []) if str(item)]
    if reason == "new_type_reference":
        return [str(item) for item in list(candidate.get("java_new_type_references") or []) if str(item)]
    if reason == "new_constant_reference":
        return [str(item) for item in list(candidate.get("java_new_constant_references") or []) if str(item)]
    if reason == "new_helper_call":  # Phase1-B: new_helper_call 지원
        return [str(item) for item in list(candidate.get("java_new_helper_calls") or []) if str(item)]
    return []


def _java_candidate_has_rescue_artifact(candidate: Dict[str, Any]) -> bool:
    semantic_features = dict(candidate.get("java_semantic_features") or {})
    if bool(candidate.get("java_decl_recovery_used", False)):
        return True
    if bool(candidate.get("java_sentinel_extract_hit", False)):
        return True
    if bool(candidate.get("java_explanation_tail_stripped", False)):
        return True
    if bool(semantic_features.get("markdown_or_explanation_tail", False)):
        return True
    if bool(semantic_features.get("model_artifact", False)):
        return True
    if bool(semantic_features.get("invented_helper_declaration", False)):
        return True
    return False


def _java_candidate_rescue_precheck(candidate: Dict[str, Any]) -> bool:
    if not bool(candidate.get("java_compile_valid", True)):
        return False
    if bool(candidate.get("java_rescue_survivor", False)):
        return False
    reasons = {str(item) for item in list(candidate.get("java_compile_validation_reasons") or []) if str(item)}
    if reasons != (reasons & JAVA_RESCUE_ALLOWED_REASON_KEYS):
        return False
    if len(reasons) < 1 or len(reasons) > 2:   # Phase1-B: 1→1~2개 reason 허용
        return False
    strict_reason = next(iter(reasons))
    # Phase1-B: new_helper_calls 차단 제거 — RESCUE_ALLOWED_REASON_KEYS에 포함되어 있으므로 허용
    if bool(candidate.get("java_compile_validation_hard_reasons")):
        return False
    if bool(candidate.get("java_semantic_features", {}).get("api_signature_drift", False)):
        return False
    if float(candidate.get("java_broad_rewrite_fraction", 0.0) or 0.0) > JAVA_RESCUE_MAX_CHANGE_FRACTION:
        return False
    if _java_candidate_has_rescue_artifact(candidate):
        return False
    reason_items = _java_candidate_reason_items(candidate, strict_reason)
    if not reason_items or len(reason_items) > JAVA_RESCUE_MAX_REASON_ITEMS:
        return False
    return True


def _filter_java_rescue_support_sources(source_map: Dict[str, Any]) -> Dict[str, List[str]]:
    filtered: Dict[str, List[str]] = {}
    for symbol, raw_sources in dict(source_map or {}).items():
        sources = sorted(
            str(item)
            for item in list(raw_sources or [])
            if str(item) in JAVA_RESCUE_ALLOWED_SUPPORT_SOURCES
        )
        if sources:
            filtered[str(symbol)] = sources
    return filtered


def _java_candidate_rescue_eligible(
    candidate: Dict[str, Any],
    rescue_validation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _java_candidate_rescue_precheck(candidate):
        return None
    if not bool(rescue_validation.get("checked", False)):
        return None
    if rescue_validation.get("valid") is not True:
        return None
    if list(rescue_validation.get("hard_reasons") or []):
        return None
    if list(rescue_validation.get("reasons") or []):
        return None
    if float(rescue_validation.get("broad_rewrite_fraction", 0.0) or 0.0) > JAVA_RESCUE_MAX_CHANGE_FRACTION:
        return None

    strict_reasons = {str(item) for item in list(candidate.get("java_compile_validation_reasons") or []) if str(item)}
    if len(strict_reasons) != 1:
        return None
    strict_reason = next(iter(strict_reasons))
    reason_items = _java_candidate_reason_items(candidate, strict_reason)
    if not reason_items:
        return None

    if strict_reason == "new_identifier_reference":
        raw_support = dict(rescue_validation.get("context_supported_new_identifier_sources") or {})
    elif strict_reason == "new_type_reference":
        raw_support = dict(rescue_validation.get("context_supported_new_type_sources") or {})
    elif strict_reason == "new_constant_reference":
        raw_support = dict(rescue_validation.get("context_supported_new_constant_sources") or {})
    else:
        return None

    filtered_support = _filter_java_rescue_support_sources(raw_support)
    rescue_support_sources = {
        symbol: filtered_support.get(symbol, [])
        for symbol in reason_items
        if filtered_support.get(symbol)
    }
    if len(rescue_support_sources) != len(reason_items):
        return None
    return {
        "java_rescue_reason": str(strict_reason),
        "java_rescue_support_sources": rescue_support_sources,
        "java_rescue_policy_version": JAVA_RESCUE_POLICY_VERSION,
    }


def resolve_plan_enforcement(args: Namespace) -> str:
    repair_branch = normalize_repair_branch(getattr(args, "repair_branch", "auto"))
    if repair_branch == "auto":
        dataset_tag = str(getattr(args, "_dataset_tag_current", "") or "").strip().lower()
        if dataset_tag == "defects4j":
            repair_branch = "java_base"
        elif dataset_tag == "bugsinpy":
            repair_branch = "python_base"
    return resolve_effective_plan_enforcement(args, repair_branch)


def _single_count_value(counts: Dict[str, int], default: str) -> str:
    normalized = {str(k): int(v) for k, v in (counts or {}).items() if str(k).strip() and int(v) > 0}
    if len(normalized) == 1:
        return next(iter(normalized))
    if len(normalized) > 1:
        return "mixed"
    return str(default)


def derive_manifest_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.manifest{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}.manifest.json")


def try_get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return str(out or "").strip()
    except Exception:
        return ""


def infer_language(context_row: Dict[str, Any], prom_row: Dict[str, Any], base_code: str) -> str:
    file_path = (
        context_row.get("file_path")
        or safe_get(prom_row, ["file_path"])
        or safe_get(prom_row, ["file", "file_path"])
        or safe_get(prom_row, ["function", "file_path"])
        or ""
    )
    if isinstance(file_path, str) and file_path.lower().endswith(".java"):
        return "java"
    if JAVA_DECL_HEAD_RE.search(base_code or ""):
        return "java"
    return "python"


def quick_has_def_or_class(code: str, language: str) -> bool:
    stripped = (code or "").lstrip()
    if not stripped:
        return False
    if language == "java":
        return bool(JAVA_DECL_HEAD_RE.search(code or ""))
    return bool(DEFCLASS_HEAD_RE.search(code or ""))


def extract_decl_name(code: str, language: str, py_node: Optional[ast.AST] = None) -> str:
    if language == "java":
        m = JAVA_NAME_RE.search(code or "")
        if not m:
            return ""
        return (m.group(1) or m.group(2) or m.group(3) or "").strip()
    if py_node is not None:
        return (getattr(py_node, "name", None) or "").strip()
    return ""

def _last_correct_marker(prompt: str) -> Optional[re.Match]:
    matches = list(re.finditer(r"^[ \t]*##correct[ \t]*$", prompt or "", flags=re.IGNORECASE | re.MULTILINE))
    return matches[-1] if matches else None

def insert_before_correct(prompt: str, insertion: str) -> str:
    """
    Insert extra guidance right before the last `##correct` marker.
    If the marker is missing, append one at the end.
    """
    insertion = (insertion or "").strip()
    if not insertion:
        return prompt or ""

    prompt = prompt or ""
    m = _last_correct_marker(prompt)
    if not m:
        return prompt.rstrip() + "\n\n" + insertion + "\n\n##correct\n"
    before = prompt[: m.start()].rstrip()
    return before + "\n\n" + insertion + "\n\n##correct\n"



# ===== PLAN placeholder handling =====
PLAN_JSON_PLACEHOLDER = "{{PLAN_JSON}}"
PLAN_ALLOWED_EDIT_TYPES = {"REPLACE", "INSERT", "DELETE"}

def _safe_int(v) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None

def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def build_default_plan_json(prom_row: Dict[str, Any], base_code: str, language: str) -> str:
    """Build a small, safe default PLAN JSON when the upstream step didn't fill it."""
    try:
        repair_mode = str((prom_row or {}).get("repair_mode") or (prom_row or {}).get("hard_tag") or "").strip().lower()
        if repair_mode not in ("single", "hard"):
            # Heuristic fallback
            repair_mode = "single" if bool((prom_row or {}).get("single_line")) else "hard"

        snippet_lines = textwrap.dedent(base_code or "").splitlines()
        n_lines = max(1, len(snippet_lines))

        buggy_loc = _safe_int(safe_get(prom_row, ["buggy_line_location"]))
        func_start = (
            _safe_int(safe_get(prom_row, ["function", "function_before_start_line"]))
            or _safe_int(safe_get(prom_row, ["function", "function_after_start_line"]))
        )
        local_bug = None
        if buggy_loc is not None and func_start is not None:
            local_bug = buggy_loc - func_start + 1
            if local_bug < 1 or local_bug > n_lines:
                local_bug = None
        if local_bug is None:
            local_bug = 1

        if repair_mode == "single":
            start = max(1, local_bug - 2)
            end = min(n_lines, local_bug + 2)
            changed_max = 6
            allowed = ["REPLACE"]
            allow_struct = False
        else:
            start = max(1, local_bug - 6)
            end = min(n_lines, local_bug + 6)
            changed_max = 25
            allowed = ["REPLACE", "INSERT", "DELETE"]
            allow_struct = True

        forbidden: List[Dict[str, Any]] = []
        if start > 1:
            forbidden.append({"start": 1, "end": start - 1, "reason": "Keep edits localized."})
        if end < n_lines:
            forbidden.append({"start": end + 1, "end": n_lines, "reason": "Keep edits localized."})

        # Best-effort evidence extraction from AST hints (if present)
        ast_nodes: List[Dict[str, Any]] = []
        nodes = safe_get(prom_row, ["suspicious_nodes_topk"]) or safe_get(prom_row, ["suspicious_nodes"]) or []
        if isinstance(nodes, list):
            for node in nodes[:3]:
                if not isinstance(node, dict):
                    continue
                line = _safe_int(node.get("line") or node.get("lineno"))
                typ = str(node.get("type") or node.get("node_type") or "").strip()
                if line is None and isinstance(node.get("span"), dict):
                    line = _safe_int(node["span"].get("start_line"))
                if line is not None or typ:
                    ast_nodes.append({"line": line, "type": typ})

        plan = {
            "target_locations": [{"start": int(start), "end": int(end)}],
            "edit_budget": {"changed_lines_max": int(changed_max), "allow_structure_changes": bool(allow_struct)},
            "allowed_edit_types": allowed,
            "forbidden_regions": forbidden,
            "structural_constraints": {
                "must_keep_signature": True,
                "must_keep_name": True,
                "no_new_imports": True,
                "language": str(language or ""),
            },
            "evidence_links": {
                "ast_nodes": ast_nodes,
                "retrieval_top": [1, 2],
            },
        }
        return json.dumps(plan, ensure_ascii=False, indent=2)
    except Exception:
        # Minimal last resort
        return "{}"

def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object")
    return obj

def validate_plan_obj(plan_obj: Any) -> Dict[str, Any]:
    if not isinstance(plan_obj, dict):
        raise ValueError("plan_json must be a dict")

    target_locations = plan_obj.get("target_locations")
    if not isinstance(target_locations, list) or not target_locations:
        raise ValueError("target_locations must be a non-empty list")
    normalized_targets: List[Dict[str, int]] = []
    for idx, item in enumerate(target_locations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"target_locations[{idx}] must be an object")
        start = _safe_int(item.get("start"))
        end = _safe_int(item.get("end"))
        if start is None or end is None or start < 1 or end < start:
            raise ValueError(f"target_locations[{idx}] must contain valid start/end")
        normalized_targets.append({"start": int(start), "end": int(end)})

    edit_budget = plan_obj.get("edit_budget")
    if not isinstance(edit_budget, dict):
        raise ValueError("edit_budget must be an object")
    changed_lines_max = _safe_int(edit_budget.get("changed_lines_max"))
    if changed_lines_max is None or changed_lines_max < 1:
        raise ValueError("edit_budget.changed_lines_max must be a positive integer")
    normalized_budget = dict(edit_budget)
    normalized_budget["changed_lines_max"] = int(changed_lines_max)

    allowed_edit_types = plan_obj.get("allowed_edit_types")
    if not isinstance(allowed_edit_types, list) or not allowed_edit_types:
        raise ValueError("allowed_edit_types must be a non-empty list")
    normalized_allowed: List[str] = []
    for raw_tag in allowed_edit_types:
        tag = str(raw_tag or "").strip().upper()
        if tag not in PLAN_ALLOWED_EDIT_TYPES:
            raise ValueError(f"Unsupported allowed_edit_type: {raw_tag}")
        normalized_allowed.append(tag)

    normalized = dict(plan_obj)
    normalized["target_locations"] = normalized_targets
    normalized["edit_budget"] = normalized_budget
    normalized["allowed_edit_types"] = normalized_allowed
    return normalized

def build_default_plan_obj(prom_row: Dict[str, Any], base_code: str, language: str) -> Dict[str, Any]:
    raw = build_default_plan_json(prom_row or {}, base_code=base_code, language=language)
    parsed = json.loads(raw)
    return validate_plan_obj(parsed)

def parse_plan_dict(
    prom_row: Dict[str, Any],
    base_code: str = "",
    language: str = "",
) -> Tuple[Dict[str, Any], str]:
    prom_row = prom_row or {}
    explicit_plan_source = str(prom_row.get("plan_source") or "").strip()
    for key in ("plan_json", "plan", "repair_plan"):
        if key not in prom_row:
            continue
        raw_plan = prom_row.get(key)
        if raw_plan is None or raw_plan == "":
            continue
        try:
            if isinstance(raw_plan, dict):
                parsed = raw_plan
            elif isinstance(raw_plan, str):
                parsed = json.loads(raw_plan)
            else:
                continue
            return validate_plan_obj(parsed), (explicit_plan_source or "unknown")
        except Exception:
            continue
    return build_default_plan_obj(prom_row, base_code=base_code, language=language), "fallback_default"

def parse_plan_obj(
    prom_row: Dict[str, Any],
    base_code: str = "",
    language: str = "",
) -> Tuple[Dict[str, Any], str]:
    return parse_plan_dict(prom_row, base_code=base_code, language=language)

def normalize_edit_type(tag: str) -> str:
    tag_norm = str(tag or "").strip().lower()
    if tag_norm == "replace":
        return "REPLACE"
    if tag_norm == "insert":
        return "INSERT"
    if tag_norm == "delete":
        return "DELETE"
    return tag_norm.upper()

def normalize_edit_op(tag: str) -> str:
    return normalize_edit_type(tag)

def compute_changed_lines_for_edit(edit: Dict[str, Any]) -> int:
    op = normalize_edit_type(edit.get("op"))
    start = _safe_int(edit.get("start")) or 0
    end = _safe_int(edit.get("end")) or start
    if op == "INSERT":
        new_text = str(edit.get("new_text") or "")
        return max(1, len(new_text.splitlines()) if new_text else 1)
    return max(1, end - start + 1)

def compute_changed_lines(edit_summary: Dict[str, Any]) -> int:
    edits = edit_summary.get("edits") if isinstance(edit_summary, dict) else []
    return int(sum(compute_changed_lines_for_edit(edit) for edit in (edits or [])))

def in_any_range(x: int, ranges: List[Dict[str, int]]) -> bool:
    return any(int(r["start"]) <= int(x) <= int(r["end"]) for r in ranges)

def _normalize_target_ranges(targets: Any) -> List[Dict[str, int]]:
    normalized: List[Dict[str, int]] = []
    if not isinstance(targets, list):
        return normalized
    for item in targets:
        if not isinstance(item, dict):
            continue
        start = _safe_int(item.get("start"))
        end = _safe_int(item.get("end"))
        if start is None or end is None:
            continue
        if start > end:
            start, end = end, start
        if start < 1:
            start = 1
        normalized.append({"start": int(start), "end": int(end)})
    return normalized

def _clamp_anchor(line_no: int, n_lines: int) -> int:
    if n_lines <= 0:
        return 1
    if line_no < 1:
        return 1
    if line_no > n_lines:
        return int(n_lines)
    return int(line_no)

def check_plan(
    plan_dict: Dict[str, Any],
    edit_summary: Dict[str, Any],
    n_lines: int,
    target_match: str = "overlap",
    language: str = "python",
) -> Tuple[bool, List[str], Dict[str, Any]]:
    violations: List[str] = []
    targets: List[Dict[str, int]] = []
    changed_lines = 0
    edit_type_violation_count = 0
    out_of_target_count = 0
    violation_type_breakdown = make_plan_violation_counter()

    try:
        plan_dict = validate_plan_obj(plan_dict)
        targets = _normalize_target_ranges(plan_dict.get("target_locations", []))
    except Exception as exc:
        violations.append(f"invalid_plan:{exc}")
        violation_type_breakdown["other"] += 1
        stats = {
            "changed_lines": int(compute_changed_lines(edit_summary)),
            "changed_lines_max": None,
            "out_of_target_count": 0,
            "edit_type_violation_count": 0,
            "violation_type_breakdown": {k: int(violation_type_breakdown.get(k, 0)) for k in PLAN_VIOLATION_TYPE_KEYS},
        }
        return False, violations, stats

    match_mode = str(target_match or "overlap").strip().lower()
    if match_mode not in {"contain", "overlap"}:
        match_mode = "overlap"
    allowed = set(plan_dict.get("allowed_edit_types", ["REPLACE", "INSERT", "DELETE"]) or ["REPLACE", "INSERT", "DELETE"])
    changed_max_raw = safe_get(plan_dict, ["edit_budget", "changed_lines_max"])
    changed_lines_max = _safe_int(changed_max_raw)
    target_slack = JAVA_PLAN_TARGET_SLACK if language == "java" else 0
    changed_lines_slack = JAVA_PLAN_CHANGED_LINES_SLACK if language == "java" else 0
    expanded_targets = [
        {
            "start": max(1, int(t["start"]) - target_slack),
            "end": max(1, min(n_lines, int(t["end"]) + target_slack)) if n_lines > 0 else int(t["end"]) + target_slack,
        }
        for t in targets
    ]
    effective_changed_lines_max = (
        changed_lines_max + changed_lines_slack if changed_lines_max is not None else None
    )
    edits = edit_summary.get("edits") if isinstance(edit_summary, dict) else []

    for edit in edits or []:
        edit_type = normalize_edit_type(edit.get("op"))
        changed_lines += compute_changed_lines_for_edit(edit)
        if edit_type not in allowed:
            edit_type_violation_count += 1
            violations.append(f"edit_type_not_allowed:{edit_type}")
            violation_type_breakdown["edit_type"] += 1

        start = _safe_int(edit.get("start")) or 1
        end = _safe_int(edit.get("end")) or start
        start_clamped = _clamp_anchor(start, n_lines)
        end_clamped = _clamp_anchor(end, n_lines)
        if end_clamped < start_clamped:
            end_clamped = start_clamped

        in_target = True
        if edit_type == "INSERT":
            anchors = [start_clamped, _clamp_anchor(start_clamped - 1, n_lines)]
            in_target = any(in_any_range(anchor, expanded_targets) for anchor in anchors)
            if not in_target:
                out_of_target_count += 1
                violations.append(f"out_of_target:INSERT@{start}")
                violation_type_breakdown["out_of_target"] += 1
        else:
            if match_mode == "contain":
                in_target = any(
                    start_clamped >= int(t["start"]) and end_clamped <= int(t["end"])
                    for t in expanded_targets
                )
            else:
                in_target = any(
                    not (end_clamped < int(t["start"]) or start_clamped > int(t["end"]))
                    for t in expanded_targets
                )
            if not in_target:
                out_of_target_count += 1
                violations.append(f"out_of_target:{edit_type}@{start_clamped}-{end_clamped}")
                violation_type_breakdown["out_of_target"] += 1

    if effective_changed_lines_max is not None and changed_lines > effective_changed_lines_max:
        violations.append(f"changed_lines_max_exceeded:{changed_lines}>{effective_changed_lines_max}")
        violation_type_breakdown["changed_lines_exceed"] += 1

    stats = {
        "changed_lines": int(changed_lines),
        "changed_lines_max": changed_lines_max,
        "changed_lines_max_effective": effective_changed_lines_max,
        "out_of_target_count": int(out_of_target_count),
        "edit_type_violation_count": int(edit_type_violation_count),
        "violation_type_breakdown": {k: int(violation_type_breakdown.get(k, 0)) for k in PLAN_VIOLATION_TYPE_KEYS},
        "target_match": match_mode,
        "target_slack": int(target_slack),
    }
    violations = list(dict.fromkeys(violations))
    return (len(violations) == 0), violations, stats

def fill_plan_placeholder(
    prompt: str,
    prom_row: Dict[str, Any],
    base_code: str,
    language: str,
    plan_obj: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = prompt or ""
    if PLAN_JSON_PLACEHOLDER not in prompt:
        return prompt

    if plan_obj is None:
        plan_obj, _plan_source = parse_plan_dict(prom_row or {}, base_code=base_code, language=language)
    plan_str = json.dumps(plan_obj, ensure_ascii=False, indent=2)

    return prompt.replace(PLAN_JSON_PLACEHOLDER, plan_str)

def change_fraction(base_code: str, code: str) -> float:
    """
    Roughly estimate how much of the function was touched (line-based).

    NOTE:
      - We count INSERT operations too (previously they were under-counted),
        because inserts are real edits that should influence ranking/filters.
    """
    try:
        base_lines = (base_code or "").splitlines()
        new_lines = (code or "").splitlines()
        if not base_lines:
            return 1.0
        matcher = difflib.SequenceMatcher(None, base_lines, new_lines)
        changed = 0
        for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
            if tag == "equal":
                continue
            # count both removed and added lines; INSERT has (ahi-alo)==0, so we take max(...)
            changed += max(ahi - alo, bhi - blo)
        return changed / max(1, len(base_lines))
    except Exception:
        return 0.0

def length_ratio(base_code: str, code: str) -> float:
    try:
        base_lines = len(base_code.splitlines())
        new_lines = len(code.splitlines())
        if base_lines == 0:
            return 1.0
        return new_lines / max(1, base_lines)
    except Exception:
        return 1.0

def score_candidate(
    context_row: Dict[str, Any],
    prom_row: Dict[str, Any],
    base_code: str,
    code: str,
    ast_pass: bool,
    language: str = "python",
    canon_base: Optional[str] = None,
    canon_code: Optional[str] = None,
) -> int:
    """
    Heuristic scoring (reranking only; hard constraints are filtered elsewhere):
      - AST pass: required
      - Signature/name match: strong bonus
      - Prefer small-but-nonzero diffs
      - Prefer similar length and high textual similarity
      - Penalize leaving the buggy line unchanged (soft)
    """
    try:
        if not code.strip():
            return -1000
        score = 0
        if ast_pass:
            score += 100
        else:
            return -500

        buggy_line = (safe_get(prom_row, ["buggy_line_content"]) or "").strip()
        buggy_in_base = buggy_line and buggy_line in base_code
        if buggy_line:
            if buggy_line in code and buggy_in_base:
                score -= 15
            elif buggy_in_base and buggy_line not in code:
                score += 15
        buggy_ctx = (safe_get(prom_row, ["buggy_line_context"]) or "").strip()
        if buggy_ctx and buggy_ctx not in code and buggy_line and buggy_line not in code:
            score -= 10

        fn_name = (safe_get(context_row, ["function", "function_name"]) or
                   context_row.get("function_name") or "").strip()
        if fn_name:
            if language == "java":
                got_name = extract_decl_name(code, language="java")
                if got_name == fn_name:
                    score += 35
                else:
                    score -= 120  # name mismatch is almost always wrong
            elif re.search(rf"^\s*(def|class)\s+{re.escape(fn_name)}\b", code, re.M):
                score += 35
            else:
                score -= 120  # name mismatch is almost always wrong

        if base_code:
            base_norm = canon_base if canon_base is not None else canon(base_code)
            code_norm = canon_code if canon_code is not None else canon(code)
            seq_matcher = difflib.SequenceMatcher(None, base_norm, code_norm)
            score += int(seq_matcher.ratio() * 45)
            diff_frac = change_fraction(base_code, code)
            if diff_frac < 0.005:
                score -= 500  # near-no-op
            elif diff_frac <= 0.03:
                score += 40
            elif diff_frac <= 0.12:
                score += 30
            elif diff_frac <= 0.25:
                score += 20
            elif diff_frac <= 0.5:
                score += 5
            else:
                score -= 10  # too large
            len_ratio_val = length_ratio(base_code, code)
            if 0.8 <= len_ratio_val <= 1.2:
                score += 20
            elif 0.6 <= len_ratio_val <= 1.5:
                score += 8
            elif len_ratio_val < 0.25 or len_ratio_val > 2.5:
                score -= 30

        if "pass" in code or "..." in code or "NotImplementedError" in code:
            score -= 300
        if _contains_model_artifact(code):
            score -= 500
        if MODEL_ARTIFACT_COMMENT_HINT_RE.search(code):
            score -= 80
        for pat in BANNED_TOKENS:
            if pat.search(code):
                score -= 80
        if re.search(r"^\s*(import|from)\s+", code, flags=re.M):
            score -= 200

        n_lines = len(code.strip().splitlines())
        if n_lines <= 2:
            score -= 30
        elif n_lines > 200:
            score -= 25

        return score
    except Exception:
        return 0

def _parse_single_toplevel_def_or_class(py_text: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(py_text)
    except Exception:
        return None
    if len(tree.body) != 1:
        return None
    node = tree.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node
    return None

def _first_toplevel_def_or_class(py_text: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(py_text)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node
    return None


def _parse_python_first_toplevel_def_or_class(py_text: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    try:
        tree = ast.parse(py_text)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node, None
    return None, None

def _dump_ast_list(nodes: List[ast.AST]) -> List[str]:
    return [ast.dump(n, include_attributes=False) for n in (nodes or [])]

def _signature_fingerprint(node: Optional[ast.AST]) -> Optional[str]:
    """
    Stable fingerprint for the definition header (type/name/args/returns/decorators/bases).
    """
    if node is None:
        return None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        payload = {
            "type": type(node).__name__,
            "name": node.name,
            "args": ast.dump(node.args, include_attributes=False),
            "returns": ast.dump(node.returns, include_attributes=False) if node.returns is not None else None,
            "decorators": _dump_ast_list(node.decorator_list),
        }
        return json.dumps(payload, sort_keys=True)
    if isinstance(node, ast.ClassDef):
        payload = {
            "type": "ClassDef",
            "name": node.name,
            "bases": _dump_ast_list(node.bases),
            "keywords": [ast.dump(k, include_attributes=False) for k in (node.keywords or [])],
            "decorators": _dump_ast_list(node.decorator_list),
        }
        return json.dumps(payload, sort_keys=True)
    return None

def validate_candidate_code(
    *,
    code: str,
    base_code: str,
    expected_name: str,
    language: str = "python",
    parsed_node: Optional[ast.AST] = None,
    canon_base: Optional[str] = None,
    canon_code: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Minimal viability filters (keep broad candidate diversity for pass@k):
      - parseable AST (checked by caller)
      - not a no-op vs base
      - contains a top-level def/class with the expected name
    """
    if not code or not code.strip():
        return False, "empty_code"
    base_norm = canon_base if canon_base is not None else canon(base_code)
    code_norm = canon_code if canon_code is not None else canon(code)
    if code_norm == base_norm:
        return False, "no_change"

    if language == "java":
        got_name = extract_decl_name(code, language="java")
        if not got_name:
            return False, "no_def_or_class"
        if expected_name and got_name != expected_name:
            return False, "name_mismatch"
        issue = _detect_java_candidate_issue(base_code, code)
        if issue:
            return False, issue
        return True, ""

    node = parsed_node if parsed_node is not None else _first_toplevel_def_or_class(code)
    if node is None:
        return False, "no_def_or_class"
    if expected_name and getattr(node, "name", None) != expected_name:
        return False, "name_mismatch"
    return True, ""

# ===== LLM client =====
def build_sampling_params(kwargs: Dict[str, Any]) -> Any:
    """Create vLLM SamplingParams with backwards-compatible kwargs.

    Some vLLM versions may not support newer arguments (e.g., stop, top_k,
    presence_penalty, frequency_penalty). We drop unsupported kwargs one-by-one.
    """
    if SamplingParams is None:
        raise RuntimeError("SamplingParams is unavailable (vLLM import failed).")
    params_kwargs = dict(kwargs or {})
    for _ in range(16):
        try:
            return SamplingParams(**params_kwargs)
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if not m:
                # older vLLM: sometimes 'seed' triggers a different TypeError message
                if "seed" in params_kwargs:
                    params_kwargs.pop("seed", None)
                    continue
                raise
            bad = m.group(1)
            if bad in params_kwargs:
                params_kwargs.pop(bad, None)
                continue
            raise
    return SamplingParams(**params_kwargs)

class QwenClient:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        backend: str = "auto",
        hf_model_name: str = "",
        hf_device: str = "auto",
        hf_dtype: str = "auto",
    ):
        self.system_prompt = (system_prompt or "").strip()
        self.backend, self.backend_selected = create_backend(
            backend=backend,
            model_name=model_name,
            system_prompt=self.system_prompt,
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            hf_model_name=(hf_model_name or model_name),
            hf_device=hf_device,
            hf_dtype=hf_dtype,
            logger=logging,
        )
        self.hf_model_name = (
            str(getattr(self.backend, "model_name", "") or "")
            if self.backend_selected == "hf"
            else ""
        )
        self.metrics = self.backend.metrics
        logging.info(
            "Initialized LLM backend=%s model=%s",
            self.backend_selected,
            getattr(self.backend, "model_name", model_name),
        )

    def generate(self, prompts: List[str], gen_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        seed_val = gen_args.get("seed", None)
        if seed_val is None:
            seed_val = random.randint(0, 10_000_000)
        records = self.backend.generate_records(
            list(prompts or []),
            temperature=float(gen_args.get("temperature", 0.8)),
            top_p=float(gen_args.get("top_p", 0.9)),
            max_new_tokens=int(gen_args.get("max_new_tokens", 480)),
            seed=int(seed_val),
            stop=gen_args.get("stop", None),
            do_sample=bool(gen_args.get("do_sample", True)),
            n=max(1, int(gen_args.get("n", 1))),
            repetition_penalty=float(gen_args.get("repetition_penalty", 1.05)),
            max_input_tokens=int(gen_args.get("max_input_tokens", 4096) or 4096),
            presence_penalty=float(gen_args.get("presence_penalty", 0.0) or 0.0),
            frequency_penalty=float(gen_args.get("frequency_penalty", 0.0) or 0.0),
            top_k=int(gen_args.get("top_k", 0) or 0),
        )
        return [
            {
                "text": record.text,
                "tokens_in": int(record.tokens_in),
                "tokens_out": int(record.tokens_out),
            }
            for record in records
        ]

# ===== Prompt builder =====
def build_guided_prompt(context_row: Dict[str, Any], base_code: str, prom_row: Dict[str, Any]) -> str:
    full_prompt = prom_row.get("prompt") if isinstance(prom_row, dict) else ""
    file_path = context_row.get("file_path") or safe_get(prom_row, ["file_path"]) or ""
    func_name = context_row.get("function_name") or safe_get(prom_row, ["function", "function_name"]) or ""
    bug_desc  = safe_get(prom_row, ["desc"]) or ""
    prompt_ctx = prom_row.get("prompt") or ""
    if prompt_ctx and len(prompt_ctx) > 400:
        prompt_ctx = prompt_ctx[:400]
    buggy_line = (safe_get(prom_row, ["buggy_line_content"]) or "").strip()
    buggy_loc = safe_get(prom_row, ["buggy_line_location"]) or ""
    buggy_ctx = safe_get(prom_row, ["buggy_line_context"]) or ""
    language = infer_language(context_row, prom_row, base_code)
    repair_branch = resolve_effective_repair_branch(None, context_row, prom_row, language)
    if language == "java":
        full_output_rule = (
            "After ##correct, output ONLY the repaired full method or class code "
            "(including its signature), strictly between @@BEGIN_JAVA_CODE@@ and "
            "@@END_JAVA_CODE@@, with no commentary and no markdown fences."
        )
    else:
        full_output_rule = "After ##correct, output ONLY the complete corrected function or class (including its signature)."

    guide = (
        "Fix the buggy code below in its original language.\n"
        "- Keep the original function/method/class signature and avoid unrelated edits\n"
        "- Keep changes minimal but sufficient to fix the bug\n"
        "- Avoid placeholder or non-functional edits (e.g., ellipsis or TODO-style stubs)\n"
        "- Modify only the relevant lines (plus adjacent lines only when needed for syntax)\n"
        f"- {full_output_rule}\n"
        "- Do NOT output a diff. Do NOT output a one-line snippet. Do NOT include explanations.\n"
        "- Code fences are optional.\n"
    )
    if language == "java" and repair_branch in JAVA_REPAIR_BRANCHES:
        guide += (
            f"- Put the final Java code between {JAVA_CODE_BEGIN_MARKER} and {JAVA_CODE_END_MARKER}.\n"
            "- Reuse the original identifiers, helper calls, and types whenever possible.\n"
            "- Preserve the original throws clause unless the fix truly requires changing it.\n"
            "- Prefer Java 6/7-compatible syntax for older Defects4J projects.\n"
            "- Do NOT invent helper methods, helper classes, or unrelated APIs unless unavoidable.\n"
        )

    if isinstance(full_prompt, str) and full_prompt.strip():
        plan_obj, _plan_source = parse_plan_dict(prom_row, base_code=base_code, language=language)
        allowed_edit_types = ", ".join(plan_obj.get("allowed_edit_types") or ["REPLACE"])
        strict_note = (
            "IMPORTANT:\n"
            "- Preserve the original language and declaration/signature style.\n"
            "- Keep changes minimal and avoid unrelated refactors.\n"
            f"- Respect PLAN_JSON exactly. Allowed edit types: {allowed_edit_types}.\n"
            f"- {full_output_rule}\n"
            "- Do NOT output a diff. Do NOT output a one-line snippet. Do NOT include explanations.\n"
            "- For Java, preserve the original method signature and checked exceptions unless the fix truly requires changing them.\n"
        )
        if language == "java" and repair_branch in JAVA_REPAIR_BRANCHES:
            strict_note += (
                f"- Put the final Java code between {JAVA_CODE_BEGIN_MARKER} and {JAVA_CODE_END_MARKER}, with nothing before or after those markers.\n"
                "- Reuse the original identifiers, helper calls, and types.\n"
                "- Do NOT invent helper methods, helper classes, or unrelated APIs unless unavoidable.\n"
                "- Prefer Java 6/7-compatible syntax for older Defects4J projects.\n"
            )
        prompt_base = fill_plan_placeholder(
            full_prompt.strip(),
            prom_row=prom_row,
            base_code=base_code,
            language=language,
            plan_obj=plan_obj,
        )
        prompt = insert_before_correct(prompt_base, strict_note)
        return prompt

    ctx = []
    if bug_desc:
        ctx.append(f"[Bug]\n{bug_desc.strip()}")
    if file_path:
        ctx.append(f"[File]\n{file_path}")
    if func_name:
        ctx.append(f"[Function]\n{func_name}")
    if buggy_line:
        tag = f"line {buggy_loc}: {buggy_line}" if buggy_loc else buggy_line
        ctx.append(f"[BuggyLine]\n{tag}")
    if buggy_ctx and buggy_ctx != buggy_line:
        ctx.append(f"[LocalContext]\n{buggy_ctx}")
    if prompt_ctx:
        # Keep the full context from steps 4/5 to retain location/similarity hints
        ctx.append(f"[ExtraContext]\n{prompt_ctx.strip()}")

    prompt = f"{guide}\n[Original]\n{textwrap.dedent(base_code).strip()}\n\n" + ("\n\n".join(ctx))
    # Add ##correct tag for easier parsing downstream
    if "##correct" not in prompt:
        prompt = prompt.rstrip() + "\n\n##correct\n"
    return prompt

# ===== Data loading =====
def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {str(i): v for i, v in enumerate(data)}
    else:
        raise ValueError("Unsupported JSON top-level.")


def _is_prompt_json_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() != ".json":
        return False
    lower_name = path.name.lower()
    if lower_name.endswith(".manifest.json"):
        return False
    if lower_name.endswith(".metrics.json"):
        return False
    return True


def _prompt_json_candidates(prompts_dir: Path, dataset_key: str = "") -> List[Path]:
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        return []
    patterns: List[str] = []
    if dataset_key:
        patterns.extend([
            f"*{dataset_key}*PlanAgent*.json",
            f"*{dataset_key}*GeneratePromport.json",
        ])
    else:
        patterns.extend([
            "*PlanAgent*.json",
            "*GeneratePromport.json",
        ])
    files: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(prompts_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            if _is_prompt_json_candidate(path) and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def resolve_prompts_path(prompts_arg: Path) -> Optional[Path]:
    if prompts_arg and prompts_arg.exists():
        if prompts_arg.is_file():
            if not _is_prompt_json_candidate(prompts_arg):
                raise ValueError(
                    f"prompts_file must be a Stage-3/4 prompt JSON, not an auxiliary JSON: {prompts_arg}"
                )
            return prompts_arg
        if prompts_arg.is_dir():
            files = _prompt_json_candidates(prompts_arg)
            if files:
                return files[0]

    arg_text = str(prompts_arg or "")
    if arg_text and any(ch in arg_text for ch in ["*", "?", "["]):
        files = [
            Path(p)
            for p in sorted(glob.glob(arg_text), key=os.path.getmtime, reverse=True)
            if _is_prompt_json_candidate(Path(p))
        ]
        if files:
            return files[0]

    files = _prompt_json_candidates(DEFAULT_PROMPTS_PATH)
    if files:
        return files[0]
    return None


def infer_dataset_tag(prompts_path: Path) -> str:
    name = prompts_path.name.lower()
    if "bugsinpy" in name:
        return "bugsinpy"
    if "defects4j" in name:
        return "defects4j"
    return prompts_path.stem


def resolve_full_prompt_paths(prompts_dir: Path) -> List[Path]:
    """
    Resolve full-run prompt files for both datasets from a directory.
    Preference: non-smoke files, newest mtime per dataset.
    """
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        return []

    resolved: List[Path] = []
    bugsinpy_files = [
        p for p in _prompt_json_candidates(prompts_dir, "bugsinpy")
        if "smoke" not in p.name.lower()
    ]
    if bugsinpy_files:
        resolved.append(bugsinpy_files[0])

    defects4j_priority_patterns = [
        ("java_semantic", ["*defects4j*java_semantic*PlanAgent*.json", "*defects4j*java_semantic*_GeneratePromport.json"]),
        ("java_base", ["*defects4j*java_base*PlanAgent*.json", "*defects4j*java_base*_GeneratePromport.json"]),
        ("hard", ["*defects4j*hard*PlanAgent*.json", "*defects4j*hard*_GeneratePromport.json"]),
        ("single", ["*defects4j*single*PlanAgent*.json", "*defects4j*single*_GeneratePromport.json"]),
    ]
    found_defects4j = False
    for label, patterns in defects4j_priority_patterns:
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(
                [
                    p for p in prompts_dir.glob(pattern)
                    if _is_prompt_json_candidate(p) and "smoke" not in p.name.lower()
                ]
            )
        deduped = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
        if not deduped:
            continue
        if len(deduped) > 1:
            sample = ", ".join(str(p) for p in deduped[:5])
            raise FileExistsError(
                f"Multiple defects4j prompt JSONs matched priority '{label}'. "
                f"Specify --prompts_file explicitly or clean old outputs. Matches: {sample}"
            )
        resolved.append(deduped[0])
        found_defects4j = True
        break
    if not found_defects4j:
        generic_defects4j = [
            p for p in _prompt_json_candidates(prompts_dir, "defects4j")
            if "smoke" not in p.name.lower()
        ]
        if generic_defects4j:
            resolved.append(generic_defects4j[0])
    return resolved


def derive_full_out_path(base_out: Path, dataset_tag: str) -> Path:
    if base_out == DEFAULT_OUT_PATH:
        return RESULTS_ROOT / "5" / f"5.full.{dataset_tag}.single.PatchesResults_vllm.json"
    suffix = "".join(base_out.suffixes) or ".json"
    stem = base_out.name[: -len(suffix)] if suffix and base_out.name.endswith(suffix) else base_out.stem
    return base_out.with_name(f"{stem}.{dataset_tag}{suffix}")


def derive_full_metrics_path(base_metrics: Optional[Path], dataset_tag: str) -> Optional[Path]:
    if base_metrics is None:
        return None
    suffix = "".join(base_metrics.suffixes) or ".json"
    stem = base_metrics.name[: -len(suffix)] if suffix and base_metrics.name.endswith(suffix) else base_metrics.stem
    return base_metrics.with_name(f"{stem}.{dataset_tag}{suffix}")


def write_run_manifest(
    *,
    args: Namespace,
    prompts_path: Path,
    out_path: Path,
    metrics_path: Optional[Path],
    dataset_tag: str,
    results: Optional[Dict[str, Any]] = None,
) -> Path:
    dataset_tag_lower = str(dataset_tag or "").strip().lower()
    manifest_language = "java" if dataset_tag_lower == "defects4j" else "python"
    branch_meta = resolve_repair_branch_metadata(args, None, None, manifest_language)
    requested_repair_branch = str(branch_meta.get("requested") or normalize_repair_branch(getattr(args, "repair_branch", "auto")))
    effective_manifest_branch = str(branch_meta.get("effective") or requested_repair_branch)
    effective_plan = str(resolve_plan_enforcement(args))
    java_survivor_backoff_mode = str(
        resolve_effective_java_survivor_backoff(args, effective_manifest_branch, effective_plan)
    )
    repair_branch_counts: Dict[str, int] = {}
    repair_branch_source_counts: Dict[str, int] = {}
    language_branch_counts: Dict[str, int] = {}
    effective_plan_enforcements: Dict[str, int] = {}
    java_survivor_backoff_counts: Dict[str, int] = {}
    if isinstance(results, dict):
        for row in results.values():
            if not isinstance(row, dict):
                continue
            repair_branch = str(row.get("repair_branch") or "unknown")
            repair_branch_source = str(row.get("repair_branch_source") or "unknown")
            language_branch = str(row.get("language_branch") or "unknown")
            plan_enforcement = str(row.get("plan_enforcement") or "unknown")
            backoff_mode = str(row.get("java_survivor_backoff_mode") or "off")
            repair_branch_counts[repair_branch] = int(repair_branch_counts.get(repair_branch, 0)) + 1
            repair_branch_source_counts[repair_branch_source] = int(repair_branch_source_counts.get(repair_branch_source, 0)) + 1
            language_branch_counts[language_branch] = int(language_branch_counts.get(language_branch, 0)) + 1
            effective_plan_enforcements[plan_enforcement] = int(effective_plan_enforcements.get(plan_enforcement, 0)) + 1
            java_survivor_backoff_counts[backoff_mode] = int(java_survivor_backoff_counts.get(backoff_mode, 0)) + 1
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "argv": list(getattr(args, "_argv", sys.argv)),
        "dataset": str(dataset_tag),
        "resolved_prompts_file": str(prompts_path),
        "out_file": str(out_path),
        "metrics_out": str(metrics_path) if metrics_path else None,
        "repair_branch_requested": str(requested_repair_branch),
        "repair_branch_effective": _single_count_value(repair_branch_counts, effective_manifest_branch),
        "repair_branch_source": _single_count_value(
            repair_branch_source_counts,
            str(branch_meta.get("source") or "auto"),
        ),
        "repair_branch": _single_count_value(repair_branch_counts, effective_manifest_branch),
        "plan_enforcement": str(effective_plan),
        "plan_target_match": str(getattr(args, "plan_target_match", "overlap") or "overlap"),
        "min_survivors_per_bug": int(getattr(args, "min_survivors_per_bug", 3) or 3),
        "fill_with_violations_penalty": float(getattr(args, "fill_with_violations_penalty", 25.0) or 25.0),
        "backend_selected": str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto"),
        "hf_model_name": str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or ""),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "seeds": str(getattr(args, "seeds", "") or ""),
        "max_per_bug": int(args.max_per_bug),
        "final_top_k": int(args.final_top_k),
        "limit_bugs": int(getattr(args, "limit_bugs", 0) or 0),
        "scheduler": str(getattr(args, "scheduler", "fixed") or "fixed"),
        "java_filter_compile_valid": bool(getattr(args, "java_filter_compile_valid", False)),
        "java_rerank_compile_valid": bool(getattr(args, "java_rerank_compile_valid", False)),
        "java_compile_validation_mode": str(resolve_java_compile_validation_mode(args)),
        "java_candidate_compile_check": bool(getattr(args, "java_candidate_compile_check", False)),
        "java_semantic_rerank_enabled": bool(effective_manifest_branch == "java_semantic"),
        "java_semantic_rerank_features": list(JAVA_SEMANTIC_RERANK_FEATURES),
        "java_branch_no_fill": bool(
            effective_manifest_branch in JAVA_REPAIR_BRANCHES
            and effective_plan != "filter_then_fill"
            and java_survivor_backoff_mode == "off"
        ),
        "java_survivor_backoff_mode": str(java_survivor_backoff_mode),
        "java_compile_feedback_once": bool(getattr(args, "java_compile_feedback_once", False)),
        "java_compile_feedback_available": bool(resolve_java_compile_feedback_available(args, "java")),
        "repair_branch_counts": repair_branch_counts,
        "repair_branch_source_counts": repair_branch_source_counts,
        "language_branch_counts": language_branch_counts,
        "effective_plan_enforcement_counts": effective_plan_enforcements,
        "java_survivor_backoff_mode_counts": java_survivor_backoff_counts,
        "git_commit": try_get_git_commit(),
    }
    manifest_path = derive_manifest_path(out_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_inputs_from_prompts(prom_dict: Dict[str, Any]) -> Dict[str, Any]:
    inputs_dict: Dict[str, Any] = {}
    for bug_id, row in prom_dict.items():
        if not isinstance(row, dict):
            continue
        base_code = safe_get(row, ["function", "function_before"]) or row.get("code") or ""
        inputs_dict[bug_id] = {
            "code": base_code,
            "file_path": (
                row.get("file_path")
                or safe_get(row, ["file", "file_path"])
                or safe_get(row, ["function", "file_path"])
            ),
            "function_name": row.get("function_name") or safe_get(row, ["function", "function_name"]),
            "candidates": [],
        }
    return inputs_dict


def limit_bug_rows(rows: Dict[str, Any], limit_bugs: int) -> Dict[str, Any]:
    if int(limit_bugs or 0) <= 0:
        return rows
    items = list((rows or {}).items())[: int(limit_bugs)]
    return {bug_id: row for bug_id, row in items}


def set_seed(seed: int):
    random.seed(seed)
    if torch is None:
        return
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass




# ===== Multi-run schedule helpers (seed + sampling profiles) =====

def _parse_kv_config(s: str) -> dict:
    """Parse a comma-separated key=value string into a dict."""
    out = {}
    for part in (p.strip() for p in (s or '').split(',')):
        if not part:
            continue
        if '=' not in part:
            # allow bare tag name
            out.setdefault('tag', part)
            continue
        k, v = part.split('=', 1)
        out[k.strip().lower()] = v.strip()
    return out


def parse_run_config(cfg: str, default_tag: str) -> Dict[str, Any]:
    """Parse --run_config entry.

    Example:
      --run_config tag=A,seed=42,temp=0.7,top_p=0.95,presence_penalty=0.0

    Supported keys:
      tag/name, seed, temp/temperature, top_p, top_k,
      presence_penalty/presence, frequency_penalty/frequency
    """
    kv = _parse_kv_config(cfg)
    tag = kv.get('tag') or kv.get('name') or default_tag

    def _f(key, default):
        if key not in kv:
            return default
        try:
            return float(kv[key])
        except Exception:
            return default

    def _i(key, default):
        if key not in kv:
            return default
        try:
            return int(float(kv[key]))
        except Exception:
            return default

    seed = _i('seed', None)
    temp = _f('temp', _f('temperature', None))
    top_p = _f('top_p', None)
    top_k = _i('top_k', None)
    pres = _f('presence_penalty', _f('presence', None))
    freq = _f('frequency_penalty', _f('frequency', None))

    return {
        'tag': str(tag),
        'seed': seed,
        'temperature': temp,
        'top_p': top_p,
        'top_k': top_k,
        'presence_penalty': pres,
        'frequency_penalty': freq,
    }


def build_run_specs(args: Namespace) -> List[Dict[str, Any]]:
    """Build multi-run specs.

    Priority:
      1) --run_config (repeatable)
      2) --multi_run_preset
      3) --seeds/--seed (multi-seed, shared sampling params)
    """
    specs: List[Dict[str, Any]] = []

    run_cfgs = list(getattr(args, 'run_config', []) or [])
    preset = str(getattr(args, 'multi_run_preset', '') or '').strip()

    if run_cfgs:
        for i, cfg in enumerate(run_cfgs, start=1):
            spec = parse_run_config(cfg, default_tag=f'run{i}')
            if spec.get('seed') is None:
                spec['seed'] = int(getattr(args, 'seed', 42))
            spec['temperature'] = float(spec.get('temperature') if spec.get('temperature') is not None else getattr(args, 'temperature', 0.8))
            spec['top_p'] = float(spec.get('top_p') if spec.get('top_p') is not None else getattr(args, 'top_p', 0.9))
            spec['presence_penalty'] = float(spec.get('presence_penalty') if spec.get('presence_penalty') is not None else float(getattr(args, 'presence_penalty', 0.0) or 0.0))
            spec['frequency_penalty'] = float(spec.get('frequency_penalty') if spec.get('frequency_penalty') is not None else float(getattr(args, 'frequency_penalty', 0.0) or 0.0))
            if spec.get('top_k') is None:
                spec['top_k'] = int(getattr(args, 'top_k', 0) or 0)
            specs.append(spec)
        return specs

    if preset == '3run_default':
        base_seed = int(getattr(args, 'seed', 42))
        presets = [
            {'tag': 'A', 'seed': base_seed + 0, 'temperature': 0.70, 'top_p': 0.95, 'presence_penalty': 0.0, 'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0), 'top_k': int(getattr(args, 'top_k', 0) or 0)},
            {'tag': 'B', 'seed': base_seed + 1, 'temperature': 0.90, 'top_p': 0.90, 'presence_penalty': 0.2, 'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0), 'top_k': int(getattr(args, 'top_k', 0) or 0)},
            {'tag': 'C', 'seed': base_seed + 2, 'temperature': 1.05, 'top_p': 0.85, 'presence_penalty': 0.4, 'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0), 'top_k': int(getattr(args, 'top_k', 0) or 0)},
        ]
        return presets

    seeds_list = [int(s.strip()) for s in str(getattr(args, 'seeds', '') or '').split(',') if s.strip()]
    if not seeds_list:
        seeds_list = [int(getattr(args, 'seed', 42))]

    for s in seeds_list:
        specs.append({
            'tag': f'seed{s}',
            'seed': int(s),
            'temperature': float(getattr(args, 'temperature', 0.8)),
            'top_p': float(getattr(args, 'top_p', 0.9)),
            'presence_penalty': float(getattr(args, 'presence_penalty', 0.0) or 0.0),
            'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0),
            'top_k': int(getattr(args, 'top_k', 0) or 0),
        })
    return specs


def compute_retrieval_confidence(prom_row: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
    top_items = safe_get(prom_row, ["bm25", "top"]) or []
    top1 = 0.0
    top2 = 0.0
    if isinstance(top_items, list) and top_items:
        if len(top_items) >= 1 and isinstance(top_items[0], dict):
            top1 = _safe_float(top_items[0].get("score_reranked"), 0.0)
        if len(top_items) >= 2 and isinstance(top_items[1], dict):
            top2 = _safe_float(top_items[1].get("score_reranked"), 0.0)
    margin = float(top1 - top2)
    if margin >= float(getattr(args, "conf_margin_high", 0.5)):
        conf_level = "high"
    elif margin >= float(getattr(args, "conf_margin_med", 0.2)):
        conf_level = "med"
    else:
        conf_level = "low"
    return {
        "top1": float(top1),
        "top2": float(top2),
        "margin": float(margin),
        "conf_level": conf_level,
    }


def resolve_bug_profile(prom_row: Dict[str, Any], args: Namespace, target: int) -> Dict[str, Any]:
    scheduler = str(getattr(args, "scheduler", "fixed") or "fixed")
    retrieval = compute_retrieval_confidence(prom_row or {}, args)
    repair_mode = str((prom_row or {}).get("repair_mode") or (prom_row or {}).get("hard_tag") or "").strip().lower()
    if scheduler != "cost_aware":
        profile = "HARD" if repair_mode == "hard" else "MED"
        return {
            "scheduler": scheduler,
            "repair_mode": repair_mode,
            "target_local": int(max(1, target)),
            "profile": profile,
            "conf_level": retrieval["conf_level"],
            "bm25_top1": retrieval["top1"],
            "bm25_margin": retrieval["margin"],
        }

    hard_target = int(getattr(args, "hard_target", None) if getattr(args, "hard_target", None) is not None else target)
    if repair_mode == "hard" or retrieval["conf_level"] == "low":
        profile = "HARD"
        target_local = min(int(target), max(1, hard_target))
    elif retrieval["conf_level"] == "high":
        profile = "EASY"
        target_local = min(int(target), max(1, int(getattr(args, "easy_target", 12))))
    else:
        profile = "MED"
        target_local = min(int(target), max(1, int(getattr(args, "med_target", 30))))

    return {
        "scheduler": scheduler,
        "repair_mode": repair_mode,
        "target_local": int(max(1, target_local)),
        "profile": profile,
        "conf_level": retrieval["conf_level"],
        "bm25_top1": retrieval["top1"],
        "bm25_margin": retrieval["margin"],
    }


def _score_of(c: Dict[str, Any]) -> float:
    try:
        return float(c.get('score', -10**9) or -10**9)
    except Exception:
        return -10**9


def select_topk_with_coverage(
    candidates: List[Dict[str, Any]],
    k: int,
    coverage_key: str = 'source_run',
    keep_best_greedy: bool = True,
) -> List[Dict[str, Any]]:
    """Select top-k with coverage across groups.

    - Optionally keep the best greedy candidate at slot0.
    - Then pick one candidate per coverage group (e.g., source_run) by score.
    - Fill remaining slots by score.
    """
    k = max(1, int(k))
    if not candidates:
        return []

    ordered = _sorted_candidates_by_priority(candidates, score_key="score")
    ordered.sort(key=lambda c: 0 if str(c.get('gen_mode', '')) == 'greedy' else 1)

    chosen: List[Dict[str, Any]] = []

    rest = ordered
    if keep_best_greedy:
        greedy = [c for c in ordered if str(c.get('gen_mode', '')) == 'greedy']
        if greedy:
            best_g = _sorted_candidates_by_priority(greedy, score_key="score")[0]
            chosen.append(best_g)
            rest = [c for c in ordered if c is not best_g and str(c.get('gen_mode', '')) != 'greedy']
        else:
            rest = [c for c in ordered if str(c.get('gen_mode', '')) != 'greedy']

    if len(chosen) >= k:
        return chosen[:k]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in rest:
        g = c.get(coverage_key)
        g = 'NA' if g is None else str(g)
        groups.setdefault(g, []).append(c)

    for gk in list(groups.keys()):
        groups[gk] = _sorted_candidates_by_priority(groups[gk], score_key="score")

    group_keys = sorted(
        groups.keys(),
        key=cmp_to_key(lambda left, right: _candidate_priority_compare(groups[left][0], groups[right][0], score_key="score")),
    )
    for gk in group_keys:
        if len(chosen) >= k:
            break
        if groups[gk]:
            chosen.append(groups[gk].pop(0))

    remaining: List[Dict[str, Any]] = []
    for lst in groups.values():
        remaining.extend(lst)
    remaining = _sorted_candidates_by_priority(remaining, score_key="score")
    for c in remaining:
        if len(chosen) >= k:
            break
        chosen.append(c)

    return chosen[:k]
def merge_patch_results(
    input_paths: List[Path],
    prom_dict: Dict[str, Any],
    max_per_bug: int,
    rerank: bool = True,
    args: Optional[Namespace] = None,
) -> Dict[str, Any]:
    """
    Merge multiple patch result JSONs (e.g., multi-seed runs), re-score, dedup, and keep top-K per bug.
    """
    merge_args = args or Namespace(
        java_filter_compile_valid=False,
        java_rerank_compile_valid=False,
        java_candidate_compile_check=False,
        no_rerank=(not rerank),
    )
    merged: Dict[str, Any] = {}
    seen_hashes: Dict[str, set] = {}

    for path in input_paths:
        if not path.exists():
            logging.warning(f"[merge] Missing file: {path}")
            continue
        logging.info(f"[merge] Loading {path}")
        data = load_json(path)
        for bug_id, row in data.items():
            prom_row = prom_dict.get(bug_id) or {}
            base_code = safe_get(prom_row, ["function", "function_before"]) or row.get("code") or ""
            fn_name = row.get("function_name") or safe_get(prom_row, ["function", "function_name"])
            file_path = row.get("file_path") or prom_row.get("file_path")
            language = infer_language(
                {"file_path": file_path, "function_name": fn_name},
                prom_row,
                base_code,
            )
            repair_branch_meta = resolve_repair_branch_metadata(merge_args, row, prom_row, language)
            repair_branch = str(repair_branch_meta.get("effective") or resolve_effective_repair_branch(merge_args, row, prom_row, language))
            language_branch = str(row.get("language_branch") or language_branch_for_language(language))

            base_node = _parse_single_toplevel_def_or_class(base_code)
            expected_name = (str(fn_name or "")).strip() or (getattr(base_node, "name", "") if base_node is not None else "")
            if (not expected_name) and language == "java":
                expected_name = extract_decl_name(base_code, language="java")
            java_defaults = init_java_validity_stats(merge_args, language)
            java_defaults["java_semantic_rerank_enabled"] = bool(java_semantic_rerank_enabled(repair_branch, language))
            java_defaults["java_survivor_backoff_mode"] = str(
                resolve_effective_java_survivor_backoff(
                    merge_args,
                    repair_branch,
                    resolve_effective_plan_enforcement(merge_args, repair_branch),
                )
            )
            java_defaults["java_branch_no_fill"] = bool(
                language == "java"
                and resolve_effective_plan_enforcement(merge_args, repair_branch) != "filter_then_fill"
                and java_defaults["java_survivor_backoff_mode"] == "off"
            )
            java_defaults["java_compile_feedback_once"] = bool(java_compile_feedback_requested(merge_args, language, repair_branch))
            java_defaults["java_compile_feedback_available"] = bool(resolve_java_compile_feedback_available(merge_args, language))
            java_requested_mode = str(java_defaults.get("java_compile_validation_mode") or "none")
            java_filter_active = bool(java_defaults.get("java_filter_compile_valid", False))
            java_rerank_active = bool(java_defaults.get("java_rerank_compile_valid", False) and rerank)
            java_semantic_active = bool(java_defaults.get("java_semantic_rerank_enabled", False) and rerank)

            if bug_id not in merged:
                merged[bug_id] = {
                    "code": base_code,
                    "file_path": file_path,
                    "function_name": fn_name,
                    "candidates": [],
                    "language": str(row.get("language") or language),
                    "language_branch": str(language_branch),
                    "repair_branch_requested": str(
                        row.get("repair_branch_requested")
                        or repair_branch_meta.get("requested")
                        or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
                    ),
                    "repair_branch_effective": str(
                        row.get("repair_branch_effective")
                        or row.get("repair_branch")
                        or repair_branch
                    ),
                    "repair_branch_source": str(
                        row.get("repair_branch_source")
                        or repair_branch_meta.get("source")
                        or "auto"
                    ),
                    "repair_branch": str(repair_branch),
                    "allowed_edit_types": list(row.get("allowed_edit_types") or ["REPLACE"]),
                    "drop_counts": {k: 0 for k in DROP_REASON_KEYS},
                    "target_candidates": int(row.get("target_candidates") or max_per_bug),
                    "valid_candidates": 0,
                    "fill_rate": 0.0,
                    "llm_calls_total": 0,
                    "elapsed_seconds": 0.0,
                    "gen_n": int(row.get("gen_n") or 1),
                    "max_gen_passes": int(row.get("max_gen_passes") or 1),
                    "temperature_used": [],
                    "top_p_used": [],
                    "unique_after_stage": {},
                    "scheduler": str(row.get("scheduler") or "fixed"),
                    "conf_level": str(row.get("conf_level") or "low"),
                    "bm25_top1": float(row.get("bm25_top1") or 0.0),
                    "bm25_margin": float(row.get("bm25_margin") or 0.0),
                    "profile": str(row.get("profile") or "MED"),
                    "profiles_tried": list(row.get("profiles_tried") or []),
                    "plan_source": str(row.get("plan_source") or "fallback_default"),
                    "plan_enforcement": str(row.get("plan_enforcement") or "filter_then_fill"),
                    "plan_target_match": str(row.get("plan_target_match") or "overlap"),
                    "plan_fill_used": bool(row.get("plan_fill_used", False)),
                    "plan_fill_added_n": int(row.get("plan_fill_added_n") or 0),
                    "plan_fill_penalty": float(row.get("plan_fill_penalty") or 0.0),
                    "plan_violation_breakdown": {k: 0 for k in PLAN_VIOLATION_TYPE_KEYS},
                    "java_filter_compile_valid": bool(java_defaults.get("java_filter_compile_valid", False)),
                    "java_rerank_compile_valid": bool(java_defaults.get("java_rerank_compile_valid", False)),
                    "java_candidate_compile_check": bool(java_defaults.get("java_candidate_compile_check", False)),
                    "java_compile_validation_mode": str(java_defaults.get("java_compile_validation_mode") or "none"),
                    "java_semantic_rerank_enabled": bool(java_defaults.get("java_semantic_rerank_enabled", False)),
                    "java_semantic_rerank_features": list(java_defaults.get("java_semantic_rerank_features") or []),
                    "java_branch_no_fill": bool(java_defaults.get("java_branch_no_fill", False)),
                    "java_survivor_backoff_mode": str(java_defaults.get("java_survivor_backoff_mode") or "off"),
                    "java_survivor_backoff_used": bool(row.get("java_survivor_backoff_used", False)),
                    "java_survivor_backoff_added_count": int(row.get("java_survivor_backoff_added_count") or 0),
                    "zero_candidate_after_filter": bool(row.get("zero_candidate_after_filter", False)),
                    "java_sentinel_extract_hit": int(row.get("java_sentinel_extract_hit") or 0),
                    "java_decl_salvage_used": int(row.get("java_decl_salvage_used") or 0),
                    "java_explanation_tail_stripped": int(row.get("java_explanation_tail_stripped") or 0),
                    "java_decl_recovery_used": int(row.get("java_decl_recovery_used") or 0),
                    "java_decl_recovery_mode_counts": dict(row.get("java_decl_recovery_mode_counts") or {}),
                    "java_compile_feedback_once": bool(java_defaults.get("java_compile_feedback_once", False)),
                    "java_compile_feedback_available": bool(java_defaults.get("java_compile_feedback_available", False)),
                    "java_compile_feedback_attempted_n": 0,
                    "java_compile_feedback_applied_n": 0,
                    "java_validity_checked_n": 0,
                    "java_validity_valid_n": 0,
                    "java_validity_invalid_n": 0,
                    "java_validity_rejected_n": 0,
                    "java_validity_reranked_n": 0,
                    "java_validity_reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
                    "backend_selected": str(row.get("backend_selected") or "auto"),
                    "hf_model_name": str(row.get("hf_model_name") or ""),
                }
                seen_hashes[bug_id] = set()
            merged[bug_id]["allowed_edit_types"] = list(row.get("allowed_edit_types") or merged[bug_id].get("allowed_edit_types") or ["REPLACE"])
            merged[bug_id]["target_candidates"] = max(
                int(merged[bug_id].get("target_candidates") or 0),
                int(row.get("target_candidates") or max_per_bug),
            )
            merged[bug_id]["llm_calls_total"] = int(merged[bug_id].get("llm_calls_total") or 0) + int(row.get("llm_calls_total") or 0)
            merged[bug_id]["elapsed_seconds"] = float(merged[bug_id].get("elapsed_seconds") or 0.0) + float(row.get("elapsed_seconds") or 0.0)
            merged[bug_id]["gen_n"] = int(max(int(merged[bug_id].get("gen_n") or 1), int(row.get("gen_n") or 1)))
            merged[bug_id]["max_gen_passes"] = int(max(int(merged[bug_id].get("max_gen_passes") or 1), int(row.get("max_gen_passes") or 1)))
            merged[bug_id]["temperature_used"] = list(merged[bug_id].get("temperature_used") or []) + list(row.get("temperature_used") or [])
            merged[bug_id]["top_p_used"] = list(merged[bug_id].get("top_p_used") or []) + list(row.get("top_p_used") or [])
            merged[bug_id]["scheduler"] = str(row.get("scheduler") or merged[bug_id].get("scheduler") or "fixed")
            merged[bug_id]["conf_level"] = str(row.get("conf_level") or merged[bug_id].get("conf_level") or "low")
            merged[bug_id]["language"] = str(row.get("language") or merged[bug_id].get("language") or language)
            merged[bug_id]["language_branch"] = str(row.get("language_branch") or merged[bug_id].get("language_branch") or language_branch)
            merged[bug_id]["repair_branch_requested"] = str(
                row.get("repair_branch_requested")
                or merged[bug_id].get("repair_branch_requested")
                or repair_branch_meta.get("requested")
                or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
            )
            merged[bug_id]["repair_branch_effective"] = str(
                row.get("repair_branch_effective")
                or row.get("repair_branch")
                or merged[bug_id].get("repair_branch_effective")
                or repair_branch
            )
            merged[bug_id]["repair_branch_source"] = str(
                row.get("repair_branch_source")
                or merged[bug_id].get("repair_branch_source")
                or repair_branch_meta.get("source")
                or "auto"
            )
            merged[bug_id]["repair_branch"] = str(row.get("repair_branch") or merged[bug_id].get("repair_branch") or repair_branch)
            merged[bug_id]["bm25_top1"] = max(float(merged[bug_id].get("bm25_top1") or 0.0), float(row.get("bm25_top1") or 0.0))
            merged[bug_id]["bm25_margin"] = max(float(merged[bug_id].get("bm25_margin") or 0.0), float(row.get("bm25_margin") or 0.0))
            merged[bug_id]["profile"] = str(row.get("profile") or merged[bug_id].get("profile") or "MED")
            merged[bug_id]["plan_source"] = str(row.get("plan_source") or merged[bug_id].get("plan_source") or "fallback_default")
            merged[bug_id]["plan_enforcement"] = str(
                row.get("plan_enforcement")
                or merged[bug_id].get("plan_enforcement")
                or resolve_effective_plan_enforcement(merge_args, repair_branch)
            )
            merged[bug_id]["plan_target_match"] = str(row.get("plan_target_match") or merged[bug_id].get("plan_target_match") or "overlap")
            merged[bug_id]["plan_fill_used"] = bool(merged[bug_id].get("plan_fill_used", False) or row.get("plan_fill_used", False))
            merged[bug_id]["plan_fill_added_n"] = int(merged[bug_id].get("plan_fill_added_n") or 0) + int(row.get("plan_fill_added_n") or 0)
            merged[bug_id]["plan_fill_penalty"] = max(float(merged[bug_id].get("plan_fill_penalty") or 0.0), float(row.get("plan_fill_penalty") or 0.0))
            merged[bug_id]["backend_selected"] = str(row.get("backend_selected") or merged[bug_id].get("backend_selected") or "auto")
            merged[bug_id]["hf_model_name"] = str(row.get("hf_model_name") or merged[bug_id].get("hf_model_name") or "")
            merged[bug_id]["java_semantic_rerank_enabled"] = bool(
                merged[bug_id].get("java_semantic_rerank_enabled", False)
                or row.get("java_semantic_rerank_enabled", False)
                or java_defaults.get("java_semantic_rerank_enabled", False)
            )
            merged[bug_id]["java_semantic_rerank_features"] = list(
                row.get("java_semantic_rerank_features")
                or merged[bug_id].get("java_semantic_rerank_features")
                or list(JAVA_SEMANTIC_RERANK_FEATURES)
            )
            merged[bug_id]["java_branch_no_fill"] = bool(
                merged[bug_id].get("java_branch_no_fill", False)
                or row.get("java_branch_no_fill", False)
                or java_defaults.get("java_branch_no_fill", False)
            )
            merged[bug_id]["java_survivor_backoff_mode"] = str(
                row.get("java_survivor_backoff_mode")
                or merged[bug_id].get("java_survivor_backoff_mode")
                or java_defaults.get("java_survivor_backoff_mode")
                or "off"
            )
            merged[bug_id]["java_survivor_backoff_used"] = bool(
                merged[bug_id].get("java_survivor_backoff_used", False)
                or row.get("java_survivor_backoff_used", False)
            )
            merged[bug_id]["java_survivor_backoff_added_count"] = int(
                merged[bug_id].get("java_survivor_backoff_added_count") or 0
            ) + int(row.get("java_survivor_backoff_added_count") or 0)
            merged[bug_id]["zero_candidate_after_filter"] = bool(
                merged[bug_id].get("zero_candidate_after_filter", False)
                or row.get("zero_candidate_after_filter", False)
            )
            merged[bug_id]["java_sentinel_extract_hit"] = int(
                merged[bug_id].get("java_sentinel_extract_hit", 0)
            ) + int(row.get("java_sentinel_extract_hit") or 0)
            merged[bug_id]["java_decl_salvage_used"] = int(
                merged[bug_id].get("java_decl_salvage_used", 0)
            ) + int(row.get("java_decl_salvage_used") or 0)
            merged[bug_id]["java_explanation_tail_stripped"] = int(
                merged[bug_id].get("java_explanation_tail_stripped", 0)
            ) + int(row.get("java_explanation_tail_stripped") or 0)
            merged[bug_id]["java_decl_recovery_used"] = int(
                merged[bug_id].get("java_decl_recovery_used", 0)
            ) + int(row.get("java_decl_recovery_used") or 0)
            merged[bug_id]["java_decl_recovery_mode_counts"] = merge_counter_dicts(
                merged[bug_id].get("java_decl_recovery_mode_counts") or {},
                row.get("java_decl_recovery_mode_counts") or {},
                list(set(list((merged[bug_id].get("java_decl_recovery_mode_counts") or {}).keys()) + list((row.get("java_decl_recovery_mode_counts") or {}).keys()))),
            )
            merged[bug_id]["java_compile_feedback_once"] = bool(
                merged[bug_id].get("java_compile_feedback_once", False)
                or row.get("java_compile_feedback_once", False)
                or java_defaults.get("java_compile_feedback_once", False)
            )
            merged[bug_id]["java_compile_feedback_available"] = bool(
                merged[bug_id].get("java_compile_feedback_available", False)
                or row.get("java_compile_feedback_available", False)
                or java_defaults.get("java_compile_feedback_available", False)
            )
            merged[bug_id]["java_compile_feedback_attempted_n"] = int(
                merged[bug_id].get("java_compile_feedback_attempted_n", 0)
            ) + int(row.get("java_compile_feedback_attempted_n") or 0)
            merged[bug_id]["java_compile_feedback_applied_n"] = int(
                merged[bug_id].get("java_compile_feedback_applied_n", 0)
            ) + int(row.get("java_compile_feedback_applied_n") or 0)
            merged[bug_id]["plan_violation_breakdown"] = merge_counter_dicts(
                merged[bug_id].get("plan_violation_breakdown") or {},
                row.get("plan_violation_breakdown") or {},
                PLAN_VIOLATION_TYPE_KEYS,
            )
            merged_profiles = list(merged[bug_id].get("profiles_tried") or [])
            for tag in list(row.get("profiles_tried") or []):
                tag_s = str(tag)
                if tag_s not in merged_profiles:
                    merged_profiles.append(tag_s)
            merged[bug_id]["profiles_tried"] = merged_profiles
            in_stage = row.get("unique_after_stage") or {}
            if isinstance(in_stage, dict):
                for stage_k, stage_v in in_stage.items():
                    k = str(stage_k)
                    try:
                        v = int(stage_v)
                    except Exception:
                        continue
                    prev = int((merged[bug_id].get("unique_after_stage") or {}).get(k, 0))
                    merged[bug_id]["unique_after_stage"][k] = max(prev, v)
            in_drop = row.get("drop_counts") or {}
            if isinstance(in_drop, dict):
                for k in DROP_REASON_KEYS:
                    merged[bug_id]["drop_counts"][k] = int(merged[bug_id]["drop_counts"].get(k, 0)) + int(in_drop.get(k, 0) or 0)

            ctx_row = {"function_name": fn_name, "file_path": file_path, "function": {"function_name": fn_name}}

            for cand_idx, cand in enumerate(row.get("candidates", []), start=1):
                code = cand.get("code") or ""
                if not code.strip():
                    continue
                h = hashlib.sha1(canon(code).encode("utf-8")).hexdigest()
                if h in seen_hashes[bug_id]:
                    continue
                ok, err = ast_ok(code, language=language)
                if not ok:
                    continue
                is_valid, _reason = validate_candidate_code(
                    code=code,
                    base_code=base_code,
                    expected_name=expected_name,
                    language=language,
                )
                if not is_valid:
                    continue
                java_validation = {
                    "checked": False,
                    "mode": str(cand.get("java_compile_validation_mode") or "none"),
                    "valid": cand.get("java_compile_valid"),
                    "reasons": list(cand.get("java_compile_validation_reasons") or []),
                    "hard_reasons": list(cand.get("java_compile_validation_hard_reasons") or []),
                    "context_drift_summary": str(cand.get("java_context_drift_summary") or "none"),
                    "context_allowlist_summary": dict(cand.get("java_context_allowlist_summary") or {}),
                    "declared_names_in_patch": list(cand.get("java_declared_names_in_patch") or []),
                    "context_supported_new_identifiers": list(cand.get("java_context_supported_new_identifiers") or []),
                    "context_supported_new_types": list(cand.get("java_context_supported_new_types") or []),
                    "context_supported_new_constants": list(cand.get("java_context_supported_new_constants") or []),
                    "context_supported_new_identifier_sources": dict(cand.get("java_context_supported_new_identifier_sources") or {}),
                    "context_supported_new_type_sources": dict(cand.get("java_context_supported_new_type_sources") or {}),
                    "context_supported_new_constant_sources": dict(cand.get("java_context_supported_new_constant_sources") or {}),
                    "new_helper_calls": list(cand.get("java_new_helper_calls") or []),
                    "new_type_references": list(cand.get("java_new_type_references") or []),
                    "new_constant_references": list(cand.get("java_new_constant_references") or []),
                    "new_identifier_references": list(cand.get("java_new_identifier_references") or []),
                    "broad_rewrite_fraction": float(cand.get("java_broad_rewrite_fraction", 0.0) or 0.0),
                    "risk_combo_triggered": bool(cand.get("java_compile_risk_combo_triggered", False)),
                    "risk_combo_details": list(cand.get("java_compile_risk_combo_details") or []),
                    "reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
                }
                if language == "java" and java_requested_mode != "none":
                    java_validation = assess_java_compile_validity(
                        base_code=base_code,
                        code=code,
                        expected_name=expected_name,
                        requested_mode=java_requested_mode,
                        prom_row=prom_row,
                        plan_ok=bool(cand.get("plan_ok", True)),
                        filter_enabled=bool(java_filter_active),
                        hard_reject_compile_risk_combos=bool(
                            getattr(merge_args, "java_hard_reject_compile_risk_combos", False)
                        ),
                    )
                    if java_validation.get("valid") is False and java_filter_active:
                        update_java_validity_stats(merged[bug_id], java_validation, rejected=True)
                        bump_drop(merged[bug_id]["drop_counts"], "java_compile_invalid")
                        continue
                base_score = score_candidate(
                    ctx_row,
                    prom_row,
                    base_code,
                    code,
                    ok,
                    language=language,
                )
                score, java_reranked = apply_java_validity_rerank(
                    float(base_score),
                    java_validation,
                    rerank_enabled=java_rerank_active,
                )
                score, java_semantic_record = apply_java_semantic_rerank(
                    score=score,
                    context_row=ctx_row,
                    prom_row=prom_row,
                    base_code=base_code,
                    code=code,
                    repair_branch=repair_branch,
                    language=language,
                    edit_summary=dict(cand.get("edit_summary") or {}),
                    plan_ok=bool(cand.get("plan_ok", True)),
                    plan_violations=list(cand.get("plan_violations") or []),
                    plan_fill_selected=bool(cand.get("plan_fill_selected", False)),
                )
                update_java_validity_stats(merged[bug_id], java_validation, reranked=java_reranked)
                src_run = str(cand.get('source_run') or path.stem)
                src_seed = cand.get('source_seed')
                java_candidate_meta = build_java_candidate_metadata(
                    cand=cand,
                    validation_record=java_validation,
                    semantic_record=java_semantic_record,
                    java_semantic_enabled=java_semantic_active,
                )
                merged[bug_id]["candidates"].append({
                    **{k: v for k, v in cand.items() if k not in ("ast", "score", "base_score")},
                    'source_run': src_run,
                    'source_seed': src_seed,
                    'source_rank_before_merge': int(cand.get("source_rank_before_merge") or cand.get("rank") or cand_idx),
                    "code": code,
                    "candidate_code_hash": str(cand.get("candidate_code_hash") or h),
                    "normalized_candidate_hash": str(cand.get("normalized_candidate_hash") or compute_normalized_candidate_hash(code)),
                    "ast": {"ok": ok, "err": err},
                    "base_score": float(base_score),
                    "score": float(score),
                    "repair_branch_requested": str(
                        cand.get("repair_branch_requested")
                        or repair_branch_meta.get("requested")
                        or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
                    ),
                    "repair_branch_effective": str(cand.get("repair_branch_effective") or repair_branch),
                    "repair_branch_source": str(cand.get("repair_branch_source") or repair_branch_meta.get("source") or "auto"),
                    "repair_branch": str(repair_branch),
                    "language_branch": str(cand.get("language_branch") or language_branch),
                    **java_candidate_meta,
                })
                seen_hashes[bug_id].add(h)

    for bug_id, row in merged.items():
        if rerank:
            row["candidates"].sort(key=lambda c: (0 if str(c.get("gen_mode", "")) == "greedy" else 1, -float(c.get("score", -10**9) or -10**9)))
        row["candidates"] = row["candidates"][: max(1, int(max_per_bug))]
        row["valid_candidates"] = int(len(row["candidates"]))
        target = int(row.get("target_candidates") or max_per_bug)
        row["fill_rate"] = round(float(row["valid_candidates"]) / float(max(1, target)), 4)

    return merged


def merge_results_dicts(
    results_list: List[Dict[str, Any]],
    prom_dict: Dict[str, Any],
    max_per_bug: int,
    rerank: bool = True,
    args: Optional[Namespace] = None,
) -> Dict[str, Any]:
    """
    Merge multiple in-memory patch result dicts (e.g., multi-seed in one run), re-score, dedup, and keep top-K per bug.
    """
    merge_args = args or Namespace(
        java_filter_compile_valid=False,
        java_rerank_compile_valid=False,
        java_candidate_compile_check=False,
        no_rerank=(not rerank),
    )
    merged: Dict[str, Any] = {}
    seen_hashes: Dict[str, set] = {}

    for data in results_list:
        for bug_id, row in data.items():
            prom_row = prom_dict.get(bug_id) or {}
            base_code = safe_get(prom_row, ["function", "function_before"]) or row.get("code") or ""
            fn_name = row.get("function_name") or safe_get(prom_row, ["function", "function_name"])
            file_path = row.get("file_path") or prom_row.get("file_path")
            language = infer_language(
                {"file_path": file_path, "function_name": fn_name},
                prom_row,
                base_code,
            )
            repair_branch_meta = resolve_repair_branch_metadata(merge_args, row, prom_row, language)
            repair_branch = str(repair_branch_meta.get("effective") or resolve_effective_repair_branch(merge_args, row, prom_row, language))
            language_branch = str(row.get("language_branch") or language_branch_for_language(language))

            base_node = _parse_single_toplevel_def_or_class(base_code)
            expected_name = (str(fn_name or "")).strip() or (getattr(base_node, "name", "") if base_node is not None else "")
            if (not expected_name) and language == "java":
                expected_name = extract_decl_name(base_code, language="java")
            java_defaults = init_java_validity_stats(merge_args, language)
            java_defaults["java_semantic_rerank_enabled"] = bool(java_semantic_rerank_enabled(repair_branch, language))
            java_defaults["java_survivor_backoff_mode"] = str(
                resolve_effective_java_survivor_backoff(
                    merge_args,
                    repair_branch,
                    resolve_effective_plan_enforcement(merge_args, repair_branch),
                )
            )
            java_defaults["java_branch_no_fill"] = bool(
                language == "java"
                and resolve_effective_plan_enforcement(merge_args, repair_branch) != "filter_then_fill"
                and java_defaults["java_survivor_backoff_mode"] == "off"
            )
            java_defaults["java_compile_feedback_once"] = bool(java_compile_feedback_requested(merge_args, language, repair_branch))
            java_defaults["java_compile_feedback_available"] = bool(resolve_java_compile_feedback_available(merge_args, language))
            java_requested_mode = str(java_defaults.get("java_compile_validation_mode") or "none")
            java_filter_active = bool(java_defaults.get("java_filter_compile_valid", False))
            java_rerank_active = bool(java_defaults.get("java_rerank_compile_valid", False) and rerank)
            java_semantic_active = bool(java_defaults.get("java_semantic_rerank_enabled", False) and rerank)

            if bug_id not in merged:
                merged[bug_id] = {
                    "code": base_code,
                    "file_path": file_path,
                    "function_name": fn_name,
                    "candidates": [],
                    "language": str(row.get("language") or language),
                    "language_branch": str(language_branch),
                    "repair_branch_requested": str(
                        row.get("repair_branch_requested")
                        or repair_branch_meta.get("requested")
                        or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
                    ),
                    "repair_branch_effective": str(
                        row.get("repair_branch_effective")
                        or row.get("repair_branch")
                        or repair_branch
                    ),
                    "repair_branch_source": str(
                        row.get("repair_branch_source")
                        or repair_branch_meta.get("source")
                        or "auto"
                    ),
                    "repair_branch": str(repair_branch),
                    "allowed_edit_types": list(row.get("allowed_edit_types") or ["REPLACE"]),
                    "drop_counts": {k: 0 for k in DROP_REASON_KEYS},
                    "target_candidates": int(row.get("target_candidates") or max_per_bug),
                    "valid_candidates": 0,
                    "fill_rate": 0.0,
                    "llm_calls_total": 0,
                    "elapsed_seconds": 0.0,
                    "gen_n": int(row.get("gen_n") or 1),
                    "max_gen_passes": int(row.get("max_gen_passes") or 1),
                    "temperature_used": [],
                    "top_p_used": [],
                    "unique_after_stage": {},
                    "scheduler": str(row.get("scheduler") or "fixed"),
                    "conf_level": str(row.get("conf_level") or "low"),
                    "bm25_top1": float(row.get("bm25_top1") or 0.0),
                    "bm25_margin": float(row.get("bm25_margin") or 0.0),
                    "profile": str(row.get("profile") or "MED"),
                    "profiles_tried": list(row.get("profiles_tried") or []),
                    "plan_source": str(row.get("plan_source") or "fallback_default"),
                    "plan_enforcement": str(row.get("plan_enforcement") or "filter_then_fill"),
                    "plan_target_match": str(row.get("plan_target_match") or "overlap"),
                    "plan_fill_used": bool(row.get("plan_fill_used", False)),
                    "plan_fill_added_n": int(row.get("plan_fill_added_n") or 0),
                    "plan_fill_penalty": float(row.get("plan_fill_penalty") or 0.0),
                    "plan_violation_breakdown": {k: 0 for k in PLAN_VIOLATION_TYPE_KEYS},
                    "java_filter_compile_valid": bool(java_defaults.get("java_filter_compile_valid", False)),
                    "java_rerank_compile_valid": bool(java_defaults.get("java_rerank_compile_valid", False)),
                    "java_candidate_compile_check": bool(java_defaults.get("java_candidate_compile_check", False)),
                    "java_compile_validation_mode": str(java_defaults.get("java_compile_validation_mode") or "none"),
                    "java_semantic_rerank_enabled": bool(java_defaults.get("java_semantic_rerank_enabled", False)),
                    "java_semantic_rerank_features": list(java_defaults.get("java_semantic_rerank_features") or []),
                    "java_branch_no_fill": bool(java_defaults.get("java_branch_no_fill", False)),
                    "java_survivor_backoff_mode": str(java_defaults.get("java_survivor_backoff_mode") or "off"),
                    "java_survivor_backoff_used": bool(row.get("java_survivor_backoff_used", False)),
                    "java_survivor_backoff_added_count": int(row.get("java_survivor_backoff_added_count") or 0),
                    "zero_candidate_after_filter": bool(row.get("zero_candidate_after_filter", False)),
                    "java_sentinel_extract_hit": int(row.get("java_sentinel_extract_hit") or 0),
                    "java_decl_salvage_used": int(row.get("java_decl_salvage_used") or 0),
                    "java_explanation_tail_stripped": int(row.get("java_explanation_tail_stripped") or 0),
                    "java_decl_recovery_used": int(row.get("java_decl_recovery_used") or 0),
                    "java_decl_recovery_mode_counts": dict(row.get("java_decl_recovery_mode_counts") or {}),
                    "java_compile_feedback_once": bool(java_defaults.get("java_compile_feedback_once", False)),
                    "java_compile_feedback_available": bool(java_defaults.get("java_compile_feedback_available", False)),
                    "java_compile_feedback_attempted_n": 0,
                    "java_compile_feedback_applied_n": 0,
                    "java_validity_checked_n": 0,
                    "java_validity_valid_n": 0,
                    "java_validity_invalid_n": 0,
                    "java_validity_rejected_n": 0,
                    "java_validity_reranked_n": 0,
                    "java_validity_reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
                    "backend_selected": str(row.get("backend_selected") or "auto"),
                    "hf_model_name": str(row.get("hf_model_name") or ""),
                }
                seen_hashes[bug_id] = set()
            merged[bug_id]["allowed_edit_types"] = list(row.get("allowed_edit_types") or merged[bug_id].get("allowed_edit_types") or ["REPLACE"])
            merged[bug_id]["target_candidates"] = max(
                int(merged[bug_id].get("target_candidates") or 0),
                int(row.get("target_candidates") or max_per_bug),
            )
            merged[bug_id]["llm_calls_total"] = int(merged[bug_id].get("llm_calls_total") or 0) + int(row.get("llm_calls_total") or 0)
            merged[bug_id]["elapsed_seconds"] = float(merged[bug_id].get("elapsed_seconds") or 0.0) + float(row.get("elapsed_seconds") or 0.0)
            merged[bug_id]["gen_n"] = int(max(int(merged[bug_id].get("gen_n") or 1), int(row.get("gen_n") or 1)))
            merged[bug_id]["max_gen_passes"] = int(max(int(merged[bug_id].get("max_gen_passes") or 1), int(row.get("max_gen_passes") or 1)))
            merged[bug_id]["temperature_used"] = list(merged[bug_id].get("temperature_used") or []) + list(row.get("temperature_used") or [])
            merged[bug_id]["top_p_used"] = list(merged[bug_id].get("top_p_used") or []) + list(row.get("top_p_used") or [])
            merged[bug_id]["scheduler"] = str(row.get("scheduler") or merged[bug_id].get("scheduler") or "fixed")
            merged[bug_id]["conf_level"] = str(row.get("conf_level") or merged[bug_id].get("conf_level") or "low")
            merged[bug_id]["language"] = str(row.get("language") or merged[bug_id].get("language") or language)
            merged[bug_id]["language_branch"] = str(row.get("language_branch") or merged[bug_id].get("language_branch") or language_branch)
            merged[bug_id]["repair_branch_requested"] = str(
                row.get("repair_branch_requested")
                or merged[bug_id].get("repair_branch_requested")
                or repair_branch_meta.get("requested")
                or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
            )
            merged[bug_id]["repair_branch_effective"] = str(
                row.get("repair_branch_effective")
                or row.get("repair_branch")
                or merged[bug_id].get("repair_branch_effective")
                or repair_branch
            )
            merged[bug_id]["repair_branch_source"] = str(
                row.get("repair_branch_source")
                or merged[bug_id].get("repair_branch_source")
                or repair_branch_meta.get("source")
                or "auto"
            )
            merged[bug_id]["repair_branch"] = str(row.get("repair_branch") or merged[bug_id].get("repair_branch") or repair_branch)
            merged[bug_id]["bm25_top1"] = max(float(merged[bug_id].get("bm25_top1") or 0.0), float(row.get("bm25_top1") or 0.0))
            merged[bug_id]["bm25_margin"] = max(float(merged[bug_id].get("bm25_margin") or 0.0), float(row.get("bm25_margin") or 0.0))
            merged[bug_id]["profile"] = str(row.get("profile") or merged[bug_id].get("profile") or "MED")
            merged[bug_id]["plan_source"] = str(row.get("plan_source") or merged[bug_id].get("plan_source") or "fallback_default")
            merged[bug_id]["plan_enforcement"] = str(
                row.get("plan_enforcement")
                or merged[bug_id].get("plan_enforcement")
                or resolve_effective_plan_enforcement(merge_args, repair_branch)
            )
            merged[bug_id]["plan_target_match"] = str(row.get("plan_target_match") or merged[bug_id].get("plan_target_match") or "overlap")
            merged[bug_id]["plan_fill_used"] = bool(merged[bug_id].get("plan_fill_used", False) or row.get("plan_fill_used", False))
            merged[bug_id]["plan_fill_added_n"] = int(merged[bug_id].get("plan_fill_added_n") or 0) + int(row.get("plan_fill_added_n") or 0)
            merged[bug_id]["plan_fill_penalty"] = max(float(merged[bug_id].get("plan_fill_penalty") or 0.0), float(row.get("plan_fill_penalty") or 0.0))
            merged[bug_id]["backend_selected"] = str(row.get("backend_selected") or merged[bug_id].get("backend_selected") or "auto")
            merged[bug_id]["hf_model_name"] = str(row.get("hf_model_name") or merged[bug_id].get("hf_model_name") or "")
            merged[bug_id]["java_semantic_rerank_enabled"] = bool(
                merged[bug_id].get("java_semantic_rerank_enabled", False)
                or row.get("java_semantic_rerank_enabled", False)
                or java_defaults.get("java_semantic_rerank_enabled", False)
            )
            merged[bug_id]["java_semantic_rerank_features"] = list(
                row.get("java_semantic_rerank_features")
                or merged[bug_id].get("java_semantic_rerank_features")
                or list(JAVA_SEMANTIC_RERANK_FEATURES)
            )
            merged[bug_id]["java_branch_no_fill"] = bool(
                merged[bug_id].get("java_branch_no_fill", False)
                or row.get("java_branch_no_fill", False)
                or java_defaults.get("java_branch_no_fill", False)
            )
            merged[bug_id]["java_survivor_backoff_mode"] = str(
                row.get("java_survivor_backoff_mode")
                or merged[bug_id].get("java_survivor_backoff_mode")
                or java_defaults.get("java_survivor_backoff_mode")
                or "off"
            )
            merged[bug_id]["java_survivor_backoff_used"] = bool(
                merged[bug_id].get("java_survivor_backoff_used", False)
                or row.get("java_survivor_backoff_used", False)
            )
            merged[bug_id]["java_survivor_backoff_added_count"] = int(
                merged[bug_id].get("java_survivor_backoff_added_count") or 0
            ) + int(row.get("java_survivor_backoff_added_count") or 0)
            merged[bug_id]["zero_candidate_after_filter"] = bool(
                merged[bug_id].get("zero_candidate_after_filter", False)
                or row.get("zero_candidate_after_filter", False)
            )
            merged[bug_id]["java_sentinel_extract_hit"] = int(
                merged[bug_id].get("java_sentinel_extract_hit", 0)
            ) + int(row.get("java_sentinel_extract_hit") or 0)
            merged[bug_id]["java_decl_salvage_used"] = int(
                merged[bug_id].get("java_decl_salvage_used", 0)
            ) + int(row.get("java_decl_salvage_used") or 0)
            merged[bug_id]["java_explanation_tail_stripped"] = int(
                merged[bug_id].get("java_explanation_tail_stripped", 0)
            ) + int(row.get("java_explanation_tail_stripped") or 0)
            merged[bug_id]["java_decl_recovery_used"] = int(
                merged[bug_id].get("java_decl_recovery_used", 0)
            ) + int(row.get("java_decl_recovery_used") or 0)
            merged[bug_id]["java_decl_recovery_mode_counts"] = merge_counter_dicts(
                merged[bug_id].get("java_decl_recovery_mode_counts") or {},
                row.get("java_decl_recovery_mode_counts") or {},
                list(set(list((merged[bug_id].get("java_decl_recovery_mode_counts") or {}).keys()) + list((row.get("java_decl_recovery_mode_counts") or {}).keys()))),
            )
            merged[bug_id]["java_compile_feedback_once"] = bool(
                merged[bug_id].get("java_compile_feedback_once", False)
                or row.get("java_compile_feedback_once", False)
                or java_defaults.get("java_compile_feedback_once", False)
            )
            merged[bug_id]["java_compile_feedback_available"] = bool(
                merged[bug_id].get("java_compile_feedback_available", False)
                or row.get("java_compile_feedback_available", False)
                or java_defaults.get("java_compile_feedback_available", False)
            )
            merged[bug_id]["java_compile_feedback_attempted_n"] = int(
                merged[bug_id].get("java_compile_feedback_attempted_n", 0)
            ) + int(row.get("java_compile_feedback_attempted_n") or 0)
            merged[bug_id]["java_compile_feedback_applied_n"] = int(
                merged[bug_id].get("java_compile_feedback_applied_n", 0)
            ) + int(row.get("java_compile_feedback_applied_n") or 0)
            merged[bug_id]["plan_violation_breakdown"] = merge_counter_dicts(
                merged[bug_id].get("plan_violation_breakdown") or {},
                row.get("plan_violation_breakdown") or {},
                PLAN_VIOLATION_TYPE_KEYS,
            )
            merged_profiles = list(merged[bug_id].get("profiles_tried") or [])
            for tag in list(row.get("profiles_tried") or []):
                tag_s = str(tag)
                if tag_s not in merged_profiles:
                    merged_profiles.append(tag_s)
            merged[bug_id]["profiles_tried"] = merged_profiles
            in_stage = row.get("unique_after_stage") or {}
            if isinstance(in_stage, dict):
                for stage_k, stage_v in in_stage.items():
                    k = str(stage_k)
                    try:
                        v = int(stage_v)
                    except Exception:
                        continue
                    prev = int((merged[bug_id].get("unique_after_stage") or {}).get(k, 0))
                    merged[bug_id]["unique_after_stage"][k] = max(prev, v)
            in_drop = row.get("drop_counts") or {}
            if isinstance(in_drop, dict):
                for k in DROP_REASON_KEYS:
                    merged[bug_id]["drop_counts"][k] = int(merged[bug_id]["drop_counts"].get(k, 0)) + int(in_drop.get(k, 0) or 0)

            ctx_row = {"function_name": fn_name, "file_path": file_path, "function": {"function_name": fn_name}}

            for cand_idx, cand in enumerate(row.get("candidates", []), start=1):
                code = cand.get("code") or ""
                if not code.strip():
                    continue
                h = hashlib.sha1(canon(code).encode("utf-8")).hexdigest()
                if h in seen_hashes[bug_id]:
                    continue
                ok, err = ast_ok(code, language=language)
                if not ok:
                    continue
                is_valid, _reason = validate_candidate_code(
                    code=code,
                    base_code=base_code,
                    expected_name=expected_name,
                    language=language,
                )
                if not is_valid:
                    continue
                java_validation = {
                    "checked": False,
                    "mode": str(cand.get("java_compile_validation_mode") or "none"),
                    "valid": cand.get("java_compile_valid"),
                    "reasons": list(cand.get("java_compile_validation_reasons") or []),
                    "hard_reasons": list(cand.get("java_compile_validation_hard_reasons") or []),
                    "context_drift_summary": str(cand.get("java_context_drift_summary") or "none"),
                    "context_allowlist_summary": dict(cand.get("java_context_allowlist_summary") or {}),
                    "declared_names_in_patch": list(cand.get("java_declared_names_in_patch") or []),
                    "context_supported_new_identifiers": list(cand.get("java_context_supported_new_identifiers") or []),
                    "context_supported_new_types": list(cand.get("java_context_supported_new_types") or []),
                    "context_supported_new_constants": list(cand.get("java_context_supported_new_constants") or []),
                    "context_supported_new_identifier_sources": dict(cand.get("java_context_supported_new_identifier_sources") or {}),
                    "context_supported_new_type_sources": dict(cand.get("java_context_supported_new_type_sources") or {}),
                    "context_supported_new_constant_sources": dict(cand.get("java_context_supported_new_constant_sources") or {}),
                    "new_helper_calls": list(cand.get("java_new_helper_calls") or []),
                    "new_type_references": list(cand.get("java_new_type_references") or []),
                    "new_constant_references": list(cand.get("java_new_constant_references") or []),
                    "new_identifier_references": list(cand.get("java_new_identifier_references") or []),
                    "broad_rewrite_fraction": float(cand.get("java_broad_rewrite_fraction", 0.0) or 0.0),
                    "risk_combo_triggered": bool(cand.get("java_compile_risk_combo_triggered", False)),
                    "risk_combo_details": list(cand.get("java_compile_risk_combo_details") or []),
                    "reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
                }
                if language == "java" and java_requested_mode != "none":
                    java_validation = assess_java_compile_validity(
                        base_code=base_code,
                        code=code,
                        expected_name=expected_name,
                        requested_mode=java_requested_mode,
                        prom_row=prom_row,
                        plan_ok=bool(cand.get("plan_ok", True)),
                        filter_enabled=bool(java_filter_active),
                        hard_reject_compile_risk_combos=bool(
                            getattr(merge_args, "java_hard_reject_compile_risk_combos", False)
                        ),
                    )
                    if java_validation.get("valid") is False and java_filter_active:
                        update_java_validity_stats(merged[bug_id], java_validation, rejected=True)
                        bump_drop(merged[bug_id]["drop_counts"], "java_compile_invalid")
                        continue
                base_score = score_candidate(
                    ctx_row,
                    prom_row,
                    base_code,
                    code,
                    ok,
                    language=language,
                )
                score, java_reranked = apply_java_validity_rerank(
                    float(base_score),
                    java_validation,
                    rerank_enabled=java_rerank_active,
                )
                score, java_semantic_record = apply_java_semantic_rerank(
                    score=score,
                    context_row=ctx_row,
                    prom_row=prom_row,
                    base_code=base_code,
                    code=code,
                    repair_branch=repair_branch,
                    language=language,
                    edit_summary=dict(cand.get("edit_summary") or {}),
                    plan_ok=bool(cand.get("plan_ok", True)),
                    plan_violations=list(cand.get("plan_violations") or []),
                    plan_fill_selected=bool(cand.get("plan_fill_selected", False)),
                )
                update_java_validity_stats(merged[bug_id], java_validation, reranked=java_reranked)
                java_candidate_meta = build_java_candidate_metadata(
                    cand=cand,
                    validation_record=java_validation,
                    semantic_record=java_semantic_record,
                    java_semantic_enabled=java_semantic_active,
                )
                merged[bug_id]["candidates"].append({
                    **{k: v for k, v in cand.items() if k not in ("ast", "score", "base_score")},
                    "source_rank_before_merge": int(cand.get("source_rank_before_merge") or cand.get("rank") or cand_idx),
                    "code": code,
                    "candidate_code_hash": str(cand.get("candidate_code_hash") or h),
                    "normalized_candidate_hash": str(cand.get("normalized_candidate_hash") or compute_normalized_candidate_hash(code)),
                    "ast": {"ok": ok, "err": err},
                    "base_score": float(base_score),
                    "score": float(score),
                    "repair_branch_requested": str(
                        cand.get("repair_branch_requested")
                        or repair_branch_meta.get("requested")
                        or normalize_repair_branch(getattr(merge_args, "repair_branch", "auto"))
                    ),
                    "repair_branch_effective": str(cand.get("repair_branch_effective") or repair_branch),
                    "repair_branch_source": str(cand.get("repair_branch_source") or repair_branch_meta.get("source") or "auto"),
                    "repair_branch": str(repair_branch),
                    "language_branch": str(cand.get("language_branch") or language_branch),
                    **java_candidate_meta,
                })
                seen_hashes[bug_id].add(h)

    for bug_id, row in merged.items():
        if rerank:
            row["candidates"].sort(key=lambda c: (0 if str(c.get("gen_mode", "")) == "greedy" else 1, -float(c.get("score", -10**9) or -10**9)))
        row["candidates"] = row["candidates"][: max(1, int(max_per_bug))]
        row["valid_candidates"] = int(len(row["candidates"]))
        target = int(row.get("target_candidates") or max_per_bug)
        row["fill_rate"] = round(float(row["valid_candidates"]) / float(max(1, target)), 4)

    return merged


def apply_final_top_k(
    results: Dict[str, Any],
    final_top_k: int,
    coverage_top_k: bool = False,
    coverage_key: str = "source_run",
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Keep only top-K candidates per bug for final output.

    When coverage_top_k is enabled, we enforce coverage across groups defined by coverage_key
    (e.g., multi-run pooling by source_run).
    """
    k = max(1, int(final_top_k))
    source_target_total = 0
    source_valid_total = 0
    for _bug_id, row in (results or {}).items():
        candidates = list(row.get("candidates") or [])
        src_target = int(row.get("target_candidates") or len(candidates))
        src_valid = int(row.get("valid_candidates") or len(candidates))
        src_fill = float(row.get("fill_rate") or (float(src_valid) / float(max(1, src_target))))

        source_target_total += src_target
        source_valid_total += src_valid

        row["source_target_candidates"] = int(src_target)
        row["source_valid_candidates"] = int(src_valid)
        row["source_fill_rate"] = float(round(src_fill, 4))

        if coverage_top_k:
            picked = select_topk_with_coverage(candidates, k, coverage_key=str(coverage_key), keep_best_greedy=True)
            for i, c in enumerate(picked):
                if isinstance(c, dict):
                    c["candidate_id"] = int(i)
                    c["rank"] = int(i + 1)
            row["candidates"] = picked
        else:
            row["candidates"] = candidates[:k]
            for i, c in enumerate(row["candidates"]):
                if isinstance(c, dict):
                    c["candidate_id"] = int(i)
                    c["rank"] = int(i + 1)

        row["target_candidates"] = int(k)
        row["valid_candidates"] = int(len(row["candidates"]))
        row["fill_rate"] = round(float(row["valid_candidates"]) / float(k), 4)
    return results, {
        "source_target_candidates_total": int(source_target_total),
        "source_valid_candidates_total": int(source_valid_total),
    }


# ===== Candidate generation (per-bug) =====
def generate_candidates_for_bug(
    client: QwenClient,
    bug_id: str,
    row: Dict[str, Any],
    base_code: str,
    prom_row: Dict[str, Any],
    args: Namespace,
    target: Optional[int] = None,
    run_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Similar to step 5:
      - Build one prompt from this bug_id context
      - Generate one greedy + multiple sampled candidates
      - Keep unique code candidates up to args.max_per_bug (e.g., 70)
    """
    start_time = time.time()
    target = max(1, int(target if target is not None else args.max_per_bug))
    language = infer_language(row, prom_row, base_code)
    repair_branch_meta = resolve_repair_branch_metadata(args, row, prom_row, language)
    repair_branch = str(repair_branch_meta.get("effective") or resolve_effective_repair_branch(args, row, prom_row, language))
    repair_branch_requested = str(
        repair_branch_meta.get("requested") or normalize_repair_branch(getattr(args, "repair_branch", "auto"))
    )
    repair_branch_source = str(repair_branch_meta.get("source") or "auto")
    language_branch = language_branch_for_language(language)
    prompt = build_guided_prompt(row, base_code, prom_row)
    bug_profile = resolve_bug_profile(prom_row, args, target)
    target_local = int(bug_profile.get("target_local", target))
    n_lines_base = len((base_code or "").splitlines())
    plan_enforcement = resolve_effective_plan_enforcement(args, repair_branch)
    java_survivor_backoff_mode = resolve_effective_java_survivor_backoff(args, repair_branch, plan_enforcement)
    plan_target_match = str(getattr(args, "plan_target_match", "overlap") or "overlap").strip().lower()
    if plan_target_match not in {"contain", "overlap"}:
        plan_target_match = "overlap"
    min_survivors_per_bug = max(1, int(getattr(args, "min_survivors_per_bug", 3) or 3))
    final_top_k_goal = int(getattr(args, "final_top_k", 0) or 0)
    plan_fill_goal = max(min_survivors_per_bug, final_top_k_goal if final_top_k_goal > 0 else min_survivors_per_bug)
    plan_fill_goal = min(max(1, target_local), max(1, plan_fill_goal))
    fill_with_violations_penalty = float(getattr(args, "fill_with_violations_penalty", 25.0) or 25.0)

    base_canon = canon(base_code)
    base_hash = hashlib.sha1(base_canon.encode("utf-8")).hexdigest()
    base_stripped = (base_code or "").strip()

    seen_signatures: set[str] = set()
    seen_code_hashes: set[str] = set()
    signature_cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    candidates: List[Dict[str, Any]] = []
    plan_bad_candidates: List[Dict[str, Any]] = []
    java_rescue_candidates: List[Dict[str, Any]] = []
    drop_counts = make_drop_counter()
    plan_violation_breakdown = make_plan_violation_counter()
    unique_after_stage: Dict[str, int] = {}
    llm_calls_total = 0

    base_node = _parse_single_toplevel_def_or_class(base_code) if language == "python" else None
    expected_name = (row.get("function_name") or safe_get(prom_row, ["function", "function_name"]) or "").strip()
    if not expected_name:
        if language == "python" and base_node is not None:
            expected_name = getattr(base_node, "name", "") or ""
        elif language == "java":
            expected_name = extract_decl_name(base_code, language="java")
    java_validity_stats = init_java_validity_stats(args, language)
    java_validity_stats["java_semantic_rerank_enabled"] = bool(java_semantic_rerank_enabled(repair_branch, language))
    java_validity_stats["java_survivor_backoff_mode"] = str(java_survivor_backoff_mode)
    java_validity_stats["java_branch_no_fill"] = bool(
        language == "java"
        and plan_enforcement not in {"filter_then_fill"}
        and java_survivor_backoff_mode == "off"
    )
    java_validity_stats["java_compile_feedback_once"] = bool(java_compile_feedback_requested(args, language, repair_branch))
    java_validity_stats["java_compile_feedback_available"] = bool(resolve_java_compile_feedback_available(args, language))
    java_validation_requested_mode = str(java_validity_stats.get("java_compile_validation_mode") or "none")
    java_filter_active = bool(java_validity_stats.get("java_filter_compile_valid", False))
    java_rerank_active = bool(
        java_validity_stats.get("java_rerank_compile_valid", False)
        and not bool(getattr(args, "no_rerank", False))
    )
    java_semantic_active = bool(java_validity_stats.get("java_semantic_rerank_enabled", False))
    java_extract_stats = {
        "java_sentinel_extract_hit": 0,
        "java_decl_salvage_used": 0,
        "java_explanation_tail_stripped": 0,
        "java_decl_recovery_used": 0,
        "java_decl_recovery_mode_counts": {},
    }

    def _buffered_count() -> int:
        if plan_enforcement == "filter_then_fill":
            return len(candidates) + len(plan_bad_candidates)
        if (
            language == "java"
            and plan_enforcement == "filter"
            and java_survivor_backoff_mode in {"hybrid", "fill"}
        ):
            return len(candidates) + len(plan_bad_candidates)
        return len(candidates)

    def _enough() -> bool:
        return _buffered_count() >= target_local

    def _bump_extraction(reason: str) -> None:
        bump_drop(drop_counts, reason)
        if reason.startswith("extraction_failed_"):
            bump_drop(drop_counts, "extraction_failed")

    def _try_add(text: str, temperature: float, top_p: float, gen_mode: str = "sample"):
        if _enough():
            return
        if not text:
            bump_drop(drop_counts, "empty_code")
            return
        code, extract_reason, extract_meta = extract_correct_block_with_reason(
            text,
            language=language,
            base_code=base_code,
            expected_name=expected_name,
            repair_branch=repair_branch,  # Phase1-E: java_v2 body-only 추출 지원
        )
        if not code:
            _bump_extraction(extract_reason or "extraction_failed_postprocess")
            return
        if language == "java":
            for key in (
                "java_sentinel_extract_hit",
                "java_decl_salvage_used",
                "java_explanation_tail_stripped",
                "java_decl_recovery_used",
                "java_v2_body_mode",             # Phase1-E: body-only 추출 추적
            ):
                if bool(extract_meta.get(key, False)):
                    java_extract_stats[key] = int(java_extract_stats.get(key, 0) or 0) + 1
            if bool(extract_meta.get("java_decl_recovery_used", False)):
                mode = str(extract_meta.get("java_decl_recovery_mode") or "unknown")
                mode_counts = java_extract_stats.get("java_decl_recovery_mode_counts") or {}
                mode_counts[mode] = int(mode_counts.get(mode, 0) or 0) + 1
                java_extract_stats["java_decl_recovery_mode_counts"] = mode_counts
        # NOTE: previous version called code.strip() unconditionally,
        # which removed leading whitespace from the first line of the
        # candidate (Qwen Java models consistently emit the method
        # signature at column 0, while keeping body lines absolutely
        # indented). That single-line indent loss was then picked up by
        # difflib as a spurious REPLACE@1-1 edit, exploding the
        # `out_of_target` plan-violation counts and forcing plan_fill
        # to rescue 55%+ of survivors. We now strip trailing whitespace
        # and leading blank lines only, and restore the first line's
        # indent to match the base code's first-line indent when the
        # model stripped it.
        code = code.rstrip()
        while code.startswith("\n"):
            code = code[1:]
        if not code:
            bump_drop(drop_counts, "empty_code")
            return
        if language == "java" and base_code:
            base_lines_first = base_code.splitlines()
            code_lines_first = code.split("\n")
            if base_lines_first and code_lines_first:
                base_first = base_lines_first[0]
                base_first_indent = len(base_first) - len(base_first.lstrip(" "))
                cand_first = code_lines_first[0]
                cand_first_indent = len(cand_first) - len(cand_first.lstrip(" "))
                if (
                    cand_first_indent < base_first_indent
                    and cand_first.lstrip(" ") != ""
                ):
                    code_lines_first[0] = " " * base_first_indent + cand_first.lstrip(" ")
                    code = "\n".join(code_lines_first)
        if not quick_has_def_or_class(code, language):
            bump_drop(drop_counts, "no_def_or_class")
            return
        if code == base_stripped or code.strip() == base_stripped:
            bump_drop(drop_counts, "no_change")
            return

        code_canon = canon(code)
        code_hash = hashlib.sha1(code_canon.encode("utf-8")).hexdigest()
        if code_hash in seen_code_hashes:
            bump_drop(drop_counts, "duplicate")
            return

        if code_hash == base_hash:
            bump_drop(drop_counts, "no_change")
            return

        ast_pass = True
        ast_err: Optional[str] = None
        parsed_node: Optional[ast.AST] = None
        if language == "python":
            parsed_node, parse_err = _parse_python_first_toplevel_def_or_class(code)
            if parse_err:
                bump_drop(drop_counts, "syntax_fail")
                return
            if parsed_node is None:
                bump_drop(drop_counts, "no_def_or_class")
                return
            ast_pass = True
            ast_err = None

        is_valid, reason = validate_candidate_code(
            code=code,
            base_code=base_code,
            expected_name=expected_name,
            language=language,
            parsed_node=parsed_node,
            canon_base=base_canon,
            canon_code=code_canon,
        )
        if not is_valid:
            bump_drop(drop_counts, reason)
            return

        compile_feedback_record = {
            "enabled": False,
            "available": False,
            "attempted": 0,
            "applied": False,
            "status": "disabled",
        }
        if language == "java":
            code, compile_feedback_record = maybe_apply_java_compile_feedback_once(
                args=args,
                language=language,
                repair_branch=repair_branch,
                code=code,
            )
            java_validity_stats["java_compile_feedback_attempted_n"] = int(
                java_validity_stats.get("java_compile_feedback_attempted_n", 0)
            ) + int(compile_feedback_record.get("attempted", 0) or 0)
            java_validity_stats["java_compile_feedback_applied_n"] = int(
                java_validity_stats.get("java_compile_feedback_applied_n", 0)
            ) + (1 if compile_feedback_record.get("applied") else 0)

        java_validation = {
            "checked": False,
            "mode": "none",
            "valid": None,
            "reasons": [],
            "hard_reasons": [],
            "context_drift_summary": "none",
            "context_allowlist_summary": {},
            "declared_names_in_patch": [],
            "context_supported_new_identifiers": [],
            "context_supported_new_types": [],
            "context_supported_new_constants": [],
            "context_supported_new_identifier_sources": {},
            "context_supported_new_type_sources": {},
            "context_supported_new_constant_sources": {},
            "new_helper_calls": [],
            "new_type_references": [],
            "new_constant_references": [],
            "new_identifier_references": [],
            "broad_rewrite_fraction": 0.0,
            "risk_combo_triggered": False,
            "risk_combo_details": [],
            "reason_counts": {k: 0 for k in JAVA_VALIDITY_REASON_KEYS},
        }

        sig_item = signature_cache.get(code_hash)
        if sig_item is None:
            patch_signature, edit_summary = build_patch_signature(base_code, code)
            signature_cache[code_hash] = (patch_signature, edit_summary)
        else:
            patch_signature, edit_summary = sig_item
        if patch_signature in seen_signatures:
            bump_drop(drop_counts, "duplicate")
            return

        plan_dict, plan_source = parse_plan_dict(prom_row, base_code=base_code, language=language)
        plan_ok, plan_violations, plan_stats = check_plan(
            plan_dict,
            edit_summary,
            n_lines=n_lines_base,
            target_match=plan_target_match,
            language=language,
        )
        violation_type_counts = plan_stats.get("violation_type_breakdown") or {}
        if isinstance(violation_type_counts, dict):
            for key in PLAN_VIOLATION_TYPE_KEYS:
                plan_violation_breakdown[key] += int(violation_type_counts.get(key, 0) or 0)

        if language == "java" and java_validation_requested_mode != "none":
            java_validation = assess_java_compile_validity(
                base_code=base_code,
                code=code,
                expected_name=expected_name,
                requested_mode=java_validation_requested_mode,
                prom_row=prom_row,
                plan_ok=bool(plan_ok),
                filter_enabled=bool(java_filter_active),
                hard_reject_compile_risk_combos=bool(
                    getattr(args, "java_hard_reject_compile_risk_combos", False)
                ),
            )
            if java_validation.get("valid") is False and java_filter_active:
                update_java_validity_stats(java_validity_stats, java_validation, rejected=True)
                bump_drop(drop_counts, "java_compile_invalid")
                return

        base_score = float(
            score_candidate(
                row,
                prom_row,
                base_code,
                code,
                ast_pass,
                language=language,
                canon_base=base_canon,
                canon_code=code_canon,
            )
        )
        score = float(base_score)
        if (not plan_ok) and plan_enforcement == "penalty":
            pen = float(getattr(args, "plan_violation_penalty", 15.0))
            if bool(getattr(args, "plan_violation_penalty_per_violation", False)):
                pen *= max(1, len(plan_violations))
            score -= float(pen)
        score, java_reranked = apply_java_validity_rerank(
            score,
            java_validation,
            rerank_enabled=java_rerank_active,
        )
        score, java_semantic_record = apply_java_semantic_rerank(
            score=score,
            context_row=row,
            prom_row=prom_row,
            base_code=base_code,
            code=code,
            repair_branch=repair_branch,
            language=language,
            edit_summary=edit_summary,
            plan_ok=bool(plan_ok),
            plan_violations=list(plan_violations),
            plan_fill_selected=False,
        )
        update_java_validity_stats(
            java_validity_stats,
            java_validation,
            reranked=java_reranked,
        )
        java_candidate_meta = build_java_candidate_metadata(
            cand=None,
            validation_record=java_validation,
            semantic_record=java_semantic_record,
            java_semantic_enabled=java_semantic_active,
        )

        entry = {
            "text": text,
            "code": code,
            "candidate_code_hash": str(code_hash),
            "normalized_candidate_hash": compute_normalized_candidate_hash(code),
            "ast": {"ok": ast_pass, "err": ast_err},
            "base_score": float(base_score),
            "score": float(score),
            "gen_mode": str(gen_mode),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "allowed_edit_types": list(plan_dict.get("allowed_edit_types") or ["REPLACE"]),
            "patch_signature": patch_signature,
            "edit_summary": edit_summary,
            "plan_ok": bool(plan_ok),
            "plan_violations": list(plan_violations),
            "plan_stats": dict(plan_stats),
            "plan_source": str(plan_source),
            "plan_target_match": str(plan_target_match),
            "plan_violation_breakdown": {
                k: int(violation_type_counts.get(k, 0) or 0) for k in PLAN_VIOLATION_TYPE_KEYS
            },
            "repair_branch_requested": str(repair_branch_requested),
            "repair_branch_effective": str(repair_branch),
            "repair_branch_source": str(repair_branch_source),
            "repair_branch": str(repair_branch),
            "language_branch": str(language_branch),
            "source_rank_before_merge": 0,
            **java_candidate_meta,
            "java_survivor_backoff_mode": str(java_survivor_backoff_mode),
            "java_survivor_backoff_selected": False,
            "java_rescue_survivor": False,
            "java_reject_policy_version": JAVA_REJECT_POLICY_VERSION,
            "java_sentinel_extract_hit": bool(extract_meta.get("java_sentinel_extract_hit", False)),
            "java_decl_salvage_used": bool(extract_meta.get("java_decl_salvage_used", False)),
            "java_explanation_tail_stripped": bool(extract_meta.get("java_explanation_tail_stripped", False)),
            "java_decl_recovery_used": bool(extract_meta.get("java_decl_recovery_used", False)),
            "java_decl_recovery_mode": str(extract_meta.get("java_decl_recovery_mode") or ""),
            "java_compile_feedback_once": bool(compile_feedback_record.get("enabled", False)),
            "java_compile_feedback_available": bool(compile_feedback_record.get("available", False)),
            "java_compile_feedback_status": str(compile_feedback_record.get("status") or "disabled"),
        }
        seen_signatures.add(patch_signature)
        seen_code_hashes.add(code_hash)

        if language == "java" and _java_candidate_rescue_precheck(entry):
            rescue_validation = assess_java_compile_validity(
                base_code=base_code,
                code=code,
                expected_name=expected_name,
                requested_mode=java_validation_requested_mode,
                prom_row=prom_row,
                plan_ok=bool(plan_ok),
                filter_enabled=bool(java_filter_active),
                rescue_mode=True,
                hard_reject_compile_risk_combos=bool(
                    getattr(args, "java_hard_reject_compile_risk_combos", False)
                ),
            )
            rescue_score, rescue_reranked = apply_java_validity_rerank(
                float(base_score),
                rescue_validation,
                rerank_enabled=java_rerank_active,
            )
            rescue_score, rescue_semantic_record = apply_java_semantic_rerank(
                score=rescue_score,
                context_row=row,
                prom_row=prom_row,
                base_code=base_code,
                code=code,
                repair_branch=repair_branch,
                language=language,
                edit_summary=edit_summary,
                plan_ok=bool(plan_ok),
                plan_violations=list(plan_violations),
                plan_fill_selected=False,
                rescue_mode=True,
            )
            rescue_details = _java_candidate_rescue_eligible(entry, rescue_validation)
            if rescue_details:
                rescue_entry = dict(entry)
                rescue_meta = build_java_candidate_metadata(
                    cand=rescue_entry,
                    validation_record=rescue_validation,
                    semantic_record=rescue_semantic_record,
                    java_semantic_enabled=java_semantic_active,
                )
                rescue_entry.update(rescue_meta)
                rescue_entry["rescue_score"] = float(rescue_score)
                rescue_entry["java_rescue_reranked"] = bool(rescue_reranked)
                rescue_entry.update(rescue_details)
                java_rescue_candidates.append(rescue_entry)

        if (not plan_ok) and plan_enforcement == "filter":
            if language == "java" and java_survivor_backoff_mode in {"hybrid", "fill"}:
                entry["fill_score"] = float(score - fill_with_violations_penalty)
                plan_bad_candidates.append(entry)
            else:
                bump_drop(drop_counts, "plan_violation")
            return

        if (not plan_ok) and plan_enforcement == "filter_then_fill":
            entry["fill_score"] = float(score - fill_with_violations_penalty)
            plan_bad_candidates.append(entry)
            return

        candidates.append(entry)

    base_temp = float(args.temperature)
    base_top_p = float(args.top_p)
    temp_step = float(getattr(args, "temp_step", 0.3))
    top_p_step = float(getattr(args, "top_p_step", 0.05))
    stop_sequences_raw = str(getattr(args, "stop_sequences", "") or "").strip()
    stop_sequences = [s for s in (x.strip() for x in stop_sequences_raw.split(",")) if s]
    repetition_penalty = float(args.repetition_penalty)
    default_presence_penalty = float(getattr(args, "presence_penalty", 0.0) or 0.0)
    default_frequency_penalty = float(getattr(args, "frequency_penalty", 0.0) or 0.0)
    default_top_k = int(getattr(args, "top_k", 0) or 0)

    temperatures_used: List[float] = []
    top_ps_used: List[float] = []
    profiles_tried: List[str] = []
    gen_n_used: List[int] = []
    max_gen_passes_used: List[int] = []

    def _run_generation_profile(
        *,
        profile_tag: str,
        profile_temp: float,
        profile_top_p: float,
        gen_n_local: int,
        max_gen_passes_local: int,
        sampling_batch_local: int,
        presence_penalty_local: float,
        frequency_penalty_local: float,
        top_k_local: int,
        mix_greedy_local: bool,
        greedy_local: bool,
        seed_base: Optional[int],
    ) -> None:
        nonlocal llm_calls_total
        if _enough():
            return

        profiles_tried.append(str(profile_tag))
        gen_n_local = max(1, int(gen_n_local))
        max_gen_passes_local = max(1, int(max_gen_passes_local))
        sampling_batch_local = max(1, int(sampling_batch_local))
        gen_n_used.append(int(gen_n_local))
        max_gen_passes_used.append(int(max_gen_passes_local))

        if mix_greedy_local and not greedy_local:
            gen_args_greedy: Dict[str, Any] = {
                "batch_size": 1,
                "n": 1,
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": repetition_penalty,
                "max_input_tokens": 4096,
            }
            if seed_base is not None:
                gen_args_greedy["seed"] = int(seed_base)
            if stop_sequences:
                gen_args_greedy["stop"] = stop_sequences

            outs_g = client.generate([prompt], gen_args_greedy)
            llm_calls_total += 1
            temperatures_used.append(0.0)
            top_ps_used.append(1.0)
            for out in outs_g:
                _try_add(out.get("text") or "", temperature=0.0, top_p=1.0, gen_mode="greedy")
                break
            unique_after_stage[f"{profile_tag}_greedy"] = int(_buffered_count())

        temp_cap = max(0.95, float(profile_temp))
        top_p_cap = max(0.99, float(profile_top_p))

        for pass_idx in range(max_gen_passes_local):
            if _enough():
                break
            if greedy_local:
                cur_temp = 0.0
                cur_top_p = 1.0
                cur_gen_n = 1
            elif pass_idx == 0:
                cur_temp = float(profile_temp)
                cur_top_p = float(profile_top_p)
                cur_gen_n = int(gen_n_local)
            else:
                total_seen = int(len(candidates) + sum(int(v) for v in drop_counts.values()))
                dup = int(drop_counts.get("duplicate", 0))
                dup_rate = float(dup) / float(max(1, total_seen))
                extraction_fail = int(drop_counts.get("extraction_failed", 0))
                need_retry = (_buffered_count() < target_local) and (
                    dup >= 2 * target_local
                    or dup_rate >= 0.4
                    or extraction_fail >= target_local
                )
                if not need_retry:
                    break
                cur_temp = min(temp_cap, float(profile_temp) + temp_step * pass_idx)
                cur_top_p = min(top_p_cap, float(profile_top_p) + top_p_step * pass_idx)
                cur_gen_n = int(gen_n_local)

            temperatures_used.append(round(float(cur_temp), 4))
            top_ps_used.append(round(float(cur_top_p), 4))

            n_total = 1 if greedy_local else max(1, int(cur_gen_n) * int(sampling_batch_local))
            gen_args: Dict[str, Any] = {
                "batch_size": 1,
                "n": int(n_total),
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": (not bool(greedy_local)),
                "temperature": float(cur_temp),
                "top_p": float(cur_top_p),
                "repetition_penalty": repetition_penalty,
                "max_input_tokens": 4096,
                "presence_penalty": float(presence_penalty_local),
                "frequency_penalty": float(frequency_penalty_local),
                "top_k": int(top_k_local),
            }
            if seed_base is not None:
                gen_args["seed"] = int(seed_base + 100 + pass_idx)
            if stop_sequences:
                gen_args["stop"] = stop_sequences
            if float(gen_args.get("presence_penalty", 0.0) or 0.0) == 0.0:
                gen_args.pop("presence_penalty", None)
            if float(gen_args.get("frequency_penalty", 0.0) or 0.0) == 0.0:
                gen_args.pop("frequency_penalty", None)
            if int(gen_args.get("top_k", 0) or 0) <= 0:
                gen_args.pop("top_k", None)

            outs = client.generate([prompt], gen_args)
            llm_calls_total += 1
            for out in outs:
                _try_add(out.get("text") or "", temperature=cur_temp, top_p=cur_top_p)
                if _enough():
                    break
            unique_after_stage[f"{profile_tag}_pass{pass_idx + 1}"] = int(_buffered_count())
            if greedy_local:
                break

    scheduler_name = str(bug_profile.get("scheduler", "fixed"))
    greedy_override = bool(getattr(args, "greedy", False))
    run_seed_base = int(run_seed if run_seed is not None else getattr(args, "seed", 42))
    profile_name = str(bug_profile.get("profile", "MED"))

    profile_specs: List[Dict[str, Any]]
    if scheduler_name != "cost_aware":
        profile_specs = [
            {
                "tag": f"seed{run_seed_base}",
                "temperature": base_temp,
                "top_p": base_top_p,
                "presence_penalty": default_presence_penalty,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": max(1, int(getattr(args, "gen_n", 1))),
                "max_gen_passes": max(1, int(getattr(args, "max_gen_passes", 1))),
                "batch_size": max(1, int(args.batch_size)),
                "mix_greedy": bool(getattr(args, "mix_greedy_first", False)) and not greedy_override,
                "greedy": greedy_override,
                "seed_base": run_seed_base,
            }
        ]
    elif profile_name == "EASY":
        profile_specs = [
            {
                "tag": "EASY",
                "temperature": base_temp,
                "top_p": base_top_p,
                "presence_penalty": default_presence_penalty,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": min(max(1, int(getattr(args, "gen_n", 1))), 2),
                "max_gen_passes": 1,
                "batch_size": min(max(1, int(args.batch_size)), 10),
                "mix_greedy": bool(getattr(args, "mix_greedy_first", False)) and not greedy_override,
                "greedy": greedy_override,
                "seed_base": run_seed_base,
            }
        ]
    elif profile_name == "MED":
        profile_specs = [
            {
                "tag": "MED",
                "temperature": base_temp,
                "top_p": base_top_p,
                "presence_penalty": default_presence_penalty,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": min(max(3, int(getattr(args, "gen_n", 1))), 4),
                "max_gen_passes": min(max(1, int(getattr(args, "max_gen_passes", 1))), 2),
                "batch_size": min(max(1, int(args.batch_size)), 20),
                "mix_greedy": bool(getattr(args, "mix_greedy_first", False)) and not greedy_override,
                "greedy": greedy_override,
                "seed_base": run_seed_base,
            }
        ]
    else:
        hard_gen_n = max(4, int(getattr(args, "gen_n", 1)))
        hard_batch_size = max(1, int(args.batch_size))
        profile_specs = [
            {
                "tag": "A",
                "temperature": 0.70,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": 1 if greedy_override else hard_gen_n,
                "max_gen_passes": 1,
                "batch_size": hard_batch_size,
                "mix_greedy": False,
                "greedy": greedy_override,
                "seed_base": run_seed_base + 0,
            },
            {
                "tag": "B",
                "temperature": 0.90,
                "top_p": 0.90,
                "presence_penalty": 0.2,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": 1 if greedy_override else hard_gen_n,
                "max_gen_passes": 1,
                "batch_size": hard_batch_size,
                "mix_greedy": False,
                "greedy": greedy_override,
                "seed_base": run_seed_base + 1,
            },
            {
                "tag": "C",
                "temperature": 1.05,
                "top_p": 0.85,
                "presence_penalty": 0.4,
                "frequency_penalty": default_frequency_penalty,
                "top_k": default_top_k,
                "gen_n": 1 if greedy_override else hard_gen_n,
                "max_gen_passes": 1,
                "batch_size": hard_batch_size,
                "mix_greedy": False,
                "greedy": greedy_override,
                "seed_base": run_seed_base + 2,
            },
        ]

    for spec in profile_specs:
        _run_generation_profile(
            profile_tag=str(spec["tag"]),
            profile_temp=float(spec["temperature"]),
            profile_top_p=float(spec["top_p"]),
            gen_n_local=int(spec["gen_n"]),
            max_gen_passes_local=int(spec["max_gen_passes"]),
            sampling_batch_local=int(spec["batch_size"]),
            presence_penalty_local=float(spec["presence_penalty"]),
            frequency_penalty_local=float(spec["frequency_penalty"]),
            top_k_local=int(spec["top_k"]),
            mix_greedy_local=bool(spec["mix_greedy"]),
            greedy_local=bool(spec["greedy"]),
            seed_base=int(spec["seed_base"]) if spec.get("seed_base") is not None else None,
        )
        if _enough():
            break

    mix_greedy_active = any(bool(spec.get("mix_greedy")) for spec in profile_specs)
    def _sort_candidate_list(items: List[Dict[str, Any]], score_key: str) -> List[Dict[str, Any]]:
        items = list(items)
        if mix_greedy_active:
            greedy_list = [c for c in items if str(c.get("gen_mode", "")) == "greedy"]
            sample_list = [c for c in items if str(c.get("gen_mode", "")) != "greedy"]
            if not args.no_rerank:
                greedy_list = _sorted_candidates_by_priority(greedy_list, score_key=score_key)
                sample_list = _sorted_candidates_by_priority(sample_list, score_key=score_key)
            if greedy_list:
                greedy_list = greedy_list[:1]
            return greedy_list + sample_list
        if not args.no_rerank:
            items = _sorted_candidates_by_priority(items, score_key=score_key)
        return items

    candidates = _sort_candidate_list(candidates, "score")
    plan_bad_candidates = _sort_candidate_list(plan_bad_candidates, "fill_score")
    java_rescue_candidates = _sort_candidate_list(java_rescue_candidates, "rescue_score")

    plan_fill_used = False
    plan_fill_added_n = 0
    java_survivor_backoff_used = False
    java_survivor_backoff_added_count = 0
    java_rescue_used = False
    java_rescue_added_count = 0
    selected_candidates: List[Dict[str, Any]] = list(candidates)
    zero_candidate_after_filter = bool(not selected_candidates)
    fill_mode = "off"
    if language == "java" and plan_enforcement in {"filter", "filter_then_fill"}:
        fill_mode = str(java_survivor_backoff_mode)
    elif plan_enforcement == "filter_then_fill":
        fill_mode = "fill"

    fill_goal = 0
    if fill_mode == "hybrid":
        fill_goal = min(target_local, max(1, min_survivors_per_bug))
    elif fill_mode == "fill":
        fill_goal = min(target_local, max(min_survivors_per_bug, plan_fill_goal))

    strict_survivor_count = int(len(selected_candidates))
    rescue_needed = bool(language == "java" and strict_survivor_count == 0)

    if fill_goal > 0:
        if len(selected_candidates) < fill_goal:
            need = fill_goal - len(selected_candidates)
            selected_signatures = {
                str(item.get("patch_signature") or "")
                for item in selected_candidates
                if str(item.get("patch_signature") or "")
            }
            fill_chunk: List[Dict[str, Any]] = []
            for item in plan_bad_candidates:
                patch_signature = str(item.get("patch_signature") or "")
                if patch_signature and patch_signature in selected_signatures:
                    continue
                fill_chunk.append(dict(item))
                if patch_signature:
                    selected_signatures.add(patch_signature)
                if len(fill_chunk) >= need:
                    break
            for item in fill_chunk:
                item["score"] = float(item.get("fill_score", item.get("score", -10**9)))
                item["plan_fill_selected"] = True
                item["java_survivor_backoff_selected"] = bool(language == "java")
                if java_semantic_active:
                    item["score"] = float(item.get("score", -10**9)) - 12.0
                    item["java_semantic_score_adjustment"] = float(item.get("java_semantic_score_adjustment", 0.0) or 0.0) - 12.0
                    semantic_features = dict(item.get("java_semantic_features") or {})
                    semantic_features["plan_fill_selected"] = True
                    item["java_semantic_features"] = semantic_features
                selected_candidates.append(item)
            plan_fill_used = bool(fill_chunk)
            plan_fill_added_n = int(len(fill_chunk))
            if language == "java" and fill_mode in {"hybrid", "fill"}:
                java_survivor_backoff_used = bool(fill_chunk)
                java_survivor_backoff_added_count = int(len(fill_chunk))
        dropped_plan_bad = max(0, len(plan_bad_candidates) - plan_fill_added_n)
        if dropped_plan_bad > 0:
            drop_counts["plan_violation"] += int(dropped_plan_bad)

    if rescue_needed and java_rescue_candidates:
        selected_signatures = {
            str(item.get("patch_signature") or "")
            for item in selected_candidates
            if str(item.get("patch_signature") or "")
        }
        rescue_chunk: List[Dict[str, Any]] = []
        for rescue_rank, item in enumerate(java_rescue_candidates[:JAVA_RESCUE_MAX_SCAN], start=1):
            patch_signature = str(item.get("patch_signature") or "")
            if patch_signature and patch_signature in selected_signatures:
                continue
            rescue_item = dict(item)
            rescue_item["score"] = float(rescue_item.get("rescue_score", rescue_item.get("score", -10**9)))
            rescue_item["java_rescue_survivor"] = True
            rescue_item["java_survivor_backoff_selected"] = True
            rescue_item["java_rescue_rank"] = int(rescue_rank)
            rescue_item["java_rescue_policy_version"] = JAVA_RESCUE_POLICY_VERSION
            rescue_chunk.append(rescue_item)
            if patch_signature:
                selected_signatures.add(patch_signature)
            if len(rescue_chunk) >= JAVA_RESCUE_MAX_SURVIVORS_PER_BUG:
                break
        if rescue_chunk:
            selected_candidates.extend(rescue_chunk)
            java_rescue_used = True
            java_rescue_added_count = int(len(rescue_chunk))
            java_survivor_backoff_used = True
            java_survivor_backoff_added_count += int(len(rescue_chunk))

    valid_count = min(len(selected_candidates), target_local)
    if valid_count < target_local:
        logging.warning(f"[{bug_id}] Only {valid_count}/{target_local} valid candidates after filtering.")

    top_drop_pairs = sorted(drop_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_drop_text = ", ".join([f"{k}={v}" for k, v in top_drop_pairs if v > 0]) or "none"
    logging.info(f"[fill] bug={bug_id} {valid_count}/{target_local} (top drops: {top_drop_text})")

    elapsed = time.time() - start_time
    meta_plan_dict, meta_plan_source = parse_plan_dict(prom_row, base_code=base_code, language=language)
    bug_meta = {
        "drop_counts": {k: int(drop_counts.get(k, 0)) for k in DROP_REASON_KEYS},
        "target_candidates": int(target_local),
        "valid_candidates": int(valid_count),
        "fill_rate": round(float(valid_count) / float(max(1, target_local)), 4),
        "llm_calls_total": int(llm_calls_total),
        "elapsed_seconds": round(elapsed, 4),
        "language": language,
        "language_branch": str(language_branch),
        "repair_branch_requested": str(repair_branch_requested),
        "repair_branch_effective": str(repair_branch),
        "repair_branch_source": str(repair_branch_source),
        "repair_branch": str(repair_branch),
        "gen_n": int(max(gen_n_used) if gen_n_used else max(1, int(getattr(args, "gen_n", 1)))),
        "max_gen_passes": int(max(max_gen_passes_used) if max_gen_passes_used else max(1, int(getattr(args, "max_gen_passes", 1)))),
        "temperature_used": temperatures_used,
        "top_p_used": top_ps_used,
        "unique_after_stage": unique_after_stage,
        "scheduler": scheduler_name,
        "conf_level": str(bug_profile.get("conf_level", "low")),
        "bm25_top1": float(bug_profile.get("bm25_top1", 0.0)),
        "bm25_margin": float(bug_profile.get("bm25_margin", 0.0)),
        "profile": profile_name,
        "profiles_tried": profiles_tried,
        "plan_source": str(meta_plan_source),
        "plan_allowed_edit_types": list(meta_plan_dict.get("allowed_edit_types") or []),
        "plan_enforcement": str(plan_enforcement),
        "plan_target_match": str(plan_target_match),
        "plan_fill_used": bool(plan_fill_used),
        "plan_fill_added_n": int(plan_fill_added_n),
        "plan_fill_penalty": float(fill_with_violations_penalty if fill_goal > 0 else 0.0),
        "plan_violation_breakdown": {k: int(plan_violation_breakdown.get(k, 0)) for k in PLAN_VIOLATION_TYPE_KEYS},
        "java_filter_compile_valid": bool(java_validity_stats.get("java_filter_compile_valid", False)),
        "java_rerank_compile_valid": bool(java_validity_stats.get("java_rerank_compile_valid", False)),
        "java_candidate_compile_check": bool(java_validity_stats.get("java_candidate_compile_check", False)),
        "java_compile_validation_mode": str(java_validity_stats.get("java_compile_validation_mode") or "none"),
        "java_semantic_rerank_enabled": bool(java_validity_stats.get("java_semantic_rerank_enabled", False)),
        "java_semantic_rerank_features": list(java_validity_stats.get("java_semantic_rerank_features") or []),
        "java_branch_no_fill": bool(java_validity_stats.get("java_branch_no_fill", False)),
        "java_survivor_backoff_mode": str(java_survivor_backoff_mode),
        "java_survivor_backoff_used": bool(java_survivor_backoff_used),
        "java_survivor_backoff_added_count": int(java_survivor_backoff_added_count),
        "java_rescue_used": bool(java_rescue_used),
        "java_rescue_added_count": int(java_rescue_added_count),
        "java_rescue_policy_version": JAVA_RESCUE_POLICY_VERSION,
        "java_strict_survivor_count": int(strict_survivor_count),
        "zero_candidate_after_filter": bool(zero_candidate_after_filter),
        "java_sentinel_extract_hit": int(java_extract_stats.get("java_sentinel_extract_hit", 0) or 0),
        "java_decl_salvage_used": int(java_extract_stats.get("java_decl_salvage_used", 0) or 0),
        "java_explanation_tail_stripped": int(java_extract_stats.get("java_explanation_tail_stripped", 0) or 0),
        "java_decl_recovery_used": int(java_extract_stats.get("java_decl_recovery_used", 0) or 0),
        "java_decl_recovery_mode_counts": dict(java_extract_stats.get("java_decl_recovery_mode_counts") or {}),
        "java_compile_feedback_once": bool(java_validity_stats.get("java_compile_feedback_once", False)),
        "java_compile_feedback_available": bool(java_validity_stats.get("java_compile_feedback_available", False)),
        "java_compile_feedback_attempted_n": int(java_validity_stats.get("java_compile_feedback_attempted_n", 0)),
        "java_compile_feedback_applied_n": int(java_validity_stats.get("java_compile_feedback_applied_n", 0)),
        "java_validity_checked_n": int(java_validity_stats.get("java_validity_checked_n", 0)),
        "java_validity_valid_n": int(java_validity_stats.get("java_validity_valid_n", 0)),
        "java_validity_invalid_n": int(java_validity_stats.get("java_validity_invalid_n", 0)),
        "java_validity_rejected_n": int(java_validity_stats.get("java_validity_rejected_n", 0)),
        "java_validity_reranked_n": int(java_validity_stats.get("java_validity_reranked_n", 0)),
        "java_validity_reason_counts": dict(java_validity_stats.get("java_validity_reason_counts") or {}),
        "backend_selected": str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto"),
        "hf_model_name": str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or ""),
    }
    return selected_candidates[:target_local], bug_meta


def generate_all_candidates(
    client: QwenClient,
    inputs_dict: Dict[str, Any],
    prom_dict: Dict[str, Any],
    args: Namespace,
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    bug_items = list(inputs_dict.items())
    logging.info(f"Generating up to {args.max_per_bug} candidates for {len(bug_items)} bugs.")

    for bug_id, row in tqdm(bug_items, desc="generate (per bug)"):
        # Enrich with step-4 context when available
        ctx = copy.deepcopy(prom_dict.get(bug_id) or {})
        if "function_name" not in row and safe_get(ctx, ["function", "function_name"]):
            row["function_name"] = safe_get(ctx, ["function", "function_name"])
        if "file_path" not in row and ctx.get("file_path"):
            row["file_path"] = ctx.get("file_path")
        base_code = safe_get(ctx, ["function", "function_before"]) or row.get("code") or ""
        if not base_code.strip():
            logging.warning(f"[{bug_id}] base code missing — skip.")
            continue

        generated, bug_meta = generate_candidates_for_bug(
            client=client,
            bug_id=bug_id,
            row=row,
            base_code=base_code,
            prom_row=ctx,
            args=args,
            target=int(args.max_per_bug),
            run_seed=run_seed,
        )

        bug_candidates: List[Dict[str, Any]] = []
        for idx, cand in enumerate(generated[: max(1, int(args.max_per_bug))], start=1):
            cand_row = dict(cand or {})
            cand_row["source_rank_before_merge"] = int(cand_row.get("source_rank_before_merge") or cand_row.get("rank") or idx)
            cand_row["normalized_candidate_hash"] = str(
                cand_row.get("normalized_candidate_hash")
                or compute_normalized_candidate_hash(cand_row.get("code") or "")
            )
            bug_candidates.append(cand_row)

        results[bug_id] = {
            "code": base_code,
            "file_path": row.get("file_path"),
            "function_name": row.get("function_name"),
            "candidates": bug_candidates,
            "language": str(bug_meta.get("language", infer_language(row, ctx, base_code))),
            "language_branch": str(bug_meta.get("language_branch", language_branch_for_language(infer_language(row, ctx, base_code)))),
            "repair_branch_requested": str(
                bug_meta.get("repair_branch_requested", normalize_repair_branch(getattr(args, "repair_branch", "auto")))
            ),
            "repair_branch_effective": str(
                bug_meta.get("repair_branch_effective", bug_meta.get("repair_branch", resolve_effective_repair_branch(args, row, ctx, infer_language(row, ctx, base_code))))
            ),
            "repair_branch_source": str(bug_meta.get("repair_branch_source", "auto")),
            "repair_branch": str(bug_meta.get("repair_branch", resolve_effective_repair_branch(args, row, ctx, infer_language(row, ctx, base_code)))),
            "allowed_edit_types": bug_meta.get("plan_allowed_edit_types", ["REPLACE"]),
            "drop_counts": bug_meta.get("drop_counts", {k: 0 for k in DROP_REASON_KEYS}),
            "target_candidates": int(bug_meta.get("target_candidates", int(args.max_per_bug))),
            "valid_candidates": int(bug_meta.get("valid_candidates", len(bug_candidates))),
            "fill_rate": float(bug_meta.get("fill_rate", 0.0)),
            "llm_calls_total": int(bug_meta.get("llm_calls_total", 0)),
            "elapsed_seconds": float(bug_meta.get("elapsed_seconds", 0.0)),
            "gen_n": int(bug_meta.get("gen_n", int(getattr(args, "gen_n", 1)))),
            "max_gen_passes": int(bug_meta.get("max_gen_passes", int(getattr(args, "max_gen_passes", 1)))),
            "temperature_used": bug_meta.get("temperature_used", []),
            "top_p_used": bug_meta.get("top_p_used", []),
            "unique_after_stage": bug_meta.get("unique_after_stage", {}),
            "scheduler": str(bug_meta.get("scheduler", getattr(args, "scheduler", "fixed"))),
            "conf_level": str(bug_meta.get("conf_level", "low")),
            "bm25_top1": float(bug_meta.get("bm25_top1", 0.0)),
            "bm25_margin": float(bug_meta.get("bm25_margin", 0.0)),
            "profile": str(bug_meta.get("profile", "MED")),
            "profiles_tried": bug_meta.get("profiles_tried", []),
            "plan_source": str(bug_meta.get("plan_source", "fallback_default")),
            "plan_enforcement": str(bug_meta.get("plan_enforcement", resolve_plan_enforcement(args))),
            "plan_target_match": str(bug_meta.get("plan_target_match", getattr(args, "plan_target_match", "overlap"))),
            "plan_fill_used": bool(bug_meta.get("plan_fill_used", False)),
            "plan_fill_added_n": int(bug_meta.get("plan_fill_added_n", 0)),
            "plan_fill_penalty": float(bug_meta.get("plan_fill_penalty", 0.0)),
            "plan_violation_breakdown": bug_meta.get("plan_violation_breakdown", {k: 0 for k in PLAN_VIOLATION_TYPE_KEYS}),
            "java_filter_compile_valid": bool(bug_meta.get("java_filter_compile_valid", False)),
            "java_rerank_compile_valid": bool(bug_meta.get("java_rerank_compile_valid", False)),
            "java_candidate_compile_check": bool(bug_meta.get("java_candidate_compile_check", False)),
            "java_compile_validation_mode": str(bug_meta.get("java_compile_validation_mode", "none")),
            "java_semantic_rerank_enabled": bool(bug_meta.get("java_semantic_rerank_enabled", False)),
            "java_semantic_rerank_features": bug_meta.get("java_semantic_rerank_features", list(JAVA_SEMANTIC_RERANK_FEATURES)),
            "java_branch_no_fill": bool(bug_meta.get("java_branch_no_fill", False)),
            "java_survivor_backoff_mode": str(bug_meta.get("java_survivor_backoff_mode", "off")),
            "java_survivor_backoff_used": bool(bug_meta.get("java_survivor_backoff_used", False)),
            "java_survivor_backoff_added_count": int(bug_meta.get("java_survivor_backoff_added_count", 0)),
            "java_rescue_used": bool(bug_meta.get("java_rescue_used", False)),
            "java_rescue_added_count": int(bug_meta.get("java_rescue_added_count", 0)),
            "java_rescue_policy_version": str(bug_meta.get("java_rescue_policy_version", JAVA_RESCUE_POLICY_VERSION)),
            "java_strict_survivor_count": int(bug_meta.get("java_strict_survivor_count", 0)),
            "zero_candidate_after_filter": bool(bug_meta.get("zero_candidate_after_filter", False)),
            "java_compile_feedback_once": bool(bug_meta.get("java_compile_feedback_once", False)),
            "java_compile_feedback_available": bool(bug_meta.get("java_compile_feedback_available", False)),
            "java_compile_feedback_attempted_n": int(bug_meta.get("java_compile_feedback_attempted_n", 0)),
            "java_compile_feedback_applied_n": int(bug_meta.get("java_compile_feedback_applied_n", 0)),
            "java_validity_checked_n": int(bug_meta.get("java_validity_checked_n", 0)),
            "java_validity_valid_n": int(bug_meta.get("java_validity_valid_n", 0)),
            "java_validity_invalid_n": int(bug_meta.get("java_validity_invalid_n", 0)),
            "java_validity_rejected_n": int(bug_meta.get("java_validity_rejected_n", 0)),
            "java_validity_reranked_n": int(bug_meta.get("java_validity_reranked_n", 0)),
            "java_validity_reason_counts": bug_meta.get("java_validity_reason_counts", {k: 0 for k in JAVA_VALIDITY_REASON_KEYS}),
            "backend_selected": str(bug_meta.get("backend_selected", getattr(args, "_backend_selected", getattr(args, "backend", "auto")))),
            "hf_model_name": str(bug_meta.get("hf_model_name", getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")))),
        }

    return results

# ===== Pipeline =====
def get_parser():
    p = ArgumentParser(description="5.TokenPatchGeneratorAgent — prompt-guided code generation (vLLM)")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--backend", type=str, default="auto", choices=list(BACKEND_CHOICES))
    p.add_argument("--hf_model_name", type=str, default="", help="HF fallback model/path. Defaults to --model_name.")
    p.add_argument("--hf_device", type=str, default="auto", choices=list(HF_DEVICE_CHOICES))
    p.add_argument("--hf_dtype", type=str, default="auto", choices=list(HF_DTYPE_CHOICES))
    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are an expert bug fixer. "
            "Follow the user's guardrails exactly. "
            "After ##correct, output ONLY the complete corrected function or class (including its signature). "
            "Do NOT output a diff. Do NOT output a one-line snippet. Do NOT include explanations. "
            "Code fences are optional."
        ),
    )
    p.add_argument(
        "--prompts_file",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="Stage-4 PlanAgent JSON file, directory, or glob pattern. Stage-3 prompt JSON is also accepted as fallback.",
    )
    p.add_argument("--out_file", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument("--limit_bugs", type=int, default=0, help="If >0, only process the first N bugs from the prompts JSON.")
    p.add_argument("--max_per_bug", type=int, default=60, help="Search candidates per bug before final top-k cut")
    p.add_argument("--final_top_k", type=int, default=10, help="Final candidates per bug after merge/rerank (<=0 disables cut).")
    p.add_argument("--batch_size", type=int, default=20, help="Multiplier for candidates per pass. Total requested per pass = gen_n * batch_size (single prompt).")
    p.add_argument("--max_new_tokens", type=int, default=640)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    # Sampling is on by default; override with --greedy if needed
    p.add_argument("--do_sample", action="store_true", default=True, help="Deprecated; kept for backward compatibility")
    p.add_argument("--greedy", action="store_true", help="Generate only the greedy candidate (no sampling)")
    p.add_argument(
        "--mix_greedy_first",
        dest="mix_greedy_first",
        action="store_true",
        help="Force Greedy-1 + Diverse-(final_top_k-1) in one run (default: enabled). (--greedy overrides this.)",
    )
    p.add_argument(
        "--no_mix_greedy_first",
        dest="mix_greedy_first",
        action="store_false",
        help="Disable Greedy-1 + Diverse-(K-1) mixing and use pure sampling unless --greedy is set.",
    )
    p.set_defaults(mix_greedy_first=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0, help="Optional vLLM top_k (0 disables).")
    p.add_argument("--presence_penalty", type=float, default=0.0, help="vLLM presence_penalty to reduce duplicates (0 disables).")
    p.add_argument("--frequency_penalty", type=float, default=0.0, help="vLLM frequency_penalty to reduce duplicates (0 disables).")
    p.add_argument("--stop_sequences", type=str, default="", help="Comma-separated stop sequences passed to vLLM (empty disables).")
    p.add_argument("--gen_n", type=int, default=4, help="vLLM SamplingParams.n (candidates per prompt).")
    p.add_argument("--max_gen_passes", type=int, default=2, help="Max generation passes per bug.")
    p.add_argument("--temp_step", type=float, default=0.3, help="Temperature increase for retry pass.")
    p.add_argument("--top_p_step", type=float, default=0.05, help="Top-p increase for retry pass.")
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="41,42,43", help="Comma-separated seeds for multi-seed generation (leave empty to use --seed only)")
    p.add_argument("--compact", action="store_true", help="Unused flag; kept for backward compatibility")
    p.add_argument("--merge_inputs", type=str, default="", help="Comma-separated patch JSONs to merge/re-rerank instead of generating")
    p.add_argument(
        "--run_config",
        action="append",
        default=[],
        help=(
            "Repeatable per-run config for multi-run pooling. "
            "Example: --run_config tag=A,seed=42,temp=0.7,top_p=0.95,presence_penalty=0.0 "
            "(Supports: tag/name, seed, temp/temperature, top_p, top_k, presence_penalty/presence, frequency_penalty/frequency)"
        ),
    )
    p.add_argument(
        "--multi_run_preset",
        type=str,
        default="3run_default",
        choices=["", "3run_default"],
        help="Convenience preset for multi-run configs when --run_config is not provided. Default: 3run_default",
    )
    p.add_argument(
        "--coverage_top_k",
        dest="coverage_top_k",
        action="store_true",
        help="When applying final_top_k, enforce coverage across --coverage_key groups (recommended for multi-run pooling, default: enabled).",
    )
    p.add_argument(
        "--no_coverage_top_k",
        dest="coverage_top_k",
        action="store_false",
        help="Disable coverage-aware final top-k selection in multi-run pooling.",
    )
    p.set_defaults(coverage_top_k=True)
    p.add_argument(
        "--coverage_key",
        type=str,
        default="source_run",
        help="Candidate field used to form coverage groups (e.g., source_run, prompt_variant).",
    )
    p.add_argument("--metrics_out", type=Path, default=None, help="Optional path to write run metrics (JSON)")
    p.add_argument(
        "--repair_branch",
        type=str,
        default="auto",
        choices=REPAIR_BRANCH_CHOICES,
        help="Explicit shared-core repair branch selector. auto => python_base for Python, java_base for Java.",
    )
    p.add_argument(
        "--java_filter_compile_valid",
        action="store_true",
        help="For Java only, drop candidates that fail the lightweight compile-validity proxy. Default off.",
    )
    p.add_argument(
        "--java_rerank_compile_valid",
        action="store_true",
        help="For Java only, prefer compile-valid candidates during ranking using the lightweight proxy. Default off.",
    )
    p.add_argument(
        "--java_hard_reject_compile_risk_combos",
        action="store_true",
        default=False,
        help="For Java only, promote compile-risk reason combos to hard rejects while leaving single soft reasons alone. Default off.",
    )
    p.add_argument(
        "--java_candidate_compile_check",
        action="store_true",
        help="Record whether evaluation will run --java_candidate_compile_check. This does not change Stage 5 generation.",
    )
    p.add_argument(
        "--java_compile_feedback_once",
        action="store_true",
        help="Java-only opt-in extension: wire one compile-feedback repair attempt when a safe local compile path is available. Default off.",
    )
    p.add_argument(
        "--java_survivor_backoff",
        type=str,
        default="off",
        choices=list(JAVA_SURVIVOR_BACKOFF_CHOICES),
        help="Java-only survivor recovery after strict filtering: off, hybrid (refill to min_survivors_per_bug), or fill (full filter_then_fill-style refill).",
    )
    p.add_argument("--no_rerank", action="store_true", help="Disable score-based reranking; keep generation order.")
    p.add_argument(
        "--filter_plan_violations",
        action="store_true",
        help="Deprecated compatibility flag. Use --plan_enforcement instead; filter_then_fill remains the default backoff behavior.",
    )
    p.add_argument(
        "--plan_enforcement",
        type=str,
        default="filter_then_fill",
        choices=["off", "penalty", "filter", "filter_then_fill"],
        help="How to handle plan-violating candidates. Default keeps strict preference plus fill backoff.",
    )
    p.add_argument(
        "--plan_target_match",
        type=str,
        default="overlap",
        choices=["contain", "overlap"],
        help="How edit spans match target_locations. overlap is more tolerant for localized plans.",
    )
    p.add_argument(
        "--plan_violation_penalty",
        type=float,
        default=15.0,
        help="When filtering is off, subtract this score from plan-violating candidates (score scale is tens to hundreds; 15 is recommended).",
    )
    p.add_argument(
        "--plan_violation_penalty_per_violation",
        action="store_true",
        help="Multiply --plan_violation_penalty by the number of plan violations for each candidate.",
    )
    p.add_argument("--min_survivors_per_bug", type=int, default=3, help="Minimum number of candidates to preserve per bug via plan backoff fill.")
    p.add_argument(
        "--fill_with_violations_penalty",
        type=float,
        default=25.0,
        help="Penalty applied when filling from buffered plan-violating candidates under filter_then_fill.",
    )
    p.add_argument("--require_plan_json", action="store_true", help="Require plan_json usage; missing plans fall back to the default plan.")
    p.add_argument(
        "--scheduler",
        type=str,
        default="fixed",
        choices=["fixed", "cost_aware"],
        help="Per-bug scheduling mode. cost_aware uses retrieval confidence and repair difficulty.",
    )
    p.add_argument("--conf_margin_high", type=float, default=0.5)
    p.add_argument("--conf_margin_med", type=float, default=0.2)
    p.add_argument("--easy_target", type=int, default=12, help="Per-bug unique candidate target for EASY bugs.")
    p.add_argument("--med_target", type=int, default=30, help="Per-bug unique candidate target for MED bugs.")
    p.add_argument("--hard_target", type=int, default=None, help="Per-bug unique candidate target for HARD bugs (default: --max_per_bug).")
    return p


def main(args: Namespace):
    args._argv = list(sys.argv)
    args._backend_selected = str(getattr(args, "_backend_selected", "not_used") or "not_used")
    args._hf_model_name_selected = str(getattr(args, "_hf_model_name_selected", "") or "")
    if bool(getattr(args, "filter_plan_violations", False)) and resolve_plan_enforcement(args) == "filter_then_fill":
        logging.info(
            "Legacy --filter_plan_violations detected; keeping filter_then_fill backoff behavior. "
            "Use --plan_enforcement filter for strict dropping."
        )
    prompts_path = resolve_prompts_path(args.prompts_file)
    if prompts_path is None:
        if not args.merge_inputs:
            logging.error(
                "No prompt/plan JSON found. Checked: %s and %s/*PlanAgent*.json",
                args.prompts_file,
                DEFAULT_PROMPTS_PATH,
            )
            return
        prom_dict = {}
    else:
        if not args.prompts_file.exists() or prompts_path != args.prompts_file:
            logging.info("Resolved prompts_file -> %s", prompts_path)
        prom_dict = load_json(prompts_path)

    merge_paths = [Path(p.strip()) for p in args.merge_inputs.split(",") if p.strip()]
    if merge_paths:
        logging.info(f"[merge] Merging {len(merge_paths)} files with max_per_bug={args.max_per_bug}")
        merged = merge_patch_results(merge_paths, prom_dict, args.max_per_bug, rerank=not args.no_rerank, args=args)
        if int(args.final_top_k) > 0:
            merged, _cut_meta = apply_final_top_k(merged, int(args.final_top_k), coverage_top_k=bool(getattr(args, "coverage_top_k", False)), coverage_key=str(getattr(args, "coverage_key", "source_run")))
            logging.info(f"[merge] applied final_top_k={int(args.final_top_k)}")
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        args._dataset_tag_current = infer_dataset_tag(prompts_path or args.prompts_file)
        manifest_path = write_run_manifest(
            args=args,
            prompts_path=prompts_path or args.prompts_file,
            out_path=args.out_file,
            metrics_path=args.metrics_out,
            dataset_tag=infer_dataset_tag(prompts_path or args.prompts_file),
            results=merged,
        )
        logging.info(f"[merge] saved: {args.out_file}")
        logging.info(f"[merge] manifest saved: {manifest_path}")
        return

    run_specs = build_run_specs(args)
    if str(getattr(args, "scheduler", "fixed")) == "cost_aware" and not list(getattr(args, "run_config", []) or []):
        run_specs = [
            {
                'tag': f'seed{int(args.seed)}',
                'seed': int(args.seed),
                'temperature': float(args.temperature),
                'top_p': float(args.top_p),
                'presence_penalty': float(getattr(args, 'presence_penalty', 0.0) or 0.0),
                'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0),
                'top_k': int(getattr(args, 'top_k', 0) or 0),
            }
        ]
        logging.info(
            "scheduler=cost_aware without explicit --run_config: forcing a single dataset-level run spec; "
            "hard diversity profiles will be applied inside generate_candidates_for_bug()."
        )
    if not run_specs:
        run_specs = [{'tag': f'seed{int(args.seed)}', 'seed': int(args.seed), 'temperature': float(args.temperature), 'top_p': float(args.top_p), 'presence_penalty': float(getattr(args,'presence_penalty',0.0) or 0.0), 'frequency_penalty': float(getattr(args,'frequency_penalty',0.0) or 0.0), 'top_k': int(getattr(args,'top_k',0) or 0)}]

    spec_str = ", ".join([f"{s['tag']}[seed={s['seed']},T={s['temperature']},p={s['top_p']},pres={s['presence_penalty']}]" for s in run_specs])
    logging.info(
        f"Running generation for {len(run_specs)} run(s): {spec_str} "
        f"(search_per_bug={args.max_per_bug}, final_top_k={int(args.final_top_k)})"
    )

    client = QwenClient(
        args.model_name,
        system_prompt=args.system_prompt,
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        backend=str(getattr(args, "backend", "auto") or "auto"),
        hf_model_name=str(getattr(args, "hf_model_name", "") or args.model_name),
        hf_device=str(getattr(args, "hf_device", "auto") or "auto"),
        hf_dtype=str(getattr(args, "hf_dtype", "auto") or "auto"),
    )
    args._backend_selected = str(getattr(client, "backend_selected", getattr(args, "backend", "auto")) or "auto")
    args._hf_model_name_selected = str(getattr(client, "hf_model_name", "") or "")

    # Default behavior: if prompts_file is the default stage-4 directory,
    # run full bugsinpy+defects4j prompt sets in sequence.
    run_targets: List[Tuple[Path, Path, Optional[Path], str]] = []
    full_default_mode = (args.prompts_file == DEFAULT_PROMPTS_PATH and args.prompts_file.is_dir())
    if full_default_mode:
        full_paths = resolve_full_prompt_paths(args.prompts_file)
        for p in full_paths:
            ds = infer_dataset_tag(p)
            run_targets.append((p, derive_full_out_path(args.out_file, ds), derive_full_metrics_path(args.metrics_out, ds), ds))

    if not run_targets:
        if prompts_path is None:
            logging.error("No usable input data (prompt JSON is missing).")
            return
        run_targets.append((prompts_path, args.out_file, args.metrics_out, infer_dataset_tag(prompts_path)))

    if args._backend_selected == "hf" and not list(getattr(args, "run_config", []) or []):
        for _cur_prompts_path, _cur_out_path, _cur_metrics_path, dataset_tag in run_targets:
            dataset_language = "java" if str(dataset_tag).strip().lower() == "defects4j" else "python"
            dataset_repair_branch = resolve_effective_repair_branch(args, {}, {}, dataset_language)
            if dataset_repair_branch in JAVA_REPAIR_BRANCHES:
                raise RuntimeError(
                    "HF backend selected for a Java repair branch without explicit --run_config. "
                    "Pass --run_config or use backend auto/vllm for java_base/java_semantic."
                )

    for cur_prompts_path, cur_out_path, cur_metrics_path, dataset_tag in run_targets:
        logging.info("[dataset=%s] prompts=%s -> out=%s", dataset_tag, cur_prompts_path, cur_out_path)
        args._dataset_tag_current = str(dataset_tag)
        cur_prom_dict = load_json(cur_prompts_path)
        if not cur_prom_dict:
            logging.error("[dataset=%s] prompt JSON is empty or invalid: %s", dataset_tag, cur_prompts_path)
            continue
        cur_prom_dict = limit_bug_rows(cur_prom_dict, int(getattr(args, "limit_bugs", 0) or 0))

        inputs_dict = build_inputs_from_prompts(cur_prom_dict)
        if not inputs_dict:
            logging.error("[dataset=%s] no usable inputs from prompts: %s", dataset_tag, cur_prompts_path)
            continue

        dataset_run_specs = list(run_specs)
        dataset_language = "java" if str(dataset_tag).strip().lower() == "defects4j" else "python"
        dataset_repair_branch = resolve_effective_repair_branch(args, {}, {}, dataset_language)
        if args._backend_selected == "hf" and not list(getattr(args, "run_config", []) or []):
            dataset_run_specs = [{
                'tag': f'seed{int(args.seed)}',
                'seed': int(args.seed),
                'temperature': float(args.temperature),
                'top_p': float(args.top_p),
                'presence_penalty': float(getattr(args, 'presence_penalty', 0.0) or 0.0),
                'frequency_penalty': float(getattr(args, 'frequency_penalty', 0.0) or 0.0),
                'top_k': int(getattr(args, 'top_k', 0) or 0),
            }]
            logging.info(
                "[dataset=%s] HF backend selected; preserving Python smoke-friendly single-run fallback.",
                dataset_tag,
            )

        run_start = time.time()
        # reset per-run metrics so metrics_out is per dataset
        client.metrics = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "batches": 0.0,
            "max_gpu_mb": 0.0,
        }

        run_results: List[Dict[str, Any]] = []
        for spec in dataset_run_specs:
            seed = int(spec.get('seed', getattr(args, 'seed', 42)))
            tag = str(spec.get('tag', f'seed{seed}'))

            run_args = copy.deepcopy(args)
            run_args._dataset_tag_current = str(dataset_tag)
            # override per-run sampling profile
            run_args.seed = int(seed)
            run_args.temperature = float(spec.get('temperature', run_args.temperature))
            run_args.top_p = float(spec.get('top_p', run_args.top_p))
            run_args.presence_penalty = float(spec.get('presence_penalty', getattr(run_args, 'presence_penalty', 0.0) or 0.0))
            run_args.frequency_penalty = float(spec.get('frequency_penalty', getattr(run_args, 'frequency_penalty', 0.0) or 0.0))
            if spec.get('top_k') is not None:
                run_args.top_k = int(spec.get('top_k') or 0)

            set_seed(seed)
            logging.info(f"[run][{dataset_tag}] {tag} seed={seed} T={run_args.temperature} top_p={run_args.top_p} pres={run_args.presence_penalty}")
            cur_results = generate_all_candidates(client, inputs_dict, cur_prom_dict, run_args, run_seed=seed)
            # annotate provenance for pooling
            for _bid, _row in cur_results.items():
                for _c in (_row.get('candidates') or []):
                    if isinstance(_c, dict):
                        _c.setdefault('source_run', tag)
                        _c.setdefault('source_seed', seed)
            run_results.append(cur_results)

        if len(run_results) == 1:
            results = run_results[0]
        else:
            results = merge_results_dicts(run_results, cur_prom_dict, args.max_per_bug, rerank=not args.no_rerank, args=args)

        cut_meta = {
            "source_target_candidates_total": 0,
            "source_valid_candidates_total": 0,
        }
        if int(args.final_top_k) > 0:
            results, cut_meta = apply_final_top_k(results, int(args.final_top_k), coverage_top_k=bool(getattr(args, "coverage_top_k", False)), coverage_key=str(getattr(args, "coverage_key", "source_run")))
            logging.info(f"[dataset={dataset_tag}] applied final_top_k={int(args.final_top_k)}")

        cur_out_path.parent.mkdir(parents=True, exist_ok=True)
        cur_out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"[dataset={dataset_tag}] saved: {cur_out_path}")
        manifest_path = write_run_manifest(
            args=args,
            prompts_path=cur_prompts_path,
            out_path=cur_out_path,
            metrics_path=cur_metrics_path,
            dataset_tag=dataset_tag,
            results=results,
        )
        logging.info(f"[dataset={dataset_tag}] manifest saved: {manifest_path}")

        if cur_metrics_path:
            elapsed = time.time() - run_start
            total_candidates = sum(len((row.get("candidates") or [])) for row in results.values())
            total_bugs = len(results)
            total_target = sum(int((row.get("target_candidates") or 0)) for row in results.values())
            total_valid = sum(int((row.get("valid_candidates") or len(row.get("candidates") or []))) for row in results.values())
            total_llm_calls = sum(int((row.get("llm_calls_total") or 0)) for row in results.values())
            source_target_total = int(cut_meta.get("source_target_candidates_total") or total_target)
            source_valid_total = int(cut_meta.get("source_valid_candidates_total") or total_valid)
            avg_bug_elapsed = (
                sum(float((row.get("elapsed_seconds") or 0.0)) for row in results.values()) / max(1, total_bugs)
            )
            plan_fill_used_bugs = sum(1 for row in results.values() if bool(row.get("plan_fill_used", False)))
            plan_fill_added_total = sum(int(row.get("plan_fill_added_n") or 0) for row in results.values())
            plan_violation_breakdown_total = {k: 0 for k in PLAN_VIOLATION_TYPE_KEYS}
            java_validity_reason_counts_total = {k: 0 for k in JAVA_VALIDITY_REASON_KEYS}
            java_validity_checked_total = 0
            java_validity_valid_total = 0
            java_validity_invalid_total = 0
            java_validity_rejected_total = 0
            java_validity_reranked_total = 0
            repair_branch_counts: Dict[str, int] = {}
            repair_branch_source_counts: Dict[str, int] = {}
            language_branch_counts: Dict[str, int] = {}
            effective_plan_enforcement_counts: Dict[str, int] = {}
            java_survivor_backoff_mode_counts: Dict[str, int] = {}
            java_survivor_backoff_used_bugs = 0
            java_survivor_backoff_added_total = 0
            java_rescue_used_bugs = 0
            java_rescue_added_total = 0
            zero_candidate_after_filter_bugs = 0
            java_sentinel_extract_hit = 0
            java_decl_salvage_used = 0
            java_explanation_tail_stripped = 0
            java_decl_recovery_used = 0
            java_decl_recovery_mode_counts: Dict[str, int] = {}
            extraction_failed_empty_after_trim_total = 0
            extraction_failed_postprocess_total = 0
            introduced_java_incompatible_syntax_total = 0
            for row in results.values():
                plan_violation_breakdown_total = merge_counter_dicts(
                    plan_violation_breakdown_total,
                    row.get("plan_violation_breakdown") or {},
                    PLAN_VIOLATION_TYPE_KEYS,
                )
                java_validity_reason_counts_total = merge_counter_dicts(
                    java_validity_reason_counts_total,
                    row.get("java_validity_reason_counts") or {},
                    JAVA_VALIDITY_REASON_KEYS,
                )
                java_validity_checked_total += int(row.get("java_validity_checked_n") or 0)
                java_validity_valid_total += int(row.get("java_validity_valid_n") or 0)
                java_validity_invalid_total += int(row.get("java_validity_invalid_n") or 0)
                java_validity_rejected_total += int(row.get("java_validity_rejected_n") or 0)
                java_validity_reranked_total += int(row.get("java_validity_reranked_n") or 0)
                repair_branch = str(row.get("repair_branch") or "unknown")
                repair_branch_source = str(row.get("repair_branch_source") or "unknown")
                language_branch = str(row.get("language_branch") or "unknown")
                effective_plan = str(row.get("plan_enforcement") or "unknown")
                backoff_mode = str(row.get("java_survivor_backoff_mode") or "off")
                repair_branch_counts[repair_branch] = int(repair_branch_counts.get(repair_branch, 0)) + 1
                repair_branch_source_counts[repair_branch_source] = int(
                    repair_branch_source_counts.get(repair_branch_source, 0)
                ) + 1
                language_branch_counts[language_branch] = int(language_branch_counts.get(language_branch, 0)) + 1
                effective_plan_enforcement_counts[effective_plan] = int(
                    effective_plan_enforcement_counts.get(effective_plan, 0)
                ) + 1
                java_survivor_backoff_mode_counts[backoff_mode] = int(
                    java_survivor_backoff_mode_counts.get(backoff_mode, 0)
                ) + 1
                java_survivor_backoff_used_bugs += 1 if bool(row.get("java_survivor_backoff_used", False)) else 0
                java_survivor_backoff_added_total += int(row.get("java_survivor_backoff_added_count") or 0)
                java_rescue_used_bugs += 1 if bool(row.get("java_rescue_used", False)) else 0
                java_rescue_added_total += int(row.get("java_rescue_added_count") or 0)
                zero_candidate_after_filter_bugs += 1 if bool(row.get("zero_candidate_after_filter", False)) else 0
                java_sentinel_extract_hit += int(row.get("java_sentinel_extract_hit") or 0)
                java_decl_salvage_used += int(row.get("java_decl_salvage_used") or 0)
                java_explanation_tail_stripped += int(row.get("java_explanation_tail_stripped") or 0)
                java_decl_recovery_used += int(row.get("java_decl_recovery_used") or 0)
                for mode_key, mode_val in dict(row.get("java_decl_recovery_mode_counts") or {}).items():
                    key = str(mode_key or "unknown")
                    java_decl_recovery_mode_counts[key] = int(java_decl_recovery_mode_counts.get(key, 0) or 0) + int(mode_val or 0)
                row_drop = dict(row.get("drop_counts") or {})
                extraction_failed_empty_after_trim_total += int(row_drop.get("extraction_failed_empty_after_trim") or 0)
                extraction_failed_postprocess_total += int(row_drop.get("extraction_failed_postprocess") or 0)
                introduced_java_incompatible_syntax_total += int(row_drop.get("introduced_java_incompatible_syntax") or 0)
            in_tok = int(client.metrics.get("input_tokens", 0))
            out_tok = int(client.metrics.get("output_tokens", 0))
            branch_meta = resolve_repair_branch_metadata(args, None, None, dataset_language)
            requested_repair_branch = str(
                branch_meta.get("requested") or normalize_repair_branch(getattr(args, "repair_branch", "auto"))
            )
            effective_summary_branch = str(branch_meta.get("effective") or requested_repair_branch)
            effective_plan = str(resolve_plan_enforcement(args))
            java_survivor_backoff_mode = str(
                resolve_effective_java_survivor_backoff(args, effective_summary_branch, effective_plan)
            )
            summary = {
                "dataset": dataset_tag,
                "seeds": [int(s.get("seed")) for s in dataset_run_specs],
                "run_specs": dataset_run_specs,
                "bugs": total_bugs,
                "candidates": total_candidates,
                "target_candidates_total": total_target,
                "valid_candidates_total": total_valid,
                "fill_rate_overall": round(float(total_valid) / float(max(1, total_target)), 4),
                "llm_calls_total": total_llm_calls,
                "search_per_bug": int(args.max_per_bug),
                "final_top_k": int(args.final_top_k),
                "limit_bugs": int(getattr(args, "limit_bugs", 0) or 0),
                "source_generation_target_candidates_total": source_target_total,
                "source_generation_valid_candidates_total": source_valid_total,
                "source_generation_fill_rate_overall": round(float(source_valid_total) / float(max(1, source_target_total)), 4),
                "avg_bug_elapsed_seconds": round(avg_bug_elapsed, 4),
                "elapsed_seconds": round(elapsed, 3),
                "avg_seconds_per_candidate": round(elapsed / max(1, total_candidates), 4),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "avg_input_tokens_per_candidate": round(in_tok / max(1, total_candidates), 2),
                "avg_output_tokens_per_candidate": round(out_tok / max(1, total_candidates), 2),
                "max_gpu_mb": round(client.metrics.get("max_gpu_mb", 0.0), 2),
                "batches": int(client.metrics.get("batches", 0)),
                "prompts_file": str(cur_prompts_path),
                "out_file": str(cur_out_path),
                "argv": list(getattr(args, "_argv", sys.argv)),
                "backend_selected": str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto"),
                "hf_model_name": str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or ""),
                "repair_branch_requested": str(requested_repair_branch),
                "repair_branch_effective": _single_count_value(repair_branch_counts, effective_summary_branch),
                "repair_branch_source": _single_count_value(
                    repair_branch_source_counts,
                    str(branch_meta.get("source") or "auto"),
                ),
                "repair_branch": _single_count_value(repair_branch_counts, effective_summary_branch),
                "plan_enforcement": str(effective_plan),
                "repair_branch_counts": repair_branch_counts,
                "repair_branch_source_counts": repair_branch_source_counts,
                "language_branch_counts": language_branch_counts,
                "effective_plan_enforcement_counts": effective_plan_enforcement_counts,
                "plan_target_match": str(getattr(args, "plan_target_match", "overlap") or "overlap"),
                "min_survivors_per_bug": int(getattr(args, "min_survivors_per_bug", 3) or 3),
                "fill_with_violations_penalty": float(getattr(args, "fill_with_violations_penalty", 25.0) or 25.0),
                "java_filter_compile_valid": bool(getattr(args, "java_filter_compile_valid", False)),
                "java_rerank_compile_valid": bool(getattr(args, "java_rerank_compile_valid", False)),
                "java_compile_validation_mode": str(resolve_java_compile_validation_mode(args)),
                "java_candidate_compile_check": bool(getattr(args, "java_candidate_compile_check", False)),
                "java_semantic_rerank_enabled": bool(effective_summary_branch == "java_semantic"),
                "java_semantic_rerank_features": list(JAVA_SEMANTIC_RERANK_FEATURES),
                "java_branch_no_fill": bool(
                    effective_summary_branch in JAVA_REPAIR_BRANCHES
                    and effective_plan != "filter_then_fill"
                    and java_survivor_backoff_mode == "off"
                ),
                "java_survivor_backoff_mode": str(java_survivor_backoff_mode),
                "java_survivor_backoff_mode_counts": java_survivor_backoff_mode_counts,
                "java_survivor_backoff_used_bugs": int(java_survivor_backoff_used_bugs),
                "java_survivor_backoff_added_total": int(java_survivor_backoff_added_total),
                "java_rescue_policy_version": JAVA_RESCUE_POLICY_VERSION,
                "java_rescue_used_bugs": int(java_rescue_used_bugs),
                "java_rescue_added_total": int(java_rescue_added_total),
                "zero_candidate_after_filter_bugs": int(zero_candidate_after_filter_bugs),
                "java_sentinel_extract_hit": int(java_sentinel_extract_hit),
                "java_decl_salvage_used": int(java_decl_salvage_used),
                "java_explanation_tail_stripped": int(java_explanation_tail_stripped),
                "java_decl_recovery_used": int(java_decl_recovery_used),
                "java_decl_recovery_mode_counts": java_decl_recovery_mode_counts,
                "extraction_failed_empty_after_trim_total": int(extraction_failed_empty_after_trim_total),
                "extraction_failed_postprocess_total": int(extraction_failed_postprocess_total),
                "introduced_java_incompatible_syntax_total": int(introduced_java_incompatible_syntax_total),
                "java_compile_feedback_once": bool(getattr(args, "java_compile_feedback_once", False)),
                "java_compile_feedback_available": bool(resolve_java_compile_feedback_available(args, "java")),
                "java_validity_checked_total": int(java_validity_checked_total),
                "java_validity_valid_total": int(java_validity_valid_total),
                "java_validity_invalid_total": int(java_validity_invalid_total),
                "java_validity_rejected_total": int(java_validity_rejected_total),
                "java_validity_reranked_total": int(java_validity_reranked_total),
                "java_validity_reason_counts_total": java_validity_reason_counts_total,
                "java_compile_feedback_attempted_total": int(sum(int(row.get("java_compile_feedback_attempted_n") or 0) for row in results.values())),
                "java_compile_feedback_applied_total": int(sum(int(row.get("java_compile_feedback_applied_n") or 0) for row in results.values())),
                "plan_fill_used_bugs": int(plan_fill_used_bugs),
                "plan_fill_added_total": int(plan_fill_added_total),
                "plan_violation_breakdown_total": plan_violation_breakdown_total,
                "git_commit": try_get_git_commit(),
            }
            cur_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            cur_metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logging.info(f"[dataset={dataset_tag}] metrics saved: {cur_metrics_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
