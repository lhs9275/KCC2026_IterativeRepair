#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate prompts from Results/2/* (BM25-enriched).
Outputs Results/3/* with compact AST hints, grounding, and basic bug info.
Default output naming follows: 3.<dataset>.<split>_GeneratePromport.json
"""

import argparse
import copy
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Defaults ---
RESULTS_ROOT = Path(os.environ.get("PIPELINE_RESULTS_ROOT", "./Results"))

DEFAULT_INPUT_JSON_PATH = RESULTS_ROOT / "2"
DEFAULT_OUTPUT_JSON_PATH = RESULTS_ROOT / "3/3.GeneratePromport.json"

DEFAULT_MAX_FULL_CODE_LINES = 200
DEFAULT_MAX_CONTEXT_LINES = 6
DEFAULT_MAX_HINT_LINES = 20
DEFAULT_MAX_HINT_CODE_LINES = 4
DEFAULT_MAX_DIFF_LINES = 20
DEFAULT_EDIT_MODE = "single"
REPAIR_BRANCH_CHOICES = ("auto", "python_base", "java_base", "java_semantic", "java_v2")

# --- Utilities ---

def read_json_file(file_path: Path) -> Optional[Union[Dict[str, Any], List[Any]]]:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다 '{file_path}'")
    except json.JSONDecodeError:
        print(f"❌ 오류: JSON 파싱에 실패했습니다 '{file_path}'")
    return None


def write_json_file(data: Union[Dict[str, Any], List[Any]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def trim_text_by_lines(text: Optional[str], max_lines: int) -> str:
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").splitlines()
    if max_lines <= 0:
        return ""
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n... (truncated)"
    return "\n".join(lines)


def _dedup_keep_order(items: List[str], limit: Optional[int] = None) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def _as_str(x: Any, fallback: str = "") -> str:
    return x if isinstance(x, str) else fallback


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def detect_language(file_path: str) -> str:
    if isinstance(file_path, str) and file_path.lower().endswith(".java"):
        return "java"
    return "python"


def language_branch_for_language(language: str) -> str:
    return "java_branch" if str(language or "").strip().lower() == "java" else "python_branch"


def normalize_repair_branch(repair_branch: str) -> str:
    value = str(repair_branch or "auto").strip().lower()
    if value in REPAIR_BRANCH_CHOICES:
        return value
    return "auto"


def resolve_repair_branch_for_bug(
    bug_data: Dict[str, Any],
    requested_branch: str,
    language: str,
) -> str:
    normalized = normalize_repair_branch(requested_branch)
    lang = str(language or "").strip().lower()
    if lang == "java":
        if normalized in {"java_base", "java_semantic", "java_v2"}:
            return normalized
        return "java_base"
    if normalized == "python_base":
        return normalized
    return "python_base"


def argv_has_flag(argv: List[str], *flags: str) -> bool:
    for token in argv:
        token_s = str(token or "")
        for flag in flags:
            if token_s == flag or token_s.startswith(f"{flag}="):
                return True
    return False


def infer_dataset_tag_from_input(input_path: str, run_tag: str = "") -> str:
    haystacks = [str(input_path or "").lower(), str(run_tag or "").lower()]
    for value in haystacks:
        if "defects4j" in value:
            return "defects4j"
        if "bugsinpy" in value:
            return "bugsinpy"
    return ""


def resolve_stage3_branch_defaults(
    *,
    requested_branch: str,
    input_path: str,
    run_tag: str,
    single_only: bool,
    edit_mode: str,
    single_only_explicit: bool,
    edit_mode_explicit: bool,
) -> Tuple[str, bool, str]:
    normalized_branch = normalize_repair_branch(requested_branch)
    dataset_tag = infer_dataset_tag_from_input(input_path, run_tag=run_tag)
    effective_branch = normalized_branch
    if effective_branch == "auto":
        effective_branch = "java_base" if dataset_tag == "defects4j" else "python_base"

    effective_single_only = bool(single_only)
    effective_edit_mode = str(edit_mode or "auto")
    if effective_branch in {"java_base", "java_semantic"} and not edit_mode_explicit:
        effective_edit_mode = "single"
    if effective_branch == "java_v2" and not single_only_explicit:
        effective_single_only = False
        if not edit_mode_explicit:
            effective_edit_mode = "auto"
    return effective_branch, effective_single_only, effective_edit_mode

# --- Core prompt builder ---

def get_buggy_function_code(bug_data: Dict[str, Any]) -> Tuple[str, str]:
    func = bug_data.get("function", {}) or {}
    code = func.get("function_before") or ""
    if isinstance(code, str) and code.strip():
        return code, "function"
    return "", "none"


def _mark_buggy_line(
    full_code: str,
    buggy_line_content: str,
    comment_token: str,
    buggy_line_location: Optional[int] = None,
) -> str:
    if not full_code:
        return full_code
    lines = full_code.split("\n")
    if buggy_line_location is not None and 1 <= buggy_line_location <= len(lines):
        if "<--- BUGGY LINE" not in lines[buggy_line_location - 1]:
            lines[buggy_line_location - 1] = f"{lines[buggy_line_location - 1]}  {comment_token} <--- BUGGY LINE"
        return "\n".join(lines)
    if not buggy_line_content:
        return full_code
    target = buggy_line_content.strip()
    for i, line in enumerate(lines):
        if line.strip() == target:
            lines[i] = f"{line}  {comment_token} <--- BUGGY LINE"
            return "\n".join(lines)
    for i, line in enumerate(lines):
        if target and target in line:
            lines[i] = f"{line}  {comment_token} <--- BUGGY LINE"
            return "\n".join(lines)
    return full_code


def _normalize_to_local_line(
    line_value: Any,
    function_start_line: Any = None,
    snippet_line_count: Optional[int] = None,
) -> Optional[int]:
    line_int = _to_int_or_none(line_value)
    if line_int is None or line_int <= 0:
        return None

    if snippet_line_count is not None and 1 <= line_int <= snippet_line_count:
        return line_int

    function_start_int = _to_int_or_none(function_start_line)
    if function_start_int is not None:
        local_line = line_int - function_start_int + 1
        if local_line > 0 and (snippet_line_count is None or local_line <= snippet_line_count):
            return local_line

    if snippet_line_count is not None:
        return None
    return line_int


def _pick_bug_marker_line(
    nodes: List[Dict[str, Any]],
    buggy_line_location: Any,
    function_start_line: Any,
    snippet_line_count: Optional[int],
) -> Optional[int]:
    local_bug_line = _normalize_to_local_line(
        buggy_line_location,
        function_start_line=function_start_line,
        snippet_line_count=snippet_line_count,
    )
    if local_bug_line is not None:
        return local_bug_line

    valid_node_lines: List[int] = []
    buggy_node_lines: List[int] = []
    for node in nodes[:5]:
        if not isinstance(node, dict):
            continue
        node_line = _normalize_to_local_line(
            node.get("line"),
            function_start_line=function_start_line,
            snippet_line_count=snippet_line_count,
        )
        if node_line is None:
            continue
        valid_node_lines.append(node_line)
        if node.get("contains_buggy_line"):
            buggy_node_lines.append(node_line)

    if buggy_node_lines:
        return buggy_node_lines[0]
    if valid_node_lines:
        return valid_node_lines[0]
    return None


def summarize_hints(nodes: List[Dict[str, Any]], lang: str) -> str:
    if not nodes:
        return ""
    parts: List[str] = []
    for i, n in enumerate(nodes[:5], 1):
        line = n.get("line", "N/A")
        ntype = n.get("type", "N/A")
        dist = n.get("distance_to_buggy", "N/A")
        contains_buggy = bool(n.get("contains_buggy_line"))
        overlap_names = n.get("identifier_overlap_names") or []
        hints = n.get("patch_hints") or []
        ctx = trim_text_by_lines(_as_str(n.get("context"), ""), DEFAULT_MAX_HINT_CODE_LINES).strip()
        code = trim_text_by_lines(_as_str(n.get("code"), ""), DEFAULT_MAX_HINT_CODE_LINES).strip()
        code_block = f"```text\n{ctx}\n```" if ctx else f"```{lang}\n{code}\n```"
        line_bits = [f"{i}. Line {line} ({ntype}, dist: {dist})"]
        if contains_buggy:
            line_bits.append("   - Contains buggy line")
        if overlap_names:
            line_bits.append("   - Ids: " + ", ".join(overlap_names[:6]))
        if hints:
            line_bits.append("   - Hints: " + ", ".join(hints))
        line_bits.append(f"   - Code:\n{code_block}")
        parts.append("\n".join(line_bits))
    return trim_text_by_lines("\n".join(parts), DEFAULT_MAX_HINT_LINES)


def summarize_bm25_grounding(bm25_obj: Any, max_items: int = 2) -> str:
    if not isinstance(bm25_obj, dict):
        return ""
    top = bm25_obj.get("top")
    if not isinstance(top, list) or not top:
        return ""
    lines: List[str] = []
    for i, item in enumerate(top[:max_items], 1):
        if not isinstance(item, dict):
            continue
        bid = _as_str(item.get("id"), "N/A")
        proj = _as_str(item.get("project_name"), "N/A")
        file_path = _as_str(item.get("file_path"), "")
        buggy = _as_str(item.get("buggy_line_content"), "")
        score = item.get("score_reranked", None)
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
        bits = [f"{i}. {bid} | {proj} | score {score_s}"]
        if file_path:
            bits.append(f"   - file: {file_path}")
        if buggy:
            bits.append(f"   - buggy_line: {buggy}")
        lines.append("\n".join(bits))
    return "\n".join(lines)


def summarize_bound_lines(
    nodes: List[Dict[str, Any]],
    buggy_line_location: Any,
    function_start_line: Any = None,
    snippet_line_count: Optional[int] = None,
) -> str:
    lines = []
    local_bug_line = _normalize_to_local_line(
        buggy_line_location,
        function_start_line=function_start_line,
        snippet_line_count=snippet_line_count,
    )
    if local_bug_line is not None:
        lines.append(local_bug_line)
    if isinstance(nodes, list):
        for n in nodes[:5]:
            if not isinstance(n, dict):
                continue
            normalized = _normalize_to_local_line(
                n.get("line"),
                function_start_line=function_start_line,
                snippet_line_count=snippet_line_count,
            )
            if normalized is not None:
                lines.append(normalized)
    lines = sorted({x for x in lines if isinstance(x, int) and x > 0})
    if not lines:
        return ""
    return ", ".join(str(x) for x in lines[:10])


JAVA_CALL_BLACKLIST = {
    "if", "for", "while", "switch", "catch", "return", "throw", "new", "super", "this",
}


def _extract_java_signature(code: str) -> str:
    if not code:
        return ""
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("@"):
            continue
        if "(" not in line:
            continue
        if line.endswith("{"):
            return line[:-1].strip()
        return line
    return ""


def _extract_java_helper_calls(code: str, function_name: str, max_items: int = 8) -> List[str]:
    if not code:
        return []
    calls = re.findall(r"\b(?:this\.)?([a-z_][A-Za-z0-9_$]*)\s*\(", code)
    filtered = [
        name for name in calls
        if name not in JAVA_CALL_BLACKLIST and name != function_name
    ]
    return _dedup_keep_order(filtered, limit=max_items)


def _extract_java_types(code: str, max_items: int = 10) -> List[str]:
    if not code:
        return []
    types = re.findall(r"\b([A-Z][A-Za-z0-9_$]*)\b", code)
    return _dedup_keep_order(types, limit=max_items)


def summarize_java_class_context(bug_data: Dict[str, Any]) -> str:
    func = bug_data.get("function", {}) or {}
    code = _as_str(func.get("function_before"), "")
    function_name = _as_str(func.get("function_name"), "")
    parent = _as_str(func.get("function_parent"), "")

    lines: List[str] = []
    if parent:
        lines.append(f"- Enclosing member: {parent}")

    signature = _extract_java_signature(code)
    if signature:
        lines.append(f"- Current method signature: {signature}")

    fields = _dedup_keep_order(re.findall(r"\bthis\.([A-Za-z_$][A-Za-z0-9_$]*)\b", code), limit=8)
    if fields:
        lines.append("- Referenced instance fields: " + ", ".join(fields))

    helper_calls = _extract_java_helper_calls(code, function_name=function_name, max_items=8)
    if helper_calls:
        lines.append("- Referenced helper methods in this method: " + ", ".join(helper_calls))

    types = _extract_java_types(code, max_items=10)
    if types:
        lines.append("- Types referenced in this method: " + ", ".join(types))

    # Phase2-D: 보강된 컨텍스트 (enrich_java_context.py로 사전 추출된 데이터)
    java_imports = bug_data.get("java_imports") or []
    if java_imports:
        # 가장 관련 있는 import만 선별: 코드에서 참조하는 타입과 매칭
        relevant_imports = []
        for imp in java_imports:
            imp_simple = imp.split(".")[-1] if "." in imp else imp
            if imp_simple in code or imp_simple == "*":
                relevant_imports.append(imp)
        if relevant_imports:
            lines.append("- Relevant imports: " + ", ".join(relevant_imports[:10]))
        elif java_imports:
            lines.append("- Available imports (top): " + ", ".join(java_imports[:8]))

    java_class_fields = bug_data.get("java_class_fields") or []
    if java_class_fields:
        lines.append("- Class field declarations:")
        for field in java_class_fields[:10]:
            lines.append(f"    {field}")

    java_sibling_sigs = bug_data.get("java_sibling_method_signatures") or []
    if java_sibling_sigs:
        # Phase2-D fix: false positive 필터링
        _kw = {'if', 'for', 'while', 'switch', 'catch', 'return', 'throw', 'do', 'else', 'try'}
        java_sibling_sigs = [
            sig for sig in java_sibling_sigs
            if not any(sig.strip().startswith(k + ' ') or sig.strip().startswith(k + '(') for k in _kw)
        ]
        # 코드에서 호출되는 메서드만 필터링
        called_sigs = []
        other_sigs = []
        for sig in java_sibling_sigs:
            sig_name = re.search(r'(\w+)\s*\(', sig)
            if sig_name and sig_name.group(1) in code:
                called_sigs.append(sig)
            else:
                other_sigs.append(sig)
        if called_sigs:
            lines.append("- Called sibling method signatures:")
            for sig in called_sigs[:8]:
                lines.append(f"    {sig}")
        if other_sigs and len(called_sigs) < 5:
            lines.append("- Other sibling method signatures (available for use):")
            for sig in other_sigs[:5]:
                lines.append(f"    {sig}")

    return "\n".join(lines)


def extract_java_structured_context(bug_data: Dict[str, Any]) -> Dict[str, Any]:
    func = bug_data.get("function", {}) or {}
    code = _as_str(func.get("function_before"), "")
    function_name = _as_str(func.get("function_name"), "")
    fields = _dedup_keep_order(re.findall(r"\bthis\.([A-Za-z_$][A-Za-z0-9_$]*)\b", code), limit=8)
    return {
        "java_method_signature": _extract_java_signature(code),
        "java_helper_calls": _extract_java_helper_calls(code, function_name=function_name, max_items=8),
        "java_types": _extract_java_types(code, max_items=10),
        "java_instance_fields": fields,
        "java_context_summary": summarize_java_class_context(bug_data),
        # Phase2-D: 보강된 컨텍스트 전달
        "java_imports": bug_data.get("java_imports") or [],
        "java_class_fields": bug_data.get("java_class_fields") or [],
        "java_sibling_method_signatures": bug_data.get("java_sibling_method_signatures") or [],
    }


def _summarize_enriched_context(bug_data: Dict[str, Any], lang: str, function_code: str = "") -> str:
    """
    Phase2-D: 통합 파일-레벨 컨텍스트 요약 (Python & Java 공통).
    0_EnrichContext.py에서 생성된 enriched_* 키를 사용.
    """
    imports = bug_data.get("enriched_imports") or []
    fields = bug_data.get("enriched_class_fields") or []
    sibling_sigs = bug_data.get("enriched_sibling_signatures") or []

    if not imports and not fields and not sibling_sigs:
        return ""

    lines: List[str] = []

    # imports: 코드에서 참조하는 것만 필터링
    if imports:
        if function_code:
            relevant = []
            for imp in imports:
                # "from x.y import Z" → Z, "import x.y" → y
                parts = imp.replace(",", " ").split()
                names = [p.split(".")[-1] for p in parts if p not in ("from", "import", "as")]
                if any(n in function_code for n in names if n and n != "*"):
                    relevant.append(imp)
            if relevant:
                lines.append("- Relevant imports: " + "; ".join(relevant[:10]))
            elif len(imports) <= 8:
                lines.append("- Available imports: " + "; ".join(imports))
            else:
                lines.append("- Available imports (top): " + "; ".join(imports[:8]))
        else:
            lines.append("- Available imports: " + "; ".join(imports[:8]))

    # fields
    if fields:
        lines.append("- Class/module-level declarations:")
        for fld in fields[:10]:
            lines.append(f"    {fld}")

    # sibling functions/methods — 호출되는 것 우선
    # Phase2-D fix: false positive 필터링 (if/for/while/return 등은 시그니처가 아님)
    _java_kw_sigs = {'if', 'for', 'while', 'switch', 'catch', 'return', 'throw', 'do', 'else', 'try'}
    if sibling_sigs:
        if lang == "java":
            sibling_sigs = [
                sig for sig in sibling_sigs
                if not any(sig.strip().startswith(kw + ' ') or sig.strip().startswith(kw + '(') for kw in _java_kw_sigs)
            ]
        if function_code:
            called = []
            others = []
            for sig in sibling_sigs:
                # 시그니처에서 함수/메서드 이름 추출
                name_match = re.search(r'(?:def|async\s+def)?\s*(\w+)\s*\(', sig)
                name = name_match.group(1) if name_match else ""
                if name and name in function_code:
                    called.append(sig)
                else:
                    others.append(sig)
            if called:
                lines.append("- Called sibling signatures:")
                for sig in called[:8]:
                    lines.append(f"    {sig}")
            if others and len(called) < 5:
                lines.append("- Other available signatures:")
                for sig in others[:5]:
                    lines.append(f"    {sig}")
        else:
            lines.append("- Sibling signatures:")
            for sig in sibling_sigs[:8]:
                lines.append(f"    {sig}")

    return "\n".join(lines)


def resolve_edit_mode(bug_data: Dict[str, Any], requested_mode: str, single_only: bool = False) -> str:
    if bool(single_only):
        return "single"
    mode = _as_str(requested_mode, "auto").strip().lower()
    if mode in ("single", "hard"):
        return mode

    hard_tag = _as_str(bug_data.get("hard_tag"), "").lower()
    if hard_tag in ("single", "hard"):
        return hard_tag

    if bool(bug_data.get("single_line", False)):
        return "single"
    return "hard"


def create_plan_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    repair_mode: str,
    repair_branch: str,
) -> str:
    file_path = _as_str(safe_get(bug_data, ["file", "file_path"], bug_data.get("file_path", "")), "")
    lang = detect_language(file_path)
    language_branch = language_branch_for_language(lang)
    code_lang = "java" if lang == "java" else "python"
    hints = summarize_hints(safe_get(bug_data, ["suspicious_nodes_topk"], []) or [], lang=code_lang)
    bm25_grounding = summarize_bm25_grounding(bug_data.get("bm25"), max_items=2)
    hard_tag = _as_str(bug_data.get("hard_tag"), "").lower()
    single_line = bool(bug_data.get("single_line", False))
    java_context_summary = _as_str(bug_data.get("java_context_summary"), "") if lang == "java" else ""

    if lang == "java" and repair_branch in {"java_base", "java_semantic"}:
        if repair_mode == "hard":
            plan_guideline = (
                "PLAN STYLE:\n"
                "- HARD MODE: multi-line edits may be needed, but keep scope within this function and near the buggy line.\n"
                "- Prefer one narrow local span anchored to the buggy line or its immediately adjacent lines.\n"
                "- Use full-method coverage or multiple distant spans only when the hints and shown context directly justify them.\n"
                "- Keep changed_lines_max conservative (usually about 6-12) unless the same nearby block clearly needs more.\n"
                "- Do not plan invented helper/API/type/field introductions. Reuse only symbols already visible in the shown method/class context.\n\n"
            )
            allowed_edit_types = "\"allowed_edit_types\":[\"REPLACE\",\"INSERT\",\"DELETE\"],"
        else:
            plan_guideline = (
                "PLAN STYLE:\n"
                "- LOCALIZED MODE: focus on the smallest region that can fix the bug.\n"
                "- Ensure at least one target_locations range covers the buggy line.\n"
                "- Keep target_locations tightly attached to the buggy line (prefer <= 6 lines span) unless nearby control flow requires a little more.\n"
                "- Prefer one narrow span over multi-span plans. Multiple far-apart spans need direct evidence from hints/context.\n"
                "- Keep changed_lines_max small (usually about 2-6).\n"
                "- INSERT is allowed for a small guard/check, but avoid broad rewrites and helper/API invention.\n"
                "- Do not plan invented helper/API/type/field introductions. Reuse only symbols already visible in the shown method/class context.\n\n"
            )
            allowed_edit_types = "\"allowed_edit_types\":[\"REPLACE\",\"INSERT\"],"
    elif repair_mode == "hard":
        plan_guideline = (
            "PLAN STYLE:\n"
            "- HARD MODE: multi-line edits may be needed, but keep scope within this function.\n"
            "- Avoid over-narrow target_locations (do not force single-line ranges).\n"
            "- Ensure at least one target_locations range covers the buggy line.\n"
            "- Set changed_lines_max in a realistic hard range (about 12-25).\n\n"
        )
        allowed_edit_types = "\"allowed_edit_types\":[\"REPLACE\",\"INSERT\",\"DELETE\"],"
    else:
        plan_guideline = (
            "PLAN STYLE:\n"
            "- LOCALIZED MODE: focus on the smallest region that can fix the bug.\n"
            "- Ensure at least one target_locations range covers the buggy line.\n"
            "- Keep target_locations narrow (prefer <= 8 lines span) unless necessary for correctness.\n"
            "- Prefer minimal edits. Set changed_lines_max in a small range (about 3-8).\n"
            "- INSERT is allowed when you need to add a guard/check; avoid broad refactors.\n\n"
        )
        allowed_edit_types = "\"allowed_edit_types\":[\"REPLACE\",\"INSERT\"],"

    schema = (
        "{"
        "\"target_locations\":[{\"start\":int,\"end\":int}],"
        "\"edit_budget\":{\"changed_lines_max\":int},"
        + allowed_edit_types +
        "\"forbidden_regions\":[],"
        "\"structural_constraints\":{\"within_nodes\":[\"If\",\"Return\",\"Call\"]},"
        "\"evidence_links\":{\"ast_nodes\":[int],\"retrieval_top\":[int]}"
        "}\n\n"
    )

    return (
        "You are a repair planner. Output ONLY valid JSON.\n"
        "Create a plan to constrain patch search.\n"
        "JSON schema:\n"
        "All line numbers refer to the provided function snippet (1-based).\n"
        + schema
        + plan_guideline
        + "HINTS:\n" + (hints or "N/A") + "\n\n"
        + "GROUNDING (similar bugs, use as analogies only):\n"
+ "WARNING: DO NOT COPY THESE EXAMPLES EXACTLY. THEY USE DIFFERENT VARIABLES. APPLY ONLY THE 'CONCEPT' TO THE CURRENT BUG.\n"
+ (bm25_grounding or "N/A") + "\n\n"
        + "BUG INFO:\n"
        + f"- bug_id: {bug_id}\n"
        + f"- file: {file_path}\n"
        + f"- repair_mode: {repair_mode}\n"
        + f"- repair_branch: {repair_branch}\n"
        + f"- language_branch: {language_branch}\n"
        + f"- hard_tag: {hard_tag or ('single' if single_line else 'hard')}\n"
        + f"- buggy_line: {bug_data.get('buggy_line_content', '')}\n"
        + (f"- java_context_summary:\n{java_context_summary}\n" if java_context_summary else "")
    )


def create_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    max_full_code_lines: int,
    max_context_lines: int,
    repair_mode: str,
    repair_branch: str,
) -> Tuple[str, str, str]:
    project_name = _as_str(bug_data.get("project_name"), "N/A")
    file_path = _as_str(safe_get(bug_data, ["file", "file_path"], bug_data.get("file_path", "")), "N/A")
    function_name = _as_str(safe_get(bug_data, ["function", "function_name"], bug_data.get("function_name", "")), "N/A")
    buggy_line_location = bug_data.get("buggy_line_location", "N/A")
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()
    buggy_line_context = _as_str(bug_data.get("buggy_line_context"), "")

    full_buggy_code, code_source = get_buggy_function_code(bug_data)

    lang = detect_language(file_path)
    language_branch = language_branch_for_language(lang)
    comment_token = "//" if lang == "java" else "#"
    code_lang = "java" if lang == "java" else "python"
    role_line = "You are a senior Java bug-fix specialist." if lang == "java" else "You are a senior Python bug-fix specialist."
    language_line = (
        "Prefer Java 6/7-compatible syntax used by older Defects4J projects."
        if lang == "java" else
        "Python 3.6 compatible."
    )
    code_label = "method or class" if lang == "java" else "function or class"

    # Mark buggy line when possible
    line_no_int = _to_int_or_none(buggy_line_location)
    func_start = _to_int_or_none(safe_get(bug_data, ["function", "function_before_start_line"]))
    suspicious_nodes = safe_get(bug_data, ["suspicious_nodes_topk"], []) or []
    raw_code_line_count = len(full_buggy_code.splitlines()) if full_buggy_code else None

    local_bug_line = None
    if line_no_int is not None and func_start is not None:
        local_bug_line = line_no_int - func_start + 1

    if full_buggy_code:
        mark_line = _pick_bug_marker_line(
            suspicious_nodes,
            buggy_line_location,
            func_start,
            raw_code_line_count,
        )
        full_buggy_code = _mark_buggy_line(full_buggy_code, buggy_line_content, comment_token, mark_line)
    full_buggy_code_trimmed = trim_text_by_lines(full_buggy_code, max_full_code_lines)

    hints = summarize_hints(suspicious_nodes, lang=code_lang)
    bm25_grounding = summarize_bm25_grounding(bug_data.get("bm25"), max_items=2)
    bound_lines = summarize_bound_lines(
        suspicious_nodes,
        buggy_line_location,
        func_start,
        raw_code_line_count,
    )
    java_class_context = _as_str(bug_data.get("java_context_summary"), "") if lang == "java" else ""
    if lang == "java" and not java_class_context:
        java_class_context = summarize_java_class_context(bug_data)

    # Phase2-D: 통합 파일-레벨 컨텍스트 (Python & Java 공통)
    file_level_context = _summarize_enriched_context(bug_data, lang=lang, function_code=full_buggy_code)

    if lang == "python":
        description = _as_str(bug_data.get("description"), "No description provided.")
        parts: List[str] = [
            "You are a senior Python bug-fix specialist. Read the task and produce a minimal, correct patch.\n\n",
            "OUTPUT:\n",
            f"- After `##correct`, output ONLY the complete corrected {code_label}.\n",
            "- No markdown, no analysis, no extra text.\n\n",
            "STRICT CONSTRAINTS:\n",
            "- Keep the original name and signature EXACTLY.\n",
            "- Make the smallest possible change.\n",
            "- Do not add imports. Do not print. Do not log. No placeholders like 'pass'.\n",
            "- Apply the smallest possible edit (prefer changing <= 3 lines unless absolutely necessary).\n",
            "- Preserve behavior that is unrelated to the bug.\n",
            "- Python 3.6 compatible. All identifiers and strings must be in English.\n",
            "- If the bug is in conditionals/boundaries, fix the condition not the data.\n",
            "- If the bug involves mutation vs copy, use an explicit shallow copy only when required.\n",
            "- If external APIs are used, do NOT alter their contract or error types.\n\n",
            "INTERNAL CHECKLIST (do not output):\n",
            "- [ ] Off-by-one / range / indexing\n",
            "- [ ] None / empty ([], {}, \"\") handling\n",
            "- [ ] Mutable default args / aliasing (copy vs reference)\n",
            "- [ ] Type coercion (str/int/float), truthiness pitfalls\n",
            "- [ ] Sorting / order assumptions (stable sort, key)\n",
            "- [ ] Integer division vs float division\n",
            "- [ ] Early return vs fall-through logic\n",
            "- [ ] Exception type/messages preserved or narrowed safely\n",
            "- [ ] Boundary conditions on loops/slices\n",
            "- [ ] Do not change I/O behavior or global state\n\n",
        ]
        if repair_mode == "single":
            parts.extend([
                "LOCALIZED MODE:\n",
                "- Focus on fixing the logic near `<--- BUGGY LINE`.\n",
                "- You MUST modify the code to fix the bug. Do NOT return the original code unchanged.\n",
                "- Preserve the surrounding code structure but ensure the logical error is resolved.\n\n",
            ])
        else:
            parts.extend([
                "HARD MODE:\n",
                "- Multi-line edits are allowed only when necessary, but keep scope local.\n",
                "- Avoid broad refactors; preserve existing invariants and control flow shape.\n\n",
            ])

        parts.extend([
            "FOCUS ORDER WHEN READING CONTEXT:\n",
            "1) Static Analysis Hints (suspicious nodes) — inspect these lines first.\n",
            "2) The line marked with '# <--- BUGGY LINE'.\n",
            "3) Surrounding lines for dataflow and invariants.\n",
            "4) Similar bug grounding — use only as analogy, never wholesale rewrite.\n\n",
            "PLAN (JSON):\n",
            "{{PLAN_JSON}}\n\n",
            "Line numbers in PLAN refer to the function snippet shown below (1-based).\n\n",
            "BUG:\n",
            f"- Project: {project_name}\n",
            f"- File: {file_path}\n",
        ])
        if function_name and function_name != "N/A":
            parts.append(f"- Function: {function_name}\n")
        parts.append(f"- Repair mode: {repair_mode}\n")
        parts.append(f"- Repair branch: {repair_branch}\n")
        parts.append(f"- Language branch: {language_branch}\n")
        if buggy_line_content:
            parts.append(f"- Buggy line (line {buggy_line_location}): {buggy_line_content}\n")
        parts.append("\n")

        if description:
            parts.append("BUG DESCRIPTION (brief):\n")
            parts.append(trim_text_by_lines(description, 60) + "\n\n")

        if bound_lines:
            parts.append("BOUNDED EDIT REGION:\n")
            parts.append(f"- Prefer edits at/near lines: {bound_lines}\n")
            parts.append("- If you must edit outside, keep within the same function and remain minimal.\n\n")

        if buggy_line_context:
            parts.append("LOCAL CONTEXT:\n```text\n")
            parts.append(trim_text_by_lines(buggy_line_context, max_context_lines) + "\n")
            parts.append("```\n\n")

        # Phase2-D: 통합 파일-레벨 컨텍스트 (Python)
        if file_level_context:
            parts.append("FILE CONTEXT (available symbols in this module/class):\n")
            parts.append(file_level_context + "\n\n")

        if hints:
            parts.append("HINTS:\n")
            parts.append(hints + "\n\n")

        if bm25_grounding:
            parts.append("GROUNDING (similar bugs, use as analogies only):\n")
            parts.append("WARNING: DO NOT COPY THESE EXAMPLES EXACTLY. THEY USE DIFFERENT VARIABLES. APPLY ONLY THE 'CONCEPT' TO THE CURRENT BUG.\n")
            parts.append(bm25_grounding + "\n\n")

        parts.append("CODE (buggy line is marked with `<--- BUGGY LINE`):\n")
        parts.append(f"```{code_lang}\n")
        parts.append(full_buggy_code_trimmed + "\n")
        parts.append("```\n\n")
        parts.append("##correct\n")
    else:
        if repair_branch == "java_v2":
            java_branch_label = "Java v2 body-first branch"
            description = _as_str(bug_data.get("description"), "No description provided.")
            parts = [
                "You are a senior Java bug-fix specialist. Read the task and produce a minimal, correct patch.\n\n",
                f"REPAIR BRANCH:\n- {java_branch_label}. This is a Java-specific repair branch that prefers body-only output so the original declaration can be preserved deterministically.\n\n",
                "OUTPUT:\n",
                "- After `##correct`, output ONLY one repaired Java patch payload.\n",
                "- Prefer BODY-ONLY mode with these markers:\n",
                "  @@BEGIN_JAVA_BODY@@\n",
                "  ...body only...\n",
                "  @@END_JAVA_BODY@@\n",
                "- Use FULL-DECLARATION mode ONLY when the fix truly requires changing the method signature, annotations/modifiers, throws clause, or surrounding class declaration:\n",
                "  @@BEGIN_JAVA_CODE@@\n",
                "  ...full method/class...\n",
                "  @@END_JAVA_CODE@@\n",
                "- In body-only mode, do NOT repeat the method signature, annotations, modifiers, throws clause, or surrounding class wrapper.\n",
                "- No commentary, no analysis, no markdown fences, no prose, no extra text before or after the markers.\n\n",
                "STRICT CONSTRAINTS:\n",
                "- Preserve the original method/class name.\n",
                "- Preserve the original method signature unless the fix truly requires changing it.\n",
                "- Preserve checked exceptions (`throws ...`) unless the fix truly requires changing them.\n",
                "- Make the smallest possible change.\n",
                "- Do not add imports. Do not print. Do not log. No placeholders.\n",
                "- Apply the smallest possible edit (prefer changing <= 3 lines unless absolutely necessary).\n",
                "- Preserve behavior that is unrelated to the bug.\n",
                "- Prefer Java 6/7-compatible syntax used by older Defects4J projects. Keep code style and naming consistent with surrounding code.\n",
                "- Use standard Java syntax only.\n",
                "- Do NOT use diamond operator (`<>`), `var`, lambdas, streams, method references, switch expressions, records, try-with-resources, or multi-catch unless the snippet already uses them.\n",
                "- Do NOT invent fields, methods, helper methods, helper classes, or unrelated external APIs unless absolutely unavoidable.\n",
                "- Reuse existing identifiers, fields, helper calls, and types from the current method/class context whenever possible.\n",
                "- Reusing identifiers, fields, helper methods, or types that already appear in the shown method/class context is allowed and preferred.\n",
                "- Do not invent new helper methods or API calls, but reusing names already present in the provided context is safe.\n",
                "- Prefer conservative reuse of existing class-context symbols over broader rewrites.\n",
                "- Do NOT use Kotlin/Groovy-only shorthand or non-Java null-coalescing/Elvis syntax. Standard Java ternary `cond ? a : b` is allowed.\n",
                "- If the bug is in conditionals/boundaries, fix the condition not the data.\n",
                "- If external APIs are used, do NOT alter their contract or exception types.\n",
                "- Any newly introduced method call, constant, type, or field name must already appear in the shown method/class context. Otherwise, do not use it.\n",
                "- Do not replace local logic with invented parser, encoder, or helper calls.\n",
                "- Prefer a smaller conservative patch over a broader rewrite if both are plausible.\n",
                "- Do not rename the method or restructure it into a different algorithmic shape unless the existing local identifiers already support that change.\n",
                "- If you introduce a conditional or guard, reuse existing local variables and existing API calls only.\n\n",
                "INTERNAL CHECKLIST (do not output):\n",
                "- [ ] Null handling and defensive checks\n",
                "- [ ] Off-by-one / range / indexing\n",
                "- [ ] Object comparison (`equals` vs `==`) where applicable\n",
                "- [ ] Integer overflow / division by zero risks\n",
                "- [ ] Collection empty-state and iteration safety\n",
                "- [ ] Early return vs fall-through logic\n",
                "- [ ] Exception type/messages preserved or narrowed safely\n",
                "- [ ] Boundary conditions on loops/substrings/slices\n",
                "- [ ] Keep braces, syntax, and return paths valid\n",
                "- [ ] Do not change observable side effects unrelated to the bug\n\n",
            ]
        else:
            java_branch_label = "Java semantic branch" if repair_branch == "java_semantic" else "Java base branch"
            description = _as_str(bug_data.get("description"), "No description provided.")
            parts = [
                "You are a senior Java bug-fix specialist. Read the task and produce a minimal, correct patch.\n\n",
                f"REPAIR BRANCH:\n- {java_branch_label}. This is a Java-specific repair branch, not the shared Python base path.\n\n",
                "OUTPUT:\n",
                f"- After `##correct`, output ONLY one repaired full {code_label}.\n",
                "- Output the final Java code STRICTLY between these markers:\n",
                "  @@BEGIN_JAVA_CODE@@\n",
                "  ...code...\n",
                "  @@END_JAVA_CODE@@\n",
                "- No commentary, no analysis, no markdown fences, no prose, no extra text before or after the markers.\n\n",
                "STRICT CONSTRAINTS:\n",
                "- Preserve the original method/class name.\n",
                "- Preserve the original method signature unless the fix truly requires changing it.\n",
                "- Preserve checked exceptions (`throws ...`) unless the fix truly requires changing them.\n",
                "- Make the smallest possible change.\n",
                "- Do not add imports. Do not print. Do not log. No placeholders.\n",
                "- Apply the smallest possible edit (prefer changing <= 3 lines unless absolutely necessary).\n",
                "- Preserve behavior that is unrelated to the bug.\n",
                "- Prefer Java 6/7-compatible syntax used by older Defects4J projects. Keep code style and naming consistent with surrounding code.\n",
                "- Use standard Java syntax only.\n",
                "- Do NOT use diamond operator (`<>`), `var`, lambdas, streams, method references, switch expressions, records, try-with-resources, or multi-catch unless the snippet already uses them.\n",
                "- Do NOT invent fields, methods, helper methods, helper classes, or unrelated external APIs unless absolutely unavoidable.\n",
                "- Reuse existing identifiers, fields, helper calls, and types from the current method/class context whenever possible.\n",
                "- Reusing identifiers, fields, helper methods, or types that already appear in the shown method/class context is allowed and preferred.\n",
                "- Do not invent new helper methods or API calls, but reusing names already present in the provided context is safe.\n",
                "- Prefer conservative reuse of existing class-context symbols over broader rewrites.\n",
                "- Do NOT use Kotlin/Groovy-only shorthand or non-Java null-coalescing/Elvis syntax. Standard Java ternary `cond ? a : b` is allowed.\n",
                "- If the bug is in conditionals/boundaries, fix the condition not the data.\n",
                "- If external APIs are used, do NOT alter their contract or exception types.\n",
                "- Any newly introduced method call, constant, type, or field name must already appear in the shown method/class context. Otherwise, do not use it.\n",
                "- Do not replace local logic with invented parser, encoder, or helper calls.\n",
                "- Prefer a smaller conservative patch over a broader rewrite if both are plausible.\n",
                "- Do not rename the method or restructure it into a different algorithmic shape unless the existing local identifiers already support that change.\n",
                "- If you introduce a conditional or guard, reuse existing local variables and existing API calls only.\n\n",
                "INTERNAL CHECKLIST (do not output):\n",
                "- [ ] Null handling and defensive checks\n",
                "- [ ] Off-by-one / range / indexing\n",
                "- [ ] Object comparison (`equals` vs `==`) where applicable\n",
                "- [ ] Integer overflow / division by zero risks\n",
                "- [ ] Collection empty-state and iteration safety\n",
                "- [ ] Early return vs fall-through logic\n",
                "- [ ] Exception type/messages preserved or narrowed safely\n",
                "- [ ] Boundary conditions on loops/substrings/slices\n",
                "- [ ] Keep braces, syntax, and return paths valid\n",
                "- [ ] Do not change observable side effects unrelated to the bug\n\n",
            ]
        if repair_mode == "single":
            parts.extend([
                "LOCALIZED MODE:\n",
                "- Focus on fixing the logic near `<--- BUGGY LINE`.\n",
                "- You MUST modify the code to fix the bug. Do NOT return the original code unchanged.\n",
                "- Preserve the surrounding code structure but ensure the logical error is resolved.\n\n",
            ])
        else:
            parts.extend([
                "HARD MODE:\n",
                "- Multi-line edits are allowed only when necessary, but keep scope local.\n",
                "- Avoid broad refactors; preserve existing invariants and control flow shape.\n\n",
            ])

        parts.extend([
            "FOCUS ORDER WHEN READING CONTEXT:\n",
            "1) Static Analysis Hints (suspicious nodes) — inspect these lines first.\n",
            "2) The line marked with '// <--- BUGGY LINE'.\n",
            "3) Surrounding lines for dataflow and invariants.\n",
            "4) Similar bug grounding — use only as analogy, never wholesale rewrite.\n\n",
            "PLAN (JSON):\n",
            "{{PLAN_JSON}}\n\n",
            "Line numbers in PLAN refer to the function snippet shown below (1-based).\n\n",
            "BUG:\n",
            f"- Project: {project_name}\n",
            f"- File: {file_path}\n",
        ])
        if function_name and function_name != "N/A":
            parts.append(f"- Function: {function_name}\n")
        parts.append(f"- Repair mode: {repair_mode}\n")
        parts.append(f"- Repair branch: {repair_branch}\n")
        parts.append(f"- Language branch: {language_branch}\n")
        if buggy_line_content:
            parts.append(f"- Buggy line (line {buggy_line_location}): {buggy_line_content}\n")
        parts.append("\n")

        if description:
            parts.append("BUG DESCRIPTION (brief):\n")
            parts.append(trim_text_by_lines(description, 60) + "\n\n")

        if bound_lines:
            parts.append("BOUNDED EDIT REGION:\n")
            parts.append(f"- Prefer edits at/near lines: {bound_lines}\n")
            parts.append("- If you must edit outside, keep within the same function and remain minimal.\n\n")

        if buggy_line_context:
            parts.append("LOCAL CONTEXT:\n```text\n")
            parts.append(trim_text_by_lines(buggy_line_context, max_context_lines) + "\n")
            parts.append("```\n\n")

        if java_class_context:
            parts.append("CLASS CONTEXT (best-effort from current class/method):\n")
            parts.append(java_class_context + "\n\n")

        # Phase2-D: 통합 파일-레벨 컨텍스트 (Java — class context에 없는 추가 정보)
        if file_level_context and not java_class_context:
            parts.append("FILE CONTEXT (available symbols in this class):\n")
            parts.append(file_level_context + "\n\n")

        if hints:
            parts.append("HINTS:\n")
            parts.append(hints + "\n\n")

        if bm25_grounding:
            parts.append("GROUNDING (similar bugs, use as analogies only):\n")
            parts.append("WARNING: DO NOT COPY THESE EXAMPLES EXACTLY. THEY USE DIFFERENT VARIABLES. APPLY ONLY THE 'CONCEPT' TO THE CURRENT BUG.\n")
            parts.append(bm25_grounding + "\n\n")

        parts.append("CODE (buggy line is marked with `<--- BUGGY LINE`):\n")
        parts.append(f"```{code_lang}\n")
        parts.append(full_buggy_code_trimmed + "\n")
        parts.append("```\n\n")
        parts.append("##correct\n")

    prompt_text = "".join(parts)
    return prompt_text, full_buggy_code_trimmed, code_source


def normalize_top_level(json_obj: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    if isinstance(json_obj, dict):
        return json_obj
    if isinstance(json_obj, list):
        return {str(i): v for i, v in enumerate(json_obj)}
    raise TypeError("Top-level JSON must be an object or an array.")


def process_content(
    source: Union[Dict[str, Any], List[Any]],
    max_full_code_lines: int,
    max_context_lines: int,
    edit_mode: str,
    repair_branch: str,
    single_only_explicit: bool = False,
    edit_mode_explicit: bool = False,
    single_only: bool = False,
) -> Dict[str, Any]:
    processed: Dict[str, Any] = copy.deepcopy(normalize_top_level(source))

    for bug_id, bug_data in processed.items():
        if not isinstance(bug_data, dict):
            continue
        file_path = _as_str(safe_get(bug_data, ["file", "file_path"], bug_data.get("file_path", "")), "")
        language = detect_language(file_path)
        language_branch = language_branch_for_language(language)
        effective_repair_branch = resolve_repair_branch_for_bug(
            bug_data,
            repair_branch,
            language,
        )
        effective_single_only = bool(single_only)
        effective_edit_mode = str(edit_mode or "auto")
        bug_data.setdefault("java_method_signature", "")
        bug_data.setdefault("java_helper_calls", [])
        bug_data.setdefault("java_types", [])
        bug_data.setdefault("java_instance_fields", [])
        bug_data.setdefault("java_context_summary", "")
        if language == "java":
            bug_data.update(extract_java_structured_context(bug_data))
        if effective_repair_branch in {"java_base", "java_semantic"}:
            hard_tag = _as_str(bug_data.get("hard_tag"), "").lower()
            if not bool(edit_mode_explicit):
                effective_edit_mode = "hard" if hard_tag == "hard" else "single"
            if not bool(single_only_explicit):
                effective_single_only = bool(effective_edit_mode == "single")
        elif effective_repair_branch == "java_v2" and not bool(single_only_explicit):
            effective_single_only = False
            if not bool(edit_mode_explicit):
                effective_edit_mode = "auto"
        repair_mode = resolve_edit_mode(bug_data, effective_edit_mode, single_only=bool(effective_single_only))
        if bool(effective_single_only):
            # Prevent downstream hard re-inference in Stage5.
            bug_data["hard_tag"] = "single"
            bug_data["single_line"] = True
        bug_data["language"] = language
        bug_data["language_branch"] = language_branch
        bug_data["repair_branch"] = effective_repair_branch
        bug_data["repair_mode"] = repair_mode
        bug_data["single_only_effective"] = bool(effective_single_only)
        plan_prompt = create_plan_prompt(
            bug_id,
            bug_data,
            repair_mode=repair_mode,
            repair_branch=effective_repair_branch,
        )
        bug_data["plan_prompt"] = plan_prompt
        prompt, code, code_source = create_prompt(
            bug_id,
            bug_data,
            max_full_code_lines=max_full_code_lines,
            max_context_lines=max_context_lines,
            repair_mode=repair_mode,
            repair_branch=effective_repair_branch,
        )
        bug_data["prompt"] = prompt
        if code_source == "function" and isinstance(code, str) and code.strip():
            bug_data.setdefault("code", code)
            bug_data.setdefault("code_mode", "full")

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompts from AST hints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_tag", default="full", help="Run tag used for auto I/O path discovery.")
    parser.add_argument(
        "-i",
        "--input",
        "--in_json",
        dest="input",
        default=None,
        help="Input Stage-3 JSON path, glob, or comma-separated paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--out_json",
        dest="output",
        default=None,
        help="Output JSON path (auto-derived per input when multiple inputs are given).",
    )
    parser.add_argument(
        "--max-full-code-lines",
        type=int,
        default=DEFAULT_MAX_FULL_CODE_LINES,
        help="Max number of lines for full-code context in prompt.",
    )
    parser.add_argument(
        "--max-context-lines",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LINES,
        help="Max number of lines for compact local context in prompt.",
    )
    parser.add_argument(
        "--edit_mode",
        choices=("single", "hard", "auto"),
        default=DEFAULT_EDIT_MODE,
        help="Prompt repair mode. `single` enforces single-line fix guidance.",
    )
    parser.add_argument(
        "--repair_branch",
        choices=REPAIR_BRANCH_CHOICES,
        default="auto",
        help="Explicit repair branch selector. auto => python_base for Python, java_base for Java. Use java_v2 explicitly to enable the new Java-only pipeline.",
    )
    parser.add_argument(
        "--single_only",
        dest="single_only",
        action="store_true",
        help="Force single-only mode across prompt generation (ignore hard/auto inference).",
    )
    parser.add_argument(
        "--no_single_only",
        dest="single_only",
        action="store_false",
        help="Disable single-only mode and honor --edit_mode selection.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config and exit.")
    parser.set_defaults(single_only=True)
    args = parser.parse_args()
    argv = list(os.sys.argv[1:])
    single_only_explicit = argv_has_flag(argv, "--single_only", "--no_single_only")
    edit_mode_explicit = argv_has_flag(argv, "--edit_mode")
    requested_branch_display, single_only_display, effective_edit_mode_display = resolve_stage3_branch_defaults(
        requested_branch=str(args.repair_branch),
        input_path=str(args.input or ""),
        run_tag=str(args.run_tag),
        single_only=bool(args.single_only),
        edit_mode=str(args.edit_mode),
        single_only_explicit=bool(single_only_explicit),
        edit_mode_explicit=bool(edit_mode_explicit),
    )

    def _auto_output_path(run_tag: str, edit_mode: str, repair_branch: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_suffix = f".{repair_branch}" if repair_branch in {"java_base", "java_semantic", "java_v2"} else ""
        mode_suffix = f".{edit_mode}" if edit_mode in ("single", "hard") else ""
        return str(RESULTS_ROOT / "3" / f"3.{run_tag}{branch_suffix}{mode_suffix}.{ts}.json")

    def _extract_ts(name: str) -> str:
        m = re.search(r"(\d{8}_\d{6})", name)
        return m.group(1) if m else ""

    def _is_smoke_name(name: str) -> bool:
        return "smoke" in name.lower()

    def _discover_stage_inputs(
        prev_stage: int,
        run_tag: str,
        prefer_tag: str = "",
        single_only: bool = False,
    ) -> List[str]:
        files: List[Path] = []
        stage_dir = RESULTS_ROOT / str(prev_stage)
        if stage_dir.is_dir():
            files.extend([p for p in stage_dir.glob("*.json") if p.is_file()])
        if not files:
            return []
        if run_tag:
            tagged = [p for p in files if run_tag in p.name]
            if tagged:
                files = tagged
        base_files = list(files)
        if prefer_tag in ("hard", "single"):
            split_tagged = [p for p in files if f".{prefer_tag}" in p.name]
            if split_tagged:
                files = split_tagged
        elif bool(single_only):
            single_files = [p for p in files if ".single" in p.name]
            non_hard_files = [p for p in files if ".single" not in p.name and ".hard" not in p.name]
            hard_files = [p for p in files if ".hard" in p.name]
            if single_files:
                files = single_files + non_hard_files + hard_files
            elif non_hard_files:
                files = non_hard_files + hard_files
            elif hard_files:
                files = hard_files
        explicit_smoke_run = bool(run_tag) and ("smoke" in run_tag.lower())
        if not explicit_smoke_run:
            preferred_non_smoke = [p for p in files if not _is_smoke_name(p.name)]
            if preferred_non_smoke:
                files = preferred_non_smoke
            else:
                fallback_non_smoke = [p for p in base_files if not _is_smoke_name(p.name)]
                if fallback_non_smoke:
                    files = fallback_non_smoke
        if not files:
            return []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ts_list = [_extract_ts(p.name) for p in files if _extract_ts(p.name)]
        if ts_list:
            latest_ts = max(ts_list)
            grouped = [p for p in files if latest_ts in p.name]
            if grouped:
                files = grouped
        return [str(p) for p in files]

    def _split_inputs(arg: str) -> List[str]:
        if not arg:
            return []
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        if len(parts) == 1 and any(ch in parts[0] for ch in ["*", "?", "["]):
            return sorted(glob.glob(parts[0]))
        return parts

    def _results_subdir_for(input_path: str, subdir: str) -> Path:
        parts = Path(input_path).parts
        if RESULTS_ROOT.name in parts:
            idx = parts.index(RESULTS_ROOT.name)
            return Path(*parts[:idx + 1], subdir)
        return RESULTS_ROOT / subdir

    def _derive_output_path(input_path: str, edit_mode: str, repair_branch: str) -> str:
        in_path = Path(input_path)
        name = in_path.name
        stem = name[:-5] if name.lower().endswith(".json") else name

        if re.match(r"^\d+\.", stem):
            stem = re.sub(r"^\d+\.", "3.", stem, count=1)
        elif not stem.startswith("3."):
            stem = "3." + stem

        # Drop BM25 markers from the output name.
        rest = stem[2:] if stem.startswith("3.") else stem
        rest = re.sub(r"(?i)bm25result", "", rest)
        rest = re.sub(r"(?i)bm25", "", rest)
        rest = re.sub(r"[._-]{2,}", ".", rest).strip("._-")
        if not rest:
            rest = "result"
        stem = f"3.{rest}"
        if repair_branch in {"java_base", "java_semantic", "java_v2"} and f".{repair_branch}" not in stem:
            stem = f"{stem}.{repair_branch}"
        if edit_mode in ("single", "hard") and f".{edit_mode}" not in stem:
            stem = f"{stem}.{edit_mode}"

        # Requested naming style:
        # 3.<...>.single_GeneratePromport.json
        out_name = f"{stem}_GeneratePromport.json"
        out_dir = _results_subdir_for(input_path, "3")
        return str(out_dir / out_name)

    def _config_for_input(input_path: str) -> Tuple[str, bool, str]:
        return resolve_stage3_branch_defaults(
            requested_branch=str(args.repair_branch),
            input_path=input_path,
            run_tag=str(args.run_tag),
            single_only=bool(args.single_only),
            edit_mode=str(args.edit_mode),
            single_only_explicit=bool(single_only_explicit),
            edit_mode_explicit=bool(edit_mode_explicit),
        )

    auto_in = ""
    if args.input:
        inputs = _split_inputs(args.input)
        if not inputs:
            inputs = [args.input]
    else:
        if effective_edit_mode_display in ("single", "hard"):
            auto_inputs = _discover_stage_inputs(
                prev_stage=2,
                run_tag=str(args.run_tag),
                prefer_tag=str(effective_edit_mode_display),
                single_only=bool(single_only_display),
            )
        else:
            auto_inputs = _discover_stage_inputs(
                prev_stage=2,
                run_tag=str(args.run_tag),
                prefer_tag="hard",
                single_only=bool(single_only_display),
            )
        if not auto_inputs:
            auto_inputs = _discover_stage_inputs(
                prev_stage=2,
                run_tag=str(args.run_tag),
                prefer_tag="",
                single_only=bool(single_only_display),
            )
        if not auto_inputs:
            raise SystemExit(f"No Stage-2 JSON found for auto discovery in {RESULTS_ROOT / '2'}.")
        inputs = auto_inputs
        auto_in = ",".join(auto_inputs)
        print(f"[auto] in_json <- {auto_in}")

    if args.output:
        resolved_outputs = [
            (
                args.output
                if len(inputs) == 1 else _derive_output_path(input_path, _config_for_input(input_path)[2], _config_for_input(input_path)[0])
            )
            for input_path in inputs
        ]
    else:
        resolved_outputs = [
            (
                _auto_output_path(str(args.run_tag), _config_for_input(input_path)[2], _config_for_input(input_path)[0])
                if len(inputs) == 1
                else _derive_output_path(input_path, _config_for_input(input_path)[2], _config_for_input(input_path)[0])
            )
            for input_path in inputs
        ]

    print(
        f"[resolved][Stage3] run_tag={args.run_tag} "
        f"edit_mode={effective_edit_mode_display} single_only={bool(single_only_display)} "
        f"repair_branch={requested_branch_display} "
        f"max_full_code_lines={int(args.max_full_code_lines)} "
        f"max_context_lines={int(args.max_context_lines)} "
        f"dry_run={bool(args.dry_run)}"
    )
    if auto_in:
        print(f"[resolved][Stage3] auto_in_json={auto_in}")
    else:
        print(f"[resolved][Stage3] in_json={','.join(inputs)}")
    print(f"[resolved][Stage3] out_json={resolved_outputs[0] if resolved_outputs else ''}")
    if args.dry_run:
        print("[dry_run] Stage3 exiting before prompt generation.")
        return

    for input_path, output_path in zip(inputs, resolved_outputs):
        source = read_json_file(Path(input_path))
        if source is None:
            raise SystemExit(1)

        effective_branch, effective_single_only, effective_edit_mode = resolve_stage3_branch_defaults(
            requested_branch=str(args.repair_branch),
            input_path=input_path,
            run_tag=str(args.run_tag),
            single_only=bool(args.single_only),
            edit_mode=str(args.edit_mode),
            single_only_explicit=bool(single_only_explicit),
            edit_mode_explicit=bool(edit_mode_explicit),
        )

        processed = process_content(
            source=source,
            max_full_code_lines=int(args.max_full_code_lines),
            max_context_lines=int(args.max_context_lines),
            edit_mode=str(effective_edit_mode),
            repair_branch=str(effective_branch),
            single_only_explicit=bool(single_only_explicit),
            edit_mode_explicit=bool(edit_mode_explicit),
            single_only=bool(effective_single_only),
        )

        write_json_file(processed, Path(output_path))
        print(f"✅ 프롬프트가 추가된 JSON 파일 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
