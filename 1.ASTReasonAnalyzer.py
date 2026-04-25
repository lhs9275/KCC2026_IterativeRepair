#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST-only bug hint extractor.
Reads root bug metadata JSONs by default (fallback: Results/1/*) and writes
Results/1/* by default.
with compact, high-signal suspicious nodes.
Uses Python ast for .py and tree-sitter (Python/Java) when available.
"""

import argparse
import ast
import glob
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tree_sitter import Parser as _TSParser, Language as _TSLanguage
except Exception:
    _TSParser = None
    _TSLanguage = None

try:
    import tree_sitter_java as _ts_java
    _HAS_JAVA_TS = _TSParser is not None and _TSLanguage is not None
except Exception:
    _HAS_JAVA_TS = False
    _ts_java = None

try:
    import tree_sitter_python as _ts_python
    _HAS_PY_TS = _TSParser is not None and _TSLanguage is not None
except Exception:
    _HAS_PY_TS = False
    _ts_python = None

# -------------------- Constants --------------------

INTERESTING_NODES = {
    ast.If: "If",
    ast.For: "For",
    ast.While: "While",
    ast.Try: "Try",
    ast.ExceptHandler: "ExceptHandler",
    ast.With: "With",
    ast.Raise: "Raise",
    ast.Assert: "Assert",
    ast.Return: "Return",
    ast.Assign: "Assign",
    ast.AnnAssign: "AnnAssign",
    ast.AugAssign: "AugAssign",
    ast.Call: "Call",
    ast.Compare: "Compare",
    ast.BoolOp: "BoolOp",
    ast.BinOp: "BinOp",
    ast.UnaryOp: "UnaryOp",
    ast.Subscript: "Subscript",
    ast.Attribute: "Attribute",
    ast.ListComp: "ListComp",
    ast.DictComp: "DictComp",
    ast.SetComp: "SetComp",
    ast.GeneratorExp: "GeneratorExp",
}

TYPE_WEIGHT = {
    "Call": 0, "Assign": 1, "AugAssign": 1, "AnnAssign": 1,
    "If": 1, "Compare": 1, "BoolOp": 1,
    "Return": 2, "For": 2, "While": 2, "Try": 1, "ExceptHandler": 1, "With": 2,
    "Subscript": 1, "Attribute": 1, "BinOp": 2, "UnaryOp": 2,
    "Raise": 2, "Assert": 2, "Function": 2, "Class": 2,
}

PATCH_HINTS = {
    "If": ["check condition"],
    "Compare": ["check boundary"],
    "BoolOp": ["check predicate"],
    "Call": ["check arguments/return"],
    "Assign": ["check mutation/copy"],
    "AugAssign": ["check mutation/copy"],
    "Subscript": ["check index/key"],
    "Attribute": ["check attribute access"],
    "Return": ["check return value"],
    "Raise": ["check exception type/message"],
    "Assert": ["check assertion"],
    "Function": ["check signature/body"],
    "Class": ["check class body"],
}

JAVA_NODE_MAP = {
    "if_statement": "If",
    "for_statement": "For",
    "enhanced_for_statement": "For",
    "while_statement": "While",
    "do_statement": "While",
    "try_statement": "Try",
    "catch_clause": "ExceptHandler",
    "throw_statement": "Raise",
    "assert_statement": "Assert",
    "return_statement": "Return",
    "assignment_expression": "Assign",
    "update_expression": "AugAssign",
    "method_invocation": "Call",
    "object_creation_expression": "Call",
    "binary_expression": "Compare",
    "unary_expression": "UnaryOp",
    "conditional_expression": "BoolOp",
    "field_access": "Attribute",
    "array_access": "Subscript",
    "method_declaration": "Function",
    "class_declaration": "Class",
}

PY_NODE_MAP = {
    "if_statement": "If",
    "for_statement": "For",
    "while_statement": "While",
    "try_statement": "Try",
    "except_clause": "ExceptHandler",
    "with_statement": "With",
    "raise_statement": "Raise",
    "assert_statement": "Assert",
    "return_statement": "Return",
    "assignment": "Assign",
    "augmented_assignment": "AugAssign",
    "call": "Call",
    "comparison_operator": "Compare",
    "boolean_operator": "BoolOp",
    "binary_operator": "BinOp",
    "unary_operator": "UnaryOp",
    "subscript": "Subscript",
    "attribute": "Attribute",
    "list_comprehension": "ListComp",
    "dictionary_comprehension": "DictComp",
    "set_comprehension": "SetComp",
    "generator_expression": "GeneratorExp",
    "function_definition": "Function",
    "class_definition": "Class",
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
    "raise", "return", "try", "while", "with", "yield",
}

RESULTS_ROOT = Path(os.environ.get("PIPELINE_RESULTS_ROOT", "./Results"))


def _stage_dir(stage: int) -> Path:
    return RESULTS_ROOT / str(stage)

# -------------------- Helpers --------------------

def read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(data: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def trim_text_by_lines(text: Optional[str], max_lines: int) -> str:
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").splitlines()
    if max_lines <= 0:
        return ""
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n... (truncated)"
    return "\n".join(lines)


def extract_identifiers(text: str) -> List[str]:
    if not text:
        return []
    names = [m.group(0) for m in _IDENT_RE.finditer(text)]
    return [n for n in names if n not in _PY_KEYWORDS]


def detect_language(item: Dict[str, Any]) -> str:
    file_path = safe_get(item, ["file", "file_path"]) or item.get("file_path") or ""
    if isinstance(file_path, str):
        lower = file_path.lower()
        if lower.endswith(".java"):
            return "java"
        if lower.endswith(".py"):
            return "python"
    dataset = (item.get("dataset") or "").lower()
    if dataset == "defects4j":
        return "java"
    return "python"


def make_reason(node: Dict[str, Any]) -> str:
    parts: List[str] = []
    ntype = node.get("type")
    dist = node.get("distance_to_buggy")
    overlap = node.get("identifier_overlap_names") or []
    hints = node.get("patch_hints") or []
    if ntype:
        parts.append(f"type={ntype}")
    if dist is not None:
        parts.append(f"dist={dist}")
    if overlap:
        parts.append("overlap=" + ",".join(overlap[:6]))
    if hints:
        parts.append("hints=" + ",".join(hints[:4]))
    return "; ".join(parts)


def node_to_code(node: ast.AST, source: str) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if isinstance(seg, str) and seg.strip():
            return seg.strip()
    except Exception:
        pass
    return ""


def get_code_context(source: str, lineno: Optional[int], window: int = 3) -> str:
    if not source or lineno is None:
        return ""
    lines = source.splitlines()
    if lineno < 1 or lineno > len(lines):
        return ""
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    out = []
    for i in range(start, end):
        prefix = "-> " if i == lineno - 1 else "   "
        out.append(f"{i+1:4d}:{prefix}{lines[i]}")
    return "\n".join(out)


def distance_to_buggy_line(node_line: Optional[int], buggy_line: Optional[int]) -> Optional[int]:
    if node_line is None or buggy_line is None:
        return None
    return abs(node_line - buggy_line)


def rank_nodes(nodes: List[Dict[str, Any]], buggy_line: Optional[int], buggy_line_content: str) -> List[Dict[str, Any]]:
    buggy_idents = set(extract_identifiers(buggy_line_content or ""))
    scored = []
    for n in nodes:
        line = n.get("line")
        dist = distance_to_buggy_line(line, buggy_line)
        weight = TYPE_WEIGHT.get(n.get("type"), 3)
        score = (dist if dist is not None else 9999) * 10 + weight
        overlap_names: List[str] = []
        if buggy_idents:
            node_idents = set(extract_identifiers(n.get("code", "")))
            overlap_names = sorted(node_idents & buggy_idents)
        overlap = len(overlap_names)
        if n.get("contains_buggy_line"):
            score -= 20
        if n.get("match_buggy_content"):
            score -= 15
        if overlap:
            score -= min(10, overlap * 3)
        n2 = dict(n)
        n2["_score"] = score
        n2["distance_to_buggy"] = dist
        n2["identifier_overlap"] = overlap
        n2["identifier_overlap_names"] = overlap_names
        scored.append(n2)
    scored.sort(key=lambda x: x["_score"])
    return scored


def _is_strong_hint(node: Dict[str, Any]) -> bool:
    if node.get("contains_buggy_line") or node.get("match_buggy_content"):
        return True
    dist = node.get("distance_to_buggy")
    if dist is not None and dist <= 2:
        return True
    if (node.get("identifier_overlap") or 0) > 0:
        return True
    if node.get("patch_hints"):
        return True
    return False


def select_effective_nodes(nodes_ranked: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    if not nodes_ranked:
        return []
    strong = [n for n in nodes_ranked if _is_strong_hint(n)]
    if len(strong) < topk:
        seen = set(id(n) for n in strong)
        for n in nodes_ranked:
            if id(n) in seen:
                continue
            strong.append(n)
            seen.add(id(n))
            if len(strong) >= topk:
                break
    return strong[:topk]


def extract_nodes_with_ast(
    code: str,
    buggy_line: Optional[int],
    buggy_line_content: str,
    lang: str = "python",
) -> List[Dict[str, Any]]:
    if lang == "java":
        return extract_nodes_with_java_ast(code, buggy_line, buggy_line_content)
    nodes = extract_nodes_with_python_ast(code, buggy_line, buggy_line_content)
    if nodes and isinstance(nodes[0], dict) and "error" in nodes[0]:
        ts_nodes = extract_nodes_with_python_ts(code, buggy_line, buggy_line_content)
        if ts_nodes and not (isinstance(ts_nodes[0], dict) and "error" in ts_nodes[0]):
            return ts_nodes
        return ts_nodes
    return nodes


def extract_nodes_with_python_ast(code: str, buggy_line: Optional[int], buggy_line_content: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]

    nodes = []
    for node in ast.walk(tree):
        for cls, name in INTERESTING_NODES.items():
            if isinstance(node, cls):
                line = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if end is None:
                    end = line
                code_snip = node_to_code(node, code)
                match_buggy = False
                if buggy_line_content and code_snip:
                    match_buggy = buggy_line_content.strip() in code_snip
                contains_buggy = False
                if buggy_line is not None and line is not None:
                    contains_buggy = (line <= buggy_line <= (end or line))
                nodes.append({
                    "type": name,
                    "line": line,
                    "end_line": end,
                    "code": code_snip,
                    "contains_buggy_line": contains_buggy,
                    "match_buggy_content": match_buggy,
                    "patch_hints": PATCH_HINTS.get(name, []),
                })
                break
    return nodes


_JAVA_PARSER = None
_JAVA_PARSER_ERR = None


def _get_java_parser() -> Optional[Any]:
    global _JAVA_PARSER, _JAVA_PARSER_ERR
    if _JAVA_PARSER is not None or _JAVA_PARSER_ERR is not None:
        return _JAVA_PARSER
    if not _HAS_JAVA_TS:
        _JAVA_PARSER_ERR = "tree_sitter/tree_sitter_java not available"
        return None
    try:
        parser = _TSParser()
        parser.language = _TSLanguage(_ts_java.language())
        _JAVA_PARSER = parser
        return _JAVA_PARSER
    except Exception as e:
        _JAVA_PARSER_ERR = f"{type(e).__name__}: {e}"
        return None


_PY_PARSER = None
_PY_PARSER_ERR = None


def _get_python_parser() -> Optional[Any]:
    global _PY_PARSER, _PY_PARSER_ERR
    if _PY_PARSER is not None or _PY_PARSER_ERR is not None:
        return _PY_PARSER
    if not _HAS_PY_TS:
        _PY_PARSER_ERR = "tree_sitter/tree_sitter_python not available"
        return None
    try:
        parser = _TSParser()
        parser.language = _TSLanguage(_ts_python.language())
        _PY_PARSER = parser
        return _PY_PARSER
    except Exception as e:
        _PY_PARSER_ERR = f"{type(e).__name__}: {e}"
        return None


def _java_node_to_code(code_bytes: bytes, start_byte: int, end_byte: int) -> str:
    if start_byte < 0 or end_byte > len(code_bytes) or end_byte <= start_byte:
        return ""
    try:
        return code_bytes[start_byte:end_byte].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _collect_java_nodes(
    root: Any,
    code_bytes: bytes,
    line_offset: int,
    byte_offset: int,
    total_lines: int,
    buggy_line: Optional[int],
    buggy_line_content: str,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    def visit(node: Any) -> None:
        node_type = getattr(node, "type", "")
        label = JAVA_NODE_MAP.get(node_type)
        if label:
            start_line = node.start_point[0] + 1 - line_offset
            end_line = node.end_point[0] + 1 - line_offset
            start_byte = node.start_byte - byte_offset
            end_byte = node.end_byte - byte_offset
            if 1 <= start_line <= total_lines and 1 <= end_line <= total_lines:
                code_snip = _java_node_to_code(code_bytes, start_byte, end_byte)
                match_buggy = False
                if buggy_line_content and code_snip:
                    match_buggy = buggy_line_content.strip() in code_snip
                contains_buggy = False
                if buggy_line is not None:
                    contains_buggy = (start_line <= buggy_line <= end_line)
                nodes.append({
                    "type": label,
                    "line": start_line,
                    "end_line": end_line,
                    "code": code_snip,
                    "contains_buggy_line": contains_buggy,
                    "match_buggy_content": match_buggy,
                    "patch_hints": PATCH_HINTS.get(label, []),
                })
        for child in node.children:
            visit(child)

    visit(root)
    return nodes


def extract_nodes_with_java_ast(code: str, buggy_line: Optional[int], buggy_line_content: str) -> List[Dict[str, Any]]:
    parser = _get_java_parser()
    if parser is None:
        return [{"error": f"Java parser unavailable: {_JAVA_PARSER_ERR}"}]

    code_bytes = code.encode("utf-8", errors="replace")
    total_lines = len(code.splitlines()) if code else 0

    tree = parser.parse(code_bytes)
    nodes = _collect_java_nodes(
        tree.root_node,
        code_bytes,
        line_offset=0,
        byte_offset=0,
        total_lines=total_lines,
        buggy_line=buggy_line,
        buggy_line_content=buggy_line_content,
    )

    if nodes and not tree.root_node.has_error:
        return nodes

    wrapper_prefix = "class Dummy {\n"
    wrapper_suffix = "\n}\n"
    wrapped = wrapper_prefix + code + wrapper_suffix
    wrapped_bytes = wrapped.encode("utf-8", errors="replace")
    prefix_len = len(wrapper_prefix.encode("utf-8", errors="replace"))
    line_offset = wrapper_prefix.count("\n")

    tree2 = parser.parse(wrapped_bytes)
    nodes2 = _collect_java_nodes(
        tree2.root_node,
        code_bytes,
        line_offset=line_offset,
        byte_offset=prefix_len,
        total_lines=total_lines,
        buggy_line=buggy_line,
        buggy_line_content=buggy_line_content,
    )
    if nodes2:
        return nodes2
    if tree2.root_node.has_error:
        return [{"error": "Java AST parsing failed (tree-sitter)"}]
    return nodes2


def _python_node_to_code(code_bytes: bytes, start_byte: int, end_byte: int) -> str:
    if start_byte < 0 or end_byte > len(code_bytes) or end_byte <= start_byte:
        return ""
    try:
        return code_bytes[start_byte:end_byte].decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _collect_python_nodes(
    root: Any,
    code_bytes: bytes,
    total_lines: int,
    buggy_line: Optional[int],
    buggy_line_content: str,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    def visit(node: Any) -> None:
        node_type = getattr(node, "type", "")
        label = PY_NODE_MAP.get(node_type)
        if label:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            if 1 <= start_line <= total_lines and 1 <= end_line <= total_lines:
                code_snip = _python_node_to_code(code_bytes, node.start_byte, node.end_byte)
                match_buggy = False
                if buggy_line_content and code_snip:
                    match_buggy = buggy_line_content.strip() in code_snip
                contains_buggy = False
                if buggy_line is not None:
                    contains_buggy = (start_line <= buggy_line <= end_line)
                nodes.append({
                    "type": label,
                    "line": start_line,
                    "end_line": end_line,
                    "code": code_snip,
                    "contains_buggy_line": contains_buggy,
                    "match_buggy_content": match_buggy,
                    "patch_hints": PATCH_HINTS.get(label, []),
                })
        for child in node.children:
            visit(child)

    visit(root)
    return nodes


def extract_nodes_with_python_ts(code: str, buggy_line: Optional[int], buggy_line_content: str) -> List[Dict[str, Any]]:
    parser = _get_python_parser()
    if parser is None:
        return [{"error": f"Python parser unavailable: {_PY_PARSER_ERR}"}]

    code_bytes = code.encode("utf-8", errors="replace")
    total_lines = len(code.splitlines()) if code else 0
    tree = parser.parse(code_bytes)
    nodes = _collect_python_nodes(
        tree.root_node,
        code_bytes,
        total_lines=total_lines,
        buggy_line=buggy_line,
        buggy_line_content=buggy_line_content,
    )
    if nodes:
        return nodes
    if tree.root_node.has_error:
        return [{"error": "Python AST parsing failed (tree-sitter)"}]
    return nodes


def analyze_one_bug(bug_id: str, item: Dict[str, Any], topk: int) -> Dict[str, Any]:
    function_info = item.get("function", {}) or {}
    function_before = function_info.get("function_before", "") or ""
    buggy_line_location = _to_int_or_none(item.get("buggy_line_location"))
    buggy_line_content = (item.get("buggy_line_content") or "").strip()
    lang = detect_language(item)

    result = dict(item)
    result["ast_parse_ok"] = True
    result["ast_backend"] = "unknown"
    result["ast_node_count"] = 0

    if not function_before.strip():
        result["error"] = f"bug_id={bug_id} has empty function_before"
        result["suspicious_nodes_topk"] = []
        return result

    # Dedent for AST parse while preserving line count
    code = textwrap.dedent(function_before)
    lines = code.splitlines()

    # Map buggy line to function-local line if possible
    func_start = _to_int_or_none(function_info.get("function_before_start_line"))
    local_bug_line = None
    if buggy_line_location is not None and func_start is not None:
        local_bug_line = buggy_line_location - func_start + 1
    if local_bug_line is None or local_bug_line < 1 or local_bug_line > len(lines):
        if buggy_line_location is not None and 1 <= buggy_line_location <= len(lines):
            local_bug_line = buggy_line_location
        else:
            local_bug_line = None

    nodes = extract_nodes_with_ast(code, local_bug_line, buggy_line_content, lang=lang)
    if nodes and isinstance(nodes[0], dict) and "error" in nodes[0]:
        result["ast_parse_ok"] = False
        result["ast_backend"] = "error"
        result["ast_node_count"] = 0
        result["error"] = f"AST parsing failed: {nodes[0]['error']}"
        result["suspicious_nodes_topk"] = []
        return result
    result["ast_parse_ok"] = True
    result["ast_node_count"] = len(nodes)
    result["ast_backend"] = "java_ts" if lang == "java" else "python_ast_or_ts"

    ranked = rank_nodes(nodes, local_bug_line, buggy_line_content)
    picked = select_effective_nodes(ranked, topk)

    suspicious_results = []
    for node in picked:
        line = node.get("line")
        ctx = get_code_context(code, line, window=3)
        suspicious_results.append({
            "line": line,
            "type": node.get("type"),
            "contains_buggy_line": node.get("contains_buggy_line"),
            "distance_to_buggy": node.get("distance_to_buggy"),
            "identifier_overlap": node.get("identifier_overlap"),
            "identifier_overlap_names": node.get("identifier_overlap_names"),
            "patch_hints": node.get("patch_hints"),
            "reason": make_reason(node),
            "code": trim_text_by_lines(node.get("code"), 4),
            "context": trim_text_by_lines(ctx, 6),
        })

    # Update buggy line context using function-local code when possible
    result["buggy_line_context"] = get_code_context(code, local_bug_line, window=3)
    result["suspicious_nodes_topk"] = suspicious_results
    if "error" in result:
        result.pop("error", None)
    return result


def normalize_top_level(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {str(i): v for i, v in enumerate(obj)}
    raise TypeError("Top-level JSON must be an object or an array.")


def _split_inputs(arg: str) -> List[str]:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) == 1 and any(ch in parts[0] for ch in ["*", "?", "["]):
        return sorted(glob.glob(parts[0]))
    return parts


def _results_subdir_for(input_path: str, subdir: str) -> str:
    parts = Path(input_path).parts
    if RESULTS_ROOT.name in parts:
        idx = parts.index(RESULTS_ROOT.name)
        return str(Path(*parts[:idx + 1], subdir))
    return str(RESULTS_ROOT / subdir)


def _derive_output_path(input_path: str, default_output: str) -> str:
    in_path = os.path.normpath(input_path)
    name = os.path.basename(in_path)
    if name.startswith("1.DataSeterAgentResult"):
        name = name.replace("1.DataSeterAgentResult", "1.AST_Analyzed_Results", 1)
    else:
        if name.lower().endswith(".json"):
            name = name[:-5] + ".AST_Analyzed.json"
        else:
            name = name + ".AST_Analyzed.json"
    out_dir = _results_subdir_for(input_path, "1")
    return os.path.join(out_dir, name)


def _auto_output_path(run_tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(_stage_dir(1) / f"1.{run_tag}.{ts}.json")


def _extract_ts(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else ""


def _discover_root_metadata_inputs() -> List[str]:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "bugsinpy_bugs_meta_data.json",
        repo_root / "defects4j_bugs_meta_data.json",
    ]
    return [str(p) for p in candidates if p.is_file()]


def _find_latest_manifest(run_tag: str) -> Optional[str]:
    stage_dir = _stage_dir(1)
    if not stage_dir.is_dir():
        return None

    def _pick(pattern: str) -> Optional[str]:
        files = [p for p in stage_dir.glob(pattern) if p.is_file()]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0])

    if run_tag:
        tagged = _pick(f"1.{run_tag}*.manifest.json")
        if tagged:
            return tagged
    by_prefix = _pick("1.*.manifest.json")
    return by_prefix


def _discover_stage1_hard_inputs(run_tag: str, single_only: bool = False) -> List[str]:
    root_inputs = _discover_root_metadata_inputs()
    if root_inputs:
        return root_inputs

    # 1) manifest priority
    manifest_path = _find_latest_manifest(run_tag)
    if manifest_path:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            splits = man.get("splits") if isinstance(man, dict) else {}
            out: List[str] = []
            keys = (
                ("defects4j.single", "bugsinpy.single")
                if bool(single_only)
                else ("defects4j.hard", "bugsinpy.hard")
            )
            for k in keys:
                p = splits.get(k) if isinstance(splits, dict) else None
                if isinstance(p, str) and Path(p).is_file():
                    out.append(p)
            if out:
                return out
        except Exception:
            pass

    # 2) fallback: Stage-1 jsons in Results/1.
    # Prefer requested split buckets when present, then generic metadata files.
    stage_dir = _stage_dir(1)
    if not stage_dir.is_dir():
        return []

    files = [
        p for p in stage_dir.glob("*.json")
        if p.is_file() and not p.name.endswith(".manifest.json")
    ]
    if run_tag:
        tagged = [p for p in files if run_tag in p.name]
        if tagged:
            files = tagged

    if single_only:
        single_files = [p for p in files if ".single" in p.name]
        if single_files:
            files = single_files
        else:
            non_hard_files = [p for p in files if ".hard" not in p.name]
            hard_files = [p for p in files if ".hard" in p.name]
            files = non_hard_files if non_hard_files else hard_files
    else:
        hard_files = [p for p in files if ".hard" in p.name]
        if hard_files:
            files = hard_files

    if not files:
        return []

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    ts_list = [_extract_ts(p.name) for p in files if _extract_ts(p.name)]
    if ts_list:
        latest_ts = max(ts_list)
        no_ts = [p for p in files if not _extract_ts(p.name)]
        grouped = [p for p in files if latest_ts in p.name]
        if grouped:
            files = grouped + no_ts

    out_paths: List[str] = []
    seen_ds = set()
    for p in files:
        name = p.name.lower()
        ds = "defects4j" if "defects4j" in name else ("bugsinpy" if "bugsinpy" in name else "")
        if ds and ds in seen_ds:
            continue
        if ds:
            seen_ds.add(ds)
        out_paths.append(str(p))

    return out_paths if out_paths else [str(files[0])]


def _discover_latest_stage_json(prev_stage: int, run_tag: str) -> Optional[str]:
    def _pick(pattern: str) -> Optional[str]:
        stage_dir = _stage_dir(prev_stage)
        if not stage_dir.is_dir():
            return None
        files = [p for p in stage_dir.glob(pattern) if p.is_file()]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0])

    if run_tag:
        tagged = _pick(f"{prev_stage}.{run_tag}*.json")
        if tagged:
            return tagged
    by_prefix = _pick(f"{prev_stage}.*.json")
    if by_prefix:
        return by_prefix
    any_json = _pick("*.json")
    return any_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AST-only bug analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_tag", default="full", help="Run tag used for auto I/O path discovery.")
    parser.add_argument(
        "--input_json",
        "--in_json",
        dest="input_json",
        default=None,
        help="Input Stage-1 JSON path, glob, or comma-separated paths.",
    )
    parser.add_argument(
        "--output_json",
        "--out_json",
        dest="output_json",
        default=None,
        help="Output JSON path (auto-derived per input when multiple inputs are given).",
    )
    parser.add_argument(
        "--single_only",
        dest="single_only",
        action="store_true",
        help="Force single-only auto input discovery (prefer .single, never prefer .hard).",
    )
    parser.add_argument(
        "--no_single_only",
        dest="single_only",
        action="store_false",
        help="Disable single-only discovery and allow hard-priority auto input selection.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of suspicious AST nodes to keep.")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config and exit.")
    parser.set_defaults(single_only=True)
    args = parser.parse_args()

    auto_selected = False
    if args.input_json:
        inputs = _split_inputs(args.input_json)
        if not inputs:
            inputs = [args.input_json]
    else:
        auto_inputs = _discover_stage1_hard_inputs(str(args.run_tag), single_only=bool(args.single_only))
        if not auto_inputs:
            raise SystemExit(
                "No root metadata JSON found for auto discovery "
                "(bugsinpy_bugs_meta_data.json / defects4j_bugs_meta_data.json), "
                "and no fallback Stage-1 JSON found in Results/1."
            )
        inputs = auto_inputs
        auto_selected = True

    if args.output_json:
        resolved_outputs = [
            (args.output_json if len(inputs) == 1 else _derive_output_path(input_path, args.output_json))
            for input_path in inputs
        ]
    else:
        resolved_outputs = [
            (_auto_output_path(str(args.run_tag)) if len(inputs) == 1
             else _derive_output_path(input_path, _auto_output_path(str(args.run_tag))))
            for input_path in inputs
        ]

    print(
        f"[resolved][AST] run_tag={args.run_tag} topk={int(args.topk)} "
        f"single_only={bool(args.single_only)} input_count={len(inputs)} dry_run={bool(args.dry_run)}"
    )
    if auto_selected:
        print(f"[resolved][AST] auto_in_json={','.join(inputs)}")
    else:
        print(f"[resolved][AST] in_json={inputs[0] if inputs else ''}")
    print(f"[resolved][AST] out_json={resolved_outputs[0] if resolved_outputs else ''}")

    if args.dry_run:
        print("[dry_run][AST] exiting before analysis.")
        return

    for input_path, output_path in zip(inputs, resolved_outputs):
        data = read_json_file(input_path)
        bugs = normalize_top_level(data)

        results: Dict[str, Any] = {}
        for bug_id, item in bugs.items():
            if not isinstance(item, dict):
                continue
            results[bug_id] = analyze_one_bug(bug_id, item, int(args.topk))
        write_json_file(results, output_path)
        print(f"✅ 완료: {input_path} -> {output_path} (총 {len(results)}개)")


if __name__ == "__main__":
    main()
