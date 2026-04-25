#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0_EnrichContext.py — File-level structural context enrichment (Stage 0)
========================================================================
Extracts file-level structural context from the source file containing
the buggy function. Language-agnostic: works for both BugsInPy (Python)
and Defects4J (Java) with the same interface.

Extracted context per bug:
  - imports:                    Import statements available in the file
  - class_fields:               Class/module-level field/attribute declarations
  - sibling_signatures:         Other function/method signatures in the same class/module
  - class_name:                 Enclosing class name (if applicable)

Usage:
  # Auto-detect all *_bugs_meta_data.json in current directory
  python 0_EnrichContext.py

  # Explicit input
  python 0_EnrichContext.py --input defects4j_bugs_meta_data.json
  python 0_EnrichContext.py --input bugsinpy_bugs_meta_data.json

  # Without checkout (extract from function_before only — limited context)
  python 0_EnrichContext.py --no_checkout

Output:
  Overwrites the input file in-place (auto-backup to .bak).
  After running, just continue with Stage 1 — no mv needed.

This script is designed to be run ONCE before the main pipeline (Stage 1–5).
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- Optional: tree-sitter for Java ----------
try:
    from tree_sitter_languages import get_parser as _get_ts_parser
    _JAVA_TS_PARSER = _get_ts_parser("java")
    _HAS_JAVA_TS = True
except Exception:
    _JAVA_TS_PARSER = None
    _HAS_JAVA_TS = False

# ---------- Self-contained: no external adapter dependency ----------
# defects4j project name mapping (repo name → defects4j project name)
_DEFECTS4J_REPO_TO_PROJECT = {
    'jfreechart': 'Chart',
    'commons-cli': 'Cli',
    'closure-compiler': 'Closure',
    'commons-codec': 'Codec',
    'commons-collections': 'Collections',
    'commons-compress': 'Compress',
    'commons-csv': 'Csv',
    'gson': 'Gson',
    'jackson-core': 'JacksonCore',
    'jackson-databind': 'JacksonDatabind',
    'jackson-dataformat-xml': 'JacksonXml',
    'jsoup': 'Jsoup',
    'commons-jxpath': 'JxPath',
    'commons-lang': 'Lang',
    'commons-math': 'Math',
    'mockito': 'Mockito',
    'joda-time': 'Time',
}


def _map_defects4j_project_name(repo_name: str) -> str:
    """repo name (e.g. 'jfreechart') → Defects4J project name (e.g. 'Chart')"""
    return _DEFECTS4J_REPO_TO_PROJECT.get(repo_name, repo_name)


# =========================================================================
#  Language detection
# =========================================================================

def detect_language(bug_data: Dict[str, Any]) -> str:
    file_path = (bug_data.get("file", {}) or {}).get("file_path", "")
    if isinstance(file_path, str):
        if file_path.lower().endswith(".java"):
            return "java"
        if file_path.lower().endswith(".py"):
            return "python"
    if bug_data.get("defects4j_id") is not None:
        return "java"
    return "python"


def detect_dataset(bug_data: Dict[str, Any], filename_hint: str = "") -> str:
    if bug_data.get("defects4j_id") is not None or "defects4j" in filename_hint.lower():
        return "defects4j"
    if bug_data.get("bugsinpy_id") is not None or "bugsinpy" in filename_hint.lower():
        return "bugsinpy"
    lang = detect_language(bug_data)
    return "defects4j" if lang == "java" else "bugsinpy"


# =========================================================================
#  Python context extraction (using stdlib ast)
# =========================================================================

def extract_python_imports(source: str, max_items: int = 30) -> List[str]:
    """Extract import statements from Python source."""
    imports: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex
        for m in re.finditer(r'^(?:from\s+[\w.]+\s+)?import\s+.+', source, re.MULTILINE):
            imports.append(m.group(0).strip())
        return imports[:max_items]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(
                (a.name + (f" as {a.asname}" if a.asname else ""))
                for a in node.names
            )
            imports.append(f"from {module} import {names}")
    return imports[:max_items]


def extract_python_class_fields(source: str, class_name: str = "", max_items: int = 15) -> List[str]:
    """Extract class-level attributes and __init__ self.x assignments."""
    fields: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return fields

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if class_name and node.name != class_name:
            continue

        # Class-level assignments
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        fields.append(f"{target.id} (class attr)")
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                ann = ast.dump(item.annotation) if item.annotation else ""
                # Simplified annotation
                ann_str = ""
                if isinstance(item.annotation, ast.Name):
                    ann_str = f": {item.annotation.id}"
                elif isinstance(item.annotation, ast.Constant):
                    ann_str = f": {item.annotation.value}"
                fields.append(f"{item.target.id}{ann_str} (class attr)")

        # __init__ self.x assignments
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"):
                                fields.append(f"self.{target.attr}")

    return list(dict.fromkeys(fields))[:max_items]  # dedup preserve order


def extract_python_sibling_signatures(
    source: str, class_name: str = "", target_function: str = "", max_items: int = 20
) -> List[str]:
    """Extract sibling function/method signatures in the same scope."""
    sigs: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sigs

    def _sig_from_funcdef(node: ast.FunctionDef) -> str:
        args_parts = []
        for a in node.args.args:
            args_parts.append(a.arg)
        if node.args.vararg:
            args_parts.append(f"*{node.args.vararg.arg}")
        for a in node.args.kwonlyargs:
            args_parts.append(a.arg)
        if node.args.kwarg:
            args_parts.append(f"**{node.args.kwarg.arg}")
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args_parts)})"

    if class_name:
        # Methods inside the target class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name != target_function:
                            sigs.append(_sig_from_funcdef(item))
    else:
        # Top-level functions
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name != target_function:
                    sigs.append(_sig_from_funcdef(item))

    return sigs[:max_items]


def extract_python_class_name_from_parent(function_parent: str) -> str:
    """Extract class name from function_parent like 'ClassName::method_name(...)'"""
    if "::" in function_parent:
        return function_parent.split("::")[0].strip()
    return ""


# =========================================================================
#  Java context extraction (tree-sitter preferred, regex fallback)
# =========================================================================

def extract_java_imports(source: str, max_items: int = 30) -> List[str]:
    return re.findall(r'^import\s+([\w.*]+)\s*;', source, re.MULTILINE)[:max_items]


def _extract_java_class_fields_ts(source: str, class_name: str, max_items: int) -> List[str]:
    if not _HAS_JAVA_TS or not _JAVA_TS_PARSER:
        return []
    code_bytes = source.encode("utf-8")
    tree = _JAVA_TS_PARSER.parse(code_bytes)
    fields: List[str] = []

    def visit(node, in_target=False):
        if node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            name = code_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else ""
            is_target = (not class_name) or (name == class_name)
            for child in node.children:
                visit(child, in_target=is_target)
            return
        if node.type == 'field_declaration' and in_target:
            field_text = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            field_text = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', field_text).strip()
            field_text = re.split(r'\s*=\s*', field_text)[0].strip().rstrip(';').strip()
            if field_text and len(field_text) < 200:
                fields.append(field_text)
        for child in node.children:
            visit(child, in_target)

    visit(tree.root_node)
    return fields[:max_items]


def _extract_java_class_fields_regex(source: str, max_items: int) -> List[str]:
    pattern = (
        r'^\s*(?:(?:public|protected|private|static|final|transient|volatile)\s+)*'
        r'([A-Z][\w<>\[\],.?\s&]*?)\s+'
        r'([a-z_]\w*)\s*(?:=|;)'
    )
    fields = []
    for m in re.finditer(pattern, source, re.MULTILINE):
        if '(' not in m.group(0):
            fields.append(f"{m.group(1).strip()} {m.group(2).strip()}")
    return fields[:max_items]


def extract_java_class_fields(source: str, class_name: str = "", max_items: int = 15) -> List[str]:
    ts_result = _extract_java_class_fields_ts(source, class_name, max_items)
    if ts_result:
        return ts_result
    return _extract_java_class_fields_regex(source, max_items)


def _extract_java_method_sigs_ts(source: str, class_name: str, target_function: str, max_items: int) -> List[str]:
    if not _HAS_JAVA_TS or not _JAVA_TS_PARSER:
        return []
    code_bytes = source.encode("utf-8")
    tree = _JAVA_TS_PARSER.parse(code_bytes)
    sigs: List[str] = []

    def visit(node, in_target=False):
        if node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            name = code_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else ""
            is_target = (not class_name) or (name == class_name)
            for child in node.children:
                visit(child, in_target=is_target)
            return
        if node.type in ('method_declaration', 'constructor_declaration') and in_target:
            # Extract name
            name_node = node.child_by_field_name('name')
            method_name = code_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else ""
            if method_name == target_function:
                return  # FIX: 타겟 함수는 건너뛰기만 — 내부로 재귀하지 않음
            # Extract signature (up to body)
            body_node = node.child_by_field_name('body')
            if body_node:
                sig = code_bytes[node.start_byte:body_node.start_byte].decode("utf-8", errors="replace").strip()
            else:
                sig = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            sig = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', sig).strip()
            if sig and len(sig) < 300 and _is_valid_java_signature(sig):
                sigs.append(sig)
            return
        for child in node.children:
            visit(child, in_target)

    visit(tree.root_node)
    return sigs[:max_items]


# Java keywords that should never be method names
_JAVA_KEYWORDS_SET = {
    'if', 'for', 'while', 'switch', 'catch', 'return', 'throw', 'new',
    'synchronized', 'do', 'else', 'try', 'finally', 'case', 'break',
    'continue', 'assert', 'import', 'package', 'class', 'interface',
    'enum', 'extends', 'implements', 'this', 'super', 'instanceof',
}


def _is_valid_java_signature(sig: str) -> bool:
    """시그니처가 실제 메서드/생성자 선언인지 검증"""
    stripped = sig.strip()
    if not stripped:
        return False
    # 첫 토큰이 Java keyword이면 false positive
    first_word = re.match(r'(\w+)', stripped)
    if first_word and first_word.group(1) in _JAVA_KEYWORDS_SET:
        return False
    # 최소한 '('가 있어야 함
    if '(' not in stripped:
        return False
    # 메서드 이름 추출: '(' 앞의 단어
    name_match = re.search(r'(\w+)\s*\(', stripped)
    if name_match and name_match.group(1) in _JAVA_KEYWORDS_SET:
        return False
    return True


def _extract_java_method_sigs_regex(source: str, target_function: str, max_items: int) -> List[str]:
    pattern = (
        r'^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|native)\s+)*'
        r'(?:<[^>]+>\s*)?'
        r'(?:[A-Z][\w<>\[\],.?\s&]*?\s+)?'
        r'([a-zA-Z_]\w*)\s*\([^)]*\)\s*(?:throws\s+[\w,.\s]+)?\s*\{'
    )
    sigs = []
    for m in re.finditer(pattern, source, re.MULTILINE):
        name_match = re.search(r'(\w+)\s*\(', m.group(0))
        if name_match and name_match.group(1) == target_function:
            continue
        # FIX: keyword 필터링
        if name_match and name_match.group(1) in _JAVA_KEYWORDS_SET:
            continue
        sig = m.group(0).rstrip('{').strip()
        sig = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', sig).strip()
        if sig and len(sig) < 300 and _is_valid_java_signature(sig):
            sigs.append(sig)
    return sigs[:max_items]


def extract_java_sibling_signatures(
    source: str, class_name: str = "", target_function: str = "", max_items: int = 20
) -> List[str]:
    ts_result = _extract_java_method_sigs_ts(source, class_name, target_function, max_items)
    if ts_result:
        return ts_result
    return _extract_java_method_sigs_regex(source, target_function, max_items)


def extract_java_class_name(function_parent: str) -> str:
    if "::" in function_parent:
        parts = function_parent.split("::")
        return parts[0].strip()
    return ""


# =========================================================================
#  Unified enrichment logic
# =========================================================================

def enrich_one_bug(bug_data: Dict[str, Any], source_code: Optional[str]) -> Dict[str, Any]:
    """
    Extract file-level structural context for one bug.
    Works for both Python and Java.

    Returns dict with keys:
      - enriched_imports
      - enriched_class_fields
      - enriched_sibling_signatures
      - enriched_class_name
      - enriched_source_available  (bool: was full file source available?)
    """
    lang = detect_language(bug_data)
    func = bug_data.get("function", {}) or {}
    function_name = func.get("function_name", "")
    function_parent = func.get("function_parent", "")
    function_before = func.get("function_before", "")

    # Determine class name
    if lang == "java":
        class_name = extract_java_class_name(function_parent)
    else:
        class_name = extract_python_class_name_from_parent(function_parent)

    # If full source is available, use it; otherwise fall back to function_before
    code = source_code if source_code else function_before
    has_full_source = bool(source_code)

    if lang == "java":
        imports = extract_java_imports(code) if has_full_source else []
        fields = extract_java_class_fields(code, class_name=class_name)
        sibling_sigs = extract_java_sibling_signatures(
            code, class_name=class_name, target_function=function_name
        ) if has_full_source else []
    else:
        imports = extract_python_imports(code) if has_full_source else []
        fields = extract_python_class_fields(code, class_name=class_name) if has_full_source else []
        sibling_sigs = extract_python_sibling_signatures(
            code, class_name=class_name, target_function=function_name
        ) if has_full_source else []

    return {
        "enriched_imports": imports,
        "enriched_class_fields": fields,
        "enriched_sibling_signatures": sibling_sigs,
        "enriched_class_name": class_name,
        "enriched_source_available": has_full_source,
    }


# =========================================================================
#  Checkout & file reading helpers
# =========================================================================

def _read_source_from_checkout(bug_data: Dict[str, Any], dataset: str, checkout_base: str) -> Optional[str]:
    """Checkout project and read full source file. No external adapter dependency."""
    import subprocess

    id_key = "defects4j_id" if dataset == "defects4j" else "bugsinpy_id"
    bug_dataset_id = bug_data.get(id_key)
    if bug_dataset_id is None:
        return None

    repo_name = bug_data.get("project_name", "")
    if not repo_name:
        return None

    file_path = (bug_data.get("file", {}) or {}).get("file_path", "")
    if not file_path:
        return None

    if dataset == "defects4j":
        project_name = _map_defects4j_project_name(repo_name)
        checkout_path = os.path.join(checkout_base, f"{project_name}_{bug_dataset_id}")

        if not os.path.isdir(checkout_path):
            cmd = [
                "defects4j", "checkout",
                "-p", project_name,
                "-v", f"{bug_dataset_id}f",
                "-w", checkout_path,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if result.returncode != 0 or not os.path.isdir(checkout_path):
                    print(f"    [WARN] defects4j checkout failed: {project_name}-{bug_dataset_id}")
                    return None
            except Exception as e:
                print(f"    [WARN] checkout error: {e}")
                return None

        source_file = os.path.join(checkout_path, file_path)

    else:  # bugsinpy
        project_name = repo_name
        checkout_path = os.path.join(checkout_base, f"{project_name}_{bug_dataset_id}")

        if not os.path.isdir(checkout_path):
            cmd = [
                "bugsinpy-checkout",
                "-p", project_name,
                "-i", str(bug_dataset_id),
                "-v", "0",
                "-w", checkout_path,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if result.returncode != 0 or not os.path.isdir(checkout_path):
                    print(f"    [WARN] bugsinpy checkout failed: {project_name}-{bug_dataset_id}")
                    return None
            except Exception as e:
                print(f"    [WARN] checkout error: {e}")
                return None

        # BugsInPy: file is inside project_name subdirectory
        source_file = os.path.join(checkout_path, project_name, file_path)
        if not os.path.exists(source_file):
            source_file = os.path.join(checkout_path, file_path)

    try:
        if not os.path.exists(source_file):
            return None
        with open(source_file, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"    [WARN] read error: {e}")
        return None


# =========================================================================
#  Main processing
# =========================================================================

def process_json(
    input_path: str,
    output_path: str,
    *,
    use_checkout: bool = True,
    checkout_base: str = "",
) -> Dict[str, int]:
    """Process a single metadata JSON file."""

    with open(input_path, 'r', encoding='utf-8') as f:
        bugs_data = json.load(f)

    if not isinstance(bugs_data, dict):
        print(f"ERROR: Expected dict, got {type(bugs_data).__name__} in {input_path}")
        return {"total": 0, "enriched": 0, "errors": 0}

    # Detect dataset from first bug
    first_bug = next(iter(bugs_data.values()), {})
    dataset = detect_dataset(first_bug, filename_hint=input_path)
    lang = detect_language(first_bug)

    print(f"Dataset: {dataset} | Language: {lang} | Bugs: {len(bugs_data)} | Checkout: {use_checkout}")

    # Checkout base directory
    if not checkout_base:
        if dataset == "defects4j":
            checkout_base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "defects4j", "framework", "bin", "temp")
            )
        else:
            checkout_base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "BugsInPy", "framework", "bin", "temp")
            )
    os.makedirs(checkout_base, exist_ok=True)

    # Verify checkout command is available
    if use_checkout:
        check_cmd = "defects4j" if dataset == "defects4j" else "bugsinpy-checkout"
        if shutil.which(check_cmd) is not None:
            _status_print(f"  checkout command '{check_cmd}' found ✅")
        else:
            print(f"  WARNING: '{check_cmd}' not found in PATH. Falling back to no-checkout mode.")
            use_checkout = False

    stats = {"total": len(bugs_data), "enriched": 0, "errors": 0, "no_checkout": 0, "source_available": 0}

    for bug_id, bug_data in bugs_data.items():
        if not isinstance(bug_data, dict):
            continue

        source_code = None
        if use_checkout:
            source_code = _read_source_from_checkout(bug_data, dataset, checkout_base)

        if source_code is None and use_checkout:
            stats["no_checkout"] += 1

        try:
            context = enrich_one_bug(bug_data, source_code)

            # Store with unified key names
            bug_data["enriched_imports"] = context["enriched_imports"]
            bug_data["enriched_class_fields"] = context["enriched_class_fields"]
            bug_data["enriched_sibling_signatures"] = context["enriched_sibling_signatures"]
            bug_data["enriched_class_name"] = context["enriched_class_name"]
            bug_data["enriched_source_available"] = context["enriched_source_available"]

            # Also keep legacy Java-specific keys for backward compatibility
            if lang == "java":
                bug_data["java_imports"] = context["enriched_imports"]
                bug_data["java_class_fields"] = context["enriched_class_fields"]
                bug_data["java_sibling_method_signatures"] = context["enriched_sibling_signatures"]
                bug_data["java_class_name"] = context["enriched_class_name"]

            n_imp = len(context["enriched_imports"])
            n_fld = len(context["enriched_class_fields"])
            n_sig = len(context["enriched_sibling_signatures"])
            src = "file" if context["enriched_source_available"] else "func"
            if context["enriched_source_available"]:
                stats["source_available"] += 1

            stats["enriched"] += 1
            if int(bug_id) % 20 == 0 or n_imp + n_fld + n_sig > 0:
                print(f"  [{bug_id}] imports={n_imp} fields={n_fld} sigs={n_sig} src={src}")

        except Exception as e:
            print(f"  [{bug_id}] ERROR: {e}")
            stats["errors"] += 1

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bugs_data, f, ensure_ascii=False, indent=2)

    return stats


def _auto_discover_inputs(script_dir: Path) -> List[Path]:
    """Auto-discover *_bugs_meta_data.json files."""
    candidates = [
        script_dir / "bugsinpy_bugs_meta_data.json",
        script_dir / "defects4j_bugs_meta_data.json",
    ]
    # Also check dataset/ subdirectory
    for ds in ("bugsinpy", "defects4j"):
        candidates.append(script_dir / "dataset" / ds / f"{ds}_bugs_meta_data.json")
        candidates.append(script_dir.parent / "dataset" / ds / f"{ds}_bugs_meta_data.json")

    return [p for p in candidates if p.is_file()]


def _status_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: File-level structural context enrichment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        nargs="*",
        default=None,
        help="Input metadata JSON path(s). Auto-discovers if not specified.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path. Default: overwrite input file in-place.",
    )
    parser.add_argument(
        "--no_checkout",
        action="store_true",
        help="Skip checkout; extract context from function_before only (limited).",
    )
    parser.add_argument(
        "--checkout_base",
        default="",
        help="Base directory for checkouts. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Skip creating .bak backup before overwriting.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Discover inputs
    if args.input:
        inputs = [Path(p) for p in args.input]
    else:
        inputs = _auto_discover_inputs(script_dir)
        if not inputs:
            print("ERROR: No *_bugs_meta_data.json found. Specify --input explicitly.")
            sys.exit(1)
        print(f"Auto-discovered {len(inputs)} input file(s):")
        for p in inputs:
            print(f"  {p}")

    use_checkout = not args.no_checkout

    # Process each input
    total_stats = {"total": 0, "enriched": 0, "errors": 0, "no_checkout": 0}

    for input_path in inputs:
        if args.output and len(inputs) == 1:
            output_path = args.output
        else:
            # In-place: 원본 파일에 직접 덮어쓰기
            output_path = str(input_path)

        # Auto backup
        if output_path == str(input_path) and not args.no_backup:
            backup_path = str(input_path) + ".bak"
            shutil.copy2(str(input_path), backup_path)
            print(f"  Backup: {backup_path}")

        print(f"\n{'='*60}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path} {'(in-place)' if output_path == str(input_path) else ''}")
        print(f"{'='*60}")

        stats = process_json(
            str(input_path),
            str(output_path),
            use_checkout=use_checkout,
            checkout_base=args.checkout_base,
        )

        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        print(f"\n  Result: enriched={stats['enriched']}, errors={stats['errors']}, "
              f"no_checkout={stats['no_checkout']}")
        print(f"  enriched_source_available: {bool(stats.get('source_available', 0) > 0)}")

    print(f"\n{'='*60}")
    print(f"TOTAL: enriched={total_stats['enriched']}/{total_stats['total']}, "
          f"errors={total_stats['errors']}, no_checkout={total_stats['no_checkout']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
