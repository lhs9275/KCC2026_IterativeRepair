"""
Evaluation harness for iterative repair.

Wraps the ICSE evaluation pipeline to support in-loop compile/test
within the iterative repair cycle. Handles checkout, patch application,
compile, test, and workspace cleanup.
"""

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

# Add icse_lib/icse_eval path for dataset_adapter, defects4j_command, etc.
_ICSE_EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "icse_lib", "icse_eval"))
if _ICSE_EVAL_DIR not in sys.path:
    sys.path.insert(0, _ICSE_EVAL_DIR)

from dataset_adapter import DatasetAdapter, Defects4J, BugsInPy
from defects4j_command import (
    defects4j_checkout,
    defects4j_compile_detailed,
    defects4j_test,
    _classify_compile_error_family,
    command_with_timeout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAIL_REASON_ENUM = {
    "pass", "empty_code", "syntax_fail", "apply_fail",
    "compile_fail", "test_fail", "test_error", "timeout",
    "unknown_error",
}
COMPILE_LOG_TRUNCATE_LIMIT = 4000
_FAILING_TEST_NAME_RE = re.compile(r"([A-Za-z0-9_.$]+::[A-Za-z0-9_.$]+)")


# ---------------------------------------------------------------------------
# Utility functions (adapted from ICSE evaluate.py)
# ---------------------------------------------------------------------------

def normalize_fail_reason(reason: str) -> str:
    reason = str(reason or "").strip()
    return reason if reason in FAIL_REASON_ENUM else "unknown_error"


def truncate_log_text(text: Any, limit: int = COMPILE_LOG_TRUNCATE_LIMIT) -> str:
    rendered = str(text or "")
    if len(rendered) <= limit:
        return rendered
    suffix = "...[truncated]"
    return rendered[:max(0, limit - len(suffix))] + suffix


def read_optional_text(path: str, limit: int = COMPILE_LOG_TRUNCATE_LIMIT) -> str:
    try:
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return truncate_log_text(f.read(), limit=limit)
    except Exception:
        return ""


def decode_process_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def extract_failing_test_cases(text: str) -> List[str]:
    seen = set()
    tests: List[str] = []
    for line in str(text or "").splitlines():
        match = _FAILING_TEST_NAME_RE.search(line)
        if not match:
            continue
        test_case = match.group(1)
        if test_case not in seen:
            seen.add(test_case)
            tests.append(test_case)
    return tests


def split_failure_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if line.startswith("--- "):
            if current:
                blocks.append(current)
            current = [line]
            continue
        if current:
            current.append(line)

    if current:
        blocks.append(current)
    return blocks


def summarize_first_failure(text: str, max_stack_frames: int = 3) -> Dict[str, str]:
    blocks = split_failure_blocks(text)
    if not blocks:
        return {
            "test_case": "",
            "message": "",
            "stack": "",
        }

    first_block = [line.strip() for line in blocks[0] if line.strip()]
    if not first_block:
        return {
            "test_case": "",
            "message": "",
            "stack": "",
        }

    test_case = ""
    header = first_block[0]
    if header.startswith("--- "):
        test_case = header[4:].strip()

    message = ""
    stack_frames: List[str] = []
    for line in first_block[1:]:
        if line.startswith("at "):
            stack_frames.append(line)
            continue
        if not message:
            message = line

    noise_tokens = (
        "junit.framework",
        "org.apache.tools.ant",
        "jdk.internal.reflect",
        "java.base/",
        "java.lang.reflect",
        "sun.reflect",
    )
    useful_stack_frames = [
        frame for frame in stack_frames
        if not any(token in frame.lower() for token in noise_tokens)
    ]
    selected_stack_frames = useful_stack_frames or stack_frames

    return {
        "test_case": test_case,
        "message": message,
        "stack": "\n".join(selected_stack_frames[:max_stack_frames]),
    }


def collect_first_failing_test_output(
    adapter: DatasetAdapter,
    project_path: str,
    test_case: str,
    timeout: int,
) -> str:
    if not test_case:
        return ""
    try:
        out, err = adapter.test_one(project_path, test_case, timeout=timeout)
    except Exception:
        logger.debug("Failed to rerun first failing test %s", test_case, exc_info=True)
        return ""

    stdout = decode_process_output(out)
    stderr = decode_process_output(err)
    combined = stdout
    if stderr:
        combined = f"{combined}\n{stderr}" if combined else stderr
    return truncate_log_text(combined, limit=COMPILE_LOG_TRUNCATE_LIMIT)


def normalize_compile_detail(raw_detail: Any) -> Dict[str, Any]:
    detail = dict(raw_detail or {}) if isinstance(raw_detail, dict) else {}
    ok = bool(detail.get("ok", False))
    return {
        "ok": bool(ok),
        "returncode": detail.get("returncode"),
        "stdout": truncate_log_text(detail.get("stdout", "")),
        "stderr": truncate_log_text(detail.get("stderr", "")),
        "elapsed_sec": detail.get("elapsed_sec"),
        "error_family": str(detail.get("error_family") or ("" if ok else "unknown_compile_fail")),
    }


def adjust_indent(code, new_indent):
    """Shift code so first non-empty line sits at new_indent spaces."""
    if code is None:
        return code
    lines = code.split("\n")
    first_idx = None
    first_indent = 0
    for i, l in enumerate(lines):
        if l.strip():
            first_idx = i
            first_indent = len(l) - len(l.lstrip(" "))
            break
    if first_idx is None:
        return code
    delta = int(new_indent) - int(first_indent)
    if delta == 0:
        return code
    body_indents = [
        len(l) - len(l.lstrip(" ")) for j, l in enumerate(lines)
        if j != first_idx and l.strip()
    ]
    # Only skip body-reindent when the body is STRICTLY deeper than new_indent
    # (i.e., there is real content nested below the signature). Otherwise a
    # malformed LLM output with body at the same indent as the signature would
    # be left flat (signature and body at same level → broken Java).
    body_absolute_ok = (
        bool(body_indents)
        and min(body_indents) >= int(new_indent)
        and max(body_indents) > int(new_indent)
    )
    if delta > 0 and body_absolute_ok:
        lines[first_idx] = " " * int(new_indent) + lines[first_idx].lstrip(" ")
        return "\n".join(lines)
    if delta > 0:
        pad = " " * delta
        return "\n".join((pad + l) if l.strip() else l for l in lines)
    strip_n = -delta
    out_lines = []
    for l in lines:
        if not l.strip():
            out_lines.append(l)
            continue
        k = 0
        while k < strip_n and k < len(l) and l[k] == " ":
            k += 1
        out_lines.append(l[k:])
    return "\n".join(out_lines)


def _normalize_source_line(line: str) -> str:
    return str(line).rstrip("\r\n").rstrip()


def _java_brace_count(line: str) -> tuple[int, int]:
    """Count { and } in a Java source line, ignoring string/char literals and // comments."""
    opens = closes = 0
    in_string = in_char = False
    i = 0
    while i < len(line):
        c = line[i]
        if in_string:
            if c == "\\" :
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
            elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            elif c == "{":
                opens += 1
            elif c == "}":
                closes += 1
        i += 1
    return opens, closes


def _find_exact_block_span(
    file_lines: List[str],
    block_text: str,
    preferred_start: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    block_lines = str(block_text or "").splitlines()
    if not file_lines or not block_lines:
        return None

    normalized_file = [_normalize_source_line(line) for line in file_lines]
    normalized_block = [_normalize_source_line(line) for line in block_lines]
    block_len = len(normalized_block)
    matches: List[tuple[int, int]] = []

    for start_idx in range(0, len(normalized_file) - block_len + 1):
        if normalized_file[start_idx:start_idx + block_len] == normalized_block:
            matches.append((start_idx + 1, start_idx + block_len))

    if not matches:
        return None
    if preferred_start is None:
        return matches[0]
    return min(matches, key=lambda span: abs(span[0] - int(preferred_start)))


def _find_signature_anchor_indices(file_lines: List[str], block_lines: List[str]) -> List[int]:
    meaningful = [line for line in block_lines if line.strip()]
    if not meaningful:
        return []

    probe: List[str] = []
    for line in meaningful:
        probe.append(_normalize_source_line(line))
        if "{" in line or len(probe) >= 3:
            break

    if not probe:
        return []

    normalized_file = [_normalize_source_line(line) for line in file_lines]
    matches: List[int] = []
    probe_len = len(probe)
    for start_idx in range(0, len(normalized_file) - probe_len + 1):
        if normalized_file[start_idx:start_idx + probe_len] == probe:
            matches.append(start_idx + 1)

    if matches:
        return matches

    first_probe = probe[0].strip()
    if not first_probe:
        return []
    for idx, line in enumerate(normalized_file, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == first_probe or stripped.startswith(first_probe):
            matches.append(idx)
    return matches


def _find_signature_based_span(
    file_lines: List[str],
    block_text: str,
    preferred_start: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    block_lines = str(block_text or "").splitlines()
    if not file_lines or not block_lines:
        return None

    anchor_indices = _find_signature_anchor_indices(file_lines, block_lines)
    if not anchor_indices:
        return None
    if preferred_start is not None:
        anchor_indices = sorted(anchor_indices, key=lambda idx: abs(idx - int(preferred_start)))

    for start in anchor_indices:
        brace_balance = 0
        opened = False
        for idx in range(start, len(file_lines) + 1):
            line = file_lines[idx - 1]
            o, c = _java_brace_count(line)
            brace_balance += o - c
            if o > 0:
                opened = True
            if opened and brace_balance <= 0:
                return start, idx

    return None


def _resolve_function_span(
    file_lines: List[str],
    bug_meta_data: Dict[str, Any],
) -> tuple[int, int]:
    fn_meta = bug_meta_data.get("function", {}) or {}
    preferred_start = fn_meta.get("function_before_start_line") or fn_meta.get("function_after_start_line")
    exact_span = _find_exact_block_span(
        file_lines=file_lines,
        block_text=fn_meta.get("function_before", ""),
        preferred_start=int(preferred_start) if preferred_start else None,
    )
    if exact_span is not None:
        return exact_span

    signature_span = _find_signature_based_span(
        file_lines=file_lines,
        block_text=fn_meta.get("function_before", ""),
        preferred_start=int(preferred_start) if preferred_start else None,
    )
    if signature_span is not None:
        return signature_span

    candidate_spans: List[tuple[int, int]] = []
    for start_key, end_key in (
        ("function_before_start_line", "function_before_end_line"),
        ("function_after_start_line", "function_after_end_line"),
    ):
        start = fn_meta.get(start_key)
        end = fn_meta.get(end_key)
        if isinstance(start, int) and isinstance(end, int) and start > 0 and end >= start:
            candidate_spans.append((start, end))

    function_before_lines = len(str(fn_meta.get("function_before", "") or "").splitlines())
    if bug_meta_data.get("project_name") == "jfreechart":
        fallback_special = handle_defects4j_special_cases(bug_meta_data, 0, 0)
        if fallback_special != (0, 0):
            candidate_spans.append(fallback_special)

    if candidate_spans:
        if function_before_lines > 0:
            return min(
                candidate_spans,
                key=lambda span: abs(((span[1] - span[0]) + 1) - function_before_lines),
            )
        return candidate_spans[0]

    raise ValueError("Unable to resolve function span from metadata")


def handle_defects4j_special_cases(bug_meta_data, default_start, default_end):
    project = bug_meta_data.get("project_name")
    defects4j_bug_id = str(bug_meta_data.get("defects4j_id"))
    if project == "jfreechart":
        special_cases = {
            "1": (1790, 1822), "9": (918, 956), "12": (143, 158),
            "13": (422, 489), "24": (123, 129),
        }
        return special_cases.get(defects4j_bug_id, (default_start, default_end))
    return default_start, default_end


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_candidate(
    adapter: DatasetAdapter,
    project_path: str,
    bug_meta_data: Dict[str, Any],
    candidate_code: str,
    test_timeout: int = 300,
) -> Dict[str, Any]:
    """
    Evaluate a single candidate patch: apply → compile → test.

    This is essentially execution_tests_detailed from ICSE evaluate.py,
    adapted for in-loop usage.

    Returns dict with keys:
        flag, fail_reason, test_executed, elapsed_sec,
        compile_attempted, compile_ok, compile_returncode,
        compile_stdout, compile_stderr, compile_error_family
    """
    saved_cwd = os.getcwd()
    target_file_path = os.path.join(project_path, bug_meta_data['file']['file_path'])
    target_file_path_backup = target_file_path + '.backup'
    started_at = time.time()

    result = {
        "flag": "Error: test execution",
        "fail_reason": "unknown_error",
        "test_executed": False,
        "elapsed_sec": None,
        "compile_attempted": False,
        "compile_ok": None,
        "compile_returncode": None,
        "compile_stdout": "",
        "compile_stderr": "",
        "compile_error_family": "",
        "test_output": "",
        "failing_tests_count": None,
        "failing_test_names": [],
        "first_failing_test": "",
        "first_failure_message": "",
        "first_failure_stack": "",
        "first_failing_test_output": "",
    }
    compile_attempted = False

    if not candidate_code or not candidate_code.strip():
        result["flag"] = "Error: empty candidate"
        result["fail_reason"] = "empty_code"
        result["elapsed_sec"] = round(time.time() - started_at, 6)
        return result

    if not os.path.exists(target_file_path):
        logger.error("Target file not found: %s", target_file_path)
        result["flag"] = "Error: file not found"
        result["fail_reason"] = "apply_fail"
        result["elapsed_sec"] = round(time.time() - started_at, 6)
        return result

    # Python syntax check
    if adapter.dataset_name == "bugsinpy":
        file_path = bug_meta_data.get("file", {}).get("file_path", "")
        if file_path.endswith(".py"):
            try:
                ast.parse(textwrap.dedent(candidate_code or ""))
            except (SyntaxError, Exception):
                result["flag"] = "Error: syntax"
                result["fail_reason"] = "syntax_fail"
                result["elapsed_sec"] = round(time.time() - started_at, 6)
                return result

    try:
        subprocess.run(['cp', target_file_path, target_file_path_backup], check=True)

        with open(target_file_path, 'r', encoding='utf-8') as f:
            old_file_lines = f.readlines()

        function_start, function_end = _resolve_function_span(old_file_lines, bug_meta_data)

        start_line = old_file_lines[function_start - 1]
        start_indent = len(start_line) - len(start_line.lstrip(" "))
        inference_code_indent = adjust_indent(candidate_code, start_indent)

        new_file_lines = (
            old_file_lines[:function_start - 1]
            + [inference_code_indent, '\n']
            + old_file_lines[function_end:]
        )
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(new_file_lines))

        compile_attempted = True
        result["compile_attempted"] = True
        compile_detail = normalize_compile_detail(
            adapter.compile_detailed(project_path)
            if hasattr(adapter, "compile_detailed")
            else {"ok": adapter.compile(project_path)}
        )
        compile_ok = bool(compile_detail.get("ok", False))
        result["compile_ok"] = bool(compile_ok)
        result["compile_returncode"] = compile_detail.get("returncode")
        result["compile_stdout"] = str(compile_detail.get("stdout", ""))
        result["compile_stderr"] = str(compile_detail.get("stderr", ""))
        result["compile_error_family"] = str(compile_detail.get("error_family", ""))

        if not compile_ok:
            result["flag"] = 'Fail'
            result["fail_reason"] = 'compile_fail'
        else:
            result["test_executed"] = True
            test_flag = adapter.test(project_path)
            test_flag_s = str(test_flag)
            failing_tests_path = os.path.join(project_path, "failing_tests")
            result["test_output"] = read_optional_text(failing_tests_path)
            failing_tests = extract_failing_test_cases(result["test_output"])
            first_failure = summarize_first_failure(result["test_output"])
            result["failing_tests_count"] = len(failing_tests)
            result["failing_test_names"] = failing_tests[:5]
            result["first_failing_test"] = failing_tests[0] if failing_tests else ""
            if not result["first_failing_test"] and first_failure["test_case"]:
                result["first_failing_test"] = first_failure["test_case"]
            result["first_failure_message"] = first_failure["message"]
            result["first_failure_stack"] = first_failure["stack"]
            if test_flag_s == 'Plausible':
                result["flag"] = 'Pass'
                result["fail_reason"] = 'pass'
            elif 'timeout' in test_flag_s.lower():
                result["flag"] = 'Fail'
                result["fail_reason"] = 'timeout'
            elif 'error' in test_flag_s.lower():
                result["flag"] = 'Error: test execution'
                result["fail_reason"] = 'test_error'
            else:
                result["flag"] = 'Fail'
                result["fail_reason"] = 'test_fail'

            if result["fail_reason"] in {"test_fail", "test_error"} and result["first_failing_test"]:
                result["first_failing_test_output"] = collect_first_failing_test_output(
                    adapter=adapter,
                    project_path=project_path,
                    test_case=result["first_failing_test"],
                    timeout=test_timeout,
                )

    except Exception as e:
        logger.error("Exception during evaluate_candidate: %s", e, exc_info=True)
        if result.get("test_executed"):
            result["flag"] = 'Error: test execution'
            result["fail_reason"] = 'test_error'
        elif compile_attempted:
            result["flag"] = 'Fail'
            result["fail_reason"] = 'compile_fail'
            if not str(result.get("compile_error_family") or "").strip():
                result["compile_error_family"] = 'other_compile_fail'
        else:
            result["flag"] = 'Error: apply'
            result["fail_reason"] = 'apply_fail'
    finally:
        if os.path.exists(target_file_path_backup):
            subprocess.run(['mv', target_file_path_backup, target_file_path], check=False)
        result["fail_reason"] = normalize_fail_reason(result.get("fail_reason"))
        result["elapsed_sec"] = round(time.time() - started_at, 6)
        os.chdir(saved_cwd)  # restore cwd (defects4j_compile/test change it)

    return result


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

def setup_workspace(
    adapter: DatasetAdapter,
    project_name: str,
    bug_id: str,
    workspace_root: str,
) -> Optional[str]:
    """
    Checkout a buggy project into a unique workspace directory.
    Returns the project path, or None if checkout failed.
    """
    workspace_dir = os.path.join(
        workspace_root,
        f"{project_name}_{bug_id}_{os.getpid()}_{int(time.time() * 1000)}"
    )
    os.makedirs(workspace_dir, exist_ok=True)
    project_path = os.path.join(workspace_dir, f"{project_name}_{bug_id}")

    ok = adapter.checkout(project_name, bug_id, project_path)
    if not ok:
        logger.error("Checkout failed for %s-%s", project_name, bug_id)
        if os.path.isdir(workspace_dir):
            shutil.rmtree(workspace_dir, ignore_errors=True)
        return None

    return project_path


def cleanup_workspace(workspace_dir: str):
    """Remove a workspace directory."""
    if workspace_dir and os.path.isdir(workspace_dir):
        shutil.rmtree(workspace_dir, ignore_errors=True)


def create_adapter(dataset: str) -> DatasetAdapter:
    """Factory for dataset adapters."""
    if dataset == "defects4j":
        return Defects4J()
    elif dataset == "bugsinpy":
        return BugsInPy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
