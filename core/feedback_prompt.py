"""
Error-type-specific feedback prompt builders (v2 — concise, action-oriented).

Designed for small LLMs (~7B) that lose focus with verbose context.
Each feedback builder follows:  ERROR → WHAT WENT WRONG → HOW TO FIX → OUTPUT RULE.
"""

from typing import Any, Dict, List, Optional

from .error_classifier import FeedbackStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_STDERR_LINES = 5
_MAX_CONTEXT_ITEMS = 5
_MAX_TEST_LINES = 12
_MAX_TEST_DETAIL_LINES = 18
_MAX_TEST_NAMES = 5


def _truncate_stderr(stderr: str, max_lines: int = _MAX_STDERR_LINES) -> str:
    lines = str(stderr or "").strip().splitlines()
    # Filter out noise (ant build lines, warnings)
    useful = [l for l in lines if "error:" in l.lower() or "symbol" in l.lower()
              or "type" in l.lower() or "found" in l.lower() or "required" in l.lower()
              or "cannot" in l.lower() or "expected" in l.lower() or "exception" in l.lower()]
    if not useful:
        useful = lines
    if len(useful) <= max_lines:
        return "\n".join(useful)
    return "\n".join(useful[:max_lines])


def _truncate_test_output(text: str, max_lines: int = _MAX_TEST_LINES) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines])


def _truncate_test_detail(text: str, max_lines: int = _MAX_TEST_DETAIL_LINES) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    keywords = (
        "fail",
        "error",
        "exception",
        "assert",
        "expected",
        "actual",
        "caused by",
        "comparisonfailure",
        "at ",
    )
    useful = [line for line in lines if any(token in line.lower() for token in keywords)]
    selected = useful or lines

    if len(selected) <= max_lines:
        return "\n".join(selected)
    return "\n".join(selected[:max_lines])


def _format_test_names(test_names: Any, limit: int = _MAX_TEST_NAMES) -> str:
    if not isinstance(test_names, list):
        return ""
    names = [str(name).strip() for name in test_names if str(name).strip()]
    if not names:
        return ""
    return "\n".join(names[:limit])


def _short_context(prom_row: Dict[str, Any], key: str, limit: int = _MAX_CONTEXT_ITEMS) -> str:
    val = prom_row.get(key, [])
    if isinstance(val, str):
        val = [val] if val else []
    if not val:
        return "(none)"
    return "\n".join(str(s) for s in val[:limit])


def _get_buggy_code(prom_row: Dict[str, Any]) -> str:
    return prom_row.get("function", {}).get("function_before", "")


# ---------------------------------------------------------------------------
# Concise feedback builders
# ---------------------------------------------------------------------------

def _build_syntax_feedback(previous_code: str, eval_result: Dict[str, Any],
                           prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    return f"""COMPILE ERROR: syntax error.
{stderr}

YOUR WRONG CODE:
{previous_code}

FIX: Check for missing semicolons, unbalanced braces, or incomplete statements.
Output the COMPLETE fixed function."""


def _build_symbol_feedback(previous_code: str, eval_result: Dict[str, Any],
                           prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    siblings = _short_context(prom_row, "enriched_sibling_signatures")
    return f"""COMPILE ERROR: cannot find symbol.
{stderr}

YOUR WRONG CODE:
{previous_code}

AVAILABLE METHODS in this class:
{siblings}

FIX: Replace the unknown symbol with one from the list above. Do NOT invent methods.
Output the COMPLETE fixed function."""


def _build_type_feedback(previous_code: str, eval_result: Dict[str, Any],
                         prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    return f"""COMPILE ERROR: incompatible types.
{stderr}

YOUR WRONG CODE:
{previous_code}

FIX: Use the correct type or add an explicit cast. Check return types carefully.
Output the COMPLETE fixed function."""


def _build_generic_type_feedback(previous_code: str, eval_result: Dict[str, Any],
                                 prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    siblings = _short_context(prom_row, "enriched_sibling_signatures")
    return f"""COMPILE ERROR: generic/parameterized type mismatch.
{stderr}

YOUR WRONG CODE:
{previous_code}

AVAILABLE METHOD SIGNATURES (with their generic parameters):
{siblings}

FIX:
- Check the type parameters `<...>` — the compiler reports a bound/inference violation.
- Common patterns: use `<?>` or `<? extends T>` for read-only, `<? super T>` for write-only.
- Do NOT erase generics to raw types; keep the parameterization the surrounding code uses.
- Match the exact type argument shape shown in AVAILABLE METHOD SIGNATURES above.
Output the COMPLETE fixed function."""


def _build_signature_feedback(previous_code: str, eval_result: Dict[str, Any],
                              prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    siblings = _short_context(prom_row, "enriched_sibling_signatures")
    return f"""COMPILE ERROR: wrong method arguments.
{stderr}

YOUR WRONG CODE:
{previous_code}

AVAILABLE METHOD SIGNATURES:
{siblings}

FIX: Match the exact parameter types and count shown above.
Output the COMPLETE fixed function."""


def _build_exception_feedback(previous_code: str, eval_result: Dict[str, Any],
                              prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    return f"""COMPILE ERROR: uncaught checked exception.
{stderr}

YOUR WRONG CODE:
{previous_code}

FIX: Add try-catch or add "throws" to the method signature.
Output the COMPLETE fixed function."""


def _build_access_feedback(previous_code: str, eval_result: Dict[str, Any],
                           prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    return f"""COMPILE ERROR: access control violation (private/protected field or method).
{stderr}

YOUR WRONG CODE:
{previous_code}

FIX: Use a public getter/setter or an accessible alternative.
Output the COMPLETE fixed function."""


def _build_import_feedback(previous_code: str, eval_result: Dict[str, Any],
                           prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    imports = _short_context(prom_row, "enriched_imports")
    return f"""COMPILE ERROR: package or class does not exist.
{stderr}

YOUR WRONG CODE:
{previous_code}

AVAILABLE IMPORTS in this file:
{imports}

FIX: Use only classes from the imports above. Do NOT reference packages that are not imported.
Output the COMPLETE fixed function."""


def _build_generic_compile_feedback(previous_code: str, eval_result: Dict[str, Any],
                                    prom_row: Dict[str, Any]) -> str:
    stderr = _truncate_stderr(
        eval_result.get("compile_stderr", "") or eval_result.get("compile_stdout", "")
    )
    return f"""COMPILE ERROR:
{stderr}

YOUR WRONG CODE:
{previous_code}

FIX: Read the error message above and fix accordingly.
Output the COMPLETE fixed function."""


def _build_test_feedback(previous_code: str, eval_result: Dict[str, Any],
                         prom_row: Dict[str, Any]) -> str:
    # NOTE: Do NOT include fixed_line_content — that would be information
    # leakage (providing the developer patch as a hint). Only bug report
    # context and failing-test output are allowed.
    buggy_line = prom_row.get("buggy_line_content", "")
    buggy_code = _get_buggy_code(prom_row)
    failing_tests_count = eval_result.get("failing_tests_count")
    failing_test_names = _format_test_names(eval_result.get("failing_test_names", []))
    test_output = _truncate_test_output(eval_result.get("test_output", ""))
    first_failing_test = str(eval_result.get("first_failing_test", "") or "")
    first_failure_message = str(eval_result.get("first_failure_message", "") or "")
    first_failure_stack = _truncate_test_detail(eval_result.get("first_failure_stack", ""), max_lines=3)
    first_failing_test_output = ""
    if not first_failure_message:
        first_failing_test_output = _truncate_test_detail(
            eval_result.get("first_failing_test_output", "")
        )
    sections = []
    if failing_tests_count is not None:
        sections.append(f"FAILING TEST COUNT: {failing_tests_count}")
    if failing_test_names:
        sections.append(f"FAILING TEST NAMES:\n{failing_test_names}")
    elif test_output:
        sections.append(f"FAILING TESTS:\n{test_output}")
    else:
        sections.append("FAILING TESTS:\n(no detailed failing test output captured)")
    if first_failing_test:
        sections.append(f"FIRST FAILING TEST:\n{first_failing_test}")
    if first_failure_message:
        sections.append(f"FIRST FAILURE MESSAGE:\n{first_failure_message}")
    if first_failure_stack:
        sections.append(f"RELEVANT STACK FRAMES:\n{first_failure_stack}")
    if first_failing_test_output:
        sections.append(f"FIRST FAILING TEST DETAIL:\n{first_failing_test_output}")
    if buggy_line:
        sections.append(f"HINT: The original bug is likely on or near this line:\n  {buggy_line}")
    if buggy_code:
        sections.append(f"ORIGINAL BUGGY CODE:\n{buggy_code[:700]}")
    joined_sections = "\n\n".join(sections)
    fail_count = failing_tests_count if isinstance(failing_tests_count, int) else None
    if fail_count is not None and fail_count <= 3:
        header = (
            f"Your code COMPILES and is PARTIALLY correct — only {fail_count} "
            f"test(s) fail. Preserve the passing behavior; fix only what fails."
        )
    elif fail_count is not None:
        header = (
            f"Your code COMPILES but {fail_count} tests fail. "
            f"The logic is partially correct — keep the passing behavior and "
            f"target only the failing cases."
        )
    else:
        header = "Your code compiled but FAILED TESTS — the logic is wrong."
    return f"""{header}

{joined_sections}

YOUR WRONG CODE:
{previous_code}

FIX: Use the failing test information above to repair the logic.
Start from the ORIGINAL BUGGY CODE, not your previous rewrite.
Make a MINIMAL semantic fix near the buggy line.
Use the FIRST FAILURE MESSAGE as the main semantic target.
If only a small number of tests fail, target that behavior and preserve everything else.
Undo unnecessary changes from the wrong code.
Output the COMPLETE fixed function."""


def _build_simplify_feedback(previous_code: str, eval_result: Dict[str, Any],
                             prom_row: Dict[str, Any]) -> str:
    buggy_code = _get_buggy_code(prom_row)
    return f"""Your code caused a TIMEOUT (possible infinite loop).

YOUR WRONG CODE:
{previous_code}

ORIGINAL BUGGY CODE (for reference):
{buggy_code[:500]}

FIX: Make a MINIMAL one-line change to the original code. Do NOT add loops or recursion.
Output the COMPLETE fixed function."""


def _build_blind_feedback(previous_code: str, eval_result: Dict[str, Any],
                          prom_row: Dict[str, Any]) -> str:
    return f"""Your previous attempt was WRONG.

YOUR WRONG CODE:
{previous_code}

Try a DIFFERENT approach. Output the COMPLETE fixed function."""


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_FEEDBACK_BUILDERS = {
    "syntax_feedback": _build_syntax_feedback,
    "symbol_feedback": _build_symbol_feedback,
    "type_feedback": _build_type_feedback,
    "generic_type_feedback": _build_generic_type_feedback,
    "signature_feedback": _build_signature_feedback,
    "exception_feedback": _build_exception_feedback,
    "access_feedback": _build_access_feedback,
    "import_feedback": _build_import_feedback,
    "generic_compile_feedback": _build_generic_compile_feedback,
    "test_feedback": _build_test_feedback,
    "simplify_feedback": _build_simplify_feedback,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feedback_prompt(
    original_prompt: str,
    previous_code: str,
    strategy: FeedbackStrategy,
    eval_result: Dict[str, Any],
    prom_row: Dict[str, Any],
    iteration: int,
    blind: bool = False,
) -> str:
    if blind:
        feedback_section = _build_blind_feedback(previous_code, eval_result, prom_row)
    else:
        builder = _FEEDBACK_BUILDERS.get(strategy.strategy_name, _build_generic_compile_feedback)
        feedback_section = builder(previous_code, eval_result, prom_row)

    base_prompt = original_prompt.rstrip()
    if base_prompt.endswith("##correct"):
        base_prompt = base_prompt[:-len("##correct")].rstrip()

    return f"""{base_prompt}

---FEEDBACK (attempt {iteration})---
{feedback_section}

##correct
"""
