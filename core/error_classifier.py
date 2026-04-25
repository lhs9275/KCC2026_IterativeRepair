"""
Error classification and feedback strategy selection.

Maps compile/test error types to specific feedback strategies
that determine what context to include in retry prompts.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Compile error family classification (adapted from ICSE defects4j_command.py)
# ---------------------------------------------------------------------------

def classify_compile_error_family(stdout: str, stderr: str) -> str:
    """Classify Java compile error into a family based on error messages."""
    combined = f"{stdout}\n{stderr}".lower()
    if "cannot find symbol" in combined:
        return "cannot_find_symbol"
    # Generics-specific type errors — surface these BEFORE the broader
    # "incompatible types" bucket so that feedback can highlight type
    # parameter handling (bounds, wildcards, inference) instead of a
    # generic "cast/reassign" hint.
    if (
        "inference variable" in combined
        or "is not within its bound" in combined
        or "is not within bounds of type-variable" in combined
        or "does not conform to upper bound" in combined
        or "type argument " in combined and "is not within bounds" in combined
        or re.search(r"incompatible types[:\s][^\n]*<[^>\n]+>", combined)
    ):
        return "generic_type_mismatch"
    if (
        "no suitable method found" in combined
        or "cannot be applied to given types" in combined
        or "does not override" in combined
    ):
        return "method_signature"
    if "incompatible types" in combined:
        return "type_mismatch"
    if (
        "must be caught or declared to be thrown" in combined
        or "unreported exception" in combined
    ):
        return "checked_exception"
    if "has private access" in combined or "has protected access" in combined:
        return "access_control"
    if (
        re.search(r"package\s+.+\s+does not exist", combined)
        or re.search(r"import\s+.+\s+does not exist", combined)
    ):
        return "package_or_import"
    if (
        "';' expected" in combined
        or "'}' expected" in combined
        or "reached end of file while parsing" in combined
        or "illegal start of" in combined
    ):
        return "syntax_or_parse"
    return "other_compile_fail"


# ---------------------------------------------------------------------------
# Feedback strategy data class
# ---------------------------------------------------------------------------

@dataclass
class FeedbackStrategy:
    error_category: str          # e.g. "compile_syntax", "test_fail"
    strategy_name: str           # e.g. "syntax_feedback", "symbol_feedback"
    include_stderr: bool = False
    include_imports: bool = False
    include_fields: bool = False
    include_siblings: bool = False
    simplify_request: bool = False


# ---------------------------------------------------------------------------
# Strategy mapping table
# ---------------------------------------------------------------------------

_STRATEGY_MAP: Dict[str, FeedbackStrategy] = {
    "syntax_or_parse": FeedbackStrategy(
        error_category="compile_syntax",
        strategy_name="syntax_feedback",
        include_stderr=True,
    ),
    "cannot_find_symbol": FeedbackStrategy(
        error_category="compile_symbol",
        strategy_name="symbol_feedback",
        include_stderr=True,
        include_imports=True,
        include_fields=True,
        include_siblings=True,
    ),
    "type_mismatch": FeedbackStrategy(
        error_category="compile_type",
        strategy_name="type_feedback",
        include_stderr=True,
        include_imports=True,
        include_fields=True,
    ),
    "generic_type_mismatch": FeedbackStrategy(
        error_category="compile_generic_type",
        strategy_name="generic_type_feedback",
        include_stderr=True,
        include_imports=True,
        include_fields=True,
        include_siblings=True,
    ),
    "method_signature": FeedbackStrategy(
        error_category="compile_signature",
        strategy_name="signature_feedback",
        include_stderr=True,
        include_siblings=True,
    ),
    "checked_exception": FeedbackStrategy(
        error_category="compile_exception",
        strategy_name="exception_feedback",
        include_stderr=True,
        include_siblings=True,
    ),
    "access_control": FeedbackStrategy(
        error_category="compile_access",
        strategy_name="access_feedback",
        include_stderr=True,
        include_fields=True,
    ),
    "package_or_import": FeedbackStrategy(
        error_category="compile_import",
        strategy_name="import_feedback",
        include_stderr=True,
        include_imports=True,
    ),
    "other_compile_fail": FeedbackStrategy(
        error_category="compile_other",
        strategy_name="generic_compile_feedback",
        include_stderr=True,
        include_imports=True,
        include_fields=True,
    ),
}

_TEST_FAIL_STRATEGY = FeedbackStrategy(
    error_category="test_fail",
    strategy_name="test_feedback",
)

_TIMEOUT_STRATEGY = FeedbackStrategy(
    error_category="timeout",
    strategy_name="simplify_feedback",
    simplify_request=True,
)

_GENERIC_STRATEGY = FeedbackStrategy(
    error_category="unknown",
    strategy_name="generic_compile_feedback",
    include_stderr=True,
    include_imports=True,
    include_fields=True,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_and_select_strategy(eval_result: Dict[str, Any]) -> FeedbackStrategy:
    """
    Given an evaluation result dict (from execution_tests_detailed),
    classify the error and return the appropriate feedback strategy.
    """
    fail_reason = str(eval_result.get("fail_reason", ""))

    if fail_reason == "pass":
        raise ValueError("Cannot select feedback strategy for a passing result")

    if fail_reason == "timeout":
        return _TIMEOUT_STRATEGY

    if fail_reason == "test_fail" or fail_reason == "test_error":
        return _TEST_FAIL_STRATEGY

    if fail_reason == "compile_fail":
        error_family = str(eval_result.get("compile_error_family", ""))
        if not error_family:
            error_family = classify_compile_error_family(
                str(eval_result.get("compile_stdout", "")),
                str(eval_result.get("compile_stderr", "")),
            )
        return _STRATEGY_MAP.get(error_family, _GENERIC_STRATEGY)

    if fail_reason == "syntax_fail":
        return _STRATEGY_MAP["syntax_or_parse"]

    return _GENERIC_STRATEGY


def failure_priority(eval_result: Dict[str, Any]) -> int:
    """
    Priority for selecting the 'best' failing candidate for feedback.
    Higher = closer to correct = better feedback signal.
    """
    fail_reason = str(eval_result.get("fail_reason", ""))
    if fail_reason in ("test_fail", "test_error"):
        return 3  # compiled successfully, test failed — closest to correct
    if fail_reason == "compile_fail":
        error_family = str(eval_result.get("compile_error_family", ""))
        if error_family in (
            "cannot_find_symbol",
            "type_mismatch",
            "generic_type_mismatch",
            "method_signature",
        ):
            return 2  # structural issue, potentially fixable with context
        return 1  # syntax or other compile error
    if fail_reason == "syntax_fail":
        return 1
    return 0  # timeout, apply_fail, unknown
