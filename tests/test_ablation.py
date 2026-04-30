"""Sanity checks for TA-noMeta ablation builder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.feedback_prompt import build_feedback_prompt, _build_test_feedback, _build_test_feedback_no_meta
from core.error_classifier import FeedbackStrategy


_EVAL = {
    "failing_tests_count": 3,
    "failing_test_names": ["testFooBar", "testBaz"],
    "first_failing_test": "testFooBar",
    "first_failure_message": "expected:<1> but was:<2>",
    "first_failure_stack": "at com.example.Foo.bar(Foo.java:42)",
    "test_output": "FAILED testFooBar",
}
_PROM = {
    "buggy_line_content": "return x + 1;",
    "function": {"function_before": "int foo() { return x + 1; }"},
}


def test_no_meta_excludes_metadata():
    out = _build_test_feedback_no_meta("int foo() { return x; }", _EVAL, _PROM)
    assert "testFooBar" not in out, "test name leaked"
    assert "expected:<1>" not in out, "failure message leaked"
    assert "Foo.java:42" not in out, "stack frame leaked"
    assert "FAILING TEST" not in out, "metadata header leaked"
    assert "HINT" in out and "return x + 1;" in out, "buggy line hint missing"
    assert "ORIGINAL BUGGY CODE" in out, "original code missing"
    assert "MINIMAL semantic fix" in out, "minimal-edit instruction missing"
    print("PASS: test_no_meta_excludes_metadata")


def test_full_ta_still_includes_metadata():
    out = _build_test_feedback("int foo() { return x; }", _EVAL, _PROM)
    assert "testFooBar" in out, "full TA should keep test name"
    assert "expected:<1>" in out, "full TA should keep failure message"
    print("PASS: test_full_ta_still_includes_metadata")


def test_dispatch_flag_routes_correctly():
    strategy = FeedbackStrategy(error_category="test_fail", strategy_name="test_feedback")
    full = build_feedback_prompt("orig prompt", "prev", strategy, _EVAL, _PROM, 1)
    no_meta = build_feedback_prompt("orig prompt", "prev", strategy, _EVAL, _PROM, 1,
                                    ablation_no_meta=True)
    assert "testFooBar" in full and "testFooBar" not in no_meta
    assert "expected:<1>" in full and "expected:<1>" not in no_meta
    assert "HINT" in full and "HINT" in no_meta
    print("PASS: test_dispatch_flag_routes_correctly")


def test_blind_takes_priority_over_ablation():
    strategy = FeedbackStrategy(error_category="test_fail", strategy_name="test_feedback")
    out = build_feedback_prompt("orig prompt", "prev", strategy, _EVAL, _PROM, 1,
                                blind=True, ablation_no_meta=True)
    assert "DIFFERENT approach" in out, "blind should take priority"
    assert "testFooBar" not in out
    print("PASS: test_blind_takes_priority_over_ablation")


def test_compile_error_unaffected_by_ablation():
    strategy = FeedbackStrategy(error_category="compile_symbol", strategy_name="symbol_feedback")
    eval_compile = {"compile_stderr": "error: cannot find symbol", "fail_reason": "compile_fail"}
    out_normal = build_feedback_prompt("orig", "prev", strategy, eval_compile, _PROM, 1)
    out_ablation = build_feedback_prompt("orig", "prev", strategy, eval_compile, _PROM, 1,
                                         ablation_no_meta=True)
    assert out_normal == out_ablation, "compile feedback must be unchanged by ablation flag"
    print("PASS: test_compile_error_unaffected_by_ablation")


if __name__ == "__main__":
    test_no_meta_excludes_metadata()
    test_full_ta_still_includes_metadata()
    test_dispatch_flag_routes_correctly()
    test_blind_takes_priority_over_ablation()
    test_compile_error_unaffected_by_ablation()
    print("\nAll tests passed.")
