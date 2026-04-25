import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from core.error_classifier import (
    classify_compile_error_family,
    classify_and_select_strategy,
    failure_priority,
)


def test_classifies_cannot_find_symbol():
    stderr = "Foo.java:10: error: cannot find symbol\n  symbol:   method bar()"
    assert classify_compile_error_family("", stderr) == "cannot_find_symbol"


def test_classifies_generic_inference_variable():
    stderr = "Foo.java:5: error: inference variable T has incompatible bounds"
    assert classify_compile_error_family("", stderr) == "generic_type_mismatch"


def test_classifies_generic_upper_bound():
    stderr = "error: type argument Integer is not within bounds of type-variable T"
    assert classify_compile_error_family("", stderr) == "generic_type_mismatch"


def test_classifies_incompatible_types_with_generics():
    stderr = "error: incompatible types: List<String> cannot be converted to List<Object>"
    assert classify_compile_error_family("", stderr) == "generic_type_mismatch"


def test_classifies_incompatible_types_without_generics_is_type_mismatch():
    stderr = "error: incompatible types: int cannot be converted to String"
    assert classify_compile_error_family("", stderr) == "type_mismatch"


def test_strategy_routing_for_generic_type_mismatch():
    eval_result = {
        "fail_reason": "compile_fail",
        "compile_error_family": "generic_type_mismatch",
    }
    strategy = classify_and_select_strategy(eval_result)
    assert strategy.strategy_name == "generic_type_feedback"
    assert strategy.include_stderr is True
    assert strategy.include_siblings is True


def test_failure_priority_treats_generic_type_as_structural():
    eval_result = {
        "fail_reason": "compile_fail",
        "compile_error_family": "generic_type_mismatch",
    }
    assert failure_priority(eval_result) == 2  # structural, same tier as type_mismatch


def test_generic_type_feedback_builder_registered():
    from core.feedback_prompt import _FEEDBACK_BUILDERS
    assert "generic_type_feedback" in _FEEDBACK_BUILDERS


def test_generic_type_feedback_renders():
    from core.feedback_prompt import build_feedback_prompt
    from core.error_classifier import FeedbackStrategy

    strategy = FeedbackStrategy(
        error_category="compile_generic_type",
        strategy_name="generic_type_feedback",
        include_stderr=True,
        include_siblings=True,
    )
    eval_result = {
        "fail_reason": "compile_fail",
        "compile_error_family": "generic_type_mismatch",
        "compile_stderr": "error: inference variable T has incompatible bounds",
    }
    prom_row = {"enriched_sibling_signatures": ["<T> void foo(List<T> items)"]}
    out = build_feedback_prompt(
        original_prompt="ORIG",
        previous_code="public void bar() { return; }",
        strategy=strategy,
        eval_result=eval_result,
        prom_row=prom_row,
        iteration=2,
        blind=False,
    )
    assert "generic/parameterized type mismatch" in out
    assert "inference variable T" in out
    assert "<T> void foo(List<T> items)" in out
