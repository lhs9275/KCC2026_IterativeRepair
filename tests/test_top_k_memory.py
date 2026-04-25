import os
import sys

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from core.iterative_repair import (
    IterativeConfig,
    _ranking_tuple,
    _update_top_k,
    run_iterative_repair,
)


# ---------------------------------------------------------------------------
# Unit tests for the new helpers
# ---------------------------------------------------------------------------

def test_ranking_tuple_prefers_test_fail_over_compile_fail():
    test_fail = {"fail_reason": "test_fail", "failing_tests_count": 2}
    compile_fail = {"fail_reason": "compile_fail", "compile_error_family": "cannot_find_symbol"}
    assert _ranking_tuple(test_fail) > _ranking_tuple(compile_fail)


def test_ranking_tuple_prefers_fewer_failing_tests():
    few = {"fail_reason": "test_fail", "failing_tests_count": 1}
    many = {"fail_reason": "test_fail", "failing_tests_count": 10}
    assert _ranking_tuple(few) > _ranking_tuple(many)


def test_update_top_k_deduplicates_by_code_and_caps():
    memory = []
    for i, (code, count) in enumerate([("A", 5), ("B", 2), ("C", 8), ("A", 1)]):
        eval_result = {"fail_reason": "test_fail", "failing_tests_count": count}
        memory = _update_top_k(memory, code, eval_result, iteration=i + 1, k=3)
    codes = [m["code"] for m in memory]
    assert codes[0] == "A"  # re-inserted A with count=1 (better than original count=5)
    # B (count=2) should rank above C (count=8)
    assert codes.index("B") < codes.index("C")
    assert len(memory) <= 3


def test_update_top_k_respects_k():
    memory = []
    for i, (code, count) in enumerate([("A", 5), ("B", 4), ("C", 3), ("D", 2), ("E", 1)]):
        eval_result = {"fail_reason": "test_fail", "failing_tests_count": count}
        memory = _update_top_k(memory, code, eval_result, iteration=i + 1, k=2)
    assert len(memory) == 2
    # Top 2 by fewest-failing are E (1) and D (2)
    assert {m["code"] for m in memory} == {"E", "D"}


# ---------------------------------------------------------------------------
# End-to-end behaviour of rotate_on_stagnation inside run_iterative_repair
# ---------------------------------------------------------------------------

class _StubBackend:
    """Returns canned LLM outputs per iteration."""

    def __init__(self, outputs_by_iter):
        self._outputs_by_iter = list(outputs_by_iter)
        self._iter = 0

    def generate_records(self, prompts, **kwargs):
        from llm_backend import GenerationRecord

        outputs = self._outputs_by_iter[self._iter]
        self._iter += 1
        return [GenerationRecord(text=t, tokens_in=0, tokens_out=0) for t in outputs]


def _java_block(body_stmt: str) -> str:
    return (
        "@@BEGIN_JAVA_CODE@@\n"
        "public int foo() {\n"
        f"    {body_stmt}\n"
        "}\n"
        "@@END_JAVA_CODE@@"
    )


def test_rotate_on_stagnation_switches_feedback_seed(monkeypatch):
    """When two consecutive iterations land on the same best code, the next
    iteration's feedback should be built from a DIFFERENT top-K seed."""

    # Iteration 0: two candidates. A compiles, fails 5 tests. B compiles, fails 8 tests.
    # Iteration 1: same A again (stagnation) with failing_tests_count=5.
    # Iteration 2: we expect the stagnation rotation to seed feedback from B.
    A = _java_block("return 1;")
    B = _java_block("return 2;")
    C = _java_block("return 3;")

    backend = _StubBackend([[A, B], [A], [C]])

    eval_call_log = []

    def fake_evaluate(code):
        eval_call_log.append(code)
        if "return 1" in code:
            return {
                "fail_reason": "test_fail",
                "failing_tests_count": 5,
                "compile_error_family": "",
                "first_failure_message": "",
            }
        if "return 2" in code:
            return {
                "fail_reason": "test_fail",
                "failing_tests_count": 8,
                "compile_error_family": "",
                "first_failure_message": "",
            }
        return {
            "fail_reason": "test_fail",
            "failing_tests_count": 3,
            "compile_error_family": "",
            "first_failure_message": "",
        }

    # Capture the previous_code that the feedback builder sees on each call.
    seen_previous = []

    import core.iterative_repair as ir
    real_build = ir.build_feedback_prompt

    def spy_build(*, original_prompt, previous_code, **kwargs):
        seen_previous.append(previous_code)
        return real_build(original_prompt=original_prompt, previous_code=previous_code, **kwargs)

    monkeypatch.setattr(ir, "build_feedback_prompt", spy_build)

    config = IterativeConfig(
        strategy="error_aware",
        max_iterations=3,
        candidates_per_iteration=2,
        temperature_schedule=[0.0, 0.0, 0.0],
        top_k=3,
        rotate_on_stagnation=True,
    )

    prom_row = {"function": {"function_before": "public int foo() {\n    return 0;\n}"}}
    result = run_iterative_repair(
        backend=backend,
        bug_id="stub-1",
        original_prompt="ORIGINAL",
        prom_row=prom_row,
        evaluate_fn=fake_evaluate,
        config=config,
        language="java",
        base_code="public int foo() {\n    return 0;\n}",
        expected_name="foo",
        repair_branch="java_v2",
    )

    # Feedback was called for iter 1 (seed = A, iter0's best) and iter 2 (rotated).
    assert len(seen_previous) == 2
    # Iter 1's feedback uses A (normal best so far).
    assert "return 1" in seen_previous[0]
    # Iter 2's feedback must NOT re-use A (stagnated) — should have rotated to B.
    assert "return 1" not in seen_previous[1]
    assert "return 2" in seen_previous[1]
    assert result.total_iterations == 3


def test_rotate_on_stagnation_disabled_keeps_same_seed(monkeypatch):
    A = _java_block("return 1;")
    B = _java_block("return 2;")
    backend = _StubBackend([[A, B], [A], [A]])

    def fake_evaluate(code):
        n = 5 if "return 1" in code else 8
        return {"fail_reason": "test_fail", "failing_tests_count": n,
                "compile_error_family": "", "first_failure_message": ""}

    seen_previous = []
    import core.iterative_repair as ir
    real_build = ir.build_feedback_prompt

    def spy_build(*, original_prompt, previous_code, **kwargs):
        seen_previous.append(previous_code)
        return real_build(original_prompt=original_prompt, previous_code=previous_code, **kwargs)

    monkeypatch.setattr(ir, "build_feedback_prompt", spy_build)

    config = IterativeConfig(
        strategy="error_aware",
        max_iterations=3,
        candidates_per_iteration=2,
        temperature_schedule=[0.0, 0.0, 0.0],
        top_k=3,
        rotate_on_stagnation=False,  # disabled
    )

    prom_row = {"function": {"function_before": "public int foo() {\n    return 0;\n}"}}
    run_iterative_repair(
        backend=backend,
        bug_id="stub-2",
        original_prompt="ORIGINAL",
        prom_row=prom_row,
        evaluate_fn=fake_evaluate,
        config=config,
        language="java",
        base_code="public int foo() {\n    return 0;\n}",
        expected_name="foo",
        repair_branch="java_v2",
    )

    # With rotation disabled, every feedback call should re-feed A (stagnated #1).
    assert all("return 1" in prev for prev in seen_previous)
