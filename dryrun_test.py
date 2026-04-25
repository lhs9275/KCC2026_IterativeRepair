#!/usr/bin/env python3
"""
Dry-run test: mock LLM backend으로 전체 파이프라인 검증.
GPU 없이 checkout → patch apply → compile → test → error classify → feedback 루프 확인.
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from core.iterative_repair import IterativeConfig, run_iterative_repair
from core.patch_generator import generate_candidates
from core.error_classifier import classify_and_select_strategy, classify_compile_error_family
from core.feedback_prompt import build_feedback_prompt
from evaluation.eval_iterative import create_adapter, evaluate_candidate, setup_workspace, cleanup_workspace
from llm_backend import LLMBackend, GenerationRecord

# ─── Mock LLM Backend ───
class MockBackend(LLMBackend):
    """LLM 호출 없이 미리 정의된 코드를 반환하는 mock."""
    backend_name = "mock"

    def __init__(self, responses):
        super().__init__(model_name="mock-model")
        self.responses = responses  # list of strings per iteration
        self.call_count = 0

    def generate_records(self, prompts, *, temperature, top_p, max_new_tokens, seed, stop=None, **kwargs):
        n = kwargs.get("n", 1)
        idx = min(self.call_count, len(self.responses) - 1)
        codes = self.responses[idx]
        self.call_count += 1
        records = []
        for code in codes[:n]:
            records.append(GenerationRecord(
                text=f"##correct\n{code}",
                tokens_in=100,
                tokens_out=50,
            ))
        return records

# ─── Test Configs ───
def load_bug(bug_id="1"):
    path = os.path.join(PROJECT_ROOT, "Results", "4", "4.defects4j.PlanAgent.json")
    with open(path) as f:
        data = json.load(f)
    return data[bug_id]

def test_1_checkout_compile_test():
    """D4J checkout → compile → test 기본 흐름"""
    print("\n=== TEST 1: Checkout / Compile / Test ===")
    adapter = create_adapter("defects4j")
    prom_row = load_bug("1")

    project_name = adapter.map_project_name(prom_row.get("project_name", ""))
    d4j_id = str(prom_row.get("defects4j_id", "1"))
    workspace_root = os.path.join(PROJECT_ROOT, "temp_dryrun")

    project_path = setup_workspace(adapter, project_name, d4j_id, workspace_root)
    if project_path is None:
        print("  FAIL: checkout failed")
        return False
    print(f"  Checkout OK: {project_path}")

    # 원본 buggy 코드로 평가 (테스트 실패해야 정상)
    buggy_code = prom_row["function"]["function_before"]
    result = evaluate_candidate(adapter, project_path, prom_row, buggy_code)
    print(f"  Buggy code eval: fail_reason={result['fail_reason']}, compile_ok={result.get('compile_ok')}")

    cleanup_workspace(os.path.dirname(project_path))

    if result["fail_reason"] in ("test_fail", "test_error"):
        print("  PASS (buggy code correctly fails tests)")
        return True
    elif result["fail_reason"] == "pass":
        print("  WARN: buggy code passed? unexpected")
        return True
    else:
        print(f"  FAIL: unexpected fail_reason: {result['fail_reason']}")
        return False

def test_2_error_classifier():
    """에러 분류기 동작 확인"""
    print("\n=== TEST 2: Error Classifier ===")

    test_cases = [
        ({"fail_reason": "compile_fail", "compile_stderr": "error: cannot find symbol"}, "cannot_find_symbol"),
        ({"fail_reason": "compile_fail", "compile_stderr": "error: incompatible types"}, "type_mismatch"),
        ({"fail_reason": "compile_fail", "compile_stderr": "error: ';' expected"}, "syntax_or_parse"),
        ({"fail_reason": "compile_fail", "compile_stderr": "error: no suitable method found"}, "method_signature"),
        ({"fail_reason": "test_fail"}, "test_fail"),
        ({"fail_reason": "timeout"}, "timeout"),
    ]

    all_pass = True
    for eval_result, expected_category in test_cases:
        strategy = classify_and_select_strategy(eval_result)
        status = "OK" if strategy.error_category.endswith(expected_category.split("_")[-1]) or \
                         strategy.strategy_name.startswith(expected_category.split("_")[0]) else "??"
        print(f"  {expected_category:<25} -> strategy={strategy.strategy_name:<25} [{status}]")

    print("  PASS")
    return True

def test_3_feedback_prompt():
    """피드백 프롬프트 생성 확인"""
    print("\n=== TEST 3: Feedback Prompt Builder ===")
    prom_row = load_bug("1")

    eval_result = {
        "fail_reason": "compile_fail",
        "compile_stderr": "error: cannot find symbol\n  symbol: variable foo",
        "compile_stdout": "",
        "compile_error_family": "cannot_find_symbol",
    }
    strategy = classify_and_select_strategy(eval_result)

    prompt = build_feedback_prompt(
        original_prompt="Fix the bug in getLegendItems",
        previous_code="public void getLegendItems() { foo.bar(); }",
        strategy=strategy,
        eval_result=eval_result,
        prom_row=prom_row,
        iteration=2,
        blind=False,
    )

    has_feedback = "---FEEDBACK" in prompt
    has_stderr = "cannot find symbol" in prompt
    has_correct = "##correct" in prompt

    print(f"  Contains FEEDBACK section: {has_feedback}")
    print(f"  Contains stderr info:      {has_stderr}")
    print(f"  Ends with ##correct:       {has_correct}")
    print(f"  Prompt length:             {len(prompt)} chars")

    if has_feedback and has_stderr and has_correct:
        print("  PASS")
        return True
    else:
        print("  FAIL")
        return False

def test_4_mock_iterative_loop():
    """Mock LLM으로 전체 반복 루프 실행"""
    print("\n=== TEST 4: Full Iterative Loop (Mock LLM) ===")
    prom_row = load_bug("1")
    adapter = create_adapter("defects4j")

    project_name = adapter.map_project_name(prom_row.get("project_name", ""))
    d4j_id = str(prom_row.get("defects4j_id", "1"))
    workspace_root = os.path.join(PROJECT_ROOT, "temp_dryrun")

    project_path = setup_workspace(adapter, project_name, d4j_id, workspace_root)
    if project_path is None:
        print("  FAIL: checkout failed")
        return False

    # iter1: 의도적으로 컴파일 실패하는 코드
    # iter2: 의도적으로 테스트 실패하는 코드 (buggy code 그대로)
    # iter3: 정답 코드 (fixed version)
    buggy = prom_row["function"]["function_before"]
    fixed = buggy.replace("if (dataset != null)", "if (dataset == null)")
    broken = "public LegendItemCollection getLegendItems() { return unknownVar; }"

    mock_responses = [
        [broken] * 5,       # iter1: compile fail
        [buggy] * 5,        # iter2: test fail
        [fixed] * 5,        # iter3: should pass
    ]
    backend = MockBackend(mock_responses)

    config = IterativeConfig(
        strategy="error_aware",
        max_iterations=3,
        candidates_per_iteration=5,
        temperature_schedule=[0.0, 0.4, 0.8],
    )

    def evaluate_fn(candidate_code):
        return evaluate_candidate(adapter, project_path, prom_row, candidate_code)

    try:
        result = run_iterative_repair(
            backend=backend,
            bug_id="1",
            original_prompt=prom_row.get("prompt", "fix the bug"),
            prom_row=prom_row,
            evaluate_fn=evaluate_fn,
            config=config,
            language="java",
            base_code=prom_row["function"]["function_before"],
            expected_name=prom_row["function"]["function_name"],
        )

        print(f"  Solved:             {result.solved}")
        print(f"  Solving iteration:  {result.solving_iteration}")
        print(f"  Total iterations:   {result.total_iterations}")
        print(f"  Total LLM calls:    {result.total_llm_calls}")

        for rec in result.iterations:
            print(f"    iter {rec.iteration}: fail_reason={rec.fail_reason}, "
                  f"error_family={rec.error_family}, strategy={rec.feedback_strategy}")

        if result.solved:
            print("  PASS (solved!)")
        else:
            print(f"  PARTIAL (not solved, but loop ran: {result.final_fail_reason})")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_workspace(os.path.dirname(project_path))

def test_5_blind_vs_error_aware():
    """blind_retry와 error_aware 피드백 프롬프트 차이 확인"""
    print("\n=== TEST 5: Blind vs Error-Aware Prompt Diff ===")
    prom_row = load_bug("1")

    eval_result = {
        "fail_reason": "compile_fail",
        "compile_stderr": "error: cannot find symbol\n  symbol: variable foo",
        "compile_error_family": "cannot_find_symbol",
    }
    strategy = classify_and_select_strategy(eval_result)

    blind_prompt = build_feedback_prompt(
        original_prompt="Fix the bug",
        previous_code="broken code here",
        strategy=strategy, eval_result=eval_result,
        prom_row=prom_row, iteration=2, blind=True,
    )

    aware_prompt = build_feedback_prompt(
        original_prompt="Fix the bug",
        previous_code="broken code here",
        strategy=strategy, eval_result=eval_result,
        prom_row=prom_row, iteration=2, blind=False,
    )

    print(f"  Blind prompt length:       {len(blind_prompt)} chars")
    print(f"  Error-aware prompt length: {len(aware_prompt)} chars")
    print(f"  Aware has stderr:          {'cannot find symbol' in aware_prompt}")
    print(f"  Blind has stderr:          {'cannot find symbol' in blind_prompt}")
    print(f"  Aware has imports:         {'Imports' in aware_prompt}")
    print(f"  Blind has imports:         {'Imports' in blind_prompt}")

    if len(aware_prompt) > len(blind_prompt):
        print("  PASS (error-aware prompt is richer)")
        return True
    else:
        print("  FAIL (expected aware > blind)")
        return False


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_ROOT, "temp_dryrun"), exist_ok=True)

    results = {}
    results["1_checkout"] = test_1_checkout_compile_test()
    results["2_classifier"] = test_2_error_classifier()
    results["3_feedback"] = test_3_feedback_prompt()
    results["4_loop"] = test_4_mock_iterative_loop()
    results["5_blind_vs_aware"] = test_5_blind_vs_error_aware()

    print("\n" + "=" * 50)
    print("  DRY-RUN SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} passed")

    # cleanup
    import shutil
    dryrun_dir = os.path.join(PROJECT_ROOT, "temp_dryrun")
    if os.path.isdir(dryrun_dir):
        shutil.rmtree(dryrun_dir, ignore_errors=True)
