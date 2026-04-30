"""
Iterative Repair Loop Controller.

Core algorithm: generate → evaluate → classify error → build feedback → retry.
Supports three strategies: one_shot, blind_retry, error_aware.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .error_classifier import classify_and_select_strategy, failure_priority, FeedbackStrategy
from .feedback_prompt import build_feedback_prompt
from .patch_generator import generate_candidates

logger = logging.getLogger(__name__)
_COMPARISON_FAILURE_RE = re.compile(r"expected:<(?P<expected>.*?)> but was:<(?P<actual>.*?)>")
_PROMPT_CODE_BLOCK_RE = re.compile(
    r"(CODE \(buggy line is marked with `<--- BUGGY LINE`\):\n```[A-Za-z0-9_+-]*\n)(.*?)(\n```)",
    re.S,
)
_PROMPT_LOCAL_CONTEXT_RE = re.compile(
    r"(LOCAL CONTEXT:\n```text\n)(.*?)(\n```)",
    re.S,
)
_PROMPT_BOUNDED_REGION_RE = re.compile(
    r"(BOUNDED EDIT REGION:\n- Prefer edits at/near lines: )(.*?)(\n- If you must edit outside, keep within the same function and remain minimal\.)"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IterativeConfig:
    strategy: str = "error_aware"          # one_shot | blind_retry | error_aware
    max_iterations: int = 3
    candidates_per_iteration: int = 5
    temperature_schedule: List[float] = field(default_factory=lambda: [0.0, 0.4, 0.8])
    top_p: float = 0.95
    max_new_tokens: int = 512
    seed: int = 42
    test_timeout: int = 300
    # Top-K candidate memory: keep the K best failing candidates across
    # iterations (ranked by failure_priority, failing_tests_count, similarity).
    # When the current iteration reproduces the previous iteration's best code
    # (stagnation), the feedback loop rotates to the next-best distinct seed
    # from this memory instead of re-feeding the same stuck code.
    top_k: int = 3
    rotate_on_stagnation: bool = True
    # Ablation: strip test-failure metadata from feedback (TA-noMeta condition).
    # Only affects test_feedback strategy; compile/timeout feedback unchanged.
    ablation_no_meta: bool = False


@dataclass
class IterationRecord:
    iteration: int
    num_candidates_generated: int
    num_valid_candidates: int
    best_candidate_code: str
    best_candidate_hash: str
    fail_reason: str
    error_family: str
    compile_stderr: str
    feedback_strategy: str
    elapsed_sec: float
    failing_tests_count: Optional[int] = None
    first_failing_test: str = ""


@dataclass
class IterativeRepairResult:
    bug_id: str
    solved: bool
    solving_iteration: Optional[int]
    total_iterations: int
    total_candidates_generated: int
    total_llm_calls: int
    iterations: List[IterationRecord]
    final_code: Optional[str]
    final_fail_reason: str
    elapsed_total_sec: float
    experiment_condition: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _code_hash(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:12]


def _get_temperature(schedule: List[float], iteration: int) -> float:
    """Get temperature for given iteration (0-indexed internally)."""
    idx = min(iteration, len(schedule) - 1)
    return schedule[idx]


def _normalize_failing_tests_count(eval_result: Optional[Dict[str, Any]]) -> int:
    value = (eval_result or {}).get("failing_tests_count")
    if value is None:
        return 10 ** 9
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 10 ** 9


def _comparison_failure_similarity(eval_result: Optional[Dict[str, Any]]) -> float:
    message = str((eval_result or {}).get("first_failure_message", "") or "")
    match = _COMPARISON_FAILURE_RE.search(message)
    if not match:
        return -1.0

    expected = match.group("expected").replace("[", "").replace("]", "")
    actual = match.group("actual").replace("[", "").replace("]", "")
    return SequenceMatcher(None, expected, actual).ratio()


def _ranking_tuple(eval_result: Optional[Dict[str, Any]]) -> tuple:
    """Tuple used to rank failing candidates. Higher = better feedback signal."""
    if eval_result is None:
        return (-1, -(10 ** 9), -1.0)
    return (
        failure_priority(eval_result),
        -_normalize_failing_tests_count(eval_result),
        _comparison_failure_similarity(eval_result),
    )


def _is_better_feedback_candidate(
    current_eval: Dict[str, Any],
    best_eval: Optional[Dict[str, Any]],
) -> bool:
    if best_eval is None:
        return True
    return _ranking_tuple(current_eval) > _ranking_tuple(best_eval)


def _update_top_k(
    memory: List[Dict[str, Any]],
    code: str,
    eval_result: Dict[str, Any],
    iteration: int,
    k: int,
) -> List[Dict[str, Any]]:
    """Insert entry into sorted top-K memory (dedup by code, cap at k)."""
    rank = _ranking_tuple(eval_result)
    filtered = [m for m in memory if m["code"] != code]
    filtered.append({"rank": rank, "code": code, "eval_result": eval_result, "iteration": iteration})
    filtered.sort(key=lambda m: m["rank"], reverse=True)
    return filtered[: max(1, int(k))]


def _strip_inline_comment(line: str) -> str:
    return re.sub(r"\s*//.*$", "", str(line or "")).rstrip()


def _find_buggy_line_index(base_code: str, buggy_line_content: str) -> Optional[int]:
    lines = str(base_code or "").splitlines()
    target = str(buggy_line_content or "").strip()
    if not lines or not target:
        return None

    for idx, line in enumerate(lines, 1):
        if line.strip() == target:
            return idx

    target_no_comment = _strip_inline_comment(target)
    for idx, line in enumerate(lines, 1):
        if _strip_inline_comment(line).strip() == target_no_comment:
            return idx

    if target_no_comment:
        for idx, line in enumerate(lines, 1):
            if target_no_comment in _strip_inline_comment(line):
                return idx

    return None


def _render_local_context(base_code: str, buggy_line_index: int, radius: int = 2) -> str:
    lines = str(base_code or "").splitlines()
    if not lines:
        return ""

    start = max(1, buggy_line_index - radius)
    end = min(len(lines), buggy_line_index + radius)
    rendered: List[str] = []

    if start > 1:
        rendered.append("... (truncated)")

    for idx in range(start, end + 1):
        marker = "->" if idx == buggy_line_index else "  "
        rendered.append(f"{idx:>4}:{marker}{lines[idx - 1]}")

    if end < len(lines):
        rendered.append("... (truncated)")

    return "\n".join(rendered)


def _render_code_block(base_code: str, buggy_line_index: int, language: str) -> str:
    lines = str(base_code or "").splitlines()
    if not lines:
        return ""

    marker = "// <--- BUGGY LINE" if str(language).lower() == "java" else "# <--- BUGGY LINE"
    rendered: List[str] = []

    for idx, line in enumerate(lines, 1):
        if idx == buggy_line_index:
            rendered.append(f"{line}  {marker}")
        else:
            rendered.append(line)

    return "\n".join(rendered)


def _render_bounded_edit_region(total_lines: int, buggy_line_index: int, radius: int = 1) -> str:
    if total_lines <= 0:
        return "1"
    start = max(1, buggy_line_index - radius)
    end = min(total_lines, buggy_line_index + radius)
    return ", ".join(str(i) for i in range(start, end + 1))


def _prepare_original_prompt(original_prompt: str, prom_row: Dict[str, Any]) -> str:
    prompt = str(original_prompt or "")
    if not prompt:
        return prompt

    plan_json = prom_row.get("plan_json")
    if "{{PLAN_JSON}}" in prompt:
        prompt = prompt.replace(
            "{{PLAN_JSON}}",
            json.dumps(plan_json or {}, ensure_ascii=False, indent=2),
        )

    base_code = prom_row.get("function", {}).get("function_before", "")
    buggy_line_content = prom_row.get("buggy_line_content", "")
    buggy_line_index = _find_buggy_line_index(base_code, buggy_line_content)
    if not base_code or buggy_line_index is None:
        return prompt

    total_lines = len(base_code.splitlines())
    language = str(prom_row.get("language", "java") or "java")
    local_context = _render_local_context(base_code, buggy_line_index)
    code_block = _render_code_block(base_code, buggy_line_index, language)
    bounded_region = _render_bounded_edit_region(total_lines, buggy_line_index)

    prompt = _PROMPT_BOUNDED_REGION_RE.sub(
        lambda m: f"{m.group(1)}{bounded_region}{m.group(3)}",
        prompt,
        count=1,
    )
    prompt = _PROMPT_LOCAL_CONTEXT_RE.sub(
        lambda m: f"{m.group(1)}{local_context}{m.group(3)}",
        prompt,
        count=1,
    )
    prompt = _PROMPT_CODE_BLOCK_RE.sub(
        lambda m: f"{m.group(1)}{code_block}{m.group(3)}",
        prompt,
        count=1,
    )
    return prompt


# ---------------------------------------------------------------------------
# Main iterative repair loop
# ---------------------------------------------------------------------------

def run_iterative_repair(
    backend,
    bug_id: str,
    original_prompt: str,
    prom_row: Dict[str, Any],
    evaluate_fn,
    config: IterativeConfig,
    *,
    language: str = "java",
    base_code: str = "",
    expected_name: str = "",
    repair_branch: str = "java_base",
    stop: Optional[List[str]] = None,
    prompt_log_dir: Optional[str] = None,
) -> IterativeRepairResult:
    """
    Run iterative repair for a single bug.

    Args:
        backend: LLMBackend instance
        bug_id: Bug identifier
        original_prompt: Repair prompt from Stage 3/4
        prom_row: Prompt row with enriched context
        evaluate_fn: Callable(candidate_code) -> eval_result dict
        config: IterativeConfig
        language: Programming language
        base_code: Original buggy code
        expected_name: Expected function/method name
        repair_branch: Repair branch type
        stop: Stop sequences for LLM

    Returns:
        IterativeRepairResult
    """
    started_at = time.time()
    original_prompt = _prepare_original_prompt(original_prompt, prom_row)
    iteration_records: List[IterationRecord] = []
    total_candidates = 0
    total_llm_calls = 0
    all_seen_codes = set()

    prev_best_code = None
    prev_eval_result = None
    # Top-K failing-candidate memory across iterations — enables rotating the
    # feedback seed when the LLM gets stuck regenerating the same "best" code.
    top_k_memory: List[Dict[str, Any]] = []

    max_iters = 1 if config.strategy == "one_shot" else config.max_iterations

    for iteration in range(max_iters):
        iter_start = time.time()

        # Build prompt
        if iteration == 0 or prev_best_code is None:
            current_prompt = original_prompt
        else:
            strategy = classify_and_select_strategy(prev_eval_result)
            blind = (config.strategy == "blind_retry")
            current_prompt = build_feedback_prompt(
                original_prompt=original_prompt,
                previous_code=prev_best_code,
                strategy=strategy,
                eval_result=prev_eval_result,
                prom_row=prom_row,
                iteration=iteration + 1,
                blind=blind,
                ablation_no_meta=config.ablation_no_meta,
            )

        temperature = _get_temperature(config.temperature_schedule, iteration)
        seed = config.seed + iteration * 1000

        # Generate candidates
        candidates = generate_candidates(
            backend,
            current_prompt,
            temperature=temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            seed=seed,
            n=config.candidates_per_iteration,
            language=language,
            base_code=base_code,
            expected_name=expected_name,
            repair_branch=repair_branch,
            stop=stop,
        )

        total_llm_calls += 1

        # Dump prompt and raw responses for post-hoc analysis
        if prompt_log_dir:
            import os as _os, json as _json
            _iter_log = _os.path.join(prompt_log_dir, bug_id)
            _os.makedirs(_iter_log, exist_ok=True)
            try:
                with open(_os.path.join(_iter_log, f"iter{iteration + 1}_prompt.txt"), "w", encoding="utf-8") as _f:
                    _f.write(current_prompt)
                with open(_os.path.join(_iter_log, f"iter{iteration + 1}_responses.json"), "w", encoding="utf-8") as _f:
                    _json.dump(
                        [{"candidate_idx": i, "text": c.get("raw_output", "")} for i, c in enumerate(candidates)],
                        _f, ensure_ascii=False, indent=2,
                    )
            except Exception as _e:
                logger.warning("[%s] iter %d: prompt dump failed: %s", bug_id, iteration + 1, _e)

        empty_count = sum(1 for c in candidates if not c["code"])
        base_eq_count = sum(1 for c in candidates if c["code"] and c["code"] == base_code)
        cross_iter_dup_count = sum(
            1 for c in candidates
            if c["code"] and c["code"] != base_code and c["code"] in all_seen_codes
        )
        valid_candidates = [
            c for c in candidates
            if c["code"]
            and c["code"] != base_code
            and c["code"] not in all_seen_codes
        ]
        total_candidates += len(candidates)

        for c in valid_candidates:
            all_seen_codes.add(c["code"])

        if iteration > 0 and cross_iter_dup_count > 0:
            logger.info(
                "[%s] iter %d: dropped %d cross-iter duplicate candidate(s) "
                "(LLM regenerating previously-seen code despite feedback)",
                bug_id, iteration + 1, cross_iter_dup_count,
            )

        if not valid_candidates:
            record = IterationRecord(
                iteration=iteration + 1,
                num_candidates_generated=len(candidates),
                num_valid_candidates=0,
                best_candidate_code="",
                best_candidate_hash="",
                fail_reason="no_valid_candidates",
                error_family="",
                compile_stderr="",
                failing_tests_count=None,
                first_failing_test="",
                feedback_strategy="",
                elapsed_sec=round(time.time() - iter_start, 3),
            )
            iteration_records.append(record)
            logger.info(
                "[%s] iter %d: no valid candidates "
                "(empty=%d, base_eq=%d, cross_iter_dup=%d, total=%d)",
                bug_id, iteration + 1,
                empty_count, base_eq_count, cross_iter_dup_count, len(candidates),
            )
            # Dedup stagnation: every candidate was a cross-iter duplicate,
            # meaning the LLM regenerated prior codes despite feedback. Rotate
            # the feedback seed to a different entry from top-K so the next
            # iteration's prompt has a chance to change the output.
            if (
                config.rotate_on_stagnation
                and cross_iter_dup_count == len(candidates)
                and cross_iter_dup_count > 0
                and prev_best_code is not None
                and len(top_k_memory) >= 2
            ):
                alt = next(
                    (m for m in top_k_memory if m["code"] != prev_best_code),
                    None,
                )
                if alt is not None:
                    logger.info(
                        "[%s] iter %d: dedup stagnation — rotating feedback "
                        "seed to top_k candidate from iter %d",
                        bug_id, iteration + 1, alt["iteration"],
                    )
                    prev_best_code = alt["code"]
                    prev_eval_result = alt["eval_result"]
            continue

        # Evaluate candidates, track the best one
        best_code = None
        best_eval = None
        best_priority = -1

        for c in valid_candidates:
            eval_result = evaluate_fn(c["code"])
            fail_reason = eval_result.get("fail_reason", "unknown_error")

            if fail_reason == "pass":
                # SOLVED
                record = IterationRecord(
                    iteration=iteration + 1,
                    num_candidates_generated=len(candidates),
                    num_valid_candidates=len(valid_candidates),
                    best_candidate_code=c["code"],
                    best_candidate_hash=_code_hash(c["code"]),
                    fail_reason="pass",
                    error_family="",
                    compile_stderr="",
                    failing_tests_count=eval_result.get("failing_tests_count"),
                    first_failing_test=str(eval_result.get("first_failing_test", "")),
                    feedback_strategy="",
                    elapsed_sec=round(time.time() - iter_start, 3),
                )
                iteration_records.append(record)

                return IterativeRepairResult(
                    bug_id=bug_id,
                    solved=True,
                    solving_iteration=iteration + 1,
                    total_iterations=iteration + 1,
                    total_candidates_generated=total_candidates,
                    total_llm_calls=total_llm_calls,
                    iterations=iteration_records,
                    final_code=c["code"],
                    final_fail_reason="pass",
                    elapsed_total_sec=round(time.time() - started_at, 3),
                    experiment_condition="test_aware_no_meta" if config.ablation_no_meta else "",
                )

            # Track the best failing candidate
            if _is_better_feedback_candidate(eval_result, best_eval):
                best_priority = failure_priority(eval_result)
                best_code = c["code"]
                best_eval = eval_result

            # Also keep EVERY evaluated candidate in top-K memory so the
            # stagnation rotation has diverse seeds to fall back to.
            top_k_memory = _update_top_k(
                top_k_memory, c["code"], eval_result, iteration + 1, config.top_k,
            )

        # Record the best (non-passing) attempt
        strategy_name = ""
        if iteration > 0 and prev_eval_result is not None:
            try:
                s = classify_and_select_strategy(prev_eval_result)
                strategy_name = s.strategy_name
            except Exception as exc:
                logger.warning(
                    "[%s] iter %d: strategy classification raised %s — "
                    "fail_reason=%r, family=%r",
                    bug_id, iteration + 1, type(exc).__name__,
                    prev_eval_result.get("fail_reason"),
                    prev_eval_result.get("compile_error_family"),
                )

        record = IterationRecord(
            iteration=iteration + 1,
            num_candidates_generated=len(candidates),
            num_valid_candidates=len(valid_candidates),
            best_candidate_code=best_code or "",
            best_candidate_hash=_code_hash(best_code) if best_code else "",
            fail_reason=best_eval.get("fail_reason", "unknown_error") if best_eval else "unknown_error",
            error_family=best_eval.get("compile_error_family", "") if best_eval else "",
            compile_stderr=str(best_eval.get("compile_stderr", ""))[:500] if best_eval else "",
            failing_tests_count=best_eval.get("failing_tests_count") if best_eval else None,
            first_failing_test=str(best_eval.get("first_failing_test", "")) if best_eval else "",
            feedback_strategy=strategy_name,
            elapsed_sec=round(time.time() - iter_start, 3),
        )
        iteration_records.append(record)

        # Stagnation detection: if this iter's best code hash matches the
        # previous iter's best code hash, the feedback is not moving the LLM.
        # Rotate the feedback seed to the next distinct candidate from top-K.
        stagnated = (
            config.rotate_on_stagnation
            and len(iteration_records) >= 2
            and record.best_candidate_hash
            and record.best_candidate_hash == iteration_records[-2].best_candidate_hash
            and len(top_k_memory) >= 2
        )
        if stagnated:
            alt = next(
                (m for m in top_k_memory if m["code"] != best_code),
                None,
            )
            if alt is not None:
                logger.info(
                    "[%s] iter %d: stagnation — rotating feedback seed to "
                    "top_k candidate from iter %d (rank=%s)",
                    bug_id, iteration + 1, alt["iteration"], alt["rank"],
                )
                prev_best_code = alt["code"]
                prev_eval_result = alt["eval_result"]
            else:
                prev_best_code = best_code
                prev_eval_result = best_eval
        else:
            prev_best_code = best_code
            prev_eval_result = best_eval

        logger.info(
            "[%s] iter %d: %d valid, best=%s, error=%s",
            bug_id, iteration + 1, len(valid_candidates),
            record.fail_reason, record.error_family,
        )

    # Not solved
    final_fail = "unknown_error"
    if iteration_records:
        final_fail = iteration_records[-1].fail_reason

    return IterativeRepairResult(
        bug_id=bug_id,
        solved=False,
        solving_iteration=None,
        total_iterations=len(iteration_records),
        total_candidates_generated=total_candidates,
        total_llm_calls=total_llm_calls,
        iterations=iteration_records,
        final_code=prev_best_code,
        final_fail_reason=final_fail,
        elapsed_total_sec=round(time.time() - started_at, 3),
        experiment_condition="test_aware_no_meta" if config.ablation_no_meta else "",
    )
