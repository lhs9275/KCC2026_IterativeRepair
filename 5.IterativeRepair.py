#!/usr/bin/env python3
"""
Main entry point for iterative repair experiments.

Usage:
    python run_iterative.py --config config/ablation_error_aware.yaml
    python run_iterative.py --strategy error_aware --dataset defects4j
    python run_iterative.py --strategy one_shot --bug_id_list 1,2,3
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import yaml
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.iterative_repair import IterativeConfig, run_iterative_repair, IterativeRepairResult
from core.patch_generator import generate_candidates
from evaluation.eval_iterative import (
    create_adapter,
    evaluate_candidate,
    setup_workspace,
    cleanup_workspace,
)
from llm_backend import create_backend

logger = logging.getLogger("iterative_repair")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(args) -> Dict[str, Any]:
    """Load config from YAML file, then override with CLI args."""
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    for key in ["strategy", "max_iterations", "candidates_per_iteration",
                 "model_name", "backend", "dataset", "test_timeout",
                 "max_model_len", "gpu_memory_utilization"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    return cfg


def build_iterative_config(cfg: Dict[str, Any]) -> IterativeConfig:
    return IterativeConfig(
        strategy=cfg.get("strategy", "error_aware"),
        max_iterations=int(cfg.get("max_iterations", 3)),
        candidates_per_iteration=int(cfg.get("candidates_per_iteration", 5)),
        temperature_schedule=cfg.get("temperature_schedule", [0.0, 0.4, 0.8]),
        top_p=float(cfg.get("top_p", 0.95)),
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        seed=int(cfg.get("seed", 42)),
        test_timeout=int(cfg.get("test_timeout", 300)),
        ablation_no_meta=bool(cfg.get("ablation_no_meta", False)),
    )


# ---------------------------------------------------------------------------
# Single bug repair worker
# ---------------------------------------------------------------------------

def repair_one_bug(
    bug_id: str,
    prom_row: Dict[str, Any],
    cfg: Dict[str, Any],
    iter_config: IterativeConfig,
    backend,
    workspace_root: str,
    prompt_log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Repair a single bug with iterative loop."""
    dataset = cfg.get("dataset", "defects4j")
    adapter = create_adapter(dataset)

    # Extract bug metadata — map repo name (e.g. "jfreechart") to D4J name ("Chart")
    raw_project_name = prom_row.get("project_name", "")
    project_name = adapter.map_project_name(raw_project_name) or raw_project_name
    if dataset == "defects4j":
        d4j_id = str(prom_row.get("defects4j_id", bug_id))
    else:
        d4j_id = str(bug_id)

    # Checkout workspace
    project_path = setup_workspace(adapter, project_name, d4j_id, workspace_root)
    if project_path is None:
        return {
            "bug_id": bug_id,
            "status": "infra_fail",
            "error": f"Checkout failed for {project_name}-{d4j_id}",
        }

    workspace_dir = os.path.dirname(project_path)

    try:
        # Prepare evaluate function (closure over adapter, project_path, etc.)
        def evaluate_fn(candidate_code: str) -> Dict[str, Any]:
            return evaluate_candidate(
                adapter=adapter,
                project_path=project_path,
                bug_meta_data=prom_row,
                candidate_code=candidate_code,
                test_timeout=iter_config.test_timeout,
            )

        # Extract prompt and metadata
        original_prompt = prom_row.get("prompt", "")
        language = prom_row.get("language", "java")
        base_code = prom_row.get("function", {}).get("function_before", "")
        expected_name = prom_row.get("function", {}).get("function_name", "")
        # Override: force java_v2 for Java bugs so the extractor's body-only
        # sentinel block mode is enabled (small LLMs often emit body-only).
        repair_branch = prom_row.get("repair_branch", "java_base")
        if language == "java":
            repair_branch = "java_v2"

        # Run iterative repair
        result = run_iterative_repair(
            backend=backend,
            bug_id=bug_id,
            original_prompt=original_prompt,
            prom_row=prom_row,
            evaluate_fn=evaluate_fn,
            config=iter_config,
            language=language,
            base_code=base_code,
            expected_name=expected_name,
            repair_branch=repair_branch,
            prompt_log_dir=prompt_log_dir,
        )

        return asdict(result)

    except Exception as e:
        logger.error("Error repairing %s: %s", bug_id, e, exc_info=True)
        return {
            "bug_id": bug_id,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        cleanup_workspace(workspace_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iterative Program Repair")
    parser.add_argument("--config", type=str, default="", help="YAML config file")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["one_shot", "blind_retry", "error_aware"])
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--candidates_per_iteration", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["defects4j", "bugsinpy"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["auto", "vllm", "hf"])
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--test_timeout", type=int, default=None)
    parser.add_argument("--prompts_json", type=str, default="",
                        help="Path to Stage 4 prompts JSON (default: auto-detect from Results/4/)")
    parser.add_argument("--meta_json", type=str, default="",
                        help="Path to bugs metadata JSON")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path (default: auto-generated)")
    parser.add_argument("--workspace_root", type=str, default="temp_workspaces")
    parser.add_argument("--bug_id_list", type=str, default="",
                        help="Comma-separated bug IDs to process (empty=all)")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of parallel workers (default=1, sequential)")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--ablation_no_meta", action="store_true",
                        help="Ablation: TA feedback without test-failure metadata (TA-noMeta).")
    parser.add_argument("--bug_subset", type=str, default="",
                        help="Path to JSON file listing bug IDs to run (overrides --bug_id_list).")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Seed for shuffling bug order when using --bug_subset.")
    parser.add_argument("--time_budget_sec", type=int, default=None,
                        help="Hard time budget in seconds; loop exits gracefully when exceeded.")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PROJECT_ROOT, "logs",
                             f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ),
        ],
    )

    # Load config
    cfg = load_config(args)
    if args.ablation_no_meta:
        cfg["ablation_no_meta"] = True
    iter_config = build_iterative_config(cfg)
    dataset = cfg.get("dataset", "defects4j")
    strategy = cfg.get("strategy", "error_aware")

    logger.info("Config: strategy=%s, max_iter=%d, candidates=%d, dataset=%s",
                strategy, iter_config.max_iterations,
                iter_config.candidates_per_iteration, dataset)

    # Load prompts JSON (Stage 4 results)
    prompts_json = args.prompts_json
    if not prompts_json:
        prompts_json = os.path.join(
            PROJECT_ROOT, "Results", "4",
            f"4.{dataset}.PlanAgent.json"
        )
    if not os.path.isfile(prompts_json):
        logger.error("Prompts JSON not found: %s", prompts_json)
        sys.exit(1)

    with open(prompts_json) as f:
        all_prompts = json.load(f)
    logger.info("Loaded %d bugs from %s", len(all_prompts), prompts_json)

    # Filter bug IDs: --bug_subset (JSON file) takes priority over --bug_id_list
    if args.bug_subset and os.path.isfile(args.bug_subset):
        import json as _json_sub, random as _random_sub
        with open(args.bug_subset) as _f:
            subset_ids = set(str(x) for x in _json_sub.load(_f))
        all_prompts = {k: v for k, v in all_prompts.items() if k in subset_ids}
        _random_sub.seed(args.shuffle_seed)
        _items = list(all_prompts.items())
        _random_sub.shuffle(_items)
        all_prompts = dict(_items)
        logger.info("Filtered to %d bugs from subset file %s (shuffle_seed=%d)",
                    len(all_prompts), args.bug_subset, args.shuffle_seed)
    elif args.bug_id_list:
        selected_ids = set(args.bug_id_list.split(","))
        all_prompts = {k: v for k, v in all_prompts.items() if k in selected_ids}
        logger.info("Filtered to %d bugs", len(all_prompts))

    # Initialize LLM backend
    model_name = cfg.get("model_name", "../models/Qwen2.5-Coder-7B-Instruct")
    backend_type = cfg.get("backend", "auto")
    max_model_len = int(cfg.get("max_model_len", 8192))
    gpu_mem = float(cfg.get("gpu_memory_utilization", 0.85))

    logger.info("Initializing LLM backend: model=%s, backend=%s", model_name, backend_type)
    backend, actual_backend = create_backend(
        backend=backend_type,
        model_name=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
    )
    logger.info("Using backend: %s", actual_backend)

    # Workspace setup
    workspace_root = os.path.join(PROJECT_ROOT, args.workspace_root)
    os.makedirs(workspace_root, exist_ok=True)

    # Resolve output path early so we can checkpoint incrementally
    output_path = args.output
    if not output_path:
        output_path = os.path.join(
            PROJECT_ROOT, "Results",
            f"iterative_{strategy}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    elif not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing checkpoint if present (allows resuming a killed run)
    results = {}
    if os.path.isfile(output_path):
        try:
            with open(output_path) as f:
                results = json.load(f)
            logger.info("Resuming from checkpoint: %d bugs already done", len(results))
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s — starting fresh", output_path, e)
            results = {}

    # Derive prompt log dir for ablation (flat convention: Results/logs_<stem>/)
    prompt_log_dir = None
    if iter_config.ablation_no_meta:
        _stem = os.path.splitext(os.path.basename(output_path))[0]
        prompt_log_dir = os.path.join(os.path.dirname(output_path), f"logs_{_stem}")
        os.makedirs(prompt_log_dir, exist_ok=True)
        logger.info("Prompt/response dump enabled: %s", prompt_log_dir)

    # Run repairs
    total = len(all_prompts)
    solved_count = sum(1 for r in results.values() if r.get("solved"))
    _run_start = time.time()

    for idx, (bug_id, prom_row) in enumerate(all_prompts.items(), 1):
        if args.time_budget_sec and (time.time() - _run_start) >= args.time_budget_sec:
            logger.info("Time budget (%ds) exceeded after %d bugs — stopping.", args.time_budget_sec, idx - 1)
            break

        if bug_id in results:
            logger.info("=== [%d/%d] Bug %s — skipping (already in checkpoint) ===", idx, total, bug_id)
            continue

        logger.info("=== [%d/%d] Bug %s ===", idx, total, bug_id)

        result = repair_one_bug(
            bug_id=bug_id,
            prom_row=prom_row,
            cfg=cfg,
            iter_config=iter_config,
            backend=backend,
            workspace_root=workspace_root,
            prompt_log_dir=prompt_log_dir,
        )
        results[bug_id] = result

        if result.get("solved"):
            solved_count += 1
            logger.info("[%s] SOLVED at iteration %s", bug_id, result.get("solving_iteration"))
        else:
            logger.info("[%s] NOT SOLVED: %s", bug_id, result.get("final_fail_reason", "unknown"))

        logger.info("Progress: %d/%d solved so far", solved_count, idx)

        # Checkpoint after every bug so a crash doesn't lose progress
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)
    logger.info("Final: %d/%d solved (%.1f%%)", solved_count, total,
                100 * solved_count / max(total, 1))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"Dataset: {dataset}")
    print(f"Total bugs: {total}")
    print(f"Solved: {solved_count} ({100*solved_count/max(total,1):.1f}%)")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
