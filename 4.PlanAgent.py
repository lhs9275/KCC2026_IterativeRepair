import importlib.util
import json
import logging
import glob
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RESULTS_ROOT = Path(os.environ.get("PIPELINE_RESULTS_ROOT", "./Results"))

_STAGE5_CANDIDATE_NAMES = (
    "5.TokenPatchGeneratorAgent.py",
    "5_TokenPatchGeneratorAgent_vllm_greedy1_diverse9_MULTIRUN.py",
    "5.TokenPatchGeneratorAgent_vllm_greedy1_diverse9.MULTIRUN.py",
)


def _resolve_stage5_path() -> Path:
    here = Path(__file__).resolve().parent
    for name in _STAGE5_CANDIDATE_NAMES:
        candidate = here / name
        if candidate.exists():
            return candidate
    return here / _STAGE5_CANDIDATE_NAMES[0]


STAGE5_PATH = _resolve_stage5_path()
DEFAULT_PROMPTS_PATH = RESULTS_ROOT / "3"
DEFAULT_OUT_PATH = RESULTS_ROOT / "4/4.PlanAgent.json"
ALLOWED_EDIT_TYPES = {"REPLACE", "INSERT", "DELETE"}


def _load_stage5_module():
    spec = importlib.util.spec_from_file_location("stage5_plan_shared", STAGE5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load shared Stage5 module from {STAGE5_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGE5 = _load_stage5_module()
DEFAULT_MODEL_NAME = getattr(_STAGE5, "DEFAULT_MODEL_NAME", "../models/qwen1.5-7b-chat")
REPAIR_BRANCH_CHOICES = tuple(getattr(_STAGE5, "REPAIR_BRANCH_CHOICES", ("auto", "python_base", "java_base", "java_semantic", "java_v2")))


def load_json_dict(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {str(i): v for i, v in enumerate(data)}
    raise ValueError(f"Unsupported JSON top-level in {path}")


def _resolve_multi_match(matches: List[Path], source_label: str, pick_latest: bool) -> Path:
    if not matches:
        raise FileNotFoundError(f"No prompt JSON found for: {source_label}")
    if len(matches) == 1:
        return matches[0]
    if not pick_latest:
        sample = ", ".join(str(p) for p in matches[:5])
        raise FileExistsError(
            f"Multiple prompt JSONs matched {source_label}. "
            f"Specify a file/glob more precisely or pass --pick_latest. Matches: {sample}"
        )
    chosen = matches[0]
    logging.info("Multiple prompt JSONs matched %s; --pick_latest selected latest: %s", source_label, chosen)
    return chosen


def infer_dataset_tag(prompts_path: Path) -> str:
    name = prompts_path.name.lower()
    if "bugsinpy" in name:
        return "bugsinpy"
    if "defects4j" in name:
        return "defects4j"
    return prompts_path.stem


def resolve_full_prompt_paths(prompts_dir: Path, requested_branch: str = "auto") -> List[Path]:
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        return []

    resolved: List[Path] = []
    bugsinpy_files = sorted(
        [
            p for p in prompts_dir.glob("*bugsinpy*single*_GeneratePromport.json")
            if "smoke" not in p.name.lower()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if bugsinpy_files:
        resolved.append(bugsinpy_files[0])

    normalized_branch = normalize_repair_branch(requested_branch)
    if normalized_branch == "java_v2":
        defects4j_priority_patterns = [
            "*defects4j*java_v2*_GeneratePromport.json",
        ]
    elif normalized_branch == "java_semantic":
        defects4j_priority_patterns = [
            "*defects4j*java_semantic*_GeneratePromport.json",
            "*defects4j*java_base*_GeneratePromport.json",
            "*defects4j*hard*_GeneratePromport.json",
            "*defects4j*single*_GeneratePromport.json",
        ]
    else:
        defects4j_priority_patterns = [
            "*defects4j*java_base*_GeneratePromport.json",
            "*defects4j*java_semantic*_GeneratePromport.json",
            "*defects4j*hard*_GeneratePromport.json",
            "*defects4j*single*_GeneratePromport.json",
        ]

    found_defects4j = False
    for pattern in defects4j_priority_patterns:
        matches = sorted(
            [
                p for p in prompts_dir.glob(pattern)
                if "smoke" not in p.name.lower()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            continue
        if len(matches) > 1:
            sample = ", ".join(str(p) for p in matches[:5])
            print(
                f"[warn] Multiple defects4j prompt JSONs matched {pattern}. "
                f"Using latest: {matches[0]}. Others: {sample}"
            )
        resolved.append(matches[0])
        found_defects4j = True
        break

    if normalized_branch == "java_v2" and not found_defects4j:
        legacy_defects4j = sorted(
            [
                p for p in prompts_dir.glob("*defects4j*_GeneratePromport.json")
                if "smoke" not in p.name.lower()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if legacy_defects4j:
            raise FileNotFoundError(
                f"Requested --repair_branch java_v2, but no matching defects4j java_v2 prompt JSON was found in {prompts_dir}. "
                "Generate Stage-3 java_v2 prompts first or pass --prompts_file explicitly."
            )
    return resolved


def resolve_prompts_paths(prompts_arg: Path, pattern: str, pick_latest: bool, requested_branch: str = "auto") -> List[Path]:
    if prompts_arg.exists():
        if prompts_arg.is_file():
            return [prompts_arg]
        if prompts_arg.is_dir():
            if prompts_arg == DEFAULT_PROMPTS_PATH:
                full_paths = resolve_full_prompt_paths(prompts_arg, requested_branch=requested_branch)
                if full_paths:
                    return full_paths
            files = sorted(
                prompts_arg.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pick_latest:
                return [_resolve_multi_match(files, str(prompts_arg), pick_latest)]
            if not files:
                raise FileNotFoundError(f"No prompt JSON found for: {prompts_arg}")
            return files

    arg_text = str(prompts_arg or "")
    if arg_text and any(ch in arg_text for ch in ["*", "?", "["]):
        matches = sorted(
            [Path(p) for p in glob.glob(arg_text)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pick_latest:
            return [_resolve_multi_match(matches, arg_text, pick_latest)]
        if not matches:
            raise FileNotFoundError(f"No prompt JSON found for: {arg_text}")
        return matches

    raise FileNotFoundError(f"No prompt JSON found for: {prompts_arg}")


def derive_full_out_path(base_out: Path, dataset_tag: str) -> Path:
    if base_out.name == DEFAULT_OUT_PATH.name:
        return RESULTS_ROOT / "4" / f"4.{dataset_tag}.PlanAgent.json"
    if base_out.suffix:
        stem = base_out.name[: -len(base_out.suffix)]
        return base_out.with_name(f"{stem}.{dataset_tag}{base_out.suffix}")
    return base_out.with_name(f"{base_out.name}.{dataset_tag}.json")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def validate_plan_obj(plan_obj: Any) -> Dict[str, Any]:
    if not isinstance(plan_obj, dict):
        raise ValueError("plan_json must be a JSON object")

    target_locations = plan_obj.get("target_locations")
    if not isinstance(target_locations, list) or not target_locations:
        raise ValueError("target_locations must be a non-empty list")

    normalized_targets: List[Dict[str, int]] = []
    for idx, item in enumerate(target_locations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"target_locations[{idx}] must be an object")
        start = _safe_int(item.get("start"))
        end = _safe_int(item.get("end"))
        if start is None or end is None or start < 1 or end < start:
            raise ValueError(f"target_locations[{idx}] must contain valid start/end integers")
        normalized_targets.append({"start": int(start), "end": int(end)})

    edit_budget = plan_obj.get("edit_budget")
    if not isinstance(edit_budget, dict):
        raise ValueError("edit_budget must be an object")
    changed_lines_max = _safe_int(edit_budget.get("changed_lines_max"))
    if changed_lines_max is None or changed_lines_max < 1:
        raise ValueError("edit_budget.changed_lines_max must be a positive integer")
    normalized_budget = dict(edit_budget)
    normalized_budget["changed_lines_max"] = int(changed_lines_max)

    allowed_edit_types = plan_obj.get("allowed_edit_types")
    if not isinstance(allowed_edit_types, list) or not allowed_edit_types:
        raise ValueError("allowed_edit_types must be a non-empty list")

    normalized_allowed: List[str] = []
    for value in allowed_edit_types:
        tag = str(value or "").strip().upper()
        if tag not in ALLOWED_EDIT_TYPES:
            raise ValueError(f"Unsupported allowed_edit_type: {value}")
        normalized_allowed.append(tag)

    normalized = dict(plan_obj)
    normalized["target_locations"] = normalized_targets
    normalized["edit_budget"] = normalized_budget
    normalized["allowed_edit_types"] = normalized_allowed
    return normalized


def _ordered_edit_types(values: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for tag in ["REPLACE", "INSERT", "DELETE"]:
        if tag in values and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    for raw in values:
        tag = str(raw or "").strip().upper()
        if tag in ALLOWED_EDIT_TYPES and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    if "REPLACE" not in seen:
        ordered.insert(0, "REPLACE")
    return ordered


def _range_span(item: Dict[str, Any]) -> int:
    start = _safe_int(item.get("start")) or 1
    end = _safe_int(item.get("end")) or start
    return max(1, end - start + 1)


def _clamp_range(start: int, end: int, n_lines: int) -> Dict[str, int]:
    max_line = max(1, int(n_lines))
    start_i = max(1, min(int(start), max_line))
    end_i = max(1, min(int(end), max_line))
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return {"start": int(start_i), "end": int(end_i)}


def _expand_range_by(item: Dict[str, Any], delta: int, n_lines: int) -> Dict[str, int]:
    start = _safe_int(item.get("start")) or 1
    end = _safe_int(item.get("end")) or start
    return _clamp_range(start - int(delta), end + int(delta), n_lines)


def _expand_range_to_min_span(item: Dict[str, Any], min_span: int, n_lines: int) -> Dict[str, int]:
    cur = _clamp_range(_safe_int(item.get("start")) or 1, _safe_int(item.get("end")) or 1, n_lines)
    span = _range_span(cur)
    if span >= int(min_span):
        return cur
    need = int(min_span) - span
    grow_left = need // 2
    grow_right = need - grow_left
    return _clamp_range(cur["start"] - grow_left, cur["end"] + grow_right, n_lines)


def normalize_repair_branch(repair_branch: str) -> str:
    value = str(repair_branch or "auto").strip().lower()
    if value in REPAIR_BRANCH_CHOICES:
        return value
    return "auto"


def language_branch_for_language(language: str) -> str:
    return "java_branch" if str(language or "").strip().lower() == "java" else "python_branch"


def _argv_has_flag(args: Namespace, *flags: str) -> bool:
    argv = list(getattr(args, "_argv", sys.argv) or sys.argv)
    for token in argv:
        token_s = str(token or "")
        for flag in flags:
            if token_s == flag or token_s.startswith(f"{flag}="):
                return True
    return False


def resolve_repair_branch_metadata(row: Dict[str, Any], args: Namespace, language: str) -> Dict[str, str]:
    requested = normalize_repair_branch(getattr(args, "repair_branch", "auto"))
    row_branch = normalize_repair_branch((row or {}).get("repair_branch"))
    lang = str(language or "").strip().lower()
    cli_explicit = _argv_has_flag(args, "--repair_branch")

    if lang == "java":
        if cli_explicit and requested in {"java_base", "java_semantic", "java_v2"}:
            return {"requested": str(requested), "effective": str(requested), "source": "cli"}
        if row_branch in {"java_base", "java_semantic", "java_v2"}:
            return {"requested": str(requested), "effective": str(row_branch), "source": "row"}
        if requested in {"java_base", "java_semantic", "java_v2"}:
            return {
                "requested": str(requested),
                "effective": str(requested),
                "source": "cli" if cli_explicit else "auto",
            }
        return {"requested": str(requested), "effective": "java_base", "source": "auto"}

    if row_branch == "python_base":
        return {"requested": str(requested), "effective": "python_base", "source": "row"}
    if requested == "python_base":
        return {
            "requested": str(requested),
            "effective": "python_base",
            "source": "cli" if cli_explicit else "auto",
        }
    return {"requested": str(requested), "effective": "python_base", "source": "auto"}


def resolve_effective_repair_branch(row: Dict[str, Any], args: Namespace, language: str) -> str:
    metadata = resolve_repair_branch_metadata(row, args, language)
    return str(metadata.get("effective") or "python_base")


def _single_count_value(counts: Dict[str, int], default: str) -> str:
    normalized = {str(k): int(v) for k, v in (counts or {}).items() if str(k).strip() and int(v) > 0}
    if len(normalized) == 1:
        return next(iter(normalized))
    if len(normalized) > 1:
        return "mixed"
    return str(default)


def resolve_effective_repair_branch_for_dataset(args: Namespace, dataset_tag: str) -> Dict[str, str]:
    language = "java" if str(dataset_tag or "").strip().lower() == "defects4j" else "python"
    return resolve_repair_branch_metadata({}, args, language)


def resolve_planner_mode_effective(row: Dict[str, Any], language: str, repair_branch: str) -> str:
    repair_mode = str((row or {}).get("repair_mode") or (row or {}).get("hard_tag") or "").strip().lower()
    if repair_mode in {"single", "hard"}:
        return repair_mode
    if "single_line" in (row or {}) and (row or {}).get("single_line") is not None:
        return "single" if bool((row or {}).get("single_line")) else "hard"
    if str(language or "").strip().lower() == "java" and repair_branch in {"java_base", "java_semantic"}:
        return "single"
    return "single" if bool((row or {}).get("single_line")) else "hard"


def postprocess_plan(plan_obj: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    plan = json.loads(json.dumps(plan_obj or {}))
    notes: List[str] = []

    base_code = get_base_code(row or {})
    n_lines = max(1, len((base_code or "").splitlines()))
    repair_mode = str((row or {}).get("planner_mode_effective") or "").strip().lower()
    if repair_mode not in {"single", "hard"}:
        repair_mode = str((row or {}).get("repair_mode") or (row or {}).get("hard_tag") or "").strip().lower()
    if repair_mode not in {"single", "hard"}:
        repair_mode = "single" if bool((row or {}).get("single_line")) else "hard"

    is_single = repair_mode == "single"
    is_hard = repair_mode == "hard"
    language = str((row or {}).get("language") or infer_language(row or {}, base_code)).strip().lower()
    repair_branch = normalize_repair_branch(
        str(
            (row or {}).get("repair_branch_effective")
            or (row or {}).get("repair_branch")
            or ""
        )
    )
    java_conservative = bool(language == "java" and repair_branch in {"java_base", "java_semantic"})

    target_locations = plan.get("target_locations")
    if not isinstance(target_locations, list) or not target_locations:
        target_locations = [{"start": 1, "end": min(n_lines, 1)}]

    normalized_targets: List[Dict[str, int]] = []
    for item in target_locations:
        if not isinstance(item, dict):
            continue
        normalized_targets.append(
            _clamp_range(_safe_int(item.get("start")) or 1, _safe_int(item.get("end")) or 1, n_lines)
        )
    if not normalized_targets:
        normalized_targets = [{"start": 1, "end": min(n_lines, 1)}]

    edit_budget = plan.get("edit_budget")
    if not isinstance(edit_budget, dict):
        edit_budget = {}
    changed_lines_max = _safe_int(edit_budget.get("changed_lines_max")) or 1
    if changed_lines_max < 1:
        changed_lines_max = 1
        notes.append("raised_changed_lines_max_to_1")

    allowed_edit_types = plan.get("allowed_edit_types")
    if not isinstance(allowed_edit_types, list):
        allowed_edit_types = []
    allowed = _ordered_edit_types([str(v or "").strip().upper() for v in allowed_edit_types])

    if is_single and "INSERT" not in allowed:
        allowed.append("INSERT")
        allowed = _ordered_edit_types(allowed)
        notes.append("added_INSERT")
    if is_hard:
        before = set(allowed)
        allowed = _ordered_edit_types(allowed + ["INSERT", "DELETE"])
        if "INSERT" not in before:
            notes.append("added_INSERT")
        if "DELETE" not in before:
            notes.append("added_DELETE")

    if is_single:
        single_min = 4 if java_conservative else 6
        if changed_lines_max < single_min:
            changed_lines_max = single_min
            notes.append(f"raised_changed_lines_max_to_{single_min}")
    if is_hard:
        hard_min = 6 if java_conservative else 12
        if changed_lines_max < hard_min:
            changed_lines_max = hard_min
            notes.append(f"raised_changed_lines_max_to_{hard_min}")

    avg_span = sum(_range_span(item) for item in normalized_targets) / max(1, len(normalized_targets))
    if is_single and avg_span < (4.0 if java_conservative else 7.0):
        widen_delta = 1 if java_conservative else 2
        normalized_targets = [_expand_range_by(item, widen_delta, n_lines) for item in normalized_targets]
        notes.append(f"widened_target_by_{widen_delta}")
    if is_hard and not java_conservative:
        widened = False
        widened_targets: List[Dict[str, int]] = []
        for item in normalized_targets:
            widened_item = _expand_range_to_min_span(item, 15, n_lines)
            if widened_item != item:
                widened = True
            widened_targets.append(widened_item)
        normalized_targets = widened_targets
        if widened:
            notes.append("widened_target_to_min_span_15")

    plan["target_locations"] = normalized_targets
    plan["edit_budget"] = dict(edit_budget)
    plan["edit_budget"]["changed_lines_max"] = int(changed_lines_max)
    plan["allowed_edit_types"] = allowed

    row["_plan_postprocessed"] = bool(notes)
    row["_plan_postprocess_notes"] = notes
    return validate_plan_obj(plan)


def build_default_plan_obj(prom_row: Dict[str, Any], base_code: str, language: str) -> Dict[str, Any]:
    temp_row = dict(prom_row or {})
    planner_mode_effective = str(temp_row.get("planner_mode_effective") or "").strip().lower()
    if planner_mode_effective in {"single", "hard"}:
        temp_row["repair_mode"] = planner_mode_effective
        temp_row["hard_tag"] = planner_mode_effective
        temp_row["single_line"] = bool(planner_mode_effective == "single")
    raw = _STAGE5.build_default_plan_json(temp_row, base_code=base_code, language=language)
    parsed = json.loads(raw)
    return validate_plan_obj(parsed)


def infer_language(prom_row: Dict[str, Any], base_code: str) -> str:
    return _STAGE5.infer_language(prom_row or {}, prom_row or {}, base_code or "")


def get_base_code(prom_row: Dict[str, Any]) -> str:
    return (
        str(prom_row.get("code") or "")
        or str(_STAGE5.safe_get(prom_row, ["function", "function_before"]) or "")
        or str(_STAGE5.safe_get(prom_row, ["function", "function_after"]) or "")
    )


def maybe_validate_existing(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    existing = row.get("plan_json")
    if existing is None:
        return None, None
    try:
        if isinstance(existing, str):
            existing = extract_json_object(existing)
        return validate_plan_obj(existing), None
    except Exception as exc:
        return None, str(exc)


def generate_plan_for_row(client, bug_id: str, row: Dict[str, Any], max_new_tokens: int) -> Tuple[Dict[str, Any], str, bool, str]:
    base_code = get_base_code(row)
    language = infer_language(row, base_code)
    plan_prompt = str(row.get("plan_prompt") or "").strip()

    if not plan_prompt:
        return build_default_plan_obj(row, base_code, language), "fallback_default", True, ""

    try:
        outputs = client.generate(
            [plan_prompt],
            {
                "batch_size": 1,
                "n": 1,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_input_tokens": 4096,
                "seed": 0,
            },
        )
        raw_text = outputs[0].get("text") if outputs else ""
        plan_obj = validate_plan_obj(extract_json_object(raw_text or ""))
        return plan_obj, "llm", True, ""
    except Exception as exc:
        logging.warning("[%s] LLM plan generation failed; using default plan (%s)", bug_id, exc)
        fallback = build_default_plan_obj(row, base_code, language)
        return fallback, "fallback_default", False, str(exc)


def process_prompts(args: Namespace, prompts_path: Path) -> Dict[str, Any]:
    rows = load_json_dict(prompts_path)
    args._backend_selected = str(getattr(args, "_backend_selected", "not_used") or "not_used")
    args._hf_model_name_selected = str(getattr(args, "_hf_model_name_selected", "") or "")
    if args.limit is not None and args.limit >= 0:
        items = list(rows.items())[: int(args.limit)]
        rows = {k: v for k, v in items}

    needs_llm = False
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        existing_plan, _existing_err = maybe_validate_existing(row)
        if existing_plan is not None and not bool(args.overwrite):
            continue
        if str(row.get("plan_prompt") or "").strip():
            needs_llm = True
            break

    client = None
    if needs_llm:
        client = _STAGE5.QwenClient(
            args.model_name,
            system_prompt=(
                "Return exactly one JSON object repair plan. "
                "Do not add markdown, commentary, or code fences."
            ),
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            backend=str(getattr(args, "backend", "auto") or "auto"),
            hf_model_name=str(getattr(args, "hf_model_name", "") or args.model_name),
            hf_device=str(getattr(args, "hf_device", "auto") or "auto"),
            hf_dtype=str(getattr(args, "hf_dtype", "auto") or "auto"),
        )
        args._backend_selected = str(getattr(client, "backend_selected", getattr(args, "backend", "auto")) or "auto")
        args._hf_model_name_selected = str(getattr(client, "hf_model_name", "") or "")

        dataset_tag = infer_dataset_tag(prompts_path)
        if args._backend_selected == "hf" and str(dataset_tag).strip().lower() == "defects4j":
            effective_java_branch = "java_base"
            for _row in rows.values():
                if not isinstance(_row, dict):
                    continue
                _base_code = get_base_code(_row)
                _language = infer_language(_row, _base_code)
                if _language != "java":
                    continue
                _meta = resolve_repair_branch_metadata(_row, args, _language)
                effective_java_branch = str(_meta.get("effective") or "java_base")
                break
            if effective_java_branch in {"java_base", "java_semantic", "java_v2"}:
                raise RuntimeError(
                    "HF backend selected for a Java planning run. Stage 4 now fails fast to match Stage 5. "
                    "Use backend auto/vllm for Java, or run a Python-only prompt file."
                )

    out_rows: Dict[str, Any] = {}
    for bug_id, row in rows.items():
        if not isinstance(row, dict):
            out_rows[bug_id] = row
            continue

        out_row = dict(row)
        base_code = get_base_code(out_row)
        language = infer_language(out_row, base_code)
        repair_branch_meta = resolve_repair_branch_metadata(out_row, args, language)
        effective_repair_branch = str(repair_branch_meta.get("effective") or resolve_effective_repair_branch(out_row, args, language))
        effective_language_branch = language_branch_for_language(language)
        planner_mode_effective = resolve_planner_mode_effective(out_row, language, effective_repair_branch)
        out_row["language_branch"] = str(out_row.get("language_branch") or effective_language_branch)
        out_row["repair_branch_requested"] = str(
            repair_branch_meta.get("requested") or normalize_repair_branch(getattr(args, "repair_branch", "auto"))
        )
        out_row["repair_branch_effective"] = str(effective_repair_branch)
        out_row["repair_branch_source"] = str(repair_branch_meta.get("source") or "auto")
        out_row["repair_branch"] = str(effective_repair_branch)
        out_row["planner_mode_effective"] = planner_mode_effective
        existing_plan, existing_error = maybe_validate_existing(out_row)
        if existing_plan is not None and not bool(args.overwrite):
            out_row["plan_json"] = postprocess_plan(existing_plan, out_row)
            out_row["plan_source"] = str(out_row.get("plan_source") or "llm")
            out_row["plan_valid"] = bool(out_row.get("plan_valid", True))
            out_row["plan_error"] = str(out_row.get("plan_error") or "")
            out_row["plan_postprocessed"] = bool(out_row.pop("_plan_postprocessed", False))
            out_row["plan_postprocess_notes"] = list(out_row.pop("_plan_postprocess_notes", []))
            out_row["backend_selected"] = str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto")
            out_row["hf_model_name"] = str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or "")
            out_rows[bug_id] = out_row
            continue

        if existing_error and not bool(args.overwrite):
            logging.info("[%s] Existing plan_json invalid; regenerating (%s)", bug_id, existing_error)

        if str(out_row.get("plan_prompt") or "").strip():
            if client is None:
                client = _STAGE5.QwenClient(
                    args.model_name,
                    system_prompt=(
                        "Return exactly one JSON object repair plan. "
                        "Do not add markdown, commentary, or code fences."
                    ),
                    max_model_len=int(args.max_model_len),
                    gpu_memory_utilization=float(args.gpu_memory_utilization),
                    backend=str(getattr(args, "backend", "auto") or "auto"),
                    hf_model_name=str(getattr(args, "hf_model_name", "") or args.model_name),
                    hf_device=str(getattr(args, "hf_device", "auto") or "auto"),
                    hf_dtype=str(getattr(args, "hf_dtype", "auto") or "auto"),
                )
                args._backend_selected = str(getattr(client, "backend_selected", getattr(args, "backend", "auto")) or "auto")
                args._hf_model_name_selected = str(getattr(client, "hf_model_name", "") or "")
            plan_obj, plan_source, plan_valid, plan_error = generate_plan_for_row(
                client=client,
                bug_id=str(bug_id),
                row=out_row,
                max_new_tokens=int(args.max_new_tokens),
            )
        else:
            plan_obj = build_default_plan_obj(out_row, base_code, language)
            plan_source = "fallback_default"
            plan_valid = False if existing_error else True
            plan_error = str(existing_error or "")

        out_row["plan_json"] = postprocess_plan(plan_obj, out_row)
        out_row["plan_source"] = plan_source
        out_row["plan_valid"] = bool(plan_valid)
        out_row["plan_error"] = str(plan_error or "")
        out_row["plan_postprocessed"] = bool(out_row.pop("_plan_postprocessed", False))
        out_row["plan_postprocess_notes"] = list(out_row.pop("_plan_postprocess_notes", []))
        out_row["backend_selected"] = str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto")
        out_row["hf_model_name"] = str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or "")
        out_rows[bug_id] = out_row

    logging.info("Processed %d bugs from %s", len(out_rows), prompts_path)
    return out_rows


def try_get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return str(out or "").strip()
    except Exception:
        return ""


def derive_manifest_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.manifest{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}.manifest.json")


def _count_row_values(rows: Dict[str, Any], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in (rows or {}).values():
        if not isinstance(row, dict):
            continue
        value = str(row.get(key) or "").strip() or "unknown"
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _count_plan_postprocess_notes(rows: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in (rows or {}).values():
        if not isinstance(row, dict):
            continue
        for note in list(row.get("plan_postprocess_notes") or []):
            note_s = str(note or "").strip()
            if not note_s:
                continue
            counts[note_s] = int(counts.get(note_s, 0)) + 1
    return counts


def write_run_manifest(args: Namespace, resolved_prompts_path: Path, out_path: Path, results: Dict[str, Any]) -> Path:
    dataset_tag = infer_dataset_tag(resolved_prompts_path)
    requested_meta = resolve_effective_repair_branch_for_dataset(args, dataset_tag)
    repair_branch_counts = _count_row_values(results, "repair_branch")
    repair_branch_source_counts = _count_row_values(results, "repair_branch_source")
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "argv": list(getattr(args, "_argv", sys.argv)),
        "resolved_prompts_file": str(resolved_prompts_path),
        "out_file": str(out_path),
        "model_name": str(args.model_name),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_new_tokens": int(args.max_new_tokens),
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "repair_branch_requested": str(requested_meta.get("requested") or "auto"),
        "repair_branch_effective": _single_count_value(
            repair_branch_counts,
            str(requested_meta.get("effective") or normalize_repair_branch(getattr(args, "repair_branch", "auto"))),
        ),
        "repair_branch_source": _single_count_value(
            repair_branch_source_counts,
            str(requested_meta.get("source") or "auto"),
        ),
        "repair_branch": _single_count_value(
            repair_branch_counts,
            str(requested_meta.get("effective") or normalize_repair_branch(getattr(args, "repair_branch", "auto"))),
        ),
        "language_branch_counts": _count_row_values(results, "language_branch"),
        "repair_branch_counts": repair_branch_counts,
        "repair_branch_source_counts": repair_branch_source_counts,
        "planner_mode_effective_counts": _count_row_values(results, "planner_mode_effective"),
        "plan_postprocess_notes": _count_plan_postprocess_notes(results),
        "backend_selected": str(getattr(args, "_backend_selected", getattr(args, "backend", "auto")) or "auto"),
        "hf_model_name": str(getattr(args, "_hf_model_name_selected", getattr(args, "hf_model_name", "")) or ""),
        "git_commit": try_get_git_commit(),
    }
    manifest_path = derive_manifest_path(out_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="4.PlanAgent — deterministic plan JSON generation via backend auto/vllm/hf")
    parser.add_argument("--prompts_file", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--out_file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--pattern", type=str, default="*GeneratePromport*.json")
    parser.add_argument(
        "--pick_latest",
        action="store_true",
        help="When multiple files match a directory/glob, automatically pick the latest one (reproducibility caution).",
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "hf"])
    parser.add_argument("--hf_model_name", type=str, default="", help="HF fallback model/path. Defaults to --model_name.")
    parser.add_argument("--hf_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument(
        "--repair_branch",
        type=str,
        default="auto",
        choices=REPAIR_BRANCH_CHOICES,
        help="Explicit repair branch selector. auto keeps Python near current behavior and uses java_base for Java. Use java_v2 explicitly to enable the new Java-only pipeline.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plan_json fields when present.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N bugs for debugging.")
    return parser


def main(args: Namespace) -> None:
    args._argv = list(sys.argv)
    resolved_paths = resolve_prompts_paths(args.prompts_file, args.pattern, args.pick_latest, requested_branch=normalize_repair_branch(args.repair_branch))
    if not resolved_paths:
        raise FileNotFoundError(f"No prompt JSON found for: {args.prompts_file}")

    full_default_mode = (args.prompts_file == DEFAULT_PROMPTS_PATH and args.prompts_file.is_dir())
    multi_mode = full_default_mode or len(resolved_paths) > 1

    for resolved in resolved_paths:
        out_path = derive_full_out_path(args.out_file, infer_dataset_tag(resolved)) if multi_mode else args.out_file
        logging.info("Using prompts JSON: %s", resolved)
        results = process_prompts(args, resolved)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = write_run_manifest(args, resolved, out_path, results)
        logging.info("Saved plan output to %s", out_path)
        logging.info("Saved plan manifest to %s", manifest_path)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
