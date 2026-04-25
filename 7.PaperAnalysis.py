#!/usr/bin/env python3
"""
KCC2026 논문용 분석 스크립트.

Usage:
    python 7.PaperAnalysis.py \
        --one_shot Results/iterative_one_shot_defects4j_*.json \
        --blind_retry Results/iterative_blind_retry_defects4j_*.json \
        --error_aware Results/iterative_error_aware_defects4j_*.json \
        --meta Results/4/4.defects4j.PlanAgent.json \
        --output_dir paper_figures

Outputs:
    - Table 2: 전략별 해결률 비교 (stdout + CSV)
    - Table 3: 프로젝트별 해결률 (stdout + CSV)
    - Figure 3: iter별 누적 해결 수 (stacked bar chart)
    - Figure 4: 에러 유형별 recovery rate (bar chart)
    - 정성 분석: ONLY error_aware 대표 사례 (stdout)
    - summary.json: 전체 통계 JSON
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


PRETTY_LABELS = {
    "one_shot": "One-shot",
    "blind_retry": "Blind Retry",
    "error_aware": "Error-Aware",
    "qwen_one_shot": "Qwen One-shot",
    "qwen_blind_retry": "Qwen Blind Retry",
    "qwen_error_aware": "Qwen Error-Aware",
    "deepseek_one_shot": "DeepSeek One-shot",
    "deepseek_error_aware": "DeepSeek Error-Aware",
}


def pretty_label(name: str) -> str:
    return PRETTY_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def find_latest(pattern: str) -> str:
    import glob
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return matches[-1]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def get_solved_set(data: Dict[str, Any]) -> Set[str]:
    return {bid for bid, r in data.items() if r.get("solved")}


def get_valid_bugs(data: Dict[str, Any]) -> Dict[str, Any]:
    """infra_fail/error 제외."""
    return {bid: r for bid, r in data.items()
            if r.get("status") not in ("infra_fail", "error") and not r.get("error")}


def table2_strategy_comparison(os_data, br_data, ea_data):
    """Table 2: 전략별 해결률."""
    rows = []
    for name, d in [("one_shot", os_data), ("blind_retry", br_data), ("error_aware", ea_data)]:
        valid = get_valid_bugs(d)
        solved = sum(1 for r in valid.values() if r.get("solved"))
        total = len(valid)
        rows.append({
            "strategy": name,
            "total": total,
            "solved": solved,
            "rate": round(solved / max(total, 1), 4),
        })
    return rows


def summarize_runs(run_data_map: Dict[str, Dict[str, Any]]):
    """Arbitrary run summary."""
    rows = []
    for name, data in run_data_map.items():
        valid = get_valid_bugs(data)
        solved = sum(1 for r in valid.values() if r.get("solved"))
        total = len(valid)
        rows.append({
            "run": name,
            "total": total,
            "solved": solved,
            "rate": round(solved / max(total, 1), 4),
        })
    return rows


def table3_project_breakdown(os_data, br_data, ea_data, meta):
    """Table 3: 프로젝트별 해결률."""
    proj_map = {}
    for bid in ea_data:
        if bid in meta:
            proj_map[bid] = meta[bid].get("project_name", "unknown")

    projects = sorted(set(proj_map.values()))
    rows = []
    for proj in projects:
        bids = [bid for bid, p in proj_map.items() if p == proj
                and ea_data.get(bid, {}).get("status") not in ("infra_fail", "error")
                and not ea_data.get(bid, {}).get("error")]
        total = len(bids)
        if total == 0:
            continue
        os_solved = sum(1 for bid in bids if os_data.get(bid, {}).get("solved"))
        br_solved = sum(1 for bid in bids if br_data.get(bid, {}).get("solved"))
        ea_solved = sum(1 for bid in bids if ea_data.get(bid, {}).get("solved"))
        rows.append({
            "project": proj,
            "total": total,
            "one_shot": os_solved,
            "one_shot_rate": round(os_solved / total, 4),
            "blind_retry": br_solved,
            "blind_retry_rate": round(br_solved / total, 4),
            "error_aware": ea_solved,
            "error_aware_rate": round(ea_solved / total, 4),
        })
    rows.sort(key=lambda x: -x["total"])
    return rows


def table_project_breakdown_multi(run_data_map, meta):
    """Project breakdown for arbitrary run sets."""
    project_to_bids = defaultdict(list)
    for bid, bug_meta in meta.items():
        project_to_bids[bug_meta.get("project_name", "unknown")].append(bid)

    rows = []
    for project, bids in sorted(project_to_bids.items(), key=lambda x: (-len(x[1]), x[0])):
        row = {
            "project": project,
            "total": len(bids),
        }
        has_any_valid = False
        for run_name, data in run_data_map.items():
            valid_bids = [
                bid for bid in bids
                if bid in data and data[bid].get("status") not in ("infra_fail", "error") and not data[bid].get("error")
            ]
            solved = sum(1 for bid in valid_bids if data[bid].get("solved"))
            row[f"{run_name}_valid"] = len(valid_bids)
            row[f"{run_name}_solved"] = solved
            row[f"{run_name}_rate"] = round(solved / max(len(valid_bids), 1), 4)
            has_any_valid = has_any_valid or bool(valid_bids)
        if has_any_valid:
            rows.append(row)
    return rows


def figure3_iter_distribution(ea_data):
    """Figure 3: error_aware iter별 해결 수."""
    iter_solved = Counter()
    for bid, r in ea_data.items():
        if r.get("solved"):
            iter_solved[r.get("solving_iteration", 0)] += 1
    return dict(sorted(iter_solved.items()))


def figure4_error_family_recovery(ea_data):
    """Figure 4: 에러 유형별 recovery rate."""
    family_total = Counter()
    family_recovered = Counter()

    for bid, r in ea_data.items():
        if r.get("status") in ("infra_fail", "error") or r.get("error"):
            continue
        iters = r.get("iterations", [])
        if not iters:
            continue

        iter1 = iters[0]
        fail = iter1.get("fail_reason", "")
        family = iter1.get("error_family", "")

        if fail == "pass":
            continue

        if family:
            key = family
        elif fail in ("test_fail", "test_error"):
            key = "test_fail"
        elif fail == "compile_fail":
            key = "compile_other"
        else:
            key = fail

        family_total[key] += 1
        if r.get("solved") and r.get("solving_iteration", 1) > 1:
            family_recovered[key] += 1

    rows = []
    for fam in sorted(family_total, key=lambda x: -family_total[x]):
        t = family_total[fam]
        rec = family_recovered.get(fam, 0)
        rows.append({
            "error_family": fam,
            "total": t,
            "recovered": rec,
            "rate": round(rec / max(t, 1), 4),
        })
    return rows


def qualitative_cases_against(target_data, baseline_data_list, meta, n=5):
    """ONLY target_data solved 대표 사례."""
    baseline_solved = set()
    for baseline in baseline_data_list:
        baseline_solved |= get_solved_set(baseline)

    cases = []
    for bid, r in sorted(target_data.items(), key=lambda x: int(x[0])):
        if not r.get("solved"):
            continue
        if bid in baseline_solved:
            continue
        iters = r.get("iterations", [])
        if len(iters) < 2:
            continue

        bug_meta = meta.get(bid, {})
        cases.append({
            "bug_id": bid,
            "project": bug_meta.get("project_name", "?"),
            "function": bug_meta.get("function", {}).get("function_name", "?"),
            "buggy_line": bug_meta.get("buggy_line_content", ""),
            "fixed_line": bug_meta.get("fixed_line_content", ""),
            "solving_iteration": r.get("solving_iteration"),
            "iterations": [
                {
                    "iter": it["iteration"],
                    "valid": it["num_valid_candidates"],
                    "gen": it["num_candidates_generated"],
                    "fail_reason": it["fail_reason"],
                    "error_family": it["error_family"],
                    "feedback_strategy": it.get("feedback_strategy", ""),
                }
                for it in iters
            ],
        })
        if len(cases) >= n:
            break
    return cases


def qualitative_cases(ea_data, os_data, br_data, meta, n=5):
    """ONLY error_aware로 해결된 대표 사례."""
    return qualitative_cases_against(ea_data, [os_data, br_data], meta, n=n)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_iter_distribution(iter_data, strategies_data, output_path):
    """Figure 3: Stacked bar — bugs solved per iteration."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    strategies = ["one_shot", "blind_retry", "error_aware"]
    labels = ["One-shot", "Blind Retry", "Error-Aware"]
    colors_iter = ["#2ecc71", "#3498db", "#9b59b6"]

    all_iter_data = {}
    for strat, d in strategies_data.items():
        ic = Counter()
        for bid, r in d.items():
            if r.get("solved"):
                ic[r.get("solving_iteration", 0)] += 1
        all_iter_data[strat] = ic

    bottoms = [0] * len(strategies)
    for it in [1, 2, 3]:
        values = [all_iter_data[s].get(it, 0) for s in strategies]
        ax.bar(labels, values, bottom=bottoms, label=f"Iter {it}",
               color=colors_iter[it - 1], edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Total labels on top
    for i, s in enumerate(strategies):
        total = sum(all_iter_data[s].values())
        ax.text(i, bottoms[i] + 0.5, str(total), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Number of Bugs Solved")
    ax.set_title("Bug Resolution by Iteration")
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_error_recovery(recovery_rows, output_path):
    """Figure 4: Error family recovery rate."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not available")
        return

    rows = [r for r in recovery_rows if r["total"] >= 2]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    families = [r["error_family"] for r in rows]
    totals = [r["total"] for r in rows]
    recovered = [r["recovered"] for r in rows]

    x = range(len(families))
    w = 0.35
    ax.bar([i - w / 2 for i in x], totals, w, label="Total (iter 1 failures)", color="#3498db")
    ax.bar([i + w / 2 for i in x], recovered, w, label="Recovered (iter 2/3)", color="#2ecc71")

    # Rate labels
    for i, r in enumerate(rows):
        if r["recovered"] > 0:
            ax.text(i + w / 2, r["recovered"] + 0.3, f'{r["rate"]:.0%}',
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Error Family")
    ax.set_ylabel("Number of Bugs")
    ax.set_title("Error-Aware Recovery by Error Type")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_project_breakdown(proj_rows, output_path):
    """Table 3 as grouped bar chart."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    projects = [r["project"] for r in proj_rows]
    x = range(len(projects))
    w = 0.25

    os_rates = [r["one_shot_rate"] * 100 for r in proj_rows]
    br_rates = [r["blind_retry_rate"] * 100 for r in proj_rows]
    ea_rates = [r["error_aware_rate"] * 100 for r in proj_rows]

    ax.bar([i - w for i in x], os_rates, w, label="One-shot", color="#e74c3c")
    ax.bar([i for i in x], br_rates, w, label="Blind Retry", color="#f39c12")
    ax.bar([i + w for i in x], ea_rates, w, label="Error-Aware", color="#2ecc71")

    ax.set_xlabel("Project")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Solve Rate by Project")
    ax.set_xticks(list(x))
    ax.set_xticklabels(projects, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_run_summary(rows, output_path, title):
    """Generic run-level solved-count comparison."""
    if not HAS_MPL or not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [pretty_label(r["run"]) for r in rows]
    solved = [r["solved"] for r in rows]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#34495e"]

    bars = ax.bar(labels, solved, color=colors[:len(rows)])
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{row["solved"]}/{row["total"]}\n{row["rate"]:.1%}',
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Bugs Solved")
    ax.set_title(title)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_project_breakdown_multi(proj_rows, run_names, output_path, title):
    """Grouped bar chart for arbitrary runs by project."""
    if not HAS_MPL or not proj_rows or not run_names:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    projects = [r["project"] for r in proj_rows]
    x = list(range(len(projects)))
    width = 0.8 / len(run_names)
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#34495e"]

    for idx, run_name in enumerate(run_names):
        rates = [r[f"{run_name}_rate"] * 100 for r in proj_rows]
        offset = -0.4 + (idx + 0.5) * width
        ax.bar(
            [i + offset for i in x],
            rates,
            width,
            label=pretty_label(run_name),
            color=colors[idx % len(colors)],
        )

    ax.set_xlabel("Project")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(projects, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_table2(rows):
    print("\n" + "=" * 55)
    print("  Table 2: Strategy Comparison")
    print("=" * 55)
    print(f"  {'Strategy':<15} {'Solved':>8} {'Total':>8} {'Rate':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")
    for r in rows:
        print(f"  {r['strategy']:<15} {r['solved']:>8} {r['total']:>8} {r['rate']*100:>9.1f}%")
    print()


def print_run_summary(rows, title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  {'Run':<24} {'Solved':>8} {'Total':>8} {'Rate':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*10}")
    for row in rows:
        print(f"  {pretty_label(row['run']):<24} {row['solved']:>8} {row['total']:>8} {row['rate']*100:>9.1f}%")
    print()


def print_table3(rows):
    print("=" * 75)
    print("  Table 3: Project Breakdown")
    print("=" * 75)
    print(f"  {'Project':<18} {'Total':>6} {'one_shot':>12} {'blind':>12} {'error_aware':>14}")
    print(f"  {'-'*18} {'-'*6} {'-'*12} {'-'*12} {'-'*14}")
    for r in rows:
        print(f"  {r['project']:<18} {r['total']:>6}"
              f" {r['one_shot']:>4}({r['one_shot_rate']*100:4.0f}%)"
              f" {r['blind_retry']:>4}({r['blind_retry_rate']*100:4.0f}%)"
              f" {r['error_aware']:>5}({r['error_aware_rate']*100:4.0f}%)")
    print()


def print_project_breakdown_multi(rows, run_names, title):
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)
    header = f"  {'Project':<18} {'Total':>6}"
    for run_name in run_names:
        header += f" {pretty_label(run_name):>24}"
    print(header)

    divider = f"  {'-'*18} {'-'*6}"
    for _ in run_names:
        divider += f" {'-'*24}"
    print(divider)

    for row in rows:
        line = f"  {row['project']:<18} {row['total']:>6}"
        for run_name in run_names:
            cell = (
                f"{row[f'{run_name}_solved']:>4}/"
                f"{row[f'{run_name}_valid']:<4}"
                f" ({row[f'{run_name}_rate']*100:>5.1f}%)"
            )
            line += f" {cell:>24}"
        print(line)
    print()


def print_figure3(iter_data):
    print("=" * 40)
    print("  Figure 3: Iteration Distribution")
    print("=" * 40)
    cumul = 0
    for it, cnt in sorted(iter_data.items()):
        cumul += cnt
        print(f"  iter {it}: +{cnt} (cumulative: {cumul})")
    print()


def print_figure4(rows):
    print("=" * 65)
    print("  Figure 4: Error Family Recovery Rate")
    print("=" * 65)
    print(f"  {'Error Family':<25} {'Total':>6} {'Recovered':>10} {'Rate':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8}")
    for r in rows:
        print(f"  {r['error_family']:<25} {r['total']:>6} {r['recovered']:>10} {r['rate']*100:>7.1f}%")
    print()


def print_qualitative(cases):
    print("=" * 60)
    print("  Qualitative: ONLY error_aware cases")
    print("=" * 60)
    for c in cases:
        print(f"\n  Bug {c['bug_id']} ({c['project']} / {c['function']})")
        print(f"    Buggy: {c['buggy_line']}")
        print(f"    Fixed: {c['fixed_line']}")
        print(f"    Solved at iter {c['solving_iteration']}")
        for it in c["iterations"]:
            print(f"      iter {it['iter']}: valid={it['valid']}/{it['gen']}, "
                  f"fail={it['fail_reason']}, strategy={it['feedback_strategy']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_standard_analysis(os_path, br_path, ea_path, meta_path, output_dir):
    print("Loading:")
    print(f"  one_shot:    {os_path}")
    print(f"  blind_retry: {br_path}")
    print(f"  error_aware: {ea_path}")
    print(f"  meta:        {meta_path}")

    os_data = load_json(os_path)
    br_data = load_json(br_path)
    ea_data = load_json(ea_path)
    meta = load_json(meta_path)

    os.makedirs(output_dir, exist_ok=True)

    t2 = table2_strategy_comparison(os_data, br_data, ea_data)
    print_table2(t2)
    write_csv(t2, os.path.join(output_dir, "table2_strategy.csv"))

    t3 = table3_project_breakdown(os_data, br_data, ea_data, meta)
    print_table3(t3)
    write_csv(t3, os.path.join(output_dir, "table3_project.csv"))

    f3 = figure3_iter_distribution(ea_data)
    print_figure3(f3)

    f4 = figure4_error_family_recovery(ea_data)
    print_figure4(f4)

    strategies_data = {"one_shot": os_data, "blind_retry": br_data, "error_aware": ea_data}
    plot_iter_distribution(f3, strategies_data, os.path.join(output_dir, "fig3_iter_distribution.png"))
    plot_error_recovery(f4, os.path.join(output_dir, "fig4_error_recovery.png"))
    plot_project_breakdown(t3, os.path.join(output_dir, "fig5_project_breakdown.png"))

    cases = qualitative_cases(ea_data, os_data, br_data, meta, n=5)
    print_qualitative(cases)

    summary = {
        "table2": t2,
        "table3": t3,
        "figure3_iter_solved": f3,
        "figure4_recovery": f4,
        "qualitative_cases": cases,
        "files": {"one_shot": os_path, "blind_retry": br_path, "error_aware": ea_path},
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")
    return summary


def run_two_strategy_analysis(one_shot_path, error_aware_path, meta_path, output_dir, label):
    print("\n" + "#" * 80)
    print(f"# {label}: Two-Strategy Analysis")
    print("#" * 80)
    print("Loading:")
    print(f"  one_shot:    {one_shot_path}")
    print(f"  error_aware: {error_aware_path}")
    print(f"  meta:        {meta_path}")

    one_shot_data = load_json(one_shot_path)
    error_aware_data = load_json(error_aware_path)
    meta = load_json(meta_path)

    os.makedirs(output_dir, exist_ok=True)

    run_names = ["one_shot", "error_aware"]
    run_data_map = {
        "one_shot": one_shot_data,
        "error_aware": error_aware_data,
    }

    t2 = summarize_runs(run_data_map)
    print_run_summary(t2, f"{label}: Strategy Comparison")
    write_csv(t2, os.path.join(output_dir, "table2_strategy.csv"))

    t3 = table_project_breakdown_multi(run_data_map, meta)
    print_project_breakdown_multi(t3, run_names, f"{label}: Project Breakdown")
    write_csv(t3, os.path.join(output_dir, "table3_project.csv"))

    f3 = figure3_iter_distribution(error_aware_data)
    print_figure3(f3)

    f4 = figure4_error_family_recovery(error_aware_data)
    print_figure4(f4)

    plot_run_summary(t2, os.path.join(output_dir, "fig2_strategy_comparison.png"), f"{label}: Solve Count by Strategy")
    plot_project_breakdown_multi(
        t3,
        run_names,
        os.path.join(output_dir, "fig3_project_breakdown.png"),
        f"{label}: Solve Rate by Project",
    )
    plot_error_recovery(f4, os.path.join(output_dir, "fig4_error_recovery.png"))

    cases = qualitative_cases_against(error_aware_data, [one_shot_data], meta, n=5)
    print_qualitative(cases)

    summary = {
        "table2": t2,
        "table3": t3,
        "figure3_iter_solved": f3,
        "figure4_recovery": f4,
        "qualitative_cases": cases,
        "files": {"one_shot": one_shot_path, "error_aware": error_aware_path},
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")
    return summary


def run_combined_latest_analysis(run_paths, meta_path, output_dir):
    print("\n" + "#" * 80)
    print("# Latest All-Runs Comparison")
    print("#" * 80)
    print("Loading:")
    for run_name, path in run_paths.items():
        print(f"  {run_name}: {path}")
    print(f"  meta: {meta_path}")

    run_data_map = {run_name: load_json(path) for run_name, path in run_paths.items()}
    meta = load_json(meta_path)

    os.makedirs(output_dir, exist_ok=True)

    t1 = summarize_runs(run_data_map)
    print_run_summary(t1, "Latest All-Runs Comparison")
    write_csv(t1, os.path.join(output_dir, "table_all_runs.csv"))

    run_names = list(run_data_map.keys())
    t2 = table_project_breakdown_multi(run_data_map, meta)
    print_project_breakdown_multi(t2, run_names, "Latest All-Runs Project Breakdown")
    write_csv(t2, os.path.join(output_dir, "table_project_runs.csv"))

    plot_run_summary(t1, os.path.join(output_dir, "fig_all_runs_solved.png"), "Latest Runs: Solve Count Comparison")
    plot_project_breakdown_multi(
        t2,
        run_names,
        os.path.join(output_dir, "fig_project_runs.png"),
        "Latest Runs: Solve Rate by Project",
    )

    summary = {
        "run_summary": t1,
        "project_breakdown": t2,
        "files": run_paths,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="KCC2026 Paper Analysis")
    parser.add_argument("--one_shot", type=str, default="")
    parser.add_argument("--blind_retry", type=str, default="")
    parser.add_argument("--error_aware", type=str, default="")
    parser.add_argument("--deepseek_one_shot", type=str, default="")
    parser.add_argument("--deepseek_error_aware", type=str, default="")
    parser.add_argument("--meta", type=str, default="Results/4/4.defects4j.PlanAgent.json")
    parser.add_argument("--output_dir", type=str, default="paper_figures")
    args = parser.parse_args()

    os_path = args.one_shot or find_latest("Results/iterative_one_shot_defects4j_*.json")
    br_path = args.blind_retry or find_latest("Results/iterative_blind_retry_defects4j_*.json")
    ea_path = args.error_aware or find_latest("Results/iterative_error_aware_defects4j_*.json")

    run_standard_analysis(os_path, br_path, ea_path, args.meta, args.output_dir)

    if args.deepseek_one_shot and args.deepseek_error_aware:
        run_two_strategy_analysis(
            args.deepseek_one_shot,
            args.deepseek_error_aware,
            args.meta,
            os.path.join(args.output_dir, "deepseek_latest"),
            "DeepSeek Latest",
        )
        run_combined_latest_analysis(
            {
                "qwen_one_shot": os_path,
                "qwen_blind_retry": br_path,
                "qwen_error_aware": ea_path,
                "deepseek_one_shot": args.deepseek_one_shot,
                "deepseek_error_aware": args.deepseek_error_aware,
            },
            args.meta,
            os.path.join(args.output_dir, "latest_all"),
        )


if __name__ == "__main__":
    main()
