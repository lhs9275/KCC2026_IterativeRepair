#!/usr/bin/env python3
"""
Analyze iterative repair results and generate paper tables/figures.

Usage:
    python analyze_results.py --results Results/iterative_*.json
    python analyze_results.py --results r1.json r2.json r3.json --compare
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_single_result(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics for one experiment run."""
    total = 0
    solved = 0
    solved_at_iter = Counter()
    error_families = Counter()
    error_family_recovered = Counter()
    total_llm_calls = 0
    total_candidates = 0
    fail_reasons = Counter()

    for bug_id, r in results.items():
        if r.get("status") in ("infra_fail", "error"):
            continue
        total += 1
        total_llm_calls += r.get("total_llm_calls", 0)
        total_candidates += r.get("total_candidates_generated", 0)

        if r.get("solved"):
            solved += 1
            solving_iter = r.get("solving_iteration", 1)
            solved_at_iter[solving_iter] += 1

            # Check if it was recovered by retry (solved after iter 1)
            if solving_iter > 1:
                iterations = r.get("iterations", [])
                if len(iterations) >= 2:
                    first_error = iterations[0].get("error_family", "")
                    if first_error:
                        error_family_recovered[first_error] += 1
        else:
            fail_reasons[r.get("final_fail_reason", "unknown")] += 1

        # Track initial error families
        iterations = r.get("iterations", [])
        if iterations:
            first_iter = iterations[0]
            first_family = first_iter.get("error_family", "")
            first_reason = first_iter.get("fail_reason", "")
            if first_family:
                error_families[first_family] += 1
            elif first_reason == "compile_fail":
                error_families["unknown_compile"] += 1
            elif first_reason != "pass":
                error_families[first_reason] += 1

    # Error family recovery rates
    recovery_rates = {}
    for family, count in error_families.items():
        recovered = error_family_recovered.get(family, 0)
        recovery_rates[family] = {
            "total": count,
            "recovered_by_retry": recovered,
            "rate": round(recovered / max(count, 1), 3),
        }

    return {
        "total_bugs": total,
        "solved_bugs": solved,
        "solve_rate": round(solved / max(total, 1), 4),
        "solved_at_iteration": dict(sorted(solved_at_iter.items())),
        "avg_llm_calls": round(total_llm_calls / max(total, 1), 2),
        "avg_candidates": round(total_candidates / max(total, 1), 2),
        "error_family_recovery": recovery_rates,
        "unsolved_fail_reasons": dict(fail_reasons.most_common()),
    }


def print_summary(name: str, stats: Dict[str, Any]):
    """Pretty-print summary statistics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total bugs:  {stats['total_bugs']}")
    print(f"  Solved:      {stats['solved_bugs']} ({stats['solve_rate']*100:.1f}%)")
    print(f"  Avg LLM calls/bug: {stats['avg_llm_calls']}")
    print()

    if stats["solved_at_iteration"]:
        print("  Solved by iteration:")
        for it, cnt in sorted(stats["solved_at_iteration"].items()):
            print(f"    Iter {it}: {cnt}")
        print()

    if stats["error_family_recovery"]:
        print("  Error family recovery rates:")
        print(f"    {'Family':<25} {'Total':>6} {'Recovered':>10} {'Rate':>8}")
        print(f"    {'-'*25} {'-'*6} {'-'*10} {'-'*8}")
        for family, info in sorted(stats["error_family_recovery"].items(),
                                    key=lambda x: -x[1]["total"]):
            print(f"    {family:<25} {info['total']:>6} {info['recovered_by_retry']:>10} "
                  f"{info['rate']*100:>7.1f}%")
        print()

    if stats["unsolved_fail_reasons"]:
        print("  Unsolved fail reasons:")
        for reason, cnt in stats["unsolved_fail_reasons"].items():
            print(f"    {reason}: {cnt}")
    print()


# ---------------------------------------------------------------------------
# Comparison table (for paper Table 1)
# ---------------------------------------------------------------------------

def print_comparison_table(named_stats: List[tuple]):
    """Print comparison table for ablation study."""
    print(f"\n{'='*80}")
    print("  Table 1: Comparison of Repair Strategies")
    print(f"{'='*80}")
    header = f"  {'Strategy':<20} {'Solved':>8} {'Rate':>8} {'Avg Iter':>10} {'Avg LLM':>10}"
    print(header)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for name, stats in named_stats:
        # Average iterations to solve
        solved_at = stats["solved_at_iteration"]
        if solved_at:
            avg_iter = sum(int(k)*v for k, v in solved_at.items()) / max(sum(solved_at.values()), 1)
        else:
            avg_iter = 0

        print(f"  {name:<20} {stats['solved_bugs']:>8} "
              f"{stats['solve_rate']*100:>7.1f}% {avg_iter:>10.2f} {stats['avg_llm_calls']:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_iteration_distribution(named_stats: List[tuple], output_path: str):
    """Figure 1: Stacked bar chart of bugs solved per iteration."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = [name for name, _ in named_stats]
    max_iter = 3

    bottoms = [0] * len(strategies)
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    labels_added = set()

    for it in range(1, max_iter + 1):
        values = []
        for _, stats in named_stats:
            values.append(stats["solved_at_iteration"].get(str(it), stats["solved_at_iteration"].get(it, 0)))

        label = f"Iter {it}" if it not in labels_added else None
        labels_added.add(it)
        ax.bar(strategies, values, bottom=bottoms, label=f"Iter {it}",
               color=colors[it-1] if it <= len(colors) else colors[-1])
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Add unsolved
    unsolved = []
    for _, stats in named_stats:
        unsolved.append(stats["total_bugs"] - stats["solved_bugs"])
    ax.bar(strategies, unsolved, bottom=bottoms, label="Unsolved", color="#bdc3c7", alpha=0.6)

    ax.set_ylabel("Number of Bugs")
    ax.set_title("Bug Resolution by Iteration")
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure: {output_path}")
    plt.close()


def plot_error_family_recovery(stats: Dict[str, Any], output_path: str):
    """Figure 2: Error family recovery rate bar chart."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    recovery = stats.get("error_family_recovery", {})
    if not recovery:
        return

    families = sorted(recovery.keys(), key=lambda x: -recovery[x]["total"])
    totals = [recovery[f]["total"] for f in families]
    recovered = [recovery[f]["recovered_by_retry"] for f in families]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(families))
    width = 0.35
    ax.bar([i - width/2 for i in x], totals, width, label="Total initial failures", color="#3498db")
    ax.bar([i + width/2 for i in x], recovered, width, label="Recovered by retry", color="#2ecc71")

    ax.set_xlabel("Error Family")
    ax.set_ylabel("Number of Bugs")
    ax.set_title("Error-Aware Recovery by Error Family")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze iterative repair results")
    parser.add_argument("--results", nargs="+", required=True, help="Result JSON files")
    parser.add_argument("--names", nargs="+", default=[], help="Display names for each result file")
    parser.add_argument("--compare", action="store_true", help="Print comparison table")
    parser.add_argument("--output_dir", type=str, default="Results/figures",
                        help="Directory for output figures")
    parser.add_argument("--save_summary", type=str, default="",
                        help="Save summary statistics to JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for i, result_path in enumerate(args.results):
        name = args.names[i] if i < len(args.names) else os.path.basename(result_path).replace(".json", "")
        with open(result_path) as f:
            results = json.load(f)
        stats = analyze_single_result(results)
        print_summary(name, stats)
        all_stats.append((name, stats))

    if args.compare and len(all_stats) > 1:
        print_comparison_table(all_stats)
        plot_iteration_distribution(all_stats, os.path.join(args.output_dir, "iteration_distribution.png"))

    # Plot error recovery for the last (presumably error_aware) result
    if all_stats:
        last_name, last_stats = all_stats[-1]
        plot_error_family_recovery(last_stats, os.path.join(args.output_dir, "error_recovery.png"))

    # Save summary
    if args.save_summary:
        summary = {name: stats for name, stats in all_stats}
        with open(args.save_summary, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to {args.save_summary}")


if __name__ == "__main__":
    main()
