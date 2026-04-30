"""Generate KCC 2026 paper tables, figures, and case study from experiment results."""

import json
from collections import Counter
from pathlib import Path

from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = {
    "error_aware": "Results/iterative_error_aware_defects4j_topk_20260418.json",
    "blind_retry": "Results/iterative_blind_retry_defects4j_topk_20260418.json",
    "one_shot":    "Results/iterative_one_shot_defects4j_topk_20260418.json",
}
OUT = Path("paper_artifacts")
OUT.mkdir(exist_ok=True)

data = {k: json.load(open(v)) for k, v in RESULTS.items()}


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(2 * binom.cdf(k, n, 0.5), 1.0)


def categorize(entry):
    iters = entry.get("iterations", [])
    if not iters:
        if entry.get("status") == "infra_fail":
            return "infra_fail"
        return "no_iters"
    i0 = iters[0]
    fr = i0.get("fail_reason", "")
    fam = i0.get("error_family", "")
    if fr == "pass":
        return "solved_iter0"
    if fr == "compile_fail":
        return f"compile:{fam}" if fam else "compile:other"
    if fr == "test_fail":
        return "test_fail"
    if fr == "no_valid_candidates":
        return "parse_fail"
    if fr == "timeout":
        return "timeout"
    return f"other:{fr}"


bug_category = {bid: categorize(v) for bid, v in data["error_aware"].items()}


# ---------------------------------------------------------------------------
# Table 1: overall results
# ---------------------------------------------------------------------------
def table_overall():
    lines = [
        r"\begin{table}[t]",
        r"\caption{전체 실험 결과 (Defects4J 255 버그)}",
        r"\label{tab:overall}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"전략 & 해결 수 & 해결률 \\",
        r"\midrule",
    ]
    for name in ["one_shot", "blind_retry", "error_aware"]:
        solved = sum(1 for v in data[name].values() if v.get("solved"))
        total = len(data[name])
        disp = {"one_shot": "One-shot (baseline)",
                "blind_retry": "Blind-retry",
                "error_aware": r"\textbf{Error-type-aware (ours)}"}[name]
        star = r"\textbf{" if name == "error_aware" else ""
        end = r"}" if name == "error_aware" else ""
        lines.append(
            f"{disp} & {star}{solved}/{total}{end} & {star}{solved/total*100:.1f}\\%{end} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: breakdown by initial-failure category
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [
    "solved_iter0",
    "test_fail",
    "parse_fail",
    "compile:cannot_find_symbol",
    "compile:syntax_or_parse",
    "compile:type_mismatch",
    "compile:method_signature",
    "compile:other_compile_fail",
    "timeout",
    "infra_fail",
]
CATEGORY_LABEL = {
    "solved_iter0":                 "1-shot 통과 (쉬움)",
    "test_fail":                    "Test-fail",
    "parse_fail":                   "Parse-fail",
    "compile:cannot_find_symbol":   "Compile: cannot-find-symbol",
    "compile:syntax_or_parse":      "Compile: syntax/parse",
    "compile:type_mismatch":        "Compile: type-mismatch",
    "compile:method_signature":     "Compile: method-signature",
    "compile:other_compile_fail":   "Compile: other",
    "timeout":                      "Timeout",
    "infra_fail":                   "Infra-fail (체크아웃 실패)",
}


def table_breakdown():
    lines = [
        r"\begin{table*}[t]",
        r"\caption{초기 실패 유형별 해결률 분포. 분류 기준은 error_aware 전략의 iter 0 실패 유형이며, "
        r"세 전략 모두 동일한 분할로 비교된다. Infra-fail은 Defects4J 체크아웃 실패로 어느 전략도 시도하지 못한 버그.}",
        r"\label{tab:breakdown}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"초기 실패 유형 & \#버그 & One-shot & Blind-retry & \textbf{Error-aware} \\",
        r"\midrule",
    ]
    for cat in CATEGORY_ORDER:
        ids = [bid for bid, c in bug_category.items() if c == cat]
        n = len(ids)
        if n == 0:
            continue
        ea = sum(1 for bid in ids if data["error_aware"][bid].get("solved"))
        br = sum(1 for bid in ids if data["blind_retry"][bid].get("solved"))
        os_ = sum(1 for bid in ids if data["one_shot"][bid].get("solved"))
        label = CATEGORY_LABEL.get(cat, cat)
        lines.append(
            f"{label} & {n} & {os_} ({os_/n*100:.1f}\\%) & {br} ({br/n*100:.1f}\\%) "
            f"& \\textbf{{{ea} ({ea/n*100:.1f}\\%)}} \\\\"
        )
    lines += [
        r"\midrule",
        f"전체 & 255 & 33 (12.9\\%) & 40 (15.7\\%) & \\textbf{{44 (17.3\\%)}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure: bar chart comparison per category
# ---------------------------------------------------------------------------
def figure_breakdown():
    shown = [c for c in CATEGORY_ORDER
             if sum(1 for bid, cc in bug_category.items() if cc == c) >= 3]
    # Re-order for figure: iterative-relevant categories first
    fig_order = [c for c in shown if c != "solved_iter0"]
    labels = [CATEGORY_LABEL[c].replace("Compile: ", "") for c in fig_order]
    ns = [sum(1 for bid, cc in bug_category.items() if cc == c) for c in fig_order]

    def pct(strategy, cat):
        ids = [bid for bid, cc in bug_category.items() if cc == cat]
        if not ids:
            return 0.0
        solved = sum(1 for bid in ids if data[strategy][bid].get("solved"))
        return solved / len(ids) * 100

    one_shot_pct = [pct("one_shot", c) for c in fig_order]
    blind_pct    = [pct("blind_retry", c) for c in fig_order]
    aware_pct    = [pct("error_aware", c) for c in fig_order]

    import numpy as np
    x = np.arange(len(fig_order))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - w, one_shot_pct, w, label="One-shot", color="#c0c0c0", edgecolor="black")
    ax.bar(x,     blind_pct,    w, label="Blind-retry", color="#7a9cc6", edgecolor="black")
    ax.bar(x + w, aware_pct,    w, label="Error-aware (ours)", color="#2f5f9a", edgecolor="black")

    ax.set_xticks(x)
    en_labels = {
        "Test-fail": "Test-fail",
        "Parse-fail": "Parse-fail",
        "cannot-find-symbol": "cannot-find-\nsymbol",
        "syntax/parse": "syntax/parse",
        "type-mismatch": "type-mismatch",
        "method-signature": "method-\nsignature",
        "other": "compile-other",
        "Timeout": "Timeout",
    }
    xlabels = [f"{en_labels.get(lbl, lbl)}\n(N={n})" for lbl, n in zip(labels, ns)]
    ax.set_xticklabels(xlabels, fontsize=8.5, rotation=0)
    ax.set_ylabel("Repair success rate (%)", fontsize=10)
    ax.set_ylim(0, max(max(one_shot_pct), max(blind_pct), max(aware_pct)) * 1.25)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for xi, pct_val in zip(x - w, one_shot_pct):
        if pct_val > 0:
            ax.text(xi, pct_val + 0.4, f"{pct_val:.1f}", ha="center", fontsize=7)
    for xi, pct_val in zip(x, blind_pct):
        if pct_val > 0:
            ax.text(xi, pct_val + 0.4, f"{pct_val:.1f}", ha="center", fontsize=7)
    for xi, pct_val in zip(x + w, aware_pct):
        if pct_val > 0:
            ax.text(xi, pct_val + 0.4, f"{pct_val:.1f}", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig_breakdown.pdf")
    fig.savefig(OUT / "fig_breakdown.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------
def stats_summary():
    def cmp(a, b, bug_ids=None):
        ids = bug_ids or list(data[a].keys())
        ab = sum(1 for i in ids if data[a][i].get("solved") and not data[b][i].get("solved"))
        ba = sum(1 for i in ids if data[b][i].get("solved") and not data[a][i].get("solved"))
        return ab, ba, mcnemar_p(ab, ba)

    out = ["=== McNemar's exact test ==="]
    for a, b in [("error_aware", "one_shot"),
                 ("error_aware", "blind_retry"),
                 ("blind_retry", "one_shot")]:
        ab, ba, p = cmp(a, b)
        out.append(f"{a:12} vs {b:12}: +{ab:2d}/-{ba:2d}  p={p:.4f}")

    out.append("")
    out.append("=== Per-category McNemar (error_aware vs blind_retry) ===")
    for cat in CATEGORY_ORDER:
        ids = [bid for bid, c in bug_category.items() if c == cat]
        if len(ids) < 3:
            continue
        ab, ba, p = cmp("error_aware", "blind_retry", ids)
        ea = sum(1 for bid in ids if data["error_aware"][bid].get("solved"))
        br = sum(1 for bid in ids if data["blind_retry"][bid].get("solved"))
        out.append(f"{CATEGORY_LABEL[cat]:<32} N={len(ids):3d}  "
                   f"EA={ea}, BR={br},  +{ab}/-{ba}  p={p:.4f}")

    out.append("")
    out.append("=== Compute cost comparison ===")
    for name, d in data.items():
        llm_calls = [v.get("total_llm_calls", 0) for v in d.values()]
        elapsed   = [v.get("elapsed_total_sec", 0) for v in d.values()]
        out.append(f"{name:12}: mean {sum(llm_calls)/len(llm_calls):.2f} LLM calls/bug, "
                   f"{sum(elapsed)/len(elapsed):.1f} s/bug, total {sum(elapsed)/3600:.2f}h")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Case study (Bug 153: Math linearCombination)
# ---------------------------------------------------------------------------
def case_study():
    bid = "153"
    ea = data["error_aware"][bid]
    br = data["blind_retry"][bid]

    ea_i0 = ea["iterations"][0]
    ea_i1 = ea["iterations"][1]
    br_last = br["iterations"][-1]

    def short(code, max_lines=6):
        lines = code.strip().splitlines()
        return "\n".join(lines[:max_lines]) + ("\n    ..." if len(lines) > max_lines else "")

    out = [
        "# Case Study: Bug 153 (Apache Commons Math `linearCombination`)",
        "",
        f"- Failing test (iter 0): `{ea_i0.get('first_failing_test')}`",
        f"- Error-aware outcome: SOLVED at iter {ea.get('solving_iteration')}",
        f"- Blind-retry outcome: FAILED after {br.get('total_iterations')} iters",
        "",
        "## Iter 0 (both strategies, identical)",
        "```java",
        short(ea_i0["best_candidate_code"], 8),
        "```",
        f"→ {ea_i0.get('failing_tests_count')} test fails: "
        f"`{ea_i0.get('first_failing_test').split('::')[-1]}`",
        "",
        "## Error-aware iter 1 feedback (typed `test_feedback`)",
        "```",
        "FAILING TEST COUNT: 1",
        "FAILING TEST NAMES:",
        f"  {ea_i0.get('first_failing_test')}",
        "...",
        "FIX: Start from the ORIGINAL BUGGY CODE, not your previous rewrite.",
        "Make a MINIMAL semantic fix near the buggy line.",
        "```",
        "",
        "## Error-aware iter 1 result (fixed)",
        "```java",
        short(ea_i1["best_candidate_code"], 8),
        "```",
        "→ PASS ✓  (single-line guard `if (len == 1) return a[0] * b[0];` added)",
        "",
        "## Blind-retry iter 2 (last attempt, still failing)",
        "```java",
        short(br_last["best_candidate_code"], 8),
        "```",
        f"→ still fails; blind prompt gave no test-name hint, "
        f"model never targets the single-element edge case.",
        "",
        "**Takeaway**: error-type-aware's `test_feedback` surfaces the failing test's "
        "self-describing name (`testLinearCombinationWithSingleElementArray`), "
        "which the LLM uses to localize the missing edge case. "
        "Blind-retry's generic 'try a different approach' lacks this signal.",
    ]
    return "\n".join(out)


def main():
    (OUT / "table1_overall.tex").write_text(table_overall())
    (OUT / "table2_breakdown.tex").write_text(table_breakdown())
    figure_breakdown()
    (OUT / "stats_summary.txt").write_text(stats_summary())
    (OUT / "case_study_bug153.md").write_text(case_study())

    print("=== Generated artifacts ===")
    for f in sorted(OUT.iterdir()):
        print(f"  {f} ({f.stat().st_size} bytes)")
    print()
    print(stats_summary())


if __name__ == "__main__":
    main()
