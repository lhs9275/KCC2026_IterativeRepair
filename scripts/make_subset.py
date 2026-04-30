"""Generate a random subset of test-failure-subgroup bug IDs for ablation."""
import argparse
import json
import random
from pathlib import Path

TA_RESULTS = Path(__file__).resolve().parent.parent / \
    "Results/iterative_error_aware_defects4j_topk_20260418.json"


def main():
    parser = argparse.ArgumentParser(description="Sample test-fail subgroup bug IDs")
    parser.add_argument("--n", type=int, default=60, help="Number of bugs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ta_results", type=str, default=str(TA_RESULTS),
                        help="Path to TA (error_aware) seed=42 results JSON")
    parser.add_argument("--output", type=str, default="ablation_subset.json",
                        help="Output JSON path")
    args = parser.parse_args()

    with open(args.ta_results) as f:
        results = json.load(f)

    # Test-fail subgroup: iter 0 (iterations[0]) fail_reason == "test_fail"
    test_fail_ids = [
        bug_id for bug_id, v in results.items()
        if v.get("iterations") and v["iterations"][0].get("fail_reason") == "test_fail"
    ]
    print(f"Test-failure subgroup size: {len(test_fail_ids)}")

    random.seed(args.seed)
    subset = random.sample(test_fail_ids, min(args.n, len(test_fail_ids)))
    subset_sorted = sorted(subset, key=lambda x: int(x))

    with open(args.output, "w") as f:
        json.dump(subset_sorted, f, indent=2)
    print(f"Wrote {len(subset_sorted)} bug IDs to {args.output}")


if __name__ == "__main__":
    main()
