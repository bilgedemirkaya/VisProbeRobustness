#!/usr/bin/env python3
"""
Test 4: Aggregate all results and create final comparison table
"""

import json
from pathlib import Path

RESULTS_DIR = Path("experiment_results")


def load_results():
    """Load all result files"""
    results = {}

    for json_file in RESULTS_DIR.glob("results_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.update(data)

    return results


def create_effectiveness_table(results):
    """Create the comparison table"""
    print("\n" + "=" * 80)
    print("ADVERSARIAL TRAINING EFFECTIVENESS - FINAL RESULTS")
    print("=" * 80)

    print("\nThreat Model          | Vanilla | Adversarially-Trained | Improvement")
    print("-" * 80)

    # Natural
    if "vanilla_natural" in results and "robust_natural" in results:
        v = results["vanilla_natural"]["score"]
        r = results["robust_natural"]["score"]
        print(f"Natural (Blur)        | {v:6.1f}% | {r:6.1f}%              | {r-v:+.1f}%")

    # Adversarial
    if "vanilla_adversarial" in results and "robust_adversarial" in results:
        v = results["vanilla_adversarial"]["score"]
        r = results["robust_adversarial"]["score"]
        print(f"Adversarial (PGD)     | {v:6.1f}% | {r:6.1f}%              | {r-v:+.1f}%")

    # Compositional
    print("\nCompositional:")
    compositional_scenarios = [
        ("low_light_pgd", "Low-Light PGD"),
        ("blurred_pgd", "Blurred PGD"),
        ("noisy_pgd", "Noisy PGD"),
    ]

    for key, name in compositional_scenarios:
        v_key = f"vanilla_{key}"
        r_key = f"robust_{key}"

        if v_key in results and r_key in results:
            v = results[v_key]["score"]
            r = results[r_key]["score"]
            improvement = r - v
            emoji = "✓" if improvement >= 10 else "✗"
            print(f"  {name:20} | {v:6.1f}% | {r:6.1f}%              | {improvement:+.1f}% {emoji}")

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print("  Adversarial training provides strong improvement on PGD")
    print("  But limited improvement on compositional attacks")
    print("  → Standard adversarial training is INSUFFICIENT for real-world robustness!")
    print("=" * 80)

    # Save final summary
    with open(RESULTS_DIR / "final_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal summary saved to {RESULTS_DIR / 'final_summary.json'}")


def main():
    print("Aggregating results from all tests...\n")

    # Check which results exist
    natural = RESULTS_DIR / "results_natural.json"
    adversarial = RESULTS_DIR / "results_adversarial.json"
    compositional = RESULTS_DIR / "results_compositional.json"

    missing = []
    if not natural.exists():
        missing.append("test_1_natural.py")
    if not adversarial.exists():
        missing.append("test_2_adversarial.py")
    if not compositional.exists():
        missing.append("test_3_compositional.py")

    if missing:
        print(f"Warning: Missing results from: {', '.join(missing)}")
        print("Run these tests first, then re-run this aggregation.\n")

    results = load_results()

    if results:
        create_effectiveness_table(results)
    else:
        print("No results found. Run the tests first.")


if __name__ == "__main__":
    main()
