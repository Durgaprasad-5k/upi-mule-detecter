"""
Phase 4: Evaluation & Optimization
====================================
Combines results from Phase 2 (cycle detection) and Phase 3 (velocity detection)
to produce a ranked list of suspected mule accounts, then evaluates detection
quality against the injected ground truth.

Metrics:
    - Precision = TP / (TP + FP)  — "Of the accounts we flagged, how many are real mules?"
    - Recall    = TP / (TP + FN)  — "Of all real mules, how many did we catch?"
    - F1-Score  = 2 × (P × R) / (P + R) — Harmonic mean (target > 0.85)

Ranking Algorithm:
    risk_score = w_cycle × I(cycle_detected) + w_star × I(star_detected)
    where I(·) is the indicator function. Accounts detected by BOTH methods
    are ranked highest (risk_score = 1.0), then single-detection (0.5).

Performance (v2):
    - Vectorized ranked list construction using pandas set operations
      instead of row-by-row append loop
"""

import json
import pandas as pd
from config import GROUND_TRUTH_FILE, RESULTS_FILE, DATA_DIR
import os


def load_ground_truth(filepath: str = GROUND_TRUTH_FILE) -> dict:
    """Load the ground-truth mule account labels from Phase 1."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_metrics(
    detected: set,
    ground_truth_set: set
) -> dict:
    """
    Compute precision, recall, and F1-score for binary classification.

    Confusion Matrix:
                      Actual Mule    Actual Legit
        Flagged        TP              FP
        Not Flagged    FN              TN

    Note: We focus on precision/recall (not accuracy) because the dataset
    is heavily imbalanced — only ~130/10,000 accounts are mules (~1.3%).
    Accuracy would be misleadingly high (>98%) even with zero detections.
    """
    tp = len(detected & ground_truth_set)
    fp = len(detected - ground_truth_set)
    fn = len(ground_truth_set - detected)
    tn = 10_000 - tp - fp - fn  # Approximate (assumes 10K total accounts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def build_ranked_list(
    cycle_detected: set,
    star_detected: set,
    ground_truth: dict
) -> pd.DataFrame:
    """
    Build a ranked list of suspected mule accounts with risk scores.

    Scoring:
        - Both detections: risk_score = 1.0 (highest confidence)
        - Star only:       risk_score = 0.5
        - Cycle only:      risk_score = 0.5
        - Source column indicates which detector(s) flagged the account

    Performance (v2): Vectorized construction using pandas set operations
    instead of row-by-row append loop.
    """
    all_detected = sorted(cycle_detected | star_detected)
    gt_set = set(ground_truth['all_mules'])
    gt_star_set = set(ground_truth.get('star_mules', []))
    gt_cycle_set = set(ground_truth.get('cycle_mules', []))

    if not all_detected:
        return pd.DataFrame(columns=['Account', 'RiskScore', 'DetectionSource', 'GroundTruth', 'PatternType'])

    # Vectorized: build boolean arrays for set membership
    accounts = pd.Series(all_detected)
    in_cycle = accounts.isin(cycle_detected)
    in_star = accounts.isin(star_detected)
    is_mule = accounts.isin(gt_set)
    is_star_gt = accounts.isin(gt_star_set)
    is_cycle_gt = accounts.isin(gt_cycle_set)

    # Compute risk scores vectorized
    risk_scores = in_cycle.astype(float) * 0.5 + in_star.astype(float) * 0.5

    # Compute detection source vectorized
    source = pd.Series('STAR', index=accounts.index)
    source[in_cycle & ~in_star] = 'CYCLE'
    source[in_cycle & in_star] = 'BOTH'

    # Compute pattern type vectorized
    pattern = pd.Series('UNKNOWN', index=accounts.index)
    pattern[is_star_gt] = 'STAR (Pattern A)'
    pattern[is_cycle_gt] = 'CYCLE (Pattern B)'
    pattern[~is_mule] = 'FALSE POSITIVE'

    # Compute ground truth label vectorized
    truth = pd.Series('LEGIT', index=accounts.index)
    truth[is_mule] = 'MULE'

    df = pd.DataFrame({
        'Account': accounts.values,
        'RiskScore': risk_scores.values,
        'DetectionSource': source.values,
        'GroundTruth': truth.values,
        'PatternType': pattern.values,
    })
    df = df.sort_values('RiskScore', ascending=False).reset_index(drop=True)
    return df


def evaluate_and_report(
    cycle_detected: set,
    star_detected: set,
    ground_truth: dict,
    phase2_time: float,
    phase3_time: float
) -> dict:
    """
    Full evaluation pipeline: metrics, ranked list, and performance report.

    Returns:
        Dictionary containing all metrics and the ranked DataFrame.
    """
    all_detected = cycle_detected | star_detected
    gt_all = set(ground_truth['all_mules'])
    gt_star = set(ground_truth['star_mules'])
    gt_cycle = set(ground_truth['cycle_mules'])

    # ── Overall Metrics ──
    overall = compute_metrics(all_detected, gt_all)

    # ── Per-Pattern Metrics ──
    star_metrics = compute_metrics(star_detected, gt_star)
    cycle_metrics = compute_metrics(cycle_detected, gt_cycle)

    # ── Ranked List ──
    ranked_df = build_ranked_list(cycle_detected, star_detected, ground_truth)

    # ── Performance ──
    total_time = phase2_time + phase3_time

    # ══════════════════════════════════════════════════════════════════════
    # Print Report
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)

    # Overall metrics
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  OVERALL DETECTION METRICS                      │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  True Positives  : {overall['tp']:>5}                        │")
    print(f"  │  False Positives : {overall['fp']:>5}                        │")
    print(f"  │  False Negatives : {overall['fn']:>5}                        │")
    print(f"  │                                                 │")
    print(f"  │  Precision       : {overall['precision']:>8.4f}                   │")
    print(f"  │  Recall          : {overall['recall']:>8.4f}                   │")
    f1_status = "✓ PASS" if overall['f1_score'] > 0.85 else "✗ FAIL"
    print(f"  │  F1-Score        : {overall['f1_score']:>8.4f}  ({f1_status})      │")
    print(f"  └─────────────────────────────────────────────────┘")

    # Per-pattern breakdown
    print(f"\n  Per-Pattern Breakdown:")
    print(f"  {'─' * 50}")
    print(f"  Pattern A (Star):  P={star_metrics['precision']:.4f}  "
          f"R={star_metrics['recall']:.4f}  "
          f"F1={star_metrics['f1_score']:.4f}")
    print(f"  Pattern B (Cycle): P={cycle_metrics['precision']:.4f}  "
          f"R={cycle_metrics['recall']:.4f}  "
          f"F1={cycle_metrics['f1_score']:.4f}")

    # Performance
    print(f"\n  Performance:")
    print(f"  {'─' * 50}")
    print(f"  Phase 2 (Cycle Detection)    : {phase2_time:.3f}s")
    print(f"  Phase 3 (Velocity Detection) : {phase3_time:.3f}s")
    perf_status = "✓ PASS" if total_time < 2.0 else "✗ FAIL"
    print(f"  Combined (target < 2.0s)     : {total_time:.3f}s  ({perf_status})")

    # Detection summary
    print(f"\n  Detection Summary:")
    print(f"  {'─' * 50}")
    print(f"  Accounts flagged by CYCLE only : "
          f"{len(cycle_detected - star_detected)}")
    print(f"  Accounts flagged by STAR only  : "
          f"{len(star_detected - cycle_detected)}")
    print(f"  Accounts flagged by BOTH       : "
          f"{len(cycle_detected & star_detected)}")
    print(f"  Total unique flagged           : {len(all_detected)}")
    print(f"  Ground truth mule count        : {len(gt_all)}")

    # Ranked list preview
    print(f"\n  Top 20 Suspected Mule Accounts:")
    print(f"  {'─' * 70}")
    print(ranked_df.head(20).to_string(index=False))

    # Missed accounts (False Negatives)
    missed = gt_all - all_detected
    if missed:
        print(f"\n  ⚠ Missed accounts ({len(missed)}):")
        for acc in sorted(missed)[:10]:
            pattern = "Star" if acc in gt_star else "Cycle"
            print(f"    {acc} (Pattern: {pattern})")
        if len(missed) > 10:
            print(f"    ... and {len(missed) - 10} more")

    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    ranked_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  ✓ Saved ranked results → {RESULTS_FILE}")

    return {
        'overall': overall,
        'star_metrics': star_metrics,
        'cycle_metrics': cycle_metrics,
        'ranked_df': ranked_df,
        'total_time': total_time,
        'phase2_time': phase2_time,
        'phase3_time': phase3_time,
    }


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 4 requires results from Phase 2 and Phase 3.")
    print("Run main.py for the full pipeline.")
