"""
UPI Mule Account Detection — CLI Orchestrator
===============================================
Runs all four phases sequentially and prints the complete report.

Usage:
    python main.py
"""

import time
import sys

# Ensure UTF-8 output on Windows to prevent UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Synthetic Data Generation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1():
    from phase1_data_generation import generate_dataset, save_dataset

    print("\n" + "=" * 65)
    print("  PHASE 1: Synthetic Data Generation")
    print("=" * 65)

    t0 = time.time()
    df, ground_truth = generate_dataset()
    save_dataset(df, ground_truth)
    t1 = time.time()

    print(f"\n  Phase 1 completed in {t1 - t0:.3f}s")
    print(f"  Dataset: {len(df):,} transactions across "
          f"{len(set(df['SourceAccount']) | set(df['DestinationAccount'])):,} accounts")

    # Pre-process Timestamp for Phase 2 & 3 to save time
    import pandas as pd
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    return df, ground_truth


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Cycle Detection
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(df):
    from phase2_neo4j_cycles import detect_cycles

    print("\n" + "=" * 65)
    print("  PHASE 2: Cycle Detection (Neo4j / NetworkX Fallback)")
    print("=" * 65)

    cycle_detected, phase2_time = detect_cycles(df)

    print(f"\n  Phase 2 completed in {phase2_time:.3f}s")
    print(f"  Detected {len(cycle_detected)} cycle participant accounts")

    return cycle_detected, phase2_time


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Velocity Detection
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3(df):
    from phase3_networkx_velocity import detect_star_patterns, print_graph_stats

    print("\n" + "=" * 65)
    print("  PHASE 3: Velocity Detection (Star Topology)")
    print("=" * 65)

    star_detected, phase3_time, G = detect_star_patterns(df)
    print_graph_stats(G)

    print(f"\n  Phase 3 completed in {phase3_time:.3f}s")
    print(f"  Detected {len(star_detected)} star-topology mule accounts")

    return star_detected, phase3_time


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase4(cycle_detected, star_detected, ground_truth, phase2_time, phase3_time):
    from phase4_evaluation import evaluate_and_report

    print("\n" + "=" * 65)
    print("  PHASE 4: Evaluation & Optimization")
    print("=" * 65)

    results = evaluate_and_report(
        cycle_detected, star_detected, ground_truth,
        phase2_time, phase3_time
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  UPI Mule Account Detection — Graph Database Analyzer       ║")
    print("║  10,000 accounts · Star & Cycle topology detection          ║")
    print("╚" + "═" * 63 + "╝")

    total_start = time.time()

    # Run all phases
    df, ground_truth = run_phase1()
    
    # Run phases sequentially to avoid GIL contention which inflates wall-clock time
    cycle_detected, phase2_time = run_phase2(df)
    star_detected, phase3_time = run_phase3(df)
        
    results = run_phase4(
        cycle_detected, star_detected, ground_truth,
        phase2_time, phase3_time
    )

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 65)
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed:.3f}s")
    print("=" * 65)

    f1 = results['overall']['f1_score']
    combined = results['total_time']

    if f1 > 0.85 and combined < 2.0:
        print("  ✅ ALL TARGETS MET")
    else:
        if f1 <= 0.85:
            print(f"  ❌ F1-Score {f1:.4f} < 0.85 target")
        if combined >= 2.0:
            print(f"  ❌ Combined time {combined:.3f}s > 2.0s target")

    return results


if __name__ == "__main__":
    results = main()
