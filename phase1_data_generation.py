"""
Phase 1: Synthetic Data Generation
====================================
Generates a synthetic dataset of 10,000 UPI accounts and their transactions,
with intentionally injected mule patterns for ground-truth evaluation.

Graph Model:
    - Nodes (V): UPI accounts  |V| = 10,000
    - Edges (E): Transactions with attributes {amount, timestamp}
    - Directed graph: edges flow from sender → receiver

Injected Ground-Truth Patterns:
    - Pattern A (Star Topology): 30 hub accounts with rapid fan-out
    - Pattern B (Circular Loop): 25 multi-hop cycles (3–6 hops)

Optimization:
    All generation uses vectorized NumPy/Pandas operations:
    np.random.choice, pd.to_timedelta, pd.concat for batch construction.
    No nested for-loops in normal transaction generation.

Performance (v2):
    - Star/cycle injection: pre-allocate arrays, build DataFrame once
    - Replaced np.delete with boolean mask indexing (avoids O(n) copy)
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from config import (
    NUM_ACCOUNTS, NUM_NORMAL_TRANSACTIONS, NUM_STAR_MULES, NUM_CYCLE_GROUPS,
    STAR_MIN_DESTINATIONS, STAR_MAX_DESTINATIONS,
    CYCLE_MIN_LENGTH, CYCLE_MAX_LENGTH,
    DATA_DIR, TRANSACTIONS_FILE, GROUND_TRUTH_FILE, RANDOM_SEED
)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def generate_account_ids(n: int) -> np.ndarray:
    """Generate n unique UPI-style account IDs via vectorized string formatting."""
    return np.array([f"UPI_{i:05d}" for i in range(n)])


# ─── Normal Transaction Generation (Fully Vectorized) ─────────────────────────

def generate_normal_transactions(
    accounts: np.ndarray,
    n_transactions: int,
    start_date: datetime,
    end_date: datetime,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate background-noise transactions using fully vectorized operations.

    Distribution Rationale:
        - Amounts: Lognormal(μ=7.5, σ=1.2) models realistic UPI payments
          (median ≈ ₹1,800; long tail up to ₹50,000).
        - Timestamps: Uniform over 30-day period (simplification of real
          diurnal patterns — acceptable for mule detection benchmarking).
        - Source/Destination: Uniform random from the 10K-account pool.
          Self-loops are eliminated via vectorized rejection sampling.

    Complexity: O(n) — all operations are array-based.
    """
    # ── Vectorized source/destination selection ──
    sources = rng.choice(accounts, size=n_transactions)
    destinations = rng.choice(accounts, size=n_transactions)

    # Eliminate self-loops via vectorized rejection sampling
    mask = sources == destinations
    while mask.any():
        destinations[mask] = rng.choice(accounts, size=mask.sum())
        mask = sources == destinations

    # ── Lognormal amounts (clipped to realistic UPI range) ──
    amounts = np.clip(
        rng.lognormal(mean=7.5, sigma=1.2, size=n_transactions),
        50, 50_000
    ).round(2)

    # ── Uniform timestamps across 30-day window ──
    total_seconds = int((end_date - start_date).total_seconds())
    random_offsets = rng.integers(0, total_seconds, size=n_transactions)
    timestamps = pd.to_datetime(start_date) + pd.to_timedelta(random_offsets, unit='s')

    return pd.DataFrame({
        'SourceAccount': sources,
        'DestinationAccount': destinations,
        'Amount': amounts,
        'Timestamp': timestamps
    })


# ─── Pattern A: Star Topology Injection ───────────────────────────────────────

def inject_star_patterns(
    accounts: np.ndarray,
    n_stars: int,
    start_date: datetime,
    rng: np.random.Generator
) -> tuple[pd.DataFrame, set]:
    """
    Inject Pattern A (Star Topology) mule accounts.

    Topology Diagram:
        Victim ──(lump sum)──▸ Hub ──▸ Spoke₁
                                   ──▸ Spoke₂
                                   ──▸ Spoke₃
                                   ──▸ ...
                                   ──▸ Spokeₙ    (n ∈ [6, 12])

    Temporal Constraint:
        All outgoing edges from Hub occur within a 10-minute window
        immediately after the incoming lump sum.

    Amount Model:
        - Incoming: ₹50,000–₹200,000 (uniform)
        - Outgoing: Dirichlet-split of 85–95% of the incoming amount
          (realistic: mule keeps a small cut)

    Performance (v2):
        - Pre-allocate list with estimated capacity
        - Use boolean mask instead of np.delete for account exclusion

    Returns:
        (transactions_df, set_of_mule_hub_account_ids)
    """
    mule_hubs = set()
    # Pre-allocate with estimated capacity: ~(1 + avg_spokes) per star
    avg_spokes = (STAR_MIN_DESTINATIONS + STAR_MAX_DESTINATIONS) // 2
    all_sources = []
    all_destinations = []
    all_amounts = []
    all_timestamps = []

    # Select hub accounts (non-overlapping)
    hub_indices = rng.choice(len(accounts), size=n_stars, replace=False)
    hub_set = set(hub_indices)

    for idx in hub_indices:
        hub = accounts[idx]
        mule_hubs.add(hub)

        # Number of spokes (distinct destinations)
        n_spokes = int(rng.integers(STAR_MIN_DESTINATIONS, STAR_MAX_DESTINATIONS + 1))

        # Select spoke accounts using boolean mask (avoids O(n) np.delete copy)
        mask = np.ones(len(accounts), dtype=bool)
        mask[idx] = False
        available = accounts[mask]
        spokes = rng.choice(available, size=n_spokes, replace=False)

        # Random base timestamp within the 30-day period (with 2-day margin)
        base_offset = int(rng.integers(0, int(timedelta(days=28).total_seconds())))
        base_time = pd.Timestamp(start_date) + pd.Timedelta(seconds=base_offset)

        # ── Incoming lump sum to hub ──
        lump_sum = float(rng.uniform(50_000, 200_000))
        victim = rng.choice(available)
        all_sources.append(victim)
        all_destinations.append(hub)
        all_amounts.append(round(lump_sum, 2))
        all_timestamps.append(base_time)

        # ── Outgoing dispersals within 10-minute window ──
        dispersal_offsets = np.sort(rng.integers(30, 570, size=n_spokes))
        split_fractions = rng.dirichlet(np.ones(n_spokes))
        total_dispersed = lump_sum * float(rng.uniform(0.85, 0.95))
        dispersal_amounts = split_fractions * total_dispersed

        for spoke, offset, amount in zip(spokes, dispersal_offsets, dispersal_amounts):
            all_sources.append(hub)
            all_destinations.append(spoke)
            all_amounts.append(round(float(amount), 2))
            all_timestamps.append(base_time + pd.Timedelta(seconds=int(offset)))

    df = pd.DataFrame({
        'SourceAccount': all_sources,
        'DestinationAccount': all_destinations,
        'Amount': all_amounts,
        'Timestamp': all_timestamps
    })
    return df, mule_hubs


# ─── Pattern B: Circular Loop Injection ───────────────────────────────────────

def inject_circular_patterns(
    accounts: np.ndarray,
    n_cycles: int,
    start_date: datetime,
    used_accounts: set,
    rng: np.random.Generator
) -> tuple[pd.DataFrame, set]:
    """
    Inject Pattern B (Circular Loop) mule accounts.

    Topology Diagram (example with 4 hops):
        A ──▸ B ──▸ C ──▸ D ──▸ A     (simple directed cycle)

    Temporal Constraint:
        All edges in the cycle occur within a 5-minute window.
        This mimics rapid "layering" where funds are quickly rotated
        through multiple accounts to obscure their origin.

    Graph Theory:
        A simple directed cycle of length k requires exactly k edges
        forming a closed walk where each vertex is visited once.
        In the property graph, each edge carries a timestamp.

    Performance (v2):
        - Pre-allocate arrays, build DataFrame once at the end
        - Use set for used_in_cycles lookup (O(1) membership test)

    Returns:
        (transactions_df, set_of_all_cycle_participant_account_ids)
    """
    cycle_mules = set()
    all_sources = []
    all_destinations = []
    all_amounts = []
    all_timestamps = []

    # Filter out already-used hub accounts to avoid pattern overlap
    available_accounts = np.array([a for a in accounts if a not in used_accounts])
    used_in_cycles = set()

    for _ in range(n_cycles):
        # Cycle length: random between 3 and 6 hops
        cycle_len = int(rng.integers(CYCLE_MIN_LENGTH, CYCLE_MAX_LENGTH + 1))

        # Select fresh accounts for this cycle (no reuse across cycles)
        remaining = np.array([a for a in available_accounts if a not in used_in_cycles])
        if len(remaining) < cycle_len:
            break

        cycle_accounts = rng.choice(remaining, size=cycle_len, replace=False)
        used_in_cycles.update(cycle_accounts)
        cycle_mules.update(cycle_accounts)

        # Random base timestamp within 30-day window
        base_offset = int(rng.integers(0, int(timedelta(days=28).total_seconds())))
        base_time = pd.Timestamp(start_date) + pd.Timedelta(seconds=base_offset)

        # Amount flowing through the cycle (₹10K–₹80K)
        cycle_amount = float(rng.uniform(10_000, 80_000))

        # Edge timestamps: sorted within a 5-minute window
        edge_offsets = np.sort(rng.integers(10, 280, size=cycle_len))

        # Create edges: A→B, B→C, C→D, D→A
        for i in range(cycle_len):
            src = cycle_accounts[i]
            dst = cycle_accounts[(i + 1) % cycle_len]
            # Slight amount variation to appear semi-realistic
            amount = cycle_amount * float(rng.uniform(0.95, 1.05))

            all_sources.append(src)
            all_destinations.append(dst)
            all_amounts.append(round(amount, 2))
            all_timestamps.append(base_time + pd.Timedelta(seconds=int(edge_offsets[i])))

    df = pd.DataFrame({
        'SourceAccount': all_sources,
        'DestinationAccount': all_destinations,
        'Amount': all_amounts,
        'Timestamp': all_timestamps
    }) if all_sources else pd.DataFrame(columns=['SourceAccount', 'DestinationAccount', 'Amount', 'Timestamp'])
    return df, cycle_mules


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def generate_dataset(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict]:
    """
    Generate the complete synthetic UPI transaction dataset.

    Returns:
        (transactions_df, ground_truth_dict)

        ground_truth_dict = {
            'star_mules': [...],     # Pattern A hub accounts
            'cycle_mules': [...],    # Pattern B cycle participants
            'all_mules': [...]       # Union of both sets
        }
    """
    rng = np.random.default_rng(seed)

    start_date = datetime(2025, 6, 1)
    end_date = datetime(2025, 7, 1)

    # ── Step 1: Account pool ──
    accounts = generate_account_ids(NUM_ACCOUNTS)
    print(f"  ✓ Generated {len(accounts):,} account IDs")

    # ── Step 2: Background noise transactions (vectorized) ──
    normal_txns = generate_normal_transactions(
        accounts, NUM_NORMAL_TRANSACTIONS, start_date, end_date, rng
    )
    print(f"  ✓ Generated {len(normal_txns):,} normal transactions")

    # ── Step 3: Inject star topology mules (Pattern A) ──
    star_txns, star_mules = inject_star_patterns(
        accounts, NUM_STAR_MULES, start_date, rng
    )
    star_edges = len(star_txns) - NUM_STAR_MULES  # subtract incoming edges
    print(f"  ✓ Injected {NUM_STAR_MULES} star patterns "
          f"({len(star_txns)} txns, {len(star_mules)} hub accounts)")

    # ── Step 4: Inject circular loop mules (Pattern B) ──
    cycle_txns, cycle_mules = inject_circular_patterns(
        accounts, NUM_CYCLE_GROUPS, start_date, star_mules, rng
    )
    n_actual_cycles = len(cycle_txns) // 4  # approximate
    print(f"  ✓ Injected {NUM_CYCLE_GROUPS} cycles "
          f"({len(cycle_txns)} txns, {len(cycle_mules)} participant accounts)")

    # ── Step 5: Combine & finalize ──
    all_txns = pd.concat([normal_txns, star_txns, cycle_txns], ignore_index=True)
    all_txns = all_txns.sort_values('Timestamp').reset_index(drop=True)
    all_txns.insert(0, 'TransactionID', [f"TXN_{i:06d}" for i in range(len(all_txns))])

    # Build ground truth dictionary
    all_mules = star_mules | cycle_mules
    ground_truth = {
        'star_mules': sorted(list(star_mules)),
        'cycle_mules': sorted(list(cycle_mules)),
        'all_mules': sorted(list(all_mules))
    }

    print(f"  ✓ Total dataset: {len(all_txns):,} transactions, "
          f"{len(all_mules)} ground-truth mule accounts "
          f"({len(star_mules)} star + {len(cycle_mules)} cycle)")

    return all_txns, ground_truth


def save_dataset(df: pd.DataFrame, ground_truth: dict) -> None:
    """Save the generated dataset and ground truth to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(TRANSACTIONS_FILE, index=False)
    print(f"  ✓ Saved transactions → {TRANSACTIONS_FILE}")

    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  ✓ Saved ground truth → {GROUND_TRUTH_FILE}")


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 1: Synthetic Data Generation")
    print("=" * 65)
    df, gt = generate_dataset()
    save_dataset(df, gt)
    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"\n  Sample transactions:")
    print(df.head(10).to_string(index=False))
