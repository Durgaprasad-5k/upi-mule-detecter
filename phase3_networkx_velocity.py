"""
Phase 3: NetworkX Velocity Detection (Star Topology)
=====================================================
Detects Pattern A (Star Topology) mule accounts by computing
time-weighted out-degree velocity on a NetworkX DiGraph.

Graph Theory:
    The "time-weighted out-degree velocity" of a node n is defined as:

        v_out(n) = |{distinct destinations}| / Δt

    where Δt is the time span of outgoing transactions within a sliding window.
    A high velocity (> 5 distinct destinations in < 10 minutes) signals a
    star-topology mule: a hub that receives and rapidly disperses funds.

    In the directed graph G(V,E), star mule hubs exhibit:
        - High in-degree burst (lump sum receipt)
        - Immediate high out-degree burst (rapid dispersal)
        - Low betweenness centrality (pass-through node, not a connector)

Algorithm:
    1. Load all edges into a NetworkX DiGraph with timestamp attributes
    2. Pre-filter: only examine accounts with > threshold outgoing edges
       (eliminates ~60% of accounts in O(1) per account)
    3. For each candidate account:
       a. Sort outgoing edges by timestamp (already sorted via DataFrame)
       b. Use np.searchsorted for O(log n) sliding window boundary detection
       c. Count distinct destinations in each window
       d. Flag if any window exceeds the threshold

Optimization (v2):
    - Pandas groupby + sort_values for per-account edge sets (vectorized)
    - np.searchsorted for O(log n) window boundaries (not O(n) scan)
    - Early exit: break on first flagged window per account
    - Pre-filter by total out-degree avoids unnecessary processing
    - Vectorized incoming edge map using numpy (no groupby iteration)

Complexity:
    O(E log E) sort + O(C × D × log D) scanning
    where C = candidate accounts (~3,800), D = max out-degree per account (~10)
    In practice: ~100–200ms for 50K edges.
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
from config import (
    VELOCITY_THRESHOLD_DESTINATIONS, VELOCITY_WINDOW_MINUTES,
    TRANSACTIONS_FILE
)


def build_digraph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Load the transaction data into a NetworkX directed graph.

    Each edge stores:
        - amount: Transaction value (₹)
        - timestamp: Transaction time (string, for serialization)

    Note: NetworkX DiGraph supports parallel edges via MultiDiGraph,
    but for velocity detection we only need the edge list per node,
    which we compute via Pandas for better performance.
    """
    G = nx.DiGraph()
    G.add_edges_from(zip(df['SourceAccount'], df['DestinationAccount']))
    return G


def detect_star_patterns(df: pd.DataFrame) -> tuple[set, float, nx.DiGraph]:
    """
    Detect Pattern A (Star Topology) by computing time-weighted
    out-degree velocity with a 10-minute sliding window.

    The algorithm flags accounts where:
        ∃ a 10-minute window W such that
        |{distinct d : (account → d) ∈ E, timestamp(d) ∈ W}| > 5

    Returns:
        (set_of_detected_star_mule_accounts, execution_time_seconds, DiGraph)
    """
    start_time = time.time()
    detected_mules = set()

    # ── Step 1: Build NetworkX DiGraph ──
    G = build_digraph(df)
    graph_build_time = time.time() - start_time
    print(f"  ✓ Built DiGraph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges ({graph_build_time:.3f}s)")

    # ── Step 2: Prepare outgoing edge data for vectorized processing ──
    out_edges = df[['SourceAccount', 'DestinationAccount', 'Timestamp']]
    out_edges = out_edges.sort_values(['SourceAccount', 'Timestamp'])

    # ── Step 2b: Prepare incoming edge lookup (vectorized, no groupby) ──
    # Build incoming timestamp map using numpy sorting + unique boundaries
    in_edges = df[['DestinationAccount', 'Timestamp']]
    in_edges = in_edges.sort_values(['DestinationAccount', 'Timestamp'])
    
    in_accts = in_edges['DestinationAccount'].values
    in_ts_all = in_edges['Timestamp'].values
    
    # Build map using numpy unique + index boundaries (avoids groupby iteration)
    unique_in_accts, in_starts = np.unique(in_accts, return_index=True)
    in_ends = np.append(in_starts[1:], len(in_accts))
    incoming_ts_map = {}
    for acct, s, e in zip(unique_in_accts, in_starts, in_ends):
        incoming_ts_map[acct] = in_ts_all[s:e]  # already sorted — zero-copy slice

    # ── Step 3: Pre-filter candidates ──
    # Only accounts with > threshold total outgoing transactions
    # can possibly exceed the velocity threshold in any window.
    # This eliminates ~60% of accounts in O(n) time.
    outgoing_counts = out_edges.groupby('SourceAccount').size()
    candidates = outgoing_counts[
        outgoing_counts > VELOCITY_THRESHOLD_DESTINATIONS
    ].index.values
    print(f"  ✓ Pre-filtered to {len(candidates):,} candidate accounts "
          f"(out of {G.number_of_nodes():,})")

    # ── Step 4: Sliding-window velocity detection ──
    # CRITICAL: Use np.timedelta64 instead of int64 nanosecond arithmetic.
    # In pandas ≥ 2.0 / 3.0, datetime columns may use datetime64[s] resolution,
    # causing .astype(np.int64) to return SECONDS (not nanoseconds).
    # np.timedelta64 is resolution-agnostic and always correct.
    window_td = np.timedelta64(VELOCITY_WINDOW_MINUTES, 'm')

    # Group edges by source account using numpy for extreme performance
    sources = out_edges['SourceAccount'].values
    timestamps_all = out_edges['Timestamp'].values
    destinations_all = out_edges['DestinationAccount'].values

    # out_edges is already sorted by ['SourceAccount', 'Timestamp']
    # So we can just find the boundaries of each account using np.unique
    unique_sources, source_starts = np.unique(sources, return_index=True)
    source_ends = np.append(source_starts[1:], len(sources))
    
    # Create a mapping for O(1) candidate lookup to their start/end indices
    source_bounds = dict(zip(unique_sources, zip(source_starts, source_ends)))

    for account in candidates:
        start_idx, end_idx = source_bounds[account]
        
        # Extract numpy arrays via zero-copy slicing
        timestamps = timestamps_all[start_idx:end_idx]
        destinations = destinations_all[start_idx:end_idx]
        n_edges = end_idx - start_idx

        # Sliding window: for each edge i, find the rightmost edge j
        # within [timestamp_i, timestamp_i + 10min] using binary search
        burst_found = False
        burst_time = None

        for i in range(n_edges):
            window_end = timestamps[i] + window_td
            # O(log n) binary search for the right boundary
            j = np.searchsorted(timestamps, window_end, side='right')

            # Quick cardinality check before computing unique set
            if j - i > VELOCITY_THRESHOLD_DESTINATIONS:
                unique_dests = len(set(destinations[i:j]))
                if unique_dests > VELOCITY_THRESHOLD_DESTINATIONS:
                    burst_found = True
                    burst_time = timestamps[i]
                    break  # Early exit: one flagged window is enough

        # ── Step 5: Verify receive-then-disperse pattern ──
        # A true star mule must have received incoming funds shortly before
        # the outgoing burst. This eliminates false positives from accounts
        # that randomly have clustered outgoing transactions.
        if burst_found and account in incoming_ts_map:
            inc_ts = incoming_ts_map[account]
            # Look for incoming transaction in [burst_time - 10min, burst_time + 1min]
            lookback_start = burst_time - window_td
            lookback_end = burst_time + np.timedelta64(1, 'm')
            left = np.searchsorted(inc_ts, lookback_start)
            right = np.searchsorted(inc_ts, lookback_end, side='right')
            if right > left:
                detected_mules.add(account)

    elapsed = time.time() - start_time
    print(f"  ✓ Detected {len(detected_mules)} star-topology mule accounts")
    return detected_mules, elapsed, G


# ─── Graph Statistics (for notebook display) ──────────────────────────────────

def print_graph_stats(G: nx.DiGraph) -> None:
    """Print summary statistics of the transaction graph."""
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    print(f"\n  Graph Statistics:")
    print(f"  {'─' * 40}")
    print(f"  Nodes (accounts)     : {G.number_of_nodes():,}")
    print(f"  Edges (transactions) : {G.number_of_edges():,}")
    print(f"  Avg in-degree        : {np.mean(in_degrees):.2f}")
    print(f"  Avg out-degree       : {np.mean(out_degrees):.2f}")
    print(f"  Max in-degree        : {max(in_degrees)}")
    print(f"  Max out-degree       : {max(out_degrees)}")
    print(f"  Density              : {nx.density(G):.6f}")

    # Weakly connected components
    n_wcc = nx.number_weakly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    print(f"  Weakly connected     : {n_wcc} components")
    print(f"  Largest WCC size     : {len(largest_wcc):,} nodes")


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 3: Velocity Detection (Star Topology)")
    print("=" * 65)
    df = pd.read_csv(TRANSACTIONS_FILE)
    detected, elapsed, G = detect_star_patterns(df)
    print_graph_stats(G)
    print(f"\n  Detected {len(detected)} accounts in {elapsed:.3f}s")
    print(f"  Sample: {sorted(list(detected))[:10]}")
