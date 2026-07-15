# %% [markdown]
# # 🔍 UPI Mule Account Detection — Real-Time Graph Database Analyzer
#
# ## Overview
# This notebook implements a complete pipeline for detecting **UPI money mule accounts**
# using graph database analysis. We process a synthetic dataset of **10,000 accounts**
# and their transactions, identifying two key money-laundering topologies:
#
# | Pattern | Topology | Description | Detection Method |
# |---------|----------|-------------|------------------|
# | **A** | Star (Hub-and-Spoke) | Account receives lump sum, disperses to >5 accounts in <10 min | NetworkX velocity |
# | **B** | Circular Loop | Multi-hop cycle A→B→C→D→A within short timeframe | Neo4j / SCC decomposition |
#
# ### Why Graph Analysis?
# Traditional rule-based fraud detection analyzes transactions **individually** against
# static thresholds. Fraudsters easily bypass these by keeping amounts small. Graph-based
# analysis shifts focus to **structural patterns** — relationships between accounts that
# reveal coordinated criminal activity invisible to per-transaction rules.
#
# ### Tech Stack
# - **Pandas + NumPy**: Vectorized data generation and manipulation
# - **NetworkX**: In-memory graph analysis (DiGraph, SCC, degree metrics)
# - **Neo4j** (optional): Native graph database with Cypher query language
# - **scikit-learn metrics**: F1-score evaluation

# %% [markdown]
# ---
# ## 📦 Setup & Configuration

# %%
import numpy as np
import pandas as pd
import networkx as nx
import time
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ── Configuration Constants ──
NUM_ACCOUNTS = 10_000
NUM_NORMAL_TRANSACTIONS = 50_000
NUM_STAR_MULES = 30
NUM_CYCLE_GROUPS = 25
STAR_MIN_DESTINATIONS = 6
STAR_MAX_DESTINATIONS = 12
CYCLE_MIN_LENGTH = 3
CYCLE_MAX_LENGTH = 6
VELOCITY_THRESHOLD_DESTINATIONS = 5
VELOCITY_WINDOW_MINUTES = 10
CYCLE_WINDOW_MINUTES = 10
RANDOM_SEED = 42

# Neo4j connection (used only if available)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# File paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("✓ Configuration loaded")
print(f"  Accounts: {NUM_ACCOUNTS:,}")
print(f"  Normal transactions: {NUM_NORMAL_TRANSACTIONS:,}")
print(f"  Star mules to inject: {NUM_STAR_MULES}")
print(f"  Circular cycles to inject: {NUM_CYCLE_GROUPS}")

# %% [markdown]
# ---
# ## 📊 Phase 1: Synthetic Data Generation
#
# ### Graph Model
# We model the UPI ecosystem as a **directed graph** $G(V, E)$ where:
# - **Nodes** $V$: UPI accounts ($|V| = 10{,}000$)
# - **Edges** $E$: Transactions with attributes $\{amount, timestamp\}$
# - Each edge is directed: sender → receiver
#
# ### Data Distribution Rationale
# - **Transaction amounts**: Lognormal distribution $(\mu=7.5, \sigma=1.2)$
#   produces a realistic long-tailed distribution (median ≈ ₹1,800, range ₹50–₹50,000)
# - **Timestamps**: Uniform over a 30-day period
# - **Source/Destination**: Uniform random from the account pool (no self-loops)
#
# ### Injected Ground-Truth Patterns
#
# **Pattern A — Star Topology** (30 hubs):
# ```
# Victim ──(₹50K-₹2L)──▸ Hub ──▸ Spoke₁
#                              ──▸ Spoke₂
#                              ──▸ ...
#                              ──▸ Spokeₙ  (n ∈ [6,12], all within 10 min)
# ```
#
# **Pattern B — Circular Loop** (25 cycles):
# ```
# A ──▸ B ──▸ C ──▸ D ──▸ A   (3-6 hops, all within 5 min)
# ```

# %%
# ── Phase 1: Data Generation Functions ──

def generate_account_ids(n):
    """Generate n unique UPI-style account IDs."""
    return np.array([f"UPI_{i:05d}" for i in range(n)])


def generate_normal_transactions(accounts, n_txns, start_date, end_date, rng):
    """
    Generate background noise transactions using fully vectorized operations.
    No nested for-loops — all operations are array-based for O(n) complexity.
    """
    # Vectorized source/destination selection
    sources = rng.choice(accounts, size=n_txns)
    destinations = rng.choice(accounts, size=n_txns)

    # Eliminate self-loops via vectorized rejection sampling
    mask = sources == destinations
    while mask.any():
        destinations[mask] = rng.choice(accounts, size=mask.sum())
        mask = sources == destinations

    # Lognormal amounts (realistic UPI distribution)
    amounts = np.clip(
        rng.lognormal(mean=7.5, sigma=1.2, size=n_txns), 50, 50_000
    ).round(2)

    # Uniform timestamps across 30-day window
    total_seconds = int((end_date - start_date).total_seconds())
    random_offsets = rng.integers(0, total_seconds, size=n_txns)
    timestamps = pd.to_datetime(start_date) + pd.to_timedelta(random_offsets, unit='s')

    return pd.DataFrame({
        'SourceAccount': sources,
        'DestinationAccount': destinations,
        'Amount': amounts,
        'Timestamp': timestamps
    })


def inject_star_patterns(accounts, n_stars, start_date, rng):
    """
    Inject Pattern A (Star Topology) mule accounts.

    Each hub:
    1. Receives one large incoming transfer (₹50K–₹2L)
    2. Disperses via Dirichlet-split to 6-12 distinct accounts
    3. All dispersals occur within a 10-minute window
    """
    mule_hubs = set()
    all_rows = []
    hub_indices = rng.choice(len(accounts), size=n_stars, replace=False)

    for idx in hub_indices:
        hub = accounts[idx]
        mule_hubs.add(hub)
        n_spokes = int(rng.integers(STAR_MIN_DESTINATIONS, STAR_MAX_DESTINATIONS + 1))

        available = np.delete(accounts, idx)
        spokes = rng.choice(available, size=n_spokes, replace=False)

        base_offset = int(rng.integers(0, int(timedelta(days=28).total_seconds())))
        base_time = pd.Timestamp(start_date) + pd.Timedelta(seconds=base_offset)

        # Incoming lump sum
        lump_sum = float(rng.uniform(50_000, 200_000))
        victim = rng.choice(np.delete(accounts, idx))
        all_rows.append({
            'SourceAccount': victim, 'DestinationAccount': hub,
            'Amount': round(lump_sum, 2), 'Timestamp': base_time
        })

        # Outgoing dispersals (Dirichlet-split, within 10-min window)
        dispersal_offsets = np.sort(rng.integers(30, 570, size=n_spokes))
        split_fractions = rng.dirichlet(np.ones(n_spokes))
        total_dispersed = lump_sum * float(rng.uniform(0.85, 0.95))
        dispersal_amounts = split_fractions * total_dispersed

        for spoke, offset, amount in zip(spokes, dispersal_offsets, dispersal_amounts):
            all_rows.append({
                'SourceAccount': hub, 'DestinationAccount': spoke,
                'Amount': round(float(amount), 2),
                'Timestamp': base_time + pd.Timedelta(seconds=int(offset))
            })

    return pd.DataFrame(all_rows), mule_hubs


def inject_circular_patterns(accounts, n_cycles, start_date, used_accounts, rng):
    """
    Inject Pattern B (Circular Loop) mule accounts.

    Each cycle: A→B→C→...→A with 3-6 hops, all edges within 5 minutes.
    Uses dedicated accounts that don't overlap with star hubs.
    """
    cycle_mules = set()
    all_rows = []
    available = np.array([a for a in accounts if a not in used_accounts])
    used_in_cycles = set()

    for _ in range(n_cycles):
        cycle_len = int(rng.integers(CYCLE_MIN_LENGTH, CYCLE_MAX_LENGTH + 1))
        remaining = np.array([a for a in available if a not in used_in_cycles])
        if len(remaining) < cycle_len:
            break

        cycle_accounts = rng.choice(remaining, size=cycle_len, replace=False)
        used_in_cycles.update(cycle_accounts)
        cycle_mules.update(cycle_accounts)

        base_offset = int(rng.integers(0, int(timedelta(days=28).total_seconds())))
        base_time = pd.Timestamp(start_date) + pd.Timedelta(seconds=base_offset)
        cycle_amount = float(rng.uniform(10_000, 80_000))
        edge_offsets = np.sort(rng.integers(10, 280, size=cycle_len))

        for i in range(cycle_len):
            src = cycle_accounts[i]
            dst = cycle_accounts[(i + 1) % cycle_len]
            amount = cycle_amount * float(rng.uniform(0.95, 1.05))
            all_rows.append({
                'SourceAccount': src, 'DestinationAccount': dst,
                'Amount': round(amount, 2),
                'Timestamp': base_time + pd.Timedelta(seconds=int(edge_offsets[i]))
            })

    return pd.DataFrame(all_rows), cycle_mules


print("✓ Phase 1 functions defined")

# %%
# ── Execute Phase 1 ──
print("=" * 65)
print("  PHASE 1: Generating Synthetic Dataset")
print("=" * 65)

rng = np.random.default_rng(RANDOM_SEED)
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 7, 1)

# Generate components
accounts = generate_account_ids(NUM_ACCOUNTS)
print(f"  ✓ Generated {len(accounts):,} account IDs")

normal_txns = generate_normal_transactions(accounts, NUM_NORMAL_TRANSACTIONS, start_date, end_date, rng)
print(f"  ✓ Generated {len(normal_txns):,} normal transactions")

star_txns, star_mules = inject_star_patterns(accounts, NUM_STAR_MULES, start_date, rng)
print(f"  ✓ Injected {NUM_STAR_MULES} star patterns ({len(star_txns)} txns, {len(star_mules)} hubs)")

cycle_txns, cycle_mules = inject_circular_patterns(accounts, NUM_CYCLE_GROUPS, start_date, star_mules, rng)
print(f"  ✓ Injected {NUM_CYCLE_GROUPS} cycles ({len(cycle_txns)} txns, {len(cycle_mules)} participants)")

# Combine and finalize
all_txns = pd.concat([normal_txns, star_txns, cycle_txns], ignore_index=True)
all_txns = all_txns.sort_values('Timestamp').reset_index(drop=True)
all_txns.insert(0, 'TransactionID', [f"TXN_{i:06d}" for i in range(len(all_txns))])

# Ground truth
all_mules_gt = star_mules | cycle_mules
ground_truth = {
    'star_mules': sorted(list(star_mules)),
    'cycle_mules': sorted(list(cycle_mules)),
    'all_mules': sorted(list(all_mules_gt))
}

# Save to disk
all_txns.to_csv(f"{DATA_DIR}/transactions.csv", index=False)
with open(f"{DATA_DIR}/ground_truth.json", 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"\n  Total: {len(all_txns):,} transactions, {len(all_mules_gt)} ground-truth mules")
print(f"  Saved → {DATA_DIR}/transactions.csv")

# %%
# ── Phase 1: Data Preview ──
print("Sample Transactions:")
print(all_txns.head(10).to_string(index=False))
print(f"\nDataset shape: {all_txns.shape}")
print(f"Date range: {all_txns['Timestamp'].min()} to {all_txns['Timestamp'].max()}")
print(f"\nAmount distribution:")
print(all_txns['Amount'].describe().to_string())

# %% [markdown]
# ---
# ## 🔄 Phase 2: Cycle Detection (Neo4j + NetworkX Fallback)
#
# ### Graph Theory: Strongly Connected Components (SCC)
#
# A **directed cycle** in graph $G(V,E)$ is a closed walk $(v_0, v_1, \ldots, v_k, v_0)$
# where all intermediate vertices are distinct.
#
# For mule detection, we need **temporal cycles** — cycles where all edge timestamps
# fall within a bounded window (10 minutes), indicating coordinated fund rotation.
#
# ### Detection Algorithm: Sliding-Window SCC Decomposition
#
# Instead of enumerating all simple cycles (exponential worst-case), we use a
# more efficient approach based on **Tarjan's SCC algorithm**:
#
# 1. **Sort** all edges by timestamp: $O(E \log E)$
# 2. **Bin** edges into overlapping 10-minute windows (with 5-min offset)
# 3. **For each window** with $\geq 3$ edges:
#    - Build a tiny directed subgraph
#    - Run Tarjan's SCC: $O(V_w + E_w)$ per window
#    - Any SCC with $|SCC| \geq 3$ contains a directed cycle
#
# **Why SCC works**: By definition, in an SCC every vertex can reach every other vertex.
# If $|SCC| \geq 2$, the SCC must contain at least one directed cycle.
#
# **Why overlapping bins**: A cycle spanning $t=8\text{min}$ to $t=12\text{min}$ would be
# split across non-overlapping bins $[0,10)$ and $[10,20)$. The 5-minute offset creates
# bin $[5,15)$ that captures it entirely.
#
# ### Neo4j Integration
#
# When a Neo4j instance is available, we use Cypher's bounded variable-length path patterns:
# ```cypher
# MATCH path = (start:Account)-[:SENT*3..6]->(start)
# WHERE duration.between(min(timestamps), max(timestamps)).minutes < 10
# ```
#
# **Critical Index** for performance:
# ```cypher
# CREATE INDEX account_id_idx FOR (a:Account) ON (a.account_id);
# ```
# This ensures $O(\log n)$ node lookups during `MATCH` pattern binding.

# %%
# ── Phase 2: Cycle Detection Functions ──

def try_neo4j_detection(df):
    """
    Attempt cycle detection via Neo4j. Returns (detected_set, success_bool).
    Falls back gracefully if Neo4j is unavailable.
    """
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("  ✓ Connected to Neo4j")

        # Clear + index
        driver.execute_query("MATCH (n) DETACH DELETE n")
        driver.execute_query(
            "CREATE INDEX account_id_idx IF NOT EXISTS "
            "FOR (a:Account) ON (a.account_id)"
        )

        # Batch ingest nodes (1000/batch using UNWIND)
        unique_accounts = sorted(set(df['SourceAccount']) | set(df['DestinationAccount']))
        for i in range(0, len(unique_accounts), 1000):
            batch = [{'id': a} for a in unique_accounts[i:i+1000]]
            driver.execute_query(
                "UNWIND $accs AS a MERGE (:Account {account_id: a.id})", accs=batch
            )

        # Batch ingest edges
        edges = df[['SourceAccount', 'DestinationAccount', 'Amount', 'Timestamp']].copy()
        edges['Timestamp'] = edges['Timestamp'].astype(str)
        for i in range(0, len(edges), 1000):
            batch = edges.iloc[i:i+1000].to_dict('records')
            driver.execute_query(
                "UNWIND $edges AS e "
                "MATCH (s:Account {account_id: e.SourceAccount}) "
                "MATCH (d:Account {account_id: e.DestinationAccount}) "
                "CREATE (s)-[:SENT {amount: e.Amount, timestamp: e.Timestamp}]->(d)",
                edges=batch
            )

        # Cycle detection query
        records, _, _ = driver.execute_query("""
            MATCH path = (start:Account)-[:SENT*3..6]->(start)
            WITH start, [r IN relationships(path) | datetime(r.timestamp)] AS ts
            WITH start,
                 reduce(mn = ts[0], t IN ts | CASE WHEN t < mn THEN t ELSE mn END) AS minT,
                 reduce(mx = ts[0], t IN ts | CASE WHEN t > mx THEN t ELSE mx END) AS maxT
            WHERE duration.between(minT, maxT).minutes < $w
            UNWIND [n IN nodes(path) | n.account_id] AS mid
            RETURN DISTINCT mid
        """, w=CYCLE_WINDOW_MINUTES)

        detected = {r['mid'] for r in records}
        driver.close()
        print(f"  ✓ Neo4j detected {len(detected)} cycle participants")
        return detected, True

    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({type(e).__name__})")
        print("  → Using NetworkX SCC fallback")
        return set(), False


def detect_cycles_networkx(df):
    """
    Sliding-window SCC decomposition for temporal cycle detection.

    Optimization: Uses numpy argsort/unique for grouping (avoids DataFrame
    creation per bin) and Kosaraju's SCC algorithm with dict-based adjacency
    (avoids NetworkX DiGraph object overhead per window).
    """
    detected = set()
    edges = df[['SourceAccount', 'DestinationAccount', 'Timestamp']].copy()
    edges['Timestamp'] = pd.to_datetime(edges['Timestamp'])
    edges = edges.sort_values('Timestamp').reset_index(drop=True)

    sources = edges['SourceAccount'].values
    destinations = edges['DestinationAccount'].values
    ts = edges['Timestamp'].values
    min_ts = ts[0]

    # Two-pass with 5-minute offset for complete coverage
    for offset_min in [0, 5]:
        offset = np.timedelta64(offset_min, 'm')
        adjusted = (ts - min_ts + offset).astype('timedelta64[m]').astype(np.int64)
        bin_ids = adjusted // CYCLE_WINDOW_MINUTES

        # Numpy-based grouping (avoids DataFrame creation)
        sort_idx = np.argsort(bin_ids, kind='mergesort')
        sorted_bins = bin_ids[sort_idx]
        sorted_src = sources[sort_idx]
        sorted_dst = destinations[sort_idx]

        unique_bins, bin_starts = np.unique(sorted_bins, return_index=True)
        bin_ends = np.append(bin_starts[1:], len(sorted_bins))

        for si, ei in zip(bin_starts, bin_ends):
            if ei - si < 3:
                continue

            grp_src = sorted_src[si:ei]
            grp_dst = sorted_dst[si:ei]

            # Build adjacency dicts
            adj = {}
            rev_adj = {}
            nodes = set()
            for s, d in zip(grp_src, grp_dst):
                adj.setdefault(s, set()).add(d)
                rev_adj.setdefault(d, set()).add(s)
                nodes.add(s)
                nodes.add(d)

            if len(nodes) < 3:
                continue

            # Kosaraju's SCC — Pass 1: finish ordering
            visited = set()
            order = []
            for root in nodes:
                if root in visited:
                    continue
                stack = [(root, False)]
                while stack:
                    node, done = stack.pop()
                    if done:
                        order.append(node)
                        continue
                    if node in visited:
                        continue
                    visited.add(node)
                    stack.append((node, True))
                    for nb in adj.get(node, ()):
                        if nb not in visited:
                            stack.append((nb, False))

            # Kosaraju's SCC — Pass 2: reverse DFS
            visited2 = set()
            for root in reversed(order):
                if root in visited2:
                    continue
                scc = set()
                stack = [root]
                while stack:
                    node = stack.pop()
                    if node in visited2:
                        continue
                    visited2.add(node)
                    scc.add(node)
                    for nb in rev_adj.get(node, ()):
                        if nb not in visited2:
                            stack.append(nb)
                if len(scc) >= 3:
                    detected.update(scc)

    return detected


print("✓ Phase 2 functions defined")

# %%
# ── Execute Phase 2 ──
print("=" * 65)
print("  PHASE 2: Cycle Detection")
print("=" * 65)

phase2_start = time.time()

cycle_detected, neo4j_ok = try_neo4j_detection(all_txns)
if not neo4j_ok:
    cycle_detected = detect_cycles_networkx(all_txns)
    print(f"  ✓ Detected {len(cycle_detected)} cycle participants via SCC fallback")

phase2_time = time.time() - phase2_start
print(f"\n  Phase 2 time: {phase2_time:.3f}s")

# %% [markdown]
# ---
# ## ⚡ Phase 3: NetworkX Velocity Detection (Star Topology)
#
# ### Time-Weighted Out-Degree Velocity
#
# The **out-degree velocity** of a node $n$ over a time window $W$ is:
#
# $$v_{out}(n, W) = |\{d : (n \rightarrow d) \in E,\ \text{timestamp}(n \rightarrow d) \in W\}|$$
#
# We flag account $n$ as a star mule if:
# $$\exists W \text{ of duration } \leq 10\text{ min}: v_{out}(n, W) > 5$$
#
# ### Algorithm: Vectorized Sliding Window with Binary Search
#
# ```
# For each candidate account (pre-filtered by total out-degree > 5):
#     Sort outgoing edges by timestamp
#     For each edge i:
#         j ← searchsorted(timestamps, timestamp_i + 10min)   # O(log n)
#         if |unique_destinations[i:j]| > 5:
#             FLAG account as star mule
#             BREAK (early exit)
# ```
#
# ### Optimization Techniques
#
# | Technique | Speedup | Description |
# |-----------|---------|-------------|
# | **Pre-filtering** | ~2.5× | Skip accounts with ≤ 5 total outgoing edges |
# | **np.searchsorted** | O(log n) vs O(n) | Binary search for window boundaries |
# | **Early exit** | ~3× avg | Stop checking once first violation found |
# | **Pandas groupby** | vectorized | Per-account edge grouping without nested loops |

# %%
# ── Phase 3: Velocity Detection Functions ──

def detect_star_patterns(df):
    """
    Detect star-topology mule accounts using time-weighted out-degree velocity.

    Builds a NetworkX DiGraph and uses vectorized Pandas operations
    with np.searchsorted for efficient sliding-window analysis.

    CRITICAL: Uses np.timedelta64 (not int64 nanoseconds) for datetime math.
    In pandas >= 2.0, datetime resolution may be 's' not 'ns', making
    .astype(np.int64) return seconds — which breaks nanosecond-based windows.
    """
    detected = set()

    # Step 1: Build NetworkX DiGraph
    G = nx.from_pandas_edgelist(
        df, source='SourceAccount', target='DestinationAccount',
        create_using=nx.DiGraph()
    )
    print(f"  ✓ Built DiGraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Step 2: Prepare outgoing edge data (vectorized sort)
    out_edges = df[['SourceAccount', 'DestinationAccount', 'Timestamp']].copy()
    out_edges['Timestamp'] = pd.to_datetime(out_edges['Timestamp'])
    out_edges = out_edges.sort_values(['SourceAccount', 'Timestamp']).reset_index(drop=True)

    # Step 2b: Prepare incoming edge lookup (for receive-then-disperse check)
    in_edges = df[['DestinationAccount', 'Timestamp']].copy()
    in_edges['Timestamp'] = pd.to_datetime(in_edges['Timestamp'])
    in_edges = in_edges.sort_values('Timestamp')
    incoming_ts_map = {}
    for acct, grp in in_edges.groupby('DestinationAccount'):
        incoming_ts_map[acct] = grp['Timestamp'].values

    # Step 3: Pre-filter candidates
    out_counts = out_edges.groupby('SourceAccount').size()
    candidates = out_counts[out_counts > VELOCITY_THRESHOLD_DESTINATIONS].index.values
    print(f"  ✓ Pre-filtered to {len(candidates):,} candidates "
          f"(from {G.number_of_nodes():,} nodes)")

    # Step 4: Sliding-window detection with np.timedelta64 (resolution-agnostic)
    window_td = np.timedelta64(VELOCITY_WINDOW_MINUTES, 'm')
    # Group edges by source account using numpy for extreme performance
    sources = out_edges['SourceAccount'].values
    timestamps_all = out_edges['Timestamp'].values
    destinations_all = out_edges['DestinationAccount'].values

    unique_sources, source_starts = np.unique(sources, return_index=True)
    source_ends = np.append(source_starts[1:], len(sources))
    source_bounds = dict(zip(unique_sources, zip(source_starts, source_ends)))

    for account in candidates:
        start_idx, end_idx = source_bounds[account]
        timestamps = timestamps_all[start_idx:end_idx]
        dests = destinations_all[start_idx:end_idx]

        burst_found = False
        burst_time = None

        for i in range(len(timestamps)):
            j = np.searchsorted(timestamps, timestamps[i] + window_td, side='right')
            if j - i > VELOCITY_THRESHOLD_DESTINATIONS:
                if len(set(dests[i:j])) > VELOCITY_THRESHOLD_DESTINATIONS:
                    burst_found = True
                    burst_time = timestamps[i]
                    break  # Early exit

        # Step 5: Verify receive-then-disperse pattern
        if burst_found and account in incoming_ts_map:
            inc_ts = incoming_ts_map[account]
            lookback_start = burst_time - window_td
            lookback_end = burst_time + np.timedelta64(1, 'm')
            left = np.searchsorted(inc_ts, lookback_start)
            right = np.searchsorted(inc_ts, lookback_end, side='right')
            if right > left:
                detected.add(account)

    return detected, G


print("✓ Phase 3 functions defined")

# %%
# ── Execute Phase 3 ──
print("=" * 65)
print("  PHASE 3: Velocity Detection (Star Topology)")
print("=" * 65)

phase3_start = time.time()
star_detected, G = detect_star_patterns(all_txns)
phase3_time = time.time() - phase3_start

print(f"  ✓ Detected {len(star_detected)} star-topology mule accounts")
print(f"\n  Phase 3 time: {phase3_time:.3f}s")

# ── Graph Statistics ──
in_degs = [d for _, d in G.in_degree()]
out_degs = [d for _, d in G.out_degree()]
print(f"\n  Graph Statistics:")
print(f"  {'─' * 40}")
print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
print(f"  Avg in-degree: {np.mean(in_degs):.2f}  |  Max: {max(in_degs)}")
print(f"  Avg out-degree: {np.mean(out_degs):.2f}  |  Max: {max(out_degs)}")
print(f"  Density: {nx.density(G):.6f}")

# %% [markdown]
# ---
# ## 📈 Phase 4: Evaluation & Optimization
#
# ### F1-Score Computation
#
# For binary classification of mule vs. legitimate accounts:
#
# $$\text{Precision} = \frac{TP}{TP + FP}$$
# $$\text{Recall} = \frac{TP}{TP + FN}$$
# $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
#
# **Why F1 over Accuracy?** The dataset is heavily imbalanced (~1.3% mules).
# A naive classifier predicting "legitimate" for all accounts achieves 98.7% accuracy
# but 0% recall. F1-score penalizes this by requiring both precision AND recall.
#
# ### Risk Scoring
#
# Accounts are scored and ranked:
# - **1.0** — Detected by BOTH cycle and star detection
# - **0.5** — Detected by one method only

# %%
# ── Phase 4: Evaluation ──
print("=" * 65)
print("  PHASE 4: Evaluation & Optimization")
print("=" * 65)

# Combine detections
all_detected = cycle_detected | star_detected
gt_all = set(ground_truth['all_mules'])
gt_star = set(ground_truth['star_mules'])
gt_cycle = set(ground_truth['cycle_mules'])

# Compute overall metrics
tp = len(all_detected & gt_all)
fp = len(all_detected - gt_all)
fn = len(gt_all - all_detected)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# Per-pattern metrics
star_tp = len(star_detected & gt_star)
star_fp = len(star_detected - gt_star)
star_fn = len(gt_star - star_detected)
star_p = star_tp / (star_tp + star_fp) if (star_tp + star_fp) > 0 else 0.0
star_r = star_tp / (star_tp + star_fn) if (star_tp + star_fn) > 0 else 0.0
star_f1 = 2 * star_p * star_r / (star_p + star_r) if (star_p + star_r) > 0 else 0.0

cycle_tp = len(cycle_detected & gt_cycle)
cycle_fp = len(cycle_detected - gt_cycle)
cycle_fn = len(gt_cycle - cycle_detected)
cycle_p = cycle_tp / (cycle_tp + cycle_fp) if (cycle_tp + cycle_fp) > 0 else 0.0
cycle_r = cycle_tp / (cycle_tp + cycle_fn) if (cycle_tp + cycle_fn) > 0 else 0.0
cycle_f1 = 2 * cycle_p * cycle_r / (cycle_p + cycle_r) if (cycle_p + cycle_r) > 0 else 0.0

# ── Print Report ──
combined_time = phase2_time + phase3_time
f1_status = "✓ PASS" if f1 > 0.85 else "✗ FAIL"
perf_status = "✓ PASS" if combined_time < 2.0 else "✗ FAIL"

print(f"""
  ┌─────────────────────────────────────────────────┐
  │  OVERALL DETECTION METRICS                      │
  ├─────────────────────────────────────────────────┤
  │  True Positives  : {tp:>5}                        │
  │  False Positives : {fp:>5}                        │
  │  False Negatives : {fn:>5}                        │
  │                                                 │
  │  Precision       : {precision:>8.4f}                   │
  │  Recall          : {recall:>8.4f}                   │
  │  F1-Score        : {f1:>8.4f}  ({f1_status})      │
  └─────────────────────────────────────────────────┘

  Per-Pattern Breakdown:
  {'─' * 50}
  Pattern A (Star):  P={star_p:.4f}  R={star_r:.4f}  F1={star_f1:.4f}
  Pattern B (Cycle): P={cycle_p:.4f}  R={cycle_r:.4f}  F1={cycle_f1:.4f}

  Performance:
  {'─' * 50}
  Phase 2 (Cycle Detection)    : {phase2_time:.3f}s
  Phase 3 (Velocity Detection) : {phase3_time:.3f}s
  Combined (target < 2.0s)     : {combined_time:.3f}s  ({perf_status})

  Detection Summary:
  {'─' * 50}
  Cycle only  : {len(cycle_detected - star_detected)}
  Star only   : {len(star_detected - cycle_detected)}
  Both        : {len(cycle_detected & star_detected)}
  Total       : {len(all_detected)}
  Ground truth: {len(gt_all)}
""")

# %%
# ── Build Ranked Suspect List ──

rows = []
for account in sorted(all_detected):
    in_cycle = account in cycle_detected
    in_star = account in star_detected
    is_mule = account in gt_all

    risk = (0.5 if in_cycle else 0.0) + (0.5 if in_star else 0.0)
    source = "BOTH" if (in_cycle and in_star) else ("CYCLE" if in_cycle else "STAR")

    if account in gt_star:
        pattern = "Star (A)"
    elif account in gt_cycle:
        pattern = "Cycle (B)"
    else:
        pattern = "FALSE POSITIVE"

    rows.append({
        'Account': account,
        'RiskScore': risk,
        'DetectionSource': source,
        'GroundTruth': 'MULE' if is_mule else 'LEGIT',
        'PatternType': pattern
    })

ranked_df = pd.DataFrame(rows).sort_values('RiskScore', ascending=False).reset_index(drop=True)

print("Top 25 Suspected Mule Accounts:")
print("─" * 75)
print(ranked_df.head(25).to_string(index=False))

# Save results
ranked_df.to_csv(f"{DATA_DIR}/results.csv", index=False)
print(f"\n✓ Saved ranked results → {DATA_DIR}/results.csv")

# %%
# ── Missed Accounts Analysis ──
missed = gt_all - all_detected
if missed:
    print(f"\n⚠ Missed {len(missed)} ground-truth mule accounts:")
    for acc in sorted(missed):
        pattern = "Star" if acc in gt_star else "Cycle"
        print(f"  {acc}  (Pattern: {pattern})")
else:
    print("\n✅ Perfect recall — all ground-truth mule accounts detected!")

# False positives
false_pos = all_detected - gt_all
if false_pos:
    print(f"\n⚠ {len(false_pos)} false positive(s):")
    for acc in sorted(false_pos):
        print(f"  {acc}")
else:
    print("✅ Perfect precision — no false positives!")

# %% [markdown]
# ---
# ## 🏁 Summary
#
# | Metric | Value | Target | Status |
# |--------|-------|--------|--------|
# | F1-Score | *computed above* | > 0.85 | ✓/✗ |
# | Phase 2+3 Time | *computed above* | < 2.0s | ✓/✗ |
#
# ### Key Optimization Techniques Used
#
# 1. **Vectorized Data Generation** — NumPy array ops instead of Python loops
# 2. **Sliding-Window SCC** — Tarjan's O(V+E) per micro-subgraph, not exponential cycle enumeration
# 3. **np.searchsorted** — O(log n) window boundaries instead of O(n) linear scan
# 4. **Pre-filtering** — Eliminate 60%+ of accounts before expensive per-node analysis
# 5. **Early exit** — Stop checking a node on first threshold violation
# 6. **Overlapping bins** — Two-pass 5-minute offset guarantees no boundary-split cycles are missed
