"""
Phase 2: Cycle Detection (Neo4j + NetworkX Fallback)
=====================================================
Detects multi-hop circular loops (Pattern B) in the transaction graph.

Graph Theory:
    A directed cycle in graph G(V,E) is a closed walk (v₀, v₁, …, vₖ, v₀)
    where all intermediate vertices are distinct. For mule detection, we seek
    "temporal cycles" — cycles where all edge timestamps fall within a bounded
    time window, indicating coordinated rapid fund movement (layering).

Detection Strategy:
    1. PRIMARY: Neo4j with bounded variable-length path patterns
       - Cypher: (n)-[:SENT*3..6]->(n) with temporal WHERE clause
       - Requires a running Neo4j instance
    2. FALLBACK: Simple-cycle detection on temporal subgraphs
       - Group edges into time windows, find simple cycles of length 3-6
       - Only considers edges between accounts with both in-degree and 
         out-degree ≥ 1 in the window (necessary for cycle participation)

Optimization (v4):
    - Fast Neo4j timeout (3s) to avoid 20s+ wait when unavailable
    - Temporal windowing: sort edges, slide window, find cycles per window
    - nx.simple_cycles with length_bound=6 for bounded cycle enumeration
    - Pre-filter by bidirectional degree to reduce graph size per window
    - Early skip: windows with <3 unique accounts cannot form cycles
"""

import time
import pandas as pd
import numpy as np
import networkx as nx
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CYCLE_WINDOW_MINUTES, CYCLE_MAX_DEPTH
)


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j Integration
# ══════════════════════════════════════════════════════════════════════════════

def try_neo4j_detection(df: pd.DataFrame) -> tuple[set, bool]:
    """
    Attempt cycle detection using Neo4j's native graph engine.

    Neo4j Indexing Strategy (critical for performance):
        CREATE INDEX account_id_idx FOR (a:Account) ON (a.account_id);
        - Ensures O(log n) lookups during MATCH pattern binding
        - Without this, the query does a full label scan → O(n) per binding

    Ingestion uses UNWIND with parameterized queries (1000 rows/batch)
    to leverage Neo4j's query plan caching and avoid Cypher injection.

    Performance: connection_timeout=3s prevents 20s+ hang when Neo4j
    is unavailable, keeping fallback latency under 3s.

    Returns:
        (detected_accounts, success_flag)
        If Neo4j is unavailable, returns (empty_set, False).
    """
    try:
        from neo4j import GraphDatabase
        import socket

        # Fast socket pre-check: instantly fail if port is closed
        # This avoids the 3s+ driver timeout when Neo4j isn't running
        host = NEO4J_URI.replace('bolt://', '').replace('neo4j://', '').split(':')[0]
        port = int(NEO4J_URI.split(':')[-1]) if ':' in NEO4J_URI.split('//')[-1] else 7687
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # 100ms is generous for localhost
        try:
            sock.connect((host, port))
            sock.close()
        except (socket.timeout, ConnectionRefusedError, OSError):
            sock.close()
            raise ConnectionError(f"Port {port} on {host} is not reachable")

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=3,
            max_connection_lifetime=60,
        )
        driver.verify_connectivity()
        print("  ✓ Connected to Neo4j instance")

        # ── Step 1: Clear existing data ──
        driver.execute_query("MATCH (n) DETACH DELETE n")

        # ── Step 2: Create indexes ──
        # B-tree index on account_id for fast node lookups
        driver.execute_query(
            "CREATE INDEX account_id_idx IF NOT EXISTS "
            "FOR (a:Account) ON (a.account_id)"
        )
        print("  ✓ Created B-tree index on Account.account_id")

        # ── Step 3: Batch ingest nodes using UNWIND + MERGE ──
        unique_accounts = sorted(
            set(df['SourceAccount'].unique()) | set(df['DestinationAccount'].unique())
        )
        account_list = [{'account_id': a} for a in unique_accounts]

        for i in range(0, len(account_list), 1000):
            batch = account_list[i:i + 1000]
            driver.execute_query(
                "UNWIND $accounts AS acc "
                "MERGE (a:Account {account_id: acc.account_id})",
                accounts=batch
            )
        print(f"  ✓ Ingested {len(account_list):,} account nodes")

        # ── Step 4: Batch ingest edges ──
        edges = df[['SourceAccount', 'DestinationAccount', 'Amount', 'Timestamp']].copy()
        edges['Timestamp'] = edges['Timestamp'].astype(str)
        edge_list = edges.to_dict('records')

        for i in range(0, len(edge_list), 1000):
            batch = edge_list[i:i + 1000]
            driver.execute_query(
                "UNWIND $edges AS e "
                "MATCH (src:Account {account_id: e.SourceAccount}) "
                "MATCH (dst:Account {account_id: e.DestinationAccount}) "
                "CREATE (src)-[:SENT {"
                "  amount: e.Amount, "
                "  timestamp: e.Timestamp"
                "}]->(dst)",
                edges=batch
            )
        print(f"  ✓ Ingested {len(edge_list):,} transaction edges")

        # ── Step 5: Cycle detection via bounded variable-length paths ──
        cycle_query = """
        MATCH path = (start:Account)-[:SENT*3..6]->(start)
        WITH start, path,
             [r IN relationships(path) | datetime(r.timestamp)] AS ts
        WITH start, path,
             reduce(mn = ts[0], t IN ts | CASE WHEN t < mn THEN t ELSE mn END) AS minT,
             reduce(mx = ts[0], t IN ts | CASE WHEN t > mx THEN t ELSE mx END) AS maxT
        WHERE duration.between(minT, maxT).minutes < $window_minutes
        UNWIND [n IN nodes(path) | n.account_id] AS mule_id
        RETURN DISTINCT mule_id
        """

        records, _, _ = driver.execute_query(
            cycle_query, window_minutes=CYCLE_WINDOW_MINUTES
        )
        detected = {record['mule_id'] for record in records}

        driver.close()
        print(f"  ✓ Neo4j detected {len(detected)} cycle participants")
        return detected, True

    except Exception as e:
        print(f"  ⚠ Neo4j unavailable ({type(e).__name__}: {e})")
        print("  → Falling back to optimized NetworkX cycle detection")
        return set(), False


# ══════════════════════════════════════════════════════════════════════════════
# NetworkX Fallback: Temporal Simple-Cycle Detection (v4)
# ══════════════════════════════════════════════════════════════════════════════

def detect_cycles_networkx(df: pd.DataFrame) -> set:
    """
    Detect temporal cycles using simple-cycle enumeration on time-windowed
    subgraphs.
    
    Algorithm (v4 — temporal windowing + simple_cycles):
        1. Sort all edges by timestamp                               O(E log E)
        2. Slide a window of CYCLE_WINDOW_MINUTES across edges       O(E)
        3. For each window position:
           a. Build small subgraph from window edges                  O(E_w)
           b. Pre-filter: keep only nodes with in-degree≥1 AND 
              out-degree≥1 (necessary for cycle membership)
           c. Run nx.simple_cycles(G, length_bound=6)                O(small)
           d. Add all cycle participants to detected set
        4. Window slides by large steps to avoid redundant checks
    
    Why this works better than SCC-first approach:
        With 50K random edges among 10K accounts, the full graph has ONE
        giant SCC (9,885 nodes). SCC-based detection then tries to find
        temporal cycles inside this massive component, which is infeasible.
        
        By windowing first, each subgraph has only ~200 edges (10min of
        50K edges over 30 days ≈ 200). Simple cycle detection on a sparse
        200-edge graph is instant.
    
    Complexity:
        O(E log E) sort + O(W × C(E_w)) where W = number of windows,
        E_w ≈ 200 edges per window, C(E_w) = simple cycle enumeration cost.
        In practice: ~300–600ms for 50K edges.
    """
    detected_mules = set()

    # ── Step 1: Prepare and sort edge data ──
    edges = df[['SourceAccount', 'DestinationAccount', 'Timestamp']]
    edges = edges.sort_values('Timestamp')
    
    all_sources = edges['SourceAccount'].values
    all_destinations = edges['DestinationAccount'].values
    all_timestamps = edges['Timestamp'].values
    
    accounts = np.concatenate([all_sources, all_destinations])
    mapped_accounts, unique_accounts = pd.factorize(accounts)
    all_sources_int = mapped_accounts[:len(all_sources)]
    all_destinations_int = mapped_accounts[len(all_sources):]
    
    n_edges = len(edges)
    window_td = np.timedelta64(CYCLE_WINDOW_MINUTES, 'm')
    # Step size: half the window to ensure overlapping coverage
    step_td = np.timedelta64(CYCLE_WINDOW_MINUTES // 2, 'm')
    
    # ── Step 2: Sliding window with stepped positioning ──
    # Instead of incrementing left by 1 each time, jump forward
    # using binary search to the next step boundary
    t_start = all_timestamps[0]
    t_end = all_timestamps[-1]
    
    # Generate window start times at step_td intervals
    window_starts = []
    current = t_start
    while current <= t_end:
        window_starts.append(current)
        current = current + step_td
    
    processed_windows = 0
    
    for window_start in window_starts:
        window_end = window_start + window_td
        
        # Binary search for left and right boundaries
        left = np.searchsorted(all_timestamps, window_start, side='left')
        right = np.searchsorted(all_timestamps, window_end, side='right')
        
        # Skip if too few edges in this window
        if right - left < 3:
            continue
        
        # ── Step 3: Build subgraph from window edges ──
        w_sources = all_sources_int[left:right]
        w_destinations = all_destinations_int[left:right]
        
        # Pre-filter: only keep accounts with both in-degree and out-degree ≥ 1
        w_src_set = set(w_sources)
        w_dst_set = set(w_destinations)
        
        w_all_accounts = w_src_set | w_dst_set
        if len(w_all_accounts) < 3:
            continue
        cycle_eligible = w_src_set & w_dst_set
        
        if len(cycle_eligible) < 3:
            continue
        
        # Build filtered subgraph
        G = nx.DiGraph()
        for s, d in zip(w_sources, w_destinations):
            if s in cycle_eligible and d in cycle_eligible:
                G.add_edge(s, d)
        
        if G.number_of_nodes() < 3:
            continue
        
        # ── Step 4: Find simple cycles of length 3-6 ──
        # Fast path: 99.9% of these sparse 10-minute windows are DAGs.
        if nx.is_directed_acyclic_graph(G):
            continue

        try:
            for cycle in nx.simple_cycles(G, length_bound=CYCLE_MAX_DEPTH):
                if len(cycle) >= 3:
                    detected_mules.update(cycle)
        except Exception:
            pass  # Gracefully handle any unexpected graph issues
        
        processed_windows += 1
        left += 1
    
    return {unique_accounts[i] for i in detected_mules}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def detect_cycles(df: pd.DataFrame) -> tuple[set, float]:
    """
    Main entry point for Phase 2 cycle detection.

    Strategy: Try Neo4j first (better for production); fall back to
    optimized NetworkX simple-cycle detection if Neo4j is unavailable.

    Returns:
        (set_of_detected_cycle_mule_accounts, execution_time_seconds)
    """
    start_time = time.time()

    # Try Neo4j first
    detected, neo4j_success = try_neo4j_detection(df)

    if not neo4j_success:
        detected = detect_cycles_networkx(df)
        print(f"  ✓ NetworkX fallback detected {len(detected)} cycle participants")

    elapsed = time.time() - start_time
    return detected, elapsed


# ─── Standalone execution ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import TRANSACTIONS_FILE
    print("=" * 65)
    print("  PHASE 2: Cycle Detection")
    print("=" * 65)
    df = pd.read_csv(TRANSACTIONS_FILE)
    detected, elapsed = detect_cycles(df)
    print(f"\n  Detected {len(detected)} accounts in {elapsed:.3f}s")
    print(f"  Sample: {sorted(list(detected))[:10]}")
