# UPI Mule Detector

A graph-based system for detecting **UPI (Unified Payments Interface) mule accounts** — accounts used to launder or rapidly move fraudulent funds. It generates a synthetic transaction dataset with two known money-laundering topologies injected into it, detects those topologies using graph algorithms, and evaluates detection accuracy against ground truth.

```
╔═══════════════════════════════════════════════════════════╗
║  UPI Mule Account Detection — Graph Database Analyzer       ║
║  10,000 accounts · Star & Cycle topology detection          ║
╚═══════════════════════════════════════════════════════════╝
```

## How It Works

The system models UPI accounts as nodes and transactions as directed, timestamped edges in a graph, then looks for two fraud patterns:

| Pattern | Description | Detected By |
|---|---|---|
| **Star Topology** (Pattern A) | A hub account receives one large lump sum, then rapidly disperses it to 6–12 "spoke" accounts within a 10-minute window | Time-weighted out-degree velocity (NetworkX) |
| **Circular Loop** (Pattern B) | Funds are passed through a 3–6 hop cycle of accounts back to the origin, within a 5-minute window ("layering") | Temporal simple-cycle detection (Neo4j, with NetworkX fallback) |

The pipeline runs in four phases:

1. **Data Generation** — Creates 10,000 synthetic accounts and ~50,000 background transactions, then injects 30 star patterns and 25 cycle patterns as labeled ground truth.
2. **Cycle Detection** — Attempts to detect Pattern B using Neo4j (via bounded variable-length Cypher paths); falls back to a windowed NetworkX `simple_cycles` approach if Neo4j isn't running.
3. **Velocity Detection** — Detects Pattern A using a sliding-window scan of each account's outgoing transactions for a receive-then-rapidly-disperse signature.
4. **Evaluation** — Combines both detectors into a ranked, risk-scored list of suspects and computes precision, recall, and F1-score against ground truth.

**Performance targets:** F1-score > 0.85, combined detection time (Phase 2 + Phase 3) < 2.0s.

## Project Structure

```
upi-mule-detector/
├── main.py                      # CLI orchestrator — runs all 4 phases
├── api.py                       # FastAPI server exposing the pipeline over HTTP
├── config.py                    # Centralized tunable parameters & env-based secrets
├── phase1_data_generation.py    # Synthetic dataset + mule pattern injection
├── phase2_neo4j_cycles.py       # Cycle detection (Neo4j primary, NetworkX fallback)
├── phase3_networkx_velocity.py  # Star-topology velocity detection
├── phase4_evaluation.py         # Metrics, ranking, and report generation
├── notebook.ipynb / notebook.py # Exploratory / walkthrough notebook
├── frontend/                    # Static frontend served by the API
├── data/                        # Generated transactions, ground truth, results (created at runtime)
└── requirements.txt
```

## Requirements

- Python 3.11 recommended (avoids build-tool issues with newer versions on Windows)
- Optional: a running [Neo4j](https://neo4j.com/) instance for accelerated cycle detection (the system automatically falls back to NetworkX if Neo4j isn't reachable — no setup required to run it)

## Setup (Windows)

Open PowerShell in the project folder. Because your Windows username contains a space (`DURGA PRASAD`), always wrap paths in quotes.

```powershell
cd "C:\Users\DURGA PRASAD\.gemini\antigravity-ide\scratch\upi-mule-detector"

# Create and activate a virtual environment with Python 3.11
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> If PowerShell blocks script execution, run this once as Administrator:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Usage

### Run the full pipeline (CLI)

```powershell
python main.py
```

This generates the dataset, runs both detectors, prints a full evaluation report to the console, and saves results to `data/results.csv` and `data/ground_truth.json`.

### Run as a web API

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser (serves `frontend/index.html`).

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend dashboard |
| `/api/run` | POST | Executes the full 4-phase pipeline (rate-limited to 1 concurrent run) |
| `/api/results` | GET | Returns cached/persisted results (30s cache TTL) |
| `/api/health` | GET | Health check + pipeline running status |

### Optional: enable Neo4j-backed cycle detection

By default, cycle detection silently falls back to NetworkX if no Neo4j instance is found. To use Neo4j instead, install it (e.g. via [Neo4j Desktop](https://neo4j.com/download/) or Docker) and set environment variables before running:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "your_password"
python main.py
```

## Configuration

All tunable parameters live in `config.py`, including:

- Dataset size (`NUM_ACCOUNTS`, `NUM_NORMAL_TRANSACTIONS`)
- Injected pattern counts (`NUM_STAR_MULES`, `NUM_CYCLE_GROUPS`)
- Detection thresholds (`VELOCITY_THRESHOLD_DESTINATIONS`, `CYCLE_WINDOW_MINUTES`, etc.)
- Neo4j connection details and allowed CORS origins (read from environment variables)

## Security Notes

- The API restricts CORS to `ALLOWED_ORIGINS` (configurable via env var), sets standard security headers (CSP, X-Frame-Options, etc.), caps request body size at 1 MB, and sanitizes error responses to avoid leaking internal tracebacks.
- Neo4j ingestion uses parameterized `UNWIND` queries to avoid Cypher injection.
- Default Neo4j credentials in `config.py` are for **local development only** — override them via environment variables in any shared or production environment.

## Tech Stack

Python · Pandas / NumPy · NetworkX · Neo4j · FastAPI · Uvicorn · scikit-learn
