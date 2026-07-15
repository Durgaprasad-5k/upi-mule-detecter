"""
Configuration constants for the UPI Mule Account Detection system.
All tunable parameters are centralized here for reproducibility.

Security: Sensitive values (Neo4j credentials) are loaded from
environment variables with fallbacks for local development only.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Neo4j Connection (loaded from environment variables)
# ──────────────────────────────────────────────────────────────────────────────
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# ──────────────────────────────────────────────────────────────────────────────
# Data Generation Parameters
# ──────────────────────────────────────────────────────────────────────────────
NUM_ACCOUNTS = 10_000                # Total UPI accounts (graph nodes)
NUM_NORMAL_TRANSACTIONS = 50_000     # Background noise transactions (graph edges)
NUM_STAR_MULES = 30                  # Pattern A: star-topology mule hubs to inject
NUM_CYCLE_GROUPS = 25                # Pattern B: circular-loop cycles to inject
STAR_MIN_DESTINATIONS = 6           # Min spokes per star hub
STAR_MAX_DESTINATIONS = 12          # Max spokes per star hub
CYCLE_MIN_LENGTH = 3                # Min hops in a circular loop
CYCLE_MAX_LENGTH = 6                # Max hops in a circular loop

# ──────────────────────────────────────────────────────────────────────────────
# Detection Parameters
# ──────────────────────────────────────────────────────────────────────────────
VELOCITY_THRESHOLD_DESTINATIONS = 5  # Flag if > this many distinct destinations
VELOCITY_WINDOW_MINUTES = 10         # Time window for velocity calculation
CYCLE_WINDOW_MINUTES = 10            # Time window for cycle temporal filtering
CYCLE_MAX_DEPTH = 6                  # Max cycle length for Neo4j traversal

# ──────────────────────────────────────────────────────────────────────────────
# File Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
TRANSACTIONS_FILE = "data/transactions.csv"
GROUND_TRUTH_FILE = "data/ground_truth.json"
RESULTS_FILE = "data/results.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# API Security
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")
