"""
UPI Mule Detector — FastAPI Server
====================================
Security features:
    - CORS with restricted origins
    - Security headers (CSP, X-Frame-Options, etc.)
    - Rate limiting via asyncio semaphore (max 1 concurrent pipeline run)
    - Sanitized error responses (no internal tracebacks leaked)
    - Request body size limits via middleware

Performance features:
    - Async pipeline execution (non-blocking event loop)
    - In-memory result caching (avoids re-reading CSV on every /api/results call)
    - Phase-level timing returned in API responses
"""

import asyncio
import logging
import os
import json
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from config import ALLOWED_ORIGINS, RESULTS_FILE, GROUND_TRUTH_FILE

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("upi_mule_detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="UPI Mule Detector API", docs_url=None, redoc_url=None)

# ── CORS Middleware (restricted origins) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Security Headers Middleware ──────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers into every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ── Request Size Limit Middleware ────────────────────────────────────────────
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with oversized bodies to prevent abuse."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)

# ── Static Files ─────────────────────────────────────────────────────────────
os.makedirs("frontend", exist_ok=True)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ── Rate Limiting (max 1 concurrent pipeline execution) ──────────────────────
_pipeline_semaphore = asyncio.Semaphore(1)
_pipeline_running = False

# ── Result Cache ─────────────────────────────────────────────────────────────
_result_cache = {"data": None, "timestamp": 0.0}
CACHE_TTL_SECONDS = 30


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")


@app.post("/api/run")
async def run_pipeline():
    """
    Execute the full 4-phase mule detection pipeline.

    Security:
        - Semaphore-guarded: only 1 concurrent execution allowed
        - Runs in a thread pool to avoid blocking the async event loop
        - Errors are sanitized before returning to the client
    """
    global _pipeline_running

    if _pipeline_running:
        raise HTTPException(
            status_code=429,
            detail="Pipeline is already running. Please wait for it to complete.",
        )

    acquired = _pipeline_semaphore.locked()
    if acquired:
        raise HTTPException(status_code=429, detail="Pipeline is already running.")

    async with _pipeline_semaphore:
        _pipeline_running = True
        try:
            results = await asyncio.to_thread(_run_pipeline_sync)

            # Convert DataFrame for JSON serialization
            if "ranked_df" in results:
                results["ranked_df"] = results["ranked_df"].to_dict(orient="records")

            # Invalidate cache so /api/results picks up new data
            _result_cache["data"] = None
            _result_cache["timestamp"] = 0.0

            return {"status": "success", "results": results}

        except Exception as e:
            logger.error("Pipeline execution failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Pipeline execution failed. Check server logs for details.",
            )
        finally:
            _pipeline_running = False


def _run_pipeline_sync() -> dict:
    """
    Synchronous pipeline execution (runs in thread pool).
    Returns results dict with phase-level timing.
    """
    from main import main
    return main()


@app.get("/api/results")
async def get_results():
    """
    Return cached pipeline results.
    Re-reads from disk only if cache is stale (> CACHE_TTL_SECONDS).
    """
    now = time.time()

    # Return cached result if fresh
    if _result_cache["data"] and (now - _result_cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _result_cache["data"]

    try:
        if not os.path.exists(RESULTS_FILE) or not os.path.exists(GROUND_TRUTH_FILE):
            return {"status": "error", "message": "Results not found. Run pipeline first."}

        df = pd.read_csv(RESULTS_FILE)
        with open(GROUND_TRUTH_FILE, "r") as f:
            ground_truth = json.load(f)

        response_data = {
            "status": "success",
            "results": {
                "ranked_df": df.to_dict(orient="records"),
                "total_mules": len(ground_truth.get("all_mules", [])),
            },
        }

        # Update cache
        _result_cache["data"] = response_data
        _result_cache["timestamp"] = now

        return response_data

    except Exception as e:
        logger.error("Failed to load results: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to load results. Check server logs for details.",
        )


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "pipeline_running": _pipeline_running}
