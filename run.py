"""
Entry point for the Customer Churn project.

Usage:
  python run.py pipeline   # Run the full ML pipeline (train models, generate outputs)
  python run.py api        # Start the FastAPI server (requires pipeline to have run first)
  python run.py all        # Run pipeline then start API
"""
import sys
import subprocess
from pathlib import Path

import config


def run_pipeline():
    print("Starting ML pipeline...")
    from src.pipeline import run
    run()


def run_api():
    import uvicorn
    print(f"\n[API] Starting server at http://localhost:{config.API_PORT}")
    print(f"[API] Open your browser at: http://localhost:{config.API_PORT}\n")
    uvicorn.run("api.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "pipeline":
        run_pipeline()
    elif mode == "api":
        run_api()
    elif mode == "all":
        run_pipeline()
        run_api()
    else:
        print(__doc__)
        sys.exit(1)
