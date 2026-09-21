"""
run_server.py
Launches the Hospital Risk & Claim Intelligence Platform FastAPI server.
Access the web dashboard at: http://127.0.0.1:8000
Interactive OpenAPI documentation: http://127.0.0.1:8000/docs
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  Hospital Risk & Claim Intelligence Platform")
    print("  FastAPI Server starting on http://127.0.0.1:8000")
    print("  Web Dashboard: http://127.0.0.1:8000/")
    print("  Interactive API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
