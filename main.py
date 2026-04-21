"""
Entry point.

Run with:
    python main.py

Or directly with uvicorn:
    uvicorn multiagent_sds.api.app:app --reload --host 0.0.0.0 --port 8000
"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "multiagent_sds.api.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
