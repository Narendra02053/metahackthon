"""Hugging Face Spaces entry point.

This file serves as the entry point for deploying AstroMind to Hugging Face Spaces.
It starts the environment server and then launches the Streamlit dashboard.
"""

import subprocess
import sys
import time
import threading

def start_server():
    """Start the FastAPI environment server in a background thread."""
    from server.app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Start server in background thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait for server to start
print("Starting environment server...")
time.sleep(3)
print("Server started on port 8000")

# Import and run Streamlit dashboard
import streamlit.web.cli as stcli
sys.argv = ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
stcli.main()
