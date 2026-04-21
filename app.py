"""Hugging Face Spaces entry point for AstroMind dashboard.

This file is the main entry point for the Streamlit app on Hugging Face Spaces.
"""

import subprocess
import sys
import time
import threading
import os

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

# Import and run the dashboard directly (not through CLI)
exec(open("dashboard.py").read())
