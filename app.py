"""Hugging Face Spaces entry point for AstroMind dashboard.

This file is the main entry point for the Streamlit app on Hugging Face Spaces.
"""

import subprocess
import sys
import time
import threading
import os
import socket

def is_server_running(port=8000):
    """Check if server is already running on the specified port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except:
        return False

def start_server():
    """Start the FastAPI environment server in a background thread."""
    from server.app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Only start server if it's not already running
if not is_server_running():
    print("Starting environment server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    print("Server started on port 8000")
else:
    print("Server already running on port 8000")

# Import and run the dashboard directly (not through CLI)
exec(open("dashboard.py").read())
