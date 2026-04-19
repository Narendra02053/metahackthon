"""AstroMind FastAPI server entry point.

Creates and runs the OpenEnv-compatible FastAPI application
serving the AstroEnvironment over HTTP.

Usage:
    uvicorn server.app:app --reload --port 8000
"""

import sys
import os

# Ensure the parent directory is on the path so models.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_fastapi_app
from models import AstroAction, AstroObservation
from server.astro_environment import AstroEnvironment

# Pass the environment CLASS (not instance) to create_fastapi_app
# The server internally instantiates it on each session
app = create_fastapi_app(AstroEnvironment, AstroAction, AstroObservation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
