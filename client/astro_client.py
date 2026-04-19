"""AstroMind OpenEnv client for connecting to the AstroEnvironment server.

Implements the EnvClient interface from openenv.core.env_client,
translating between AstroAction/AstroObservation models and
the WebSocket JSON protocol used by the server.

The client connects via WebSocket for persistent sessions,
with a .sync() wrapper for synchronous usage.
"""

import sys
import os

# Ensure the parent directory is on the path so models.py is importable
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_client import EnvClient, StepResult
from models import AstroAction, AstroObservation


class AstroEnvClient(EnvClient[AstroAction, AstroObservation, dict]):
    """WebSocket client for the AstroMind environment.

    Connects to the FastAPI server running AstroEnvironment
    via WebSocket and provides a Pythonic interface for
    reset/step/state calls.

    Usage (sync):
        client = AstroEnvClient(base_url="ws://localhost:8000").sync()
        with client:
            result = client.reset()
            result = client.step(AstroAction(command="fire_thruster"))

    Usage (async):
        async with AstroEnvClient(base_url="ws://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(AstroAction(command="fire_thruster"))
    """

    def _step_payload(self, action: AstroAction) -> dict:
        """Serialize an AstroAction into the JSON payload for the step message.

        The EnvClient sends {"type": "step", "data": <this return value>}
        over the WebSocket connection.

        Args:
            action: The AstroAction to send.

        Returns:
            Dict with command and parameters fields.
        """
        return {
            "command": action.command,
            "parameters": action.parameters or {}
        }

    def _parse_result(self, payload: dict) -> StepResult[AstroObservation]:
        """Parse the server's WebSocket response into a StepResult.

        The server returns observation data along with reward and done flags.
        We extract these into the StepResult container expected by EnvClient.

        Args:
            payload: Raw JSON dict from the server response (the "data" field).

        Returns:
            StepResult containing AstroObservation, reward, and done flag.
        """
        obs_data = payload.get("observation", payload)
        # The server returns reward/done as top-level fields alongside observation.
        # AstroObservation requires them, so merge them into the observation dict.
        if "reward" not in obs_data:
            obs_data["reward"] = payload.get("reward", 0.0)
        if "done" not in obs_data:
            obs_data["done"] = payload.get("done", False)
        return StepResult(
            observation=AstroObservation(**obs_data),
            reward=payload.get("reward", None),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> dict:
        """Parse the state response into a dict.

        Args:
            payload: Raw JSON dict from the server's state response.

        Returns:
            The state dictionary as-is.
        """
        return payload
