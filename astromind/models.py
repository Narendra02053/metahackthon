from pydantic import BaseModel
from typing import Literal, Optional, List


class AstroAction(BaseModel):
    """Action space for the AstroMind Mars mission environment.

    The agent can choose from 7 commands:
    - fire_thruster: Burn fuel to reduce distance to target
    - enter_orbit: Attempt orbital insertion at current position
    - deploy_lander: Deploy lander from Mars orbit
    - collect_sample: Collect surface sample during surface_ops phase
    - run_diagnostics: Repair hull and recharge power slightly
    - communicate_earth: Send signal to Earth (informational)
    - emergency_abort: Abort the mission immediately
    """
    command: Literal[
        "fire_thruster",
        "enter_orbit",
        "deploy_lander",
        "collect_sample",
        "run_diagnostics",
        "communicate_earth",
        "emergency_abort"
    ]
    parameters: Optional[dict] = {}


class AstroObservation(BaseModel):
    """Observation space for the AstroMind Mars mission environment.

    Contains all information the agent needs to make decisions:
    - Time tracking (mission_day)
    - Current mission phase and progress
    - Resource levels (fuel, oxygen, power, hull_integrity)
    - Navigation data (distance_to_target)
    - Environmental hazards (solar storms, micrometeorites)
    - Communication status
    - RL signals (reward, done, success)
    """
    # Time
    mission_day: int

    # Mission phase
    phase: str
    phase_index: int

    # Resources (all 0-100)
    fuel: float
    oxygen: float
    power: float
    hull_integrity: float

    # Navigation
    distance_to_target: float

    # Environment hazards
    solar_storm_active: bool
    solar_storm_risk: float
    micrometeorite_hit: bool

    # Communication
    communication_delay: int

    # Mission progress
    samples_collected: int
    anomalies_detected: List[str]

    # RL signals
    reward: float
    total_reward: float
    done: bool
    success: bool
    message: str
