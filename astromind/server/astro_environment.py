"""AstroMind Mars Mission RL Environment.

Implements the OpenEnv Environment interface for a Mars mission
where an agent must pilot a spacecraft from Earth to Mars and back,
managing fuel, oxygen, power, and hull integrity across 7 mission
phases and up to 300 mission days.

This module follows the exact OpenEnv pattern:
- Extends Environment from openenv.core.env_server
- Implements reset() → Observation and step(action) → Observation
- Exposes state property returning dict
"""

import uuid
from random import random, uniform, randint

from openenv.core.env_server import Environment

from models import AstroAction, AstroObservation


# Ordered mission phases — the agent must progress through these sequentially
MISSION_PHASES = [
    "launch",        # Phase 0: Lift off from Earth
    "earth_orbit",   # Phase 1: Achieve and maintain Earth orbit
    "transit",       # Phase 2: Hohmann transfer to Mars (longest phase)
    "mars_orbit",    # Phase 3: Mars orbit insertion
    "landing",       # Phase 4: Descent to Mars surface
    "surface_ops",   # Phase 5: Surface operations & sample collection
    "return"         # Phase 6: Return journey to Earth
]

# Distance thresholds (km) required to advance from each phase
PHASE_DISTANCE_THRESHOLDS = {
    "earth_orbit": 400,          # km from Earth surface
    "transit": 0,                # just needs phase advancement command
    "mars_orbit": 1000,         # km from Mars
    "landing": 100,             # km from Mars surface
    "surface_ops": 0,           # already on surface
    "return": 225_000_000       # km back to Earth (full return)
}

# Emoji mapping for each phase (used by dashboard)
PHASE_EMOJIS = {
    "launch": "🚀",
    "earth_orbit": "🌍",
    "transit": "⭐",
    "mars_orbit": "🔴",
    "landing": "🛸",
    "surface_ops": "🪨",
    "return": "🏠"
}


class AstroEnvironment(Environment):
    """OpenEnv-compatible Mars mission RL environment.

    The agent must navigate 7 mission phases in order while managing
    4 critical resources: fuel, oxygen, power, and hull integrity.
    Random space events (solar storms, micrometeorites) add uncertainty.
    Success requires completing all phases and collecting 3 samples.
    """

    def __init__(self):
        super().__init__()
        self._reset_state()

    def _reset_state(self):
        """Initialize or reset all environment state to starting values."""
        self.mission_day = 0
        self.phase_index = 0
        self.fuel = 100.0
        self.oxygen = 100.0
        self.power = 100.0
        self.hull_integrity = 100.0
        self.distance_to_target = 225_000_000.0  # Earth-Mars distance in km
        self.samples_collected = 0
        self.anomalies = []
        self.total_reward = 0.0
        self.episode_id = str(uuid.uuid4())
        self.solar_storm_active = False
        self.micrometeorite_hit = False
        self.communication_delay = 0

    def reset(self) -> AstroObservation:
        """Reset the environment and return the initial observation.

        Called at the start of each episode. Returns an observation
        with the spacecraft on the launch pad awaiting commands.

        Returns:
            AstroObservation with initial mission state.
        """
        self._reset_state()
        return self._build_observation(
            reward=0.0,
            done=False,
            success=False,
            message="Mission initialized. Awaiting launch sequence."
        )

    def step(self, action: AstroAction) -> AstroObservation:
        """Process one action and advance the mission by one day.

        Steps through: day increment → random events → action processing →
        resource decay → failure checks → success check → return observation.

        Args:
            action: AstroAction with a command and optional parameters.

        Returns:
            AstroObservation with updated state, reward, and terminal info.
        """
        # ── Step 1: Increment mission day ──
        self.mission_day += 1

        # ── Step 2: Random space events ──
        # Solar storm: 10% chance per day, damages power
        solar_storm = random() < 0.10
        if solar_storm:
            power_loss = uniform(5, 20)
            self.power = max(0, self.power - power_loss)
            self.solar_storm_active = True
            self.anomalies.append(
                f"Day {self.mission_day}: Solar storm, lost {power_loss:.1f}% power"
            )
        else:
            self.solar_storm_active = False

        # Micrometeorite: 5% chance per day, damages hull
        micrometeorite = random() < 0.05
        if micrometeorite:
            hull_loss = uniform(3, 10)
            self.hull_integrity = max(0, self.hull_integrity - hull_loss)
            self.micrometeorite_hit = True
            self.anomalies.append(
                f"Day {self.mission_day}: Micrometeorite hit, lost {hull_loss:.1f}% hull"
            )
        else:
            self.micrometeorite_hit = False

        # ── Step 3: Process the agent's action ──
        reward = 0.0
        message = ""
        done = False
        success = False

        current_phase = MISSION_PHASES[self.phase_index]

        if action.command == "fire_thruster":
            # Burn fuel to cover distance toward target
            if self.fuel >= 2:
                fuel_cost = uniform(2, 8)
                self.fuel -= fuel_cost
                distance_covered = uniform(10_000_000, 30_000_000)
                self.distance_to_target = max(0, self.distance_to_target - distance_covered)
                reward = 10.0
                message = (
                    f"Thruster fired. Fuel: {self.fuel:.1f}%. "
                    f"Distance: {self.distance_to_target:,.0f} km"
                )
            else:
                reward = -20.0
                message = "THRUSTER FAILED: Insufficient fuel"

        elif action.command == "enter_orbit":
            phase = MISSION_PHASES[self.phase_index]

            if phase == "launch":
                # Always allow leaving launch
                self.phase_index += 1
                reward = 30.0
                message = "Earth orbit achieved"

            elif phase == "earth_orbit":
                # Always allow trans-Mars injection
                self.phase_index += 1
                reward = 30.0
                message = "Trans-Mars injection complete"

            elif phase == "transit":
                # Allow entering Mars orbit when distance < 5_000_000
                if self.distance_to_target < 5_000_000:
                    self.phase_index += 1
                    reward = 80.0
                    message = "Mars orbit insertion successful"
                else:
                    reward = -5.0
                    message = f"Too far for orbit: {self.distance_to_target:,.0f} km"

            elif phase == "landing":
                # Always allow surface ops after landing
                self.phase_index += 1
                reward = 40.0
                message = "Touchdown on Mars surface"

            else:
                reward = -5.0
                message = "Cannot enter orbit in current phase"

        elif action.command == "deploy_lander":
            # Deploy lander — only valid during mars_orbit phase
            if MISSION_PHASES[self.phase_index] == "mars_orbit":
                self.phase_index += 1
                reward = 40.0
                message = "Lander deployed. Beginning descent to Mars surface"
            else:
                reward = -15.0
                message = f"Cannot deploy lander during {MISSION_PHASES[self.phase_index]} phase"

        elif action.command == "collect_sample":
            # Collect surface sample — only valid during surface_ops phase
            if MISSION_PHASES[self.phase_index] == "surface_ops":
                self.samples_collected += 1
                reward = 30.0
                message = f"Sample {self.samples_collected} collected successfully"
                # All 3 samples collected → advance to return phase
                if self.samples_collected >= 3:
                    self.phase_index += 1
                    reward += 100.0
                    message += " — All samples collected! Initiating return sequence."
            else:
                reward = -5.0
                message = f"Cannot collect samples during {MISSION_PHASES[self.phase_index]} phase"

        elif action.command == "run_diagnostics":
            # Repair hull and recharge power slightly
            self.hull_integrity = min(100, self.hull_integrity + 5.0)
            self.power = min(100, self.power + 3.0)
            reward = 5.0
            message = (
                f"Diagnostics complete. Hull: {self.hull_integrity:.1f}%, "
                f"Power: {self.power:.1f}%"
            )

        elif action.command == "communicate_earth":
            # Send signal to Earth — informational, reveals communication delay
            delay = randint(180, 1400)
            self.communication_delay = delay
            reward = 2.0
            message = f"Signal sent to Earth. Round-trip delay: {delay}s"

        elif action.command == "emergency_abort":
            # Immediately abort the mission
            done = True
            reward = -100.0
            message = "MISSION ABORTED by agent"
            self.total_reward += reward
            return self._build_observation(reward, done=True, success=False, message=message)

        # ── Step 4: Daily resource decay ──
        # Oxygen decays steadily (crew consumption)
        self.oxygen = max(0, self.oxygen - uniform(0.5, 1.5))
        # Power decays from baseline systems
        self.power = max(0, self.power - uniform(0.3, 1.0))

        # ── Step 5: Check failure conditions (in priority order) ──
        if self.oxygen <= 0:
            return self._build_observation(
                -200, done=True, success=False,
                message=f"MISSION FAILED: Crew oxygen depleted on Day {self.mission_day}"
            )

        if self.hull_integrity <= 0:
            return self._build_observation(
                -200, done=True, success=False,
                message=f"MISSION FAILED: Hull integrity lost on Day {self.mission_day}"
            )

        current_phase = MISSION_PHASES[self.phase_index]
        if self.fuel <= 0 and current_phase not in ("return", "surface_ops"):
            return self._build_observation(
                -150, done=True, success=False,
                message=f"MISSION FAILED: Propellant exhausted on Day {self.mission_day}"
            )

        if self.mission_day >= 300:
            return self._build_observation(
                -50, done=True, success=False,
                message="MISSION FAILED: 300 day duration limit exceeded"
            )

        # ── Step 6: Check success condition ──
        # Mission succeeds when agent reaches "return" phase with 3+ samples
        if MISSION_PHASES[self.phase_index] == "return" and self.samples_collected >= 3:
            reward += 500.0
            self.total_reward += reward
            return self._build_observation(
                reward, done=True, success=True,
                message=f"MISSION SUCCESS: Mars samples returned on Day {self.mission_day}!"
            )

        # ── Step 7: Update total reward and return observation ──
        self.total_reward += reward
        return self._build_observation(reward, done=False, success=False, message=message)

    def _build_observation(self, reward: float, done: bool, success: bool, message: str) -> AstroObservation:
        """Construct an AstroObservation from the current environment state.

        Args:
            reward: Step reward value.
            done: Whether the episode is terminal.
            success: Whether the mission succeeded.
            message: Human-readable description of what happened.

        Returns:
            Fully populated AstroObservation.
        """
        return AstroObservation(
            mission_day=self.mission_day,
            phase=MISSION_PHASES[self.phase_index],
            phase_index=self.phase_index,
            fuel=self.fuel,
            oxygen=self.oxygen,
            power=self.power,
            hull_integrity=self.hull_integrity,
            distance_to_target=self.distance_to_target,
            solar_storm_active=self.solar_storm_active,
            solar_storm_risk=0.10,  # base probability
            micrometeorite_hit=self.micrometeorite_hit,
            communication_delay=self.communication_delay,
            samples_collected=self.samples_collected,
            anomalies_detected=list(self.anomalies),
            reward=reward,
            total_reward=self.total_reward,
            done=done,
            success=success,
            message=message
        )

    @property
    def state(self) -> dict:
        """Return a summary dict of the current environment state.

        Used by the OpenEnv server to expose environment state
        via the /state endpoint.
        """
        return {
            "episode_id": self.episode_id,
            "mission_day": self.mission_day,
            "phase": MISSION_PHASES[self.phase_index],
            "total_reward": self.total_reward,
            "samples_collected": self.samples_collected
        }
