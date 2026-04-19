"""Reward shaping utilities for AstroMind environment.

Provides a supplementary reward computation based on observation state,
useful for reward shaping or evaluation outside the environment's
intrinsic step-level rewards.
"""

PHASE_SCORES = {
    "launch": 0,
    "earth_orbit": 10,
    "transit": 20,
    "mars_orbit": 40,
    "landing": 60,
    "surface_ops": 80,
    "return": 100
}


def compute_reward(obs: dict) -> float:
    """Compute a shaped reward from the current observation.

    Combines resource health, phase progress, sample count, and
    distance-to-target into a single scalar reward signal.

    Args:
        obs: Dictionary with observation fields (fuel, oxygen,
             hull_integrity, samples_collected, phase,
             distance_to_target).

    Returns:
        Rounded float reward value.
    """
    reward = 0.0
    # Resource conservation bonuses
    reward += obs["fuel"] * 0.1
    reward += obs["oxygen"] * 0.15
    reward += obs["hull_integrity"] * 0.1
    # Sample collection bonus
    reward += obs["samples_collected"] * 30.0
    # Phase progression bonus
    reward += PHASE_SCORES.get(obs["phase"], 0)
    # Distance progress bonus (0 at Earth, 50 at Mars)
    progress = 1 - (obs["distance_to_target"] / 225_000_000)
    reward += max(0, progress) * 50
    return round(reward, 2)
