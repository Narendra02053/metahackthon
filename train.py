"""AstroMind Training Script.

Runs RL training episodes against the AstroMind environment server
using a rule-based smart agent as baseline. This agent is designed
to occasionally complete the full mission, demonstrating that the
environment is solvable with good planning.

In the hackathon, this rule-based agent gets replaced by an
LLM-powered agent trained with TRL (Transformer Reinforcement Learning).

Usage:
    1. Start the server:  uvicorn server.app:app --reload --port 8000
    2. Run training:     python train.py
"""

import random
import math

import matplotlib.pyplot as plt

from client.astro_client import AstroEnvClient
from models import AstroAction
from llm_agent import get_llm_action

BASE_URL = "ws://localhost:8000"

# All available actions in the AstroMind environment
ACTIONS = [
    "fire_thruster",
    "enter_orbit",
    "deploy_lander",
    "collect_sample",
    "run_diagnostics",
    "communicate_earth",
    "emergency_abort"
]

NUM_EPISODES = 500


def select_action(obs, episode=0) -> str:
    """Epsilon-greedy smart agent for baseline training.

    Early episodes explore randomly with high epsilon,
    late episodes follow smart policy with low epsilon.

    Args:
        obs: AstroObservation with current mission state.
        episode: Current episode number for epsilon decay.

    Returns:
        Command string for the selected action.
    """
    # Exploration decreases over time
    epsilon = max(0.05, 1.0 - (episode / 300))

    if random.random() < epsilon:
        # Early episodes: explore randomly
        # but never abort, weighted toward thrust
        return random.choice([
            "fire_thruster",
            "fire_thruster",
            "fire_thruster",  # weighted toward thrust
            "run_diagnostics",
            "enter_orbit",
            "communicate_earth"
        ])

    # Late episodes: follow smart policy
    if obs.oxygen < 25:
        return "run_diagnostics"
    if obs.hull_integrity < 25:
        return "run_diagnostics"
    if obs.power < 20:
        return "run_diagnostics"
    if obs.mission_day % 15 == 0:
        return "communicate_earth"

    phase = obs.phase
    if phase == "launch":
        return "fire_thruster"
    elif phase == "earth_orbit":
        return "enter_orbit"
    elif phase == "transit":
        if obs.distance_to_target > 5_000_000:
            return "fire_thruster"
        return "enter_orbit"
    elif phase == "mars_orbit":
        return "deploy_lander"
    elif phase == "landing":
        return "enter_orbit"
    elif phase == "surface_ops":
        if obs.samples_collected < 3:
            return "collect_sample"
        return "enter_orbit"
    elif phase == "return":
        return "fire_thruster"
    else:
        return "run_diagnostics"


def run_training():
    """Execute the training loop over NUM_EPISODES episodes.

    Connects to the AstroMind environment server, runs episodes
    using the rule-based agent, and plots the reward curve at the end.
    """
    rewards_per_episode = []

    print("=" * 70)
    print("  🚀 AstroMind Training — Rule-Based Baseline Agent")
    print("=" * 70)
    print(f"  Server: {BASE_URL}")
    print(f"  Episodes: {NUM_EPISODES}")
    print("=" * 70)

    # Create a sync wrapper around the async WebSocket client
    env = AstroEnvClient(base_url=BASE_URL).sync()
    with env:
        for episode in range(NUM_EPISODES):
            obs = env.reset()
            total_reward = 0.0
            step = 0

            while not obs.observation.done:
                # Select action using the epsilon-greedy agent
                action = select_action(obs.observation, episode)
                obs = env.step(AstroAction(command=action))
                total_reward += obs.reward or 0
                step += 1

                # Safety limit to prevent infinite loops
                if step > 500:
                    break

            rewards_per_episode.append(total_reward)

            # Log progress every episode
            final_phase = obs.observation.phase
            samples = obs.observation.samples_collected
            success_marker = "✅" if obs.observation.success else "❌"
            print(
                f"  Episode {episode:03d} | Steps: {step:3d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Phase: {final_phase} {success_marker} | "
                f"Samples: {samples}"
            )

    # ── Plot reward curve ──
    plt.figure(figsize=(14, 6))
    plt.style.use('dark_background')

    # Raw rewards
    plt.plot(rewards_per_episode,
             alpha=0.3, color='cyan',
             linewidth=0.8, label='Episode Reward')

    # Moving average
    window = 30
    moving_avg = []
    for i in range(len(rewards_per_episode)):
        start = max(0, i - window)
        moving_avg.append(
            sum(rewards_per_episode[start:i + 1]) /
            (i - start + 1)
        )
    plt.plot(moving_avg, color='red',
             linewidth=2.5, label=f'Moving Avg ({window})')

    # Add horizontal line at 0
    plt.axhline(y=0, color='white',
               linestyle='--', alpha=0.3,
               label='Baseline')

    # Add annotation for best episode
    best_ep = rewards_per_episode.index(
        max(rewards_per_episode))
    best_reward = max(rewards_per_episode)
    plt.annotate(f'Best: {best_reward:.0f}',
        xy=(best_ep, best_reward),
        xytext=(best_ep + 20, best_reward + 50),
        color='yellow', fontsize=10,
        arrowprops=dict(arrowstyle='->',
        color='yellow'))

    plt.title('AstroMind — Agent Learning to Plan Mars Mission',
              fontsize=14, color='white', pad=15)
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Total Reward', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('rewards.png', dpi=150,
                facecolor='#0d0d0d')
    plt.show()
    print(f"Best episode: {best_ep} with reward {best_reward:.1f}")
    print(f"Final avg reward: {moving_avg[-1]:.1f}")


def run_llm_training():
    """
    Run training with LLM agent instead of rule-based agent.
    The LLM reasons about mission state and decides actions.
    """
    rewards_per_episode = []
    
    print("=" * 70)
    print(" 🤖 AstroMind — ARIA LLM Agent Training")
    print("=" * 70)
    
    env = AstroEnvClient(base_url=BASE_URL).sync()
    
    NUM_LLM_EPISODES = 20  # LLM is slower, 20 is enough to show
    
    with env:
        for episode in range(NUM_LLM_EPISODES):
            obs = env.reset()
            total_reward = 0.0
            step = 0
            
            print(f"\n--- Episode {episode+1} ---")
            
            while not obs.observation.done:
                # LLM decides the action
                action = get_llm_action(obs.observation)
                obs = env.step(AstroAction(command=action))
                total_reward += obs.reward or 0
                step += 1
                
                if step > 500:
                    break
            
            rewards_per_episode.append(total_reward)
            status = "✅ SUCCESS" if obs.observation.success else "❌ FAILED"
            print(f"\nEpisode {episode+1}: {status} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Phase: {obs.observation.phase} | "
                  f"Steps: {step}")
    
    # Plot LLM reward curve
    plt.figure(figsize=(12, 5))
    plt.style.use('dark_background')
    plt.plot(rewards_per_episode, 
             color='cyan', linewidth=2, 
             marker='o', markersize=6,
             label='LLM Agent Reward')
    plt.axhline(y=0, color='white', 
                linestyle='--', alpha=0.3)
    plt.title('AstroMind — ARIA LLM Agent Performance',
              fontsize=14)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('llm_rewards.png', dpi=150,
                facecolor='#0d0d0d')
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--llm":
        run_llm_training()  # python train.py --llm
    else:
        run_training()      # python train.py
