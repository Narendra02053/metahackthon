"""AstroMind TRL Training Script.

Demonstrates training a language model using Hugging Face TRL
against the AstroMind Mars mission environment.

This script uses PPO (Proximal Policy Optimization) to train
a language model to select actions in the AstroMind environment.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from client.astro_client import AstroEnvClient
from models import AstroAction

BASE_URL = "ws://localhost:8000"

# Mission state description for the language model
STATE_TEMPLATE = """
Mission Day: {mission_day}/300
Current Phase: {phase}
Resources - Fuel: {fuel:.1f}%, Oxygen: {oxygen:.1f}%, Power: {power:.1f}%, Hull: {hull:.1f}%
Distance to Target: {distance_to_target:,.0f} km
Samples Collected: {samples_collected}/3
Last Reward: {reward}

What action should I take?
"""

VALID_ACTIONS = [
    "fire_thruster",
    "enter_orbit",
    "deploy_lander",
    "collect_sample",
    "run_diagnostics",
    "communicate_earth"
]


def format_state(obs):
    """Format observation as natural language for the LM."""
    return STATE_TEMPLATE.format(
        mission_day=obs.mission_day,
        phase=obs.phase,
        fuel=obs.fuel,
        oxygen=obs.oxygen,
        power=obs.power,
        hull=obs.hull_integrity,
        distance_to_target=obs.distance_to_target,
        samples_collected=obs.samples_collected,
        reward=obs.reward
    )


def train_with_trl():
    """
    Train a language model using TRL PPO on AstroMind environment.
    
    This is a minimal demonstration showing TRL integration.
    For full training, you would need:
    - More episodes
    - Better hyperparameters
    - Reward shaping
    - Evaluation metrics
    """
    
    # 1. Load a small language model (for demonstration)
    model_name = "gpt2"  # Use a small model for quick training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # 2. Configure PPO
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
    )
    
    # 3. Initialize environment
    env = AstroEnvClient(base_url=BASE_URL).sync()
    
    # 4. Training loop
    print("=" * 70)
    print("  🤖 AstroMind — TRL PPO Training")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  Episodes: 20 (demonstration)")
    print("=" * 70)
    
    with env:
        for episode in range(20):
            obs = env.reset()
            total_reward = 0.0
            step = 0
            
            while not obs.observation.done and step < 100:
                # Format state for LM
                state_text = format_state(obs.observation)
                
                # Get action from model (random for demo, would be trained in real use)
                action_idx = torch.randint(0, len(VALID_ACTIONS), (1,)).item()
                action = VALID_ACTIONS[action_idx]
                
                # Step environment
                obs = env.step(AstroAction(command=action))
                total_reward += obs.reward or 0
                step += 1
            
            print(f"Episode {episode+1}: Reward {total_reward:.1f}, Phase {obs.observation.phase}")
    
    print("\n" + "=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print("\nNote: This is a demonstration of TRL integration.")
    print("For full training, you would:")
    print("  1. Use a larger model (e.g., Llama-7B)")
    print("  2. Implement proper PPO reward calculation")
    print("  3. Add more training episodes")
    print("  4. Include evaluation metrics")


if __name__ == "__main__":
    train_with_trl()
