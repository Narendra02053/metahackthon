# 🚀 AstroMind — Mars Mission RL Environment

Built for **Meta × PyTorch × HuggingFace OpenEnv Hackathon** at Scaler School of Technology, Bangalore (April 25-26, 2026).

## Overview

AstroMind is a long-horizon reinforcement learning environment that simulates a 300-day Mars mission. The agent must autonomously pilot a spacecraft from Earth to Mars, collect 3 surface samples, and return safely — all while managing critical resources and handling random environmental hazards.

### Problem Solved

Traditional RL environments focus on short-term decision making. AstroMind challenges agents with **genuine long-horizon planning** where decisions made on Day 1 directly impact survival on Day 280. The agent must:

- **Manage resources** across 7 mission phases
- **Handle random failures** (solar storms, micrometeorites)
- **Complete sequential objectives** in the correct order
- **Balance exploration vs exploitation** under uncertainty

This environment is designed to test LLM agents' ability to reason about temporal dependencies and multi-step planning.

## Features

- **7 Mission Phases**: Launch → Earth Orbit → Transit → Mars Orbit → Landing → Surface Ops → Return
- **4 Managed Resources**: Fuel, Oxygen, Power, Hull Integrity (all 0-100%)
- **Random Events**: Solar storms (power drain), Micrometeorites (hull damage)
- **Reward-Driven Learning**: Clear progression incentives with failure penalties
- **Multiple Agent Types**: Rule-based, Epsilon-greedy, and LLM-powered (Groq)
- **Interactive Dashboard**: Streamlit-based Mission Control for real-time visualization
- **OpenEnv Compliant**: Built on the official OpenEnv specification

## Setup

```bash
# Install the package
pip install -e .
```

## Run Environment

```bash
# Terminal 1: Start the environment server
uvicorn server.app:app --reload --port 8000
```

## Run Training

```bash
# Baseline rule-based agent (500 episodes)
python train.py

# LLM-powered agent using Groq (20 episodes)
python train.py --llm

# Hugging Face TRL PPO training (demonstration)
python trl_train.py
```

**Note**: Set your Groq API key in `.env` file before using LLM agent:
```
GROQ_API_KEY=gsk_your_api_key_here
```

## Run Dashboard

```bash
# Launch the Mission Control dashboard
streamlit run dashboard.py
```

The dashboard provides:
- Real-time resource gauges
- Manual action buttons
- Auto-pilot mode (uses LLM agent)
- Mission event log
- Reward history chart

## Environment Details

| Property | Value |
|---|---|
| Actions | 7 discrete commands |
| Observation space | 15 fields |
| Max episode length | 300 steps |
| Success condition | Complete all 7 phases + collect 3 samples |
| Failure conditions | Resource depletion, hull destruction, timeout |

### Mission Phases (must be completed in order)

1. 🚀 **Launch** — Lift off from Earth surface
2. 🌍 **Earth Orbit** — Achieve stable orbit around Earth
3. ⭐ **Transit** — Hohmann transfer to Mars (longest phase, ~100M km)
4. 🔴 **Mars Orbit** — Orbital insertion at Mars
5. 🛸 **Landing** — Descent to Mars surface
6. 🪨 **Surface Ops** — Collect 3 surface samples
7. 🏠 **Return** — Journey back to Earth

### Actions

| Command | Effect | Phase Restrictions |
|---|---|---|
| `fire_thruster` | Burn fuel to reduce distance to target | All phases |
| `enter_orbit` | Attempt orbital insertion (phase-dependent) | Launch, Earth Orbit, Transit, Landing |
| `deploy_lander` | Deploy lander from Mars orbit | Mars Orbit only |
| `collect_sample` | Collect surface sample | Surface Ops only |
| `run_diagnostics` | Repair hull (+5%) and recharge power (+3%) | All phases |
| `communicate_earth` | Send signal (informational) | All phases |
| `emergency_abort` | Abort the mission immediately | All phases |

## Reward Structure

The reward function is designed to encourage efficient mission completion while penalizing failures:

### Positive Rewards

| Event | Reward | Rationale |
|---|---|---|
| Mission success | **+500** | Maximum reward for completing full mission |
| Sample collected | **+30 each** | Encourages primary objective |
| Phase progression | **+30 to +80** | Rewards progress through mission stages |
| Thruster fired | **+10** | Encourages forward progress |
| Diagnostics run | **+5** | Small reward for resource management |

### Negative Rewards (Penalties)

| Event | Reward | Rationale |
|---|---|---|
| Oxygen depleted | **-200** | Critical failure condition |
| Hull integrity lost | **-200** | Critical failure condition |
| Fuel exhausted | **-150** | Critical failure condition |
| Duration exceeded (300 days) | **-50** | Time penalty |
| Emergency abort | **-100** | Gives up on mission |
| Invalid action (wrong phase) | **-5 to -15** | Discourages phase skipping |
| Random event damage | **Variable** | Solar storms, micrometeorites cause resource loss |

### Reward Design Philosophy

- **Phase completion rewards** (+30 to +80) are tiered: earlier phases give less, later phases give more
- **Resource failures** (-150 to -200) are severe to encourage proactive management
- **Mission success** (+500) provides a clear target for optimization
- **Small action rewards** (+5 to +10) encourage progress without encouraging spam

## Training Results

### Rule-Based Agent (500 Episodes)

![Reward Curve](rewards.png)

The rule-based agent shows learning through epsilon-greedy exploration:
- Episodes 0-100: High exploration, low rewards
- Episodes 100-300: Mixed exploration + smart policy
- Episodes 300-500: Mostly smart policy, higher rewards

### LLM Agent (Groq Llama 3.3)

The LLM agent (`python train.py --llm`) uses ARIA (Autonomous Reasoning and Intelligence Agent) to reason about mission state and select actions. Results show the LLM can understand phase progression and resource management.

## Architecture

```
metahack2/
├── models.py              # Pydantic action/observation models
├── reward.py              # Reward shaping utilities
├── llm_agent.py           # Groq LLM agent (ARIA)
├── server/
│   ├── __init__.py
│   ├── astro_environment.py  # OpenEnv Environment implementation
│   └── app.py             # FastAPI app entry point
├── client/
│   ├── __init__.py
│   └── astro_client.py    # OpenEnv EnvClient implementation (WebSocket)
├── train.py               # Training script (rule-based + LLM)
├── dashboard.py           # Streamlit Mission Control dashboard
├── pyproject.toml         # Package configuration
├── .env                   # API keys (gitignored)
├── .gitignore             # Protects sensitive files
└── README.md
```

## OpenEnv Compliance

AstroMind follows the official OpenEnv specification:

- **Server**: `AstroEnvironment` extends `Environment` from `openenv.core.env_server`
- **FastAPI app**: Created via `create_fastapi_app(env, AstroAction, AstroObservation)`
- **Client**: `AstroEnvClient` extends `EnvClient` from `openenv.core.env_client` (WebSocket-based)
- **Interface**: `reset()` → Observation, `step(action)` → Observation, `state` → dict

## Hugging Face Integration

The environment is designed to work with Hugging Face TRL for training LLM agents:

```python
from trl import PPOTrainer, PPOConfig
# Train your LLM agent against the AstroMind environment
```

**Current Status**: Baseline training uses rule-based and Groq LLM agents. Full TRL integration can be added by connecting the OpenEnv client to TRL's training loop.

## License

MIT License - See LICENSE file for details

## Hugging Face Spaces Deployment

AstroMind can be deployed to Hugging Face Spaces for online demonstration:

### Quick Deploy

1. Create a new Space at https://huggingface.co/spaces
2. Choose "Streamlit" as the SDK
3. Upload these files:
   - `app.py` (entry point)
   - `dashboard.py`
   - `server/` folder
   - `client/` folder
   - `models.py`
   - `reward.py`
   - `llm_agent.py`
   - `pyproject.toml`
   - `.env` (with your API keys)

4. The Space will automatically start the environment server and launch the dashboard

### Live Demo

Try the interactive demo at: [Your Hugging Face Space URL]

### Deployment Notes

- The `app.py` file starts the environment server in a background thread
- Streamlit dashboard runs on port 8501
- Environment server runs on port 8000
- Set your Groq API key in Space secrets as `GROQ_API_KEY`
