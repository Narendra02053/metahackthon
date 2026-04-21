---
title: AstroMind - Mars Mission RL Environment
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
pinned: false
license: mit
---

# AstroMind — Mars Mission RL Environment

Interactive demo of the AstroMind Mars mission reinforcement learning environment.

## Features

- 🎮 **Interactive Dashboard**: Control the mission manually or watch AI agents pilot it
- 🤖 **AI Agents**: Rule-based, epsilon-greedy, and LLM-powered (Groq) agents
- 📊 **Real-time Visualization**: Resource gauges, mission log, reward history
- 🚀 **7 Mission Phases**: Launch → Earth Orbit → Transit → Mars Orbit → Landing → Surface Ops → Return

## How to Use

1. The environment server starts automatically
2. Use the sidebar to select actions manually
3. Enable "Auto Pilot" to watch AI agents play
4. Monitor resources and mission progress in real-time

## Training

This demo uses pre-trained agents. For training your own agents, clone the repository and run:

```bash
python train.py --llm
```

## About

Built for **Meta × PyTorch × HuggingFace OpenEnv Hackathon** at Scaler School of Technology, Bangalore (April 25-26, 2026).
