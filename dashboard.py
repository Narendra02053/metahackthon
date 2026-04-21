"""AstroMind Mission Control Dashboard.

Streamlit-based interactive dashboard for visualizing and
controlling the AstroMind Mars mission RL environment.

Provides:
- Real-time mission status display
- Resource gauges with color-coded warnings
- Manual action buttons + auto-pilot mode
- Mission event log and reward chart

Usage:
    1. Start the server:  uvicorn server.app:app --reload --port 8000
    2. Launch dashboard:  streamlit run dashboard.py
"""

import sys
import os
import time
import matplotlib.pyplot as plt

import streamlit as st

from client.astro_client import AstroEnvClient
from models import AstroAction

# ── Import the rule-based agent for auto-pilot mode ──
# We import the function directly to avoid running training
sys.path.insert(0, os.path.dirname(__file__))
from train import select_action
from llm_agent import get_llm_action


def select_action_safe(obs):
    """Wrapper for select_action that prevents emergency_abort in auto-pilot mode."""
    action = select_action(obs)
    # Never allow emergency_abort in auto-pilot
    if action == "emergency_abort":
        return "run_diagnostics"  # Safe fallback
    return action

# ── Configuration ──
BASE_URL = "ws://localhost:8000"

# Phase emoji mapping for visual display
PHASE_EMOJIS = {
    "launch": "🚀",
    "earth_orbit": "🌍",
    "transit": "⭐",
    "mars_orbit": "🔴",
    "landing": "🛸",
    "surface_ops": "🪨",
    "return": "🏠"
}

# Action button definitions: (command, emoji, label)
ACTION_BUTTONS = [
    ("fire_thruster", "🔥", "Fire Thrusters"),
    ("enter_orbit", "🌍", "Enter Orbit"),
    ("deploy_lander", "🛸", "Deploy Lander"),
    ("collect_sample", "🪨", "Collect Sample"),
    ("run_diagnostics", "🔧", "Run Diagnostics"),
    ("communicate_earth", "📡", "Communicate Earth"),
    ("emergency_abort", "🚨", "Emergency Abort"),
]

# ── Page Configuration ──
st.set_page_config(
    page_title="🚀 AstroMind Mission Control",
    page_icon="🚀",
    layout="wide"
)

# ── Dark Theme Styling ──
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0a0a2e;
        color: #e0e0e0;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d0d3b;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        color: #8888aa;
    }
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    /* Scrollable log container */
    .log-container {
        max-height: 250px;
        overflow-y: auto;
        background-color: #111144;
        border: 1px solid #333366;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #00ff88;
    }
    /* Warning text */
    .warning-text {
        color: #ff4444;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    /* Success text */
    .success-text {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.2rem;
    }
    /* Phase indicator */
    .phase-display {
        font-size: 1.5rem;
        text-align: center;
        padding: 10px;
        background: linear-gradient(135deg, #1a1a4e, #2a2a6e);
        border-radius: 10px;
        border: 1px solid #4444aa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables on first load."""
    if "env" not in st.session_state:
        st.session_state.env = AstroEnvClient(base_url=BASE_URL).sync()
        st.session_state.env.__enter__()  # open the sync client context

    if "obs" not in st.session_state:
        try:
            st.session_state.obs = st.session_state.env.reset()
        except Exception as e:
            st.error(f"Could not connect to server at {BASE_URL}: {e}")
            st.stop()

    if "mission_log" not in st.session_state:
        st.session_state.mission_log = [
            "System initialized. Awaiting commands..."
        ]

    if "rewards" not in st.session_state:
        st.session_state.rewards = []

    if "reward_history" not in st.session_state:
        st.session_state.reward_history = []

    if "auto_pilot" not in st.session_state:
        st.session_state.auto_pilot = False


def send_action(command: str):
    """Send an action to the environment and update session state.

    Args:
        command: The action command string to send.
    """
    try:
        obs = st.session_state.env.step(AstroAction(command=command))
        st.session_state.obs = obs

        # Log the action and result with detailed resource info
        log_entry = (
            f"Day {obs.observation.mission_day:3d} | "
            f"{command:20s} | "
            f"Reward: {obs.reward:6.1f} | "
            f"Fuel: {obs.observation.fuel:.1f}% | "
            f"Oxygen: {obs.observation.oxygen:.1f}% | "
            f"Power: {obs.observation.power:.1f}% | "
            f"Hull: {obs.observation.hull_integrity:.1f}%"
        )
        st.session_state.mission_log.append(log_entry)

        # Track rewards for the chart
        st.session_state.rewards.append(obs.reward or 0)
        st.session_state.reward_history.append(obs.reward or 0)

        # If episode ended, log it
        if obs.observation.done:
            if obs.observation.success:
                st.session_state.mission_log.append(
                    "🎉 ═══════ MISSION SUCCESS ═══════ 🎉"
                )
            else:
                # Determine failure reason
                failure_reason = "Unknown"
                if obs.observation.fuel <= 0:
                    failure_reason = "Fuel Depleted"
                elif obs.observation.oxygen <= 0:
                    failure_reason = "Oxygen Depleted"
                elif obs.observation.power <= 0:
                    failure_reason = "Power Depleted"
                elif obs.observation.hull_integrity <= 0:
                    failure_reason = "Hull Destroyed"
                st.session_state.mission_log.append(
                    f"💀 ═══════ MISSION FAILED ({failure_reason}) ═══════ 💀"
                )
    except Exception as e:
        st.session_state.mission_log.append(f"ERROR: {e}")


def reset_mission():
    """Reset the mission environment and clear logs."""
    try:
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.mission_log = [
            "Mission reset. Awaiting launch sequence..."
        ]
        st.session_state.rewards = []
        st.session_state.reward_history = []
    except Exception as e:
        st.session_state.mission_log.append(f"RESET ERROR: {e}")


# ── Initialize ──
init_session_state()

# ── Sidebar: Mission Commands ──
with st.sidebar:
    st.title("🎮 Mission Commands")
    st.markdown("---")

    # Action buttons
    for command, emoji, label in ACTION_BUTTONS:
        if st.button(f"{emoji} {label}", key=f"btn_{command}"):
            send_action(command)
            st.rerun()

    st.markdown("---")

    # Auto Pilot toggle
    auto_label = "⏸️ Disable Auto Pilot" if st.session_state.auto_pilot else "▶️ Enable Auto Pilot"
    if st.button(auto_label, key="btn_autopilot"):
        st.session_state.auto_pilot = not st.session_state.auto_pilot
        if st.session_state.auto_pilot:
            st.session_state.mission_log.append("🤖 Auto Pilot ENGAGED")
        else:
            st.session_state.mission_log.append("🤖 Auto Pilot DISENGAGED")
        st.rerun()

    # Reset Mission button
    st.markdown("---")
    if st.button("🔄 Reset Mission", key="btn_reset"):
        reset_mission()
        st.rerun()

    # Auto pilot status indicator
    if st.session_state.auto_pilot:
        st.markdown("---")
        st.info("🤖 Auto Pilot is **ACTIVE**")

# ── Auto Pilot Execution ──
# When auto-pilot is enabled, select and execute the best action automatically
if st.session_state.auto_pilot and not st.session_state.obs.observation.done:
    st.sidebar.info("🤖 Auto Pilot running 50 steps...")
    
    for step in range(50):
        if st.session_state.obs.observation.done:
            break
        
        # Use LLM agent in auto pilot mode
        try:
            action = get_llm_action(st.session_state.obs.observation)
        except:
            action = select_action(st.session_state.obs.observation)
        
        send_action(action)
        time.sleep(0.2)
    
    st.sidebar.success("🤖 Auto Pilot completed")
    st.rerun()

# Auto-reset after 2 seconds if mission failed
if st.session_state.auto_pilot and st.session_state.obs.observation.done and not st.session_state.obs.observation.success:
    time.sleep(2)
    reset_mission()
    st.rerun()

# ── Main Panel ──
obs_data = st.session_state.obs.observation

# Title bar
st.title("🚀 AstroMind Mission Control")
st.caption("Mars Mission RL Environment — Meta × PyTorch × HuggingFace OpenEnv Hackathon")

# ── Three-column layout ──
col1, col2, col3 = st.columns(3)

# ── Column 1: Mission Status ──
with col1:
    st.subheader("📍 Mission Status")

    # Current phase with emoji
    phase_emoji = PHASE_EMOJIS.get(obs_data.phase, "❓")
    phase_display = f"{phase_emoji} {obs_data.phase.replace('_', ' ').title()}"
    st.markdown(f'<div class="phase-display">{phase_display}</div>', unsafe_allow_html=True)

    st.metric("Mission Day", f"{obs_data.mission_day} / 300")
    st.metric("Samples Collected", f"{obs_data.samples_collected} / 3")
    st.metric("Distance to Target", f"{obs_data.distance_to_target:,.0f} km")

    # Success/failure indicator
    if obs_data.done:
        if obs_data.success:
            st.success("🎉 MISSION COMPLETED SUCCESSFULLY!")
        else:
            # Determine failure reason
            if obs_data.fuel <= 0:
                st.error("💀 MISSION FAILED: Fuel Depleted")
            elif obs_data.oxygen <= 0:
                st.error("💀 MISSION FAILED: Oxygen Depleted")
            elif obs_data.power <= 0:
                st.error("💀 MISSION FAILED: Power Depleted")
            elif obs_data.hull_integrity <= 0:
                st.error("💀 MISSION FAILED: Hull Destroyed")
            else:
                st.error("💀 MISSION FAILED")

# ── Column 2: Resources ──
with col2:
    st.subheader("🔋 Resources")

    # Fuel gauge
    fuel_color = "🔴" if obs_data.fuel < 20 else "🟡" if obs_data.fuel < 50 else "🟢"
    st.progress(obs_data.fuel / 100.0, text=f"{fuel_color} Fuel: {obs_data.fuel:.1f}%")

    # Oxygen gauge
    oxy_color = "🔴" if obs_data.oxygen < 20 else "🟡" if obs_data.oxygen < 50 else "🟢"
    st.progress(obs_data.oxygen / 100.0, text=f"{oxy_color} Oxygen: {obs_data.oxygen:.1f}%")

    # Power gauge
    pwr_color = "🔴" if obs_data.power < 15 else "🟡" if obs_data.power < 30 else "🟢"
    st.progress(obs_data.power / 100.0, text=f"{pwr_color} Power: {obs_data.power:.1f}%")

    # Hull gauge
    hull_color = "🔴" if obs_data.hull_integrity < 30 else "🟡" if obs_data.hull_integrity < 60 else "🟢"
    st.progress(obs_data.hull_integrity / 100.0, text=f"{hull_color} Hull: {obs_data.hull_integrity:.1f}%")

# ── Column 3: Events ──
with col3:
    st.subheader("⚡ Events")

    # Solar storm indicator
    if obs_data.solar_storm_active:
        st.markdown('<p class="warning-text">☀️ SOLAR STORM ACTIVE!</p>', unsafe_allow_html=True)
    else:
        st.markdown("☀️ Solar Storm: **Clear**")

    # Micrometeorite indicator
    if obs_data.micrometeorite_hit:
        st.markdown("☄️ Micrometeorite: **IMPACT!**")
    else:
        st.markdown("☄️ Micrometeorite: **None**")

    # Communication delay
    st.metric("Comm Delay", f"{obs_data.communication_delay}s")

    # Last reward
    last_reward = st.session_state.rewards[-1] if st.session_state.rewards else 0.0
    reward_color = "🟢" if last_reward > 0 else "🔴" if last_reward < 0 else "⚪"
    st.metric("Last Reward", f"{reward_color} {last_reward:.1f}")

    # Total reward
    st.metric("Total Reward", f"{obs_data.total_reward:.1f}")

# ── Bottom Section ──
st.markdown("---")

# Mission log (last 15 entries in scrollable box)
st.subheader("📋 Mission Log")
log_entries = st.session_state.mission_log[-15:]
log_html = "<div class='log-container'>" + "<br>".join(log_entries) + "</div>"
st.markdown(log_html, unsafe_allow_html=True)

# Reward chart using matplotlib
if st.session_state.reward_history:
    st.subheader("📈 Reward Progress Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state.reward_history, color='#00ff88', linewidth=2)
    ax.set_xlabel('Steps', color='#e0e0e0')
    ax.set_ylabel('Reward', color='#e0e0e0')
    ax.set_title('Reward Progress Over Time', color='#e0e0e0')
    ax.grid(True, alpha=0.3, color='#4444aa')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['bottom'].set_color('#4444aa')
    ax.spines['top'].set_color('#4444aa')
    ax.spines['left'].set_color('#4444aa')
    ax.spines['right'].set_color('#4444aa')
    fig.patch.set_facecolor('#0a0a2e')
    ax.set_facecolor('#0a0a2e')
    st.pyplot(fig)
