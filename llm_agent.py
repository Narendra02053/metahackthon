from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VALID_ACTIONS = [
    "fire_thruster",
    "enter_orbit", 
    "deploy_lander",
    "collect_sample",
    "run_diagnostics",
    "communicate_earth"
    # Note: emergency_abort excluded — agent should never give up
]

SYSTEM_PROMPT = """You are ARIA — Autonomous Reasoning 
and Intelligence Agent — the AI commander of the 
AstroMind Mars mission.

Your job is to pilot a spacecraft from Earth to Mars 
and back, collecting 3 surface samples.

MISSION PHASES (must complete in this exact order):
1. launch → fire thrusters to leave Earth
2. earth_orbit → enter orbit to prepare for transit
3. transit → fire thrusters repeatedly until 
   distance < 5,000,000 km, then enter orbit
4. mars_orbit → deploy lander immediately
5. landing → enter orbit to reach surface
6. surface_ops → collect sample 3 times, then enter orbit
7. return → fire thrusters to return home

CRITICAL RULES:
- If oxygen < 25%: run_diagnostics IMMEDIATELY
- If hull_integrity < 25%: run_diagnostics IMMEDIATELY  
- If power < 20%: run_diagnostics IMMEDIATELY
- During solar storm: prefer run_diagnostics over fire_thruster
- Never use emergency_abort
- Always follow phase order — wrong phase actions give -15 penalty

You must respond with ONLY one of these exact action names:
fire_thruster, enter_orbit, deploy_lander, 
collect_sample, run_diagnostics, communicate_earth

No explanation. No punctuation. Just the action name."""

def get_llm_action(obs) -> str:
    """
    Ask the Groq LLM to decide the next action
    based on current mission state.
    
    Args:
        obs: AstroObservation with full mission state
    
    Returns:
        action string — one of the 6 valid actions
    """
    
    # Build detailed state description for LLM
    user_message = f"""
CURRENT MISSION STATE:
━━━━━━━━━━━━━━━━━━━━
Mission Day: {obs.mission_day} / 300
Current Phase: {obs.phase} (phase {obs.phase_index} of 6)

RESOURCES:
  Fuel:            {obs.fuel:.1f}%  {'⚠️ CRITICAL' if obs.fuel < 20 else '✅'}
  Oxygen:          {obs.oxygen:.1f}%  {'⚠️ CRITICAL' if obs.oxygen < 25 else '✅'}
  Power:           {obs.power:.1f}%  {'⚠️ CRITICAL' if obs.power < 20 else '✅'}
  Hull Integrity:  {obs.hull_integrity:.1f}%  {'⚠️ CRITICAL' if obs.hull_integrity < 25 else '✅'}

NAVIGATION:
  Distance to Target: {obs.distance_to_target:,.0f} km
  Samples Collected: {obs.samples_collected} / 3

HAZARDS:
  Solar Storm Active: {'YES ⚡' if obs.solar_storm_active else 'No'}
  Micrometeorite Hit: {'YES 💥' if obs.micrometeorite_hit else 'No'}
  Communication Delay: {obs.communication_delay}s

Last reward: {obs.reward}
Last message: {obs.message}

What is your next action?"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=10,
            temperature=0.1  # Low temperature = consistent decisions
        )
        
        action = response.choices[0].message.content.strip().lower()
        
        # Remove any punctuation just in case
        action = action.replace(".", "").replace(",", "").strip()
        
        # Validate — if LLM gives invalid action, use safe fallback
        if action in VALID_ACTIONS:
            print(f"  🤖 ARIA decides: {action}")
            return action
        else:
            print(f"  ⚠️ Invalid LLM action '{action}', using fallback")
            return _safe_fallback(obs)
            
    except Exception as e:
        print(f"  ❌ Groq API error: {e}, using fallback")
        return _safe_fallback(obs)

def _safe_fallback(obs) -> str:
    """
    Rule-based fallback if LLM fails.
    Used when Groq API is unavailable.
    """
    if obs.oxygen < 25:
        return "run_diagnostics"
    if obs.hull_integrity < 25:
        return "run_diagnostics"
    if obs.power < 20:
        return "run_diagnostics"
    
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
