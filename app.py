"""Hugging Face Spaces entry point for AstroMind dashboard.

This file is the main entry point for the Streamlit app on Hugging Face Spaces.
Environment is initialized directly without external server dependency.
"""

# Import and run the dashboard directly (not through CLI)
exec(open("dashboard.py").read())
