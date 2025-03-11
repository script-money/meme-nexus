"""
MemeNexus: A Python library for interacting with various blockchain and DeFi services.

This package provides tools and utilities for:
- Fetching token information from DexScreener
- Analyzing token performance
- Monitoring trading activities
"""

import logging
import os

# Set SMC_CREDIT to 0 to disable the thank you message from SmartMoneyConcepts
os.environ["SMC_CREDIT"] = "0"

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
