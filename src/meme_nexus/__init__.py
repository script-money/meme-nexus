"""
MemeNexus: A Python library for interacting with various blockchain and DeFi services.

This package provides tools and utilities for:
- Fetching token information from DexScreener
- Analyzing token performance
- Monitoring trading activities
"""

from meme_nexus.clients.dexscreener import DexScreenerClient
from meme_nexus.core import (
    Token,
    Pair,
    Txns,
    TimeframeTxns,
    TimeframeVolume,
    TimeframePriceChange,
    Liquidity,
    Info,
    Website,
    Social,
    DexScreenerResponse,
)
from meme_nexus.exceptions import APIError

__version__ = "0.1.0"

__all__ = [
    "DexScreenerClient",
    "Token",
    "Pair",
    "Txns",
    "TimeframeTxns",
    "TimeframeVolume",
    "TimeframePriceChange",
    "Liquidity",
    "Info",
    "Website",
    "Social",
    "DexScreenerResponse",
    "APIError",
]
