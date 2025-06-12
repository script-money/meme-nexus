# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemeNexus is a Python library for interacting with meme coins and DeFi services. It provides two main client interfaces:
- **DexScreenerClient**: For fetching token data from DexScreener API
- **GeckoTerminalClient**: For fetching token data from GeckoTerminal API

The library also includes advanced charting capabilities with technical analysis indicators and utilities for formatting data.

## Development Commands

**Setup and Dependencies:**
```bash
uv sync --dev             # Install dependencies with dev group
uv pip install -e .      # Install in development mode
```

**Testing:**
```bash
uv run pytest                    # Run all tests
uv run pytest tests/clients/     # Run client tests only
uv run pytest -v                 # Verbose output
```

**Code Quality:**
```bash
uv run ruff check                # Lint code
uv run ruff format               # Format code
uv run ruff check --fix          # Auto-fix linting issues
```

## Architecture

**Client Layer** (`src/meme_nexus/clients/`)
- Async HTTP clients for external APIs
- Pydantic models for response validation
- Retry logic with tenacity for reliability

**Utilities Layer** (`src/meme_nexus/utils/`)
- `format.py`: Number and timeframe formatting utilities
- `indicators.py`: Technical analysis calculations (swing points, liquidity, order blocks, etc.)
- `draw/`: Advanced charting system with mplfinance integration
  - `main.py`: Core plotting orchestration
  - `elements.py`: Chart element creation
  - `config.py`: Styling and configuration

**Draw System Architecture:**
The draw system is modular with separate concerns:
- Chart styling and color schemes in `config.py`
- Individual chart elements (swing points, liquidity zones, etc.) in `elements.py`
- Main plotting coordination in `main.py`
- Public interface through `draw.py`

**Model Design:**
All API response models inherit from `BaseDexModel` which provides dict-like access and proper JSON serialization for datetime objects.

## Key Dependencies

- `httpx`: Async HTTP client for API calls
- `pydantic`: Data validation and serialization
- `tenacity`: Retry logic for API reliability
- `mplfinance`: Financial charting
- `smartmoneyconcepts`: Technical analysis indicators
- `ccxt`: Cryptocurrency exchange connectivity

## Python Version

Requires Python 3.11 exactly (`==3.11.*`).