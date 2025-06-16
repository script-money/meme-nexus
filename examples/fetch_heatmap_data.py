#!/usr/bin/env python3
"""
Helper script to fetch appropriate liquidation heatmap data for different timeframes.

Usage:
    uv run examples/fetch_heatmap_data.py 5m    # Fetches 1D data for 5m charts
    uv run examples/fetch_heatmap_data.py 15m   # Fetches 3D data for 15m charts
    uv run examples/fetch_heatmap_data.py 30m   # Fetches 1W data for 30m charts
    uv run examples/fetch_heatmap_data.py 1h    # Fetches 2W data for 1h charts
    uv run examples/fetch_heatmap_data.py 2h    # Fetches 1M data for 2h+ charts

Configuration:
    API key can be configured via .env file:
    COINANK_API_KEY=your_api_key_here

    If no .env file is found, the script uses a default working API key.
"""

import json
import os
import subprocess
import sys

from pathlib import Path


def load_env_config():
    """Load environment configuration with fallback to defaults."""
    # Default API key (current working value)
    default_api_key = (
        "LWIzMWUtYzU0Ny1kMjk5LWI2ZDA3Yjc2MzFhYmEyYzkwM2NjfDI4NjA4MDYzODg5MzAzNDc="
    )

    # Try to load from .env file
    env_file = Path(".env")
    api_key = default_api_key

    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("COINANK_API_KEY="):
                        api_key = line.split("=", 1)[1].strip("\"'")
                        break
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}")
            print("Using default API key")
    else:
        # Check if we should create .env file
        api_key_from_env = os.getenv("COINANK_API_KEY")
        if api_key_from_env:
            api_key = api_key_from_env

    return api_key


def validate_response(response_data: str) -> bool:
    """Validate if the API response is successful."""
    try:
        data = json.loads(response_data)

        # Check for explicit failure
        if isinstance(data, dict):
            success = data.get("success", True)
            code = data.get("code", "200")
            msg = data.get("msg", "")

            if success is False or code == "403":
                print(f"❌ API returned error: code={code}, msg={msg}")
                return False

            # Check if we have actual heatmap data
            if "data" in data and "liqHeatMap" in data["data"]:
                return True
            else:
                print("❌ Response missing expected heatmap data structure")
                return False

    except json.JSONDecodeError:
        print("❌ Invalid JSON response")
        return False

    return True


def fetch_heatmap_data(timeframe: str):
    """Fetch heatmap data for the specified timeframe."""

    # Timeframe to API interval mapping
    timeframe_to_interval = {
        "5m": "1d",
        "15m": "3d",
        "30m": "1w",
        "1h": "2w",
        "2h": "1M",
        "4h": "1M",
    }

    if timeframe not in timeframe_to_interval:
        print(f"Unsupported timeframe: {timeframe}")
        print(f"Supported timeframes: {list(timeframe_to_interval.keys())}")
        sys.exit(1)

    interval = timeframe_to_interval[timeframe]
    output_file = "examples/btc_heatmap_full.json"  # Always save as this name

    # Load API key from configuration
    api_key = load_env_config()

    # API endpoint and headers
    url = f"https://api.coinank.com/api/liqMap/getLiqHeatMap?exchangeName=Binance&symbol=BTCUSDT&interval={interval}"
    headers = [
        "-H",
        "client: web",
        "-H",
        f"coinank-apikey: {api_key}",
        "-H",
        "Referer: https://coinank.com/",
    ]

    print(f"Fetching {interval} liquidation heatmap data for {timeframe} charts...")
    print(f"API URL: {url}")
    print(f"Output file: {output_file}")

    # Construct curl command
    cmd = ["curl", url, *headers]

    try:
        # Execute curl command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Curl command failed: {result.stderr}")
            return

        # Validate response before saving
        if not validate_response(result.stdout):
            print("❌ Invalid or failed API response. Not saving file.")
            print("")
            print("🌐 Please try fetching heatmap data manually from the web:")
            print("   1. Go to https://coinank.com/liquidation-heatmap")
            print("   2. Select BTC/USDT")
            print(f"   3. Set timeframe to {interval}")
            print("   4. Export or copy the data")
            print(f"   5. Save as {output_file}")
            return

        # Save successful response
        with open(output_file, "w") as f:
            f.write(result.stdout)

        print(f"✅ Successfully saved heatmap data to {output_file}")
        print(
            f"Now you can run: "
            f"uv run examples/draw_btc_liquidation_heatmap.py {timeframe}"
        )

    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run examples/fetch_heatmap_data.py <timeframe>")
        print("Supported timeframes: 5m, 15m, 30m, 1h, 2h, 4h")
        print()
        print("Examples:")
        print("  uv run examples/fetch_heatmap_data.py 5m   # Fetch 1D data")
        print("  uv run examples/fetch_heatmap_data.py 15m  # Fetch 3D data")
        print("  python examples/fetch_heatmap_data.py 2h   # Fetch 1M data")
        sys.exit(1)

    timeframe = sys.argv[1]
    fetch_heatmap_data(timeframe)


if __name__ == "__main__":
    main()
