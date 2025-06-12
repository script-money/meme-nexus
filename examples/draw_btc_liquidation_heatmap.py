"""
BTC Liquidation Heatmap Visualization Example

This script demonstrates how to create a candlestick chart with liquidation heatmap
overlay for Bitcoin futures data. The heatmap shows liquidation intensity at different
price levels and time periods.

Data Source:
To get the latest liquidation heatmap data, use the CoinAnk API.
See the curl command in the repository documentation.

Requirements:
- btc_heatmap_full.json file in the examples directory
- Active internet connection for fetching OHLCV data
"""

import asyncio
import json
import logging
import os
import time

from datetime import datetime

import ccxt.async_support as ccxt
import pandas as pd

from meme_nexus.utils.draw import plot_candlestick
from meme_nexus.utils.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)


async def main(timeframe_config: str = "2h", limit: int = 500):
    """
    Main function to create BTC liquidation heatmap chart.

    Args:
        timeframe_config: Chart timeframe ("15m", "1h", "2h", "4h")
        limit: Number of candlesticks to fetch (default: 500)

    Process:
    1. Load liquidation heatmap data from JSON file
    2. Fetch OHLCV data using custom limit for larger range
    3. Calculate technical indicators
    4. Generate chart with heatmap overlay
    """
    # Timeframe configuration mapping
    # Note: Hyperliquid supports specific timeframes, so we use the closest available
    timeframe_configs = {
        "5m": {"aggregate": 5, "timeframe": "m"},
        "15m": {"aggregate": 15, "timeframe": "m"},
        "30m": {"aggregate": 30, "timeframe": "m"},
        "1h": {"aggregate": 1, "timeframe": "h"},
        "2h": {"aggregate": 2, "timeframe": "h"},  # Use 2h directly
        "4h": {"aggregate": 4, "timeframe": "h"},
    }

    if timeframe_config not in timeframe_configs:
        supported = list(timeframe_configs.keys())
        raise ValueError(
            f"Unsupported timeframe: {timeframe_config}. Supported: {supported}"
        )

    config = timeframe_configs[timeframe_config]
    aggregate = config["aggregate"]
    timeframe = config["timeframe"]

    logger.info(
        f"Using timeframe configuration: {timeframe_config} ({aggregate}{timeframe})"
    )

    exchange = ccxt.hyperliquid()

    try:
        # Load liquidation heatmap data
        heatmap_file = os.path.join("examples", "btc_heatmap_full.json")
        if not os.path.exists(heatmap_file):
            logger.error(f"Heatmap data file not found: {heatmap_file}")
            logger.info(
                "Please run the curl command from the docstring to fetch the data"
            )
            return

        with open(heatmap_file) as f:
            heatmap_data = json.load(f)

        logger.info("Loaded liquidation heatmap data")

        # Extract time range from heatmap data for reference
        liq_data = heatmap_data["data"]["liqHeatMap"]
        chart_times = liq_data["chartTimeArray"]
        heatmap_start_time = chart_times[0] // 1000  # Convert milliseconds to seconds
        heatmap_end_time = chart_times[-1] // 1000

        logger.info(
            f"Heatmap time range: {datetime.fromtimestamp(heatmap_start_time)} to "
            f"{datetime.fromtimestamp(heatmap_end_time)}"
        )

        # Configure symbol and exchange
        symbol = "BTC"
        ticker = f"{symbol}/USDC:USDC"  # Hyperliquid BTC perpetual

        logger.info(f"Fetching {limit} candles of {aggregate}{timeframe} data")

        # Fetch candlestick data from Hyperliquid exchange using custom limit
        # This allows for a much larger time range than just the heatmap data
        ohlcv = await exchange.fetch_ohlcv(
            ticker, f"{aggregate}{timeframe}", limit=limit
        )

        # Convert to pandas DataFrame and process timestamps
        ohlcv = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        ohlcv["timestamp"] = ohlcv["timestamp"] // 1000  # Convert ms to s
        ohlcv = ohlcv.set_index("timestamp")

        # Convert timestamp index to datetime for plotting
        ohlcv.index = pd.to_datetime(ohlcv.index, unit="s")

        logger.info(f"Data loaded successfully, {len(ohlcv)} candles")
        logger.info(f"OHLCV time range: {ohlcv.index[0]} to {ohlcv.index[-1]}")

        # Check if we have enough data
        if len(ohlcv) < 50:
            logger.warning(
                f"Only {len(ohlcv)} candles available. "
                "Some technical indicators may not work properly."
            )

        # Calculate technical indicators (swing points, order blocks, etc.)
        time0 = time.time()
        logger.info("Calculating technical indicators")

        # For shorter timeframes or limited data, disable some indicators
        if len(ohlcv) < 100:
            logger.info("Limited data available, using simplified indicator set")
            all_indicators = None
        else:
            all_indicators = calculate_all_indicators(ohlcv, only_killzone=False)
            logger.info(f"Indicators calculated in {time.time() - time0:.2f} seconds")

        # Generate the final chart with liquidation heatmap overlay
        logger.info("Drawing chart with liquidation heatmap")

        # Adjust indicators based on available data
        has_indicators = all_indicators is not None

        img_path, _, _ = plot_candlestick(
            ohlcv,
            symbol,
            timeframe,
            aggregate=aggregate,
            is_save_file=True,
            only_killzone=True,
            # Technical indicators - disable if insufficient data
            is_draw_swingpoint=has_indicators,
            is_draw_orderblock=has_indicators,
            is_draw_liquidity=False,
            is_draw_fvg=has_indicators,
            is_draw_choch=has_indicators,
            is_draw_rainbow=True,  # Always disabled due to high data requirements
            # Liquidation heatmap
            is_draw_liquidation_heatmap=True,  # Enable heatmap overlay
            liquidation_heatmap_data=heatmap_data,  # Pass heatmap data
            dark_mode=True,  # Switch back to dark mode first
            indicators=all_indicators,  # Use pre-calculated indicators
        )

        logger.info(f"Chart with liquidation heatmap saved to: {img_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        await exchange.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Get timeframe and limit from command line arguments or use defaults
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "2h"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print(f"Generating BTC liquidation heatmap chart with {timeframe} timeframe")
    print(f"Fetching {limit} candlesticks for larger time range")
    print("Usage: python examples/draw_btc_liquidation_heatmap.py [timeframe] [limit]")
    print("Supported timeframes: 5m, 15m, 30m, 1h, 2h, 4h")
    print("Default limit: 500 (increase for larger time range)")
    print()

    asyncio.run(main(timeframe, limit))
