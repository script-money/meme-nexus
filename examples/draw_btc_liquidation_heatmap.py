"""
BTC Liquidation Heatmap and Order Depth Visualization Example

This script demonstrates how to create a comprehensive candlestick chart with:
1. Liquidation heatmap overlay showing liquidation intensity at different price levels
2. Order depth charts showing buy/sell activity from both SWAP and SPOT markets
3. Technical indicators and market structure analysis

The chart displays multiple panels:
- Main candlestick chart with liquidation heatmap overlay
- Volume panel
- SWAP market buy/sell count differences (green=net buying, red=net selling)
- SWAP market buy/sell value differences
- SPOT market buy/sell count differences
- SPOT market buy/sell value differences

Data Source:
Raw liquidation heatmap and order depth data from third-party APIs.

Requirements:
- btc_heatmap_and_order_depth.json file in the examples directory
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

from meme_nexus.models import LiquidationHeatmapData
from meme_nexus.utils.draw import plot_candlestick
from meme_nexus.utils.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)


def convert_liquidation_heatmap_data(raw_data: dict) -> LiquidationHeatmapData:
    """
    Convert raw liquidation heatmap data to structured format.

    Args:
        raw_data: Raw data from API containing liquidity_data

    Returns:
        LiquidationHeatmapData: Structured heatmap data
    """
    liq_data = raw_data["liquidity_data"]["BTCUSDT_3d"]["data"]["data"]

    # Extract heatmap parameters
    heatmap_data = liq_data["liqHeatMap"]["data"]

    # Convert heatmap data format
    # Original format: [["x", "y", "value"], ...]
    # Target format: structured arrays for plotting

    # Process the heatmap data
    x_vals = []
    y_vals = []
    values = []

    for point in heatmap_data:
        if len(point) >= 3:
            x_vals.append(int(point[0]))
            y_vals.append(int(point[1]))
            values.append(float(point[2]))

    # Convert to structured format
    if not x_vals:
        # Return empty heatmap if no data
        return LiquidationHeatmapData(
            data=[], chartTimeArray=[], priceArray=[], maxLiqValue=1.0
        )

    # Use the existing arrays from the original data
    original_chart_times = liq_data["liqHeatMap"]["chartTimeArray"]
    original_price_array = liq_data["liqHeatMap"]["priceArray"]
    original_max_liq_value = liq_data["liqHeatMap"]["maxLiqValue"]

    # Convert to the expected data format: [time_idx, price_idx, liq_value]
    data_triplets = []

    for x, y, value in zip(x_vals, y_vals, values, strict=False):
        if value > 0:  # Only include non-zero values
            data_triplets.append([float(x), float(y), float(value)])

    return LiquidationHeatmapData(
        data=data_triplets,
        chartTimeArray=original_chart_times,
        priceArray=original_price_array,
        maxLiqValue=original_max_liq_value,
    )


def convert_order_depth_data(raw_data: dict) -> tuple[list[dict], list[dict]]:
    """
    Convert raw order depth data to chart format.

    Args:
        raw_data: Raw data containing orderbook_data

    Returns:
        tuple: (swap_charts, spot_charts) - List of chart dicts for additional_charts
    """
    swap_data = raw_data["orderbook_data"]["BTC_orderbook_SWAP"]["data"]["data"]
    spot_data = raw_data["orderbook_data"]["BTC_orderbook_SPOT"]["data"]["data"]

    def process_orderbook_data(data: list, title_prefix: str) -> list[dict]:
        """Process orderbook data into chart format."""
        if not data:
            return []

        timestamps = []
        order_deltas = []

        for entry in data:
            if "ts" in entry and "all" in entry:
                timestamps.append(entry["ts"] // 1000)  # Convert ms to seconds

                all_data = entry["all"]
                buy_value = all_data.get("coinBuyValue", 0)
                sell_value = all_data.get("coinSellValue", 0)

                # Calculate order delta using abs(larger)/abs(smaller) with sign
                if buy_value == 0 or sell_value == 0:
                    delta = 1.0  # Neutral when data missing
                else:
                    # Use abs(larger) / abs(smaller) to ensure result > 1 or < -1
                    abs_buy = abs(buy_value)
                    abs_sell = abs(sell_value)

                    if abs_buy >= abs_sell:
                        delta = abs_buy / abs_sell  # Positive, >= 1
                    else:
                        delta = -(abs_sell / abs_buy)  # Negative, <= -1

                # Clamp delta to reasonable range (-2.5 to 2.5)
                delta = max(-2.5, min(2.5, delta))
                order_deltas.append(delta)

        if not timestamps:
            return []

        charts = []

        # Split into buy dominance (>1) and sell dominance (<-1) charts
        import numpy as np

        buy_dominance = [delta if delta > 1 else np.nan for delta in order_deltas]
        sell_dominance = [delta if delta < -1 else np.nan for delta in order_deltas]

        # Determine market type for title
        market_type = (
            "Aggregated FUTURE OrderBook Depth Ratio Chart"
            if title_prefix == "SWAP"
            else "Aggregated SPOT OrderBook Depth Ratio Chart"
        )

        # Buy dominance chart (values > 1)
        charts.append(
            {
                "title": f"{market_type}",
                "series": buy_dominance,
                "timestamps": timestamps,
                "type": "histogram",
            }
        )

        # Sell dominance chart (negative values < -1)
        charts.append(
            {
                "title": f"{market_type}",
                "series": sell_dominance,
                "timestamps": timestamps,
                "type": "histogram",
            }
        )

        return charts

    swap_charts = process_orderbook_data(swap_data, "SWAP")
    spot_charts = process_orderbook_data(spot_data, "SPOT")

    return swap_charts, spot_charts


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
        # Load combined heatmap and order depth data
        data_file = os.path.join("examples", "btc_heatmap_and_order_depth.json")
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please ensure btc_heatmap_and_order_depth.json exists")
            return

        with open(data_file) as f:
            raw_data = json.load(f)

        # Convert liquidation heatmap data to structured model
        heatmap_data = convert_liquidation_heatmap_data(raw_data)
        logger.info(
            f"Converted liquidation heatmap data: {len(heatmap_data.data)} points, "
            f"max value: {heatmap_data.maxLiqValue}"
        )

        if not heatmap_data.data:
            logger.warning("No liquidation heatmap data points found!")

        # Convert order depth data to additional charts
        swap_charts, spot_charts = convert_order_depth_data(raw_data)
        # Only 2 charts now (1 per market)
        additional_charts = swap_charts + spot_charts
        logger.info(f"Converted order depth data: {len(additional_charts)} charts")

        # Extract time range from heatmap data for reference
        chart_times = heatmap_data.chartTimeArray
        if chart_times:
            # Convert milliseconds to seconds
            heatmap_start_time = chart_times[0] // 1000
            heatmap_end_time = chart_times[-1] // 1000

            logger.info(
                f"Heatmap time range: {datetime.fromtimestamp(heatmap_start_time)} to "
                f"{datetime.fromtimestamp(heatmap_end_time)}"
            )
        else:
            logger.info("No heatmap time data available")

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
            # Enable if heatmap data available
            is_draw_liquidation_heatmap=bool(chart_times),
            liquidation_heatmap_data=heatmap_data,  # Pass structured heatmap data
            dark_mode=True,  # Switch back to dark mode first
            indicators=all_indicators,  # Use pre-calculated indicators
            # Additional charts for order depth
            additional_charts=additional_charts,  # Add order depth charts
            zero_line_separator=1,  # Use 1 as neutral line (buy/sell balance)
        )

        logger.info(
            f"Chart with liquidation heatmap and order depth saved to: {img_path}"
        )

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

    print(
        f"Generating BTC chart with liquidation heatmap and order depth "
        f"using {timeframe} timeframe"
    )
    print(f"Fetching {limit} candlesticks for larger time range")
    print("Usage: python examples/draw_btc_liquidation_heatmap.py [timeframe] [limit]")
    print("Supported timeframes: 5m, 15m, 30m, 1h, 2h, 4h")
    print("Default limit: 500 (increase for larger time range)")
    print("Data source: examples/btc_heatmap_and_order_depth.json")
    print()

    asyncio.run(main(timeframe, limit))
