import asyncio
import logging
import time

import ccxt.async_support as ccxt
import pandas as pd

from meme_nexus.utils.draw import plot_candlestick
from meme_nexus.utils.indicators import (
    calculate_all_indicators,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    exchange = ccxt.hyperliquid()

    try:
        symbol = "BTC"
        ticker = f"{symbol}/USDC:USDC"
        aggregate = 4
        timeframe = "h"
        use_cache = False
        only_killzone = True

        csv_path = "examples/draw_cex_ohlcv.csv"

        if use_cache:
            ohlcv = pd.read_csv(csv_path, index_col=0)
        else:
            ohlcv = await exchange.fetch_ohlcv(
                ticker, f"{aggregate}{timeframe}", since=None, limit=1000
            )

            ohlcv = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            ohlcv["timestamp"] = ohlcv["timestamp"] // 1000  # Convert ms to s
            ohlcv = ohlcv.set_index("timestamp")

            ohlcv.to_csv(csv_path)

        # Convert index to datetime if needed
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv.index = pd.to_datetime(ohlcv.index, unit="s")

        logger.info("Data loaded successfully")

        # 1: Calculate all indicators at once
        time0 = time.time()
        logger.info("Calculating all indicators")
        all_indicators = calculate_all_indicators(ohlcv, only_killzone=only_killzone)
        logger.info(f"All indicators calculated with {len(all_indicators)} components")
        logger.info(f"Time taken: {time.time() - time0:.2f} seconds")

        # 2: Draw chart with pre-calculated indicators
        logger.info("Drawing chart with pre-calculated indicators")
        img_path, _, _ = plot_candlestick(
            ohlcv,
            symbol,
            timeframe,
            aggregate=aggregate,
            is_save_file=True,
            only_killzone=only_killzone,
            is_draw_swingpoint=True,
            is_draw_orderblock=True,
            is_draw_liquidity=True,
            is_draw_fvg=True,
            is_draw_choch=True,
            is_draw_rainbow=False,
            dark_mode=False,
            indicators=all_indicators,  # Pass pre-calculated indicators
        )

        logger.info(f"Image saved to: {img_path}")
    except ccxt.RateLimitExceeded:
        await exchange.sleep(1000)  # sleep 1 second
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
