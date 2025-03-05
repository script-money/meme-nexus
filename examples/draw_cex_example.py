import asyncio

import ccxt.async_support as ccxt
import pandas as pd

from meme_nexus.utils.draw import plot_candlestick


async def main():
    exchange = ccxt.binanceusdm()

    try:
        symbol = "BTC"
        ticker = f"{symbol}/USDT:USDT"
        aggregate = 4
        timeframe = "h"
        use_cache = False

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

        img_path, mime_type, base64 = plot_candlestick(
            ohlcv,
            ticker,
            timeframe,
            aggregate=aggregate,
            is_save_file=True,
            is_draw_swingpoint=True,
            is_draw_orderblock=True,
            is_draw_liquidity=True,
            is_draw_fvg=True,
            is_draw_choch=True,
            is_draw_rainbow=True,
            dark_mode=False,
        )
        print(img_path)
        print(mime_type)
        print(base64[:100])
    except ccxt.RateLimitExceeded:
        await exchange.sleep(1000)  # sleep 1 second
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
