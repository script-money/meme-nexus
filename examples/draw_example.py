import asyncio

import pandas as pd

from meme_nexus.clients.geckoterminal import GeckoTerminalClient
from meme_nexus.utils.draw import plot_candlestick


async def main():
    client = GeckoTerminalClient()
    try:
        csv_path = "examples/draw_ohlcv.csv"
        use_cache = False

        if use_cache:
            ohlcv = pd.read_csv(csv_path, index_col=0)
        else:
            timeframe, aggregate = "minute", 15
            ohlcv = await client.get_ohlcv(
                "bsc",
                "0x1d519280255d5d90f469f79dc8f5abe05f64429f",
                timeframe,
                aggregate=aggregate,
                limit=1000,
            )

            ohlcv = pd.DataFrame(
                [candle.to_list() for candle in ohlcv],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).set_index("timestamp")

            ohlcv.to_csv(csv_path)

        img_path, mime_type, base64 = plot_candlestick(
            ohlcv,
            "SHELL",
            timeframe,
            aggregate=aggregate,
            is_save_file=True,
            is_draw_swingpoint=True,
            is_draw_orderblock=True,
            is_draw_liquidity=True,
            is_draw_fvg=True,
            is_draw_choch=True,
            dark_mode=True,
        )
        print(img_path)
        print(mime_type)
        print(base64[:100])

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
