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
            ohlcv = await client.get_ohlcv(
                "solana",
                "AJCNdUWWJV32fX9FrSpdSdezBTuXmYgPUwFHFKgxYcLZ",
                "hour",
                aggregate=1,
                limit=1000,
            )

            ohlcv = pd.DataFrame(
                [candle.to_list() for candle in ohlcv],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).set_index("timestamp")

            ohlcv.to_csv(csv_path)

        img_path, mime_type, base64 = plot_candlestick(
            ohlcv,
            "Daige",
            "h",
            aggregate=1,
            is_save_file=True,
            is_draw_swingpoint=True,
            is_draw_orderblock=True,
            is_draw_liquidity=True,
            dark_mode=False,
        )
        print(img_path)
        print(mime_type)
        print(base64[:100])

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
