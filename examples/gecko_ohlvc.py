import asyncio
import logging

import pandas as pd

from meme_nexus.clients.geckoterminal import GeckoTerminalClient, OHLCVAttributes

# Use the logging configuration from meme_nexus package
logger = logging.getLogger(__name__)


async def main():
    client = GeckoTerminalClient()

    try:
        pool_address = "AJCNdUWWJV32fX9FrSpdSdezBTuXmYgPUwFHFKgxYcLZ"

        # Example 1: get last day 5min candles
        ohlcv = await client.get_ohlcv(
            "solana",
            pool_address,
            "minute",
            aggregate=5,
            limit=12 * 24,
        )
        print(len(ohlcv))

        # Example 2: get ohlcv by create at of a pool
        pools_info = await client.search_pools(
            pool_address,
            "solana",
        )
        create_at = pd.to_datetime(
            pools_info.data[0].attributes.pool_created_at
        )  # get the first search result's create at time and convert to datetime
        full_ohlcv: OHLCVAttributes = await client.get_ohlcv_history(
            "solana",
            pool_address,
            "minute",
            start_timestamp=int(create_at.timestamp()),
            aggregate=5,
        )
        print(len(full_ohlcv))

        # Example 3: get all candles by multiple requests
        full_ohlcv_2: OHLCVAttributes = await client.get_ohlcv_history(
            "solana", pool_address, "hour"
        )
        print(len(full_ohlcv_2))

    except Exception as e:
        print(f"❌ Error occurred: {e!s}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
