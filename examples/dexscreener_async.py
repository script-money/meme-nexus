"""
Example of using DexScreenerClient to fetch token information asynchronously.

This example demonstrates:
1. How to create and use DexScreenerClient
2. How to make multiple async requests
3. How to handle different response types
4. Error handling and proper resource cleanup
"""

import asyncio
import logging

from datetime import datetime

from meme_nexus.clients.dexscreener import DexScreenerClient
from meme_nexus.exceptions import APIError

logger = logging.getLogger(__name__)


async def print_token_info(client: DexScreenerClient, token_address: str) -> None:
    """Print detailed information about a token."""
    try:
        response = await client.search_by_token_address(token_address)
        if not response.pairs:
            logger.info(f"No pairs found for token {token_address}")
            return

        # Print basic information about each pair
        for pair in response.pairs:
            print(f"\n{'=' * 50}")
            print(f"Token: {pair.baseToken.symbol} ({pair.chainId})")
            print(f"DEX: {pair.dexId}")
            print(f"Price: ${pair.priceUsd}")
            print(f"24h Volume: ${pair.volume.h24:,.2f}")
            print(f"Liquidity: ${pair.liquidity.usd:,.2f}")

            # Print price changes if available
            if pair.priceChange and pair.priceChange.h24 is not None:
                print(f"24h Change: {pair.priceChange.h24:+.2f}%")

            # Print transaction counts
            print(f"24h Txs: {pair.txns.h24.buys} buys, {pair.txns.h24.sells} sells")

            # Print token information if available
            if pair.info:
                print("\nToken Info:")
                if pair.info.websites:
                    print("Websites:")
                    for website in pair.info.websites:
                        print(f"  - {website.label}: {website.url}")
                if pair.info.socials:
                    print("Social Media:")
                    for social in pair.info.socials:
                        print(f"  - {social.type}: {social.url}")

            print(f"{'=' * 50}\n")

    except APIError as e:
        logger.error(f"API error fetching data for {token_address}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for {token_address}: {e}", exc_info=True)


async def main():
    # Create client
    client = DexScreenerClient()
    try:
        # Example token addresses from different chains
        tokens = [
            "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",  # PKIN (Solana)
            "eayyd-iiaaa-aaaah-adtea-cai",  # iDoge (ICP)
            "invalid_address",  # Invalid address for testing error handling
        ]

        logger.info(f"Fetching data at {datetime.now()}")

        # Create tasks for all tokens
        tasks = [print_token_info(client, token) for token in tokens]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

    finally:
        # Always close the client to clean up resources
        await client.close()
        logger.info("Client closed successfully")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
