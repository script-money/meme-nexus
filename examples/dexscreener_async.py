"""
Example of using DexScreenerClient to fetch token information asynchronously.

This example demonstrates:
1. How to create and use DexScreenerClient
2. How to make multiple async requests
3. How to handle different response types
4. Error handling
"""

import asyncio
from datetime import datetime
from meme_nexus.clients.dexscreener import DexScreenerClient
from meme_nexus.exceptions import APIError


async def print_token_info(client: DexScreenerClient, token_address: str) -> None:
    """Print detailed information about a token."""
    try:
        response = await client.search_by_token_address(token_address)
        if not response.pairs:
            print(f"No pairs found for token {token_address}")
            return

        # Print basic information about each pair
        for pair in response.pairs:
            print(f"\n{'='*50}")
            print(f"Token: {pair.baseToken.symbol} ({pair.chainId})")
            print(f"DEX: {pair.dexId}")
            print(f"Price: ${pair.priceUsd}")
            print(f"24h Volume: ${pair.volume.h24:,.2f}")
            print(f"Liquidity: ${pair.liquidity.usd:,.2f}")

            # Print price changes if available
            if pair.priceChange.h24 is not None:
                print(f"24h Change: {pair.priceChange.h24:+.2f}%")

            # Print transaction counts
            print(
                f"24h Transactions: {pair.txns.h24.buys} buys, {pair.txns.h24.sells} sells"
            )

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

            print(f"{'='*50}\n")

    except APIError as e:
        print(f"Error fetching data for {token_address}: {e}")
    except Exception as e:
        print(f"Unexpected error for {token_address}: {e}")


async def main():
    # Create client
    client = DexScreenerClient()

    # Example token addresses from different chains
    tokens = [
        "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",  # PKIN (Solana)
        "eayyd-iiaaa-aaaah-adtea-cai",  # iDoge (ICP)
        "invalid_address",  # Invalid address
    ]

    print(f"Fetching data at {datetime.now()}\n")

    # Create tasks for all tokens
    tasks = [print_token_info(client, token) for token in tokens]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
