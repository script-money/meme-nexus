import asyncio

from meme_nexus.clients.geckoterminal import GeckoTerminalClient


async def analyze_pool(pool):
    """Format and analyze single pool data"""
    print(f"\n{'='*40}")
    print(f"Pool Name: {pool.attributes.name}")
    if pool.relationships.network:
        print(f"Network: {pool.relationships.network.data.id}")
    if pool.relationships.dex:
        print(f"DEX: {pool.relationships.dex.data.id}")
    print(f"Address: {pool.attributes.address}")
    print(f"Created At: {pool.attributes.pool_created_at.strftime('%Y-%m-%d %H:%M')}")

    # Price information
    print("\nPrice Information:")
    print(f"Base Token Price: ${pool.attributes.base_token_price_usd:.8f}")
    print(f"Quote Token Price: ${pool.attributes.quote_token_price_usd:.2f}")

    # Market metrics
    if pool.attributes.market_cap_usd:
        print(f"Market Cap: ${pool.attributes.market_cap_usd:,.2f}")
    if pool.attributes.fdv_usd:
        print(f"Fully Diluted Value: ${pool.attributes.fdv_usd:,.2f}")

    # Trading metrics
    print("\nTrading Metrics:")
    print(f"24h Volume: ${pool.attributes.volume_usd.h24:,.2f}")
    print(f"Liquidity: ${pool.attributes.reserve_in_usd:,.2f}")
    print(f"Price Change (24h): {pool.attributes.price_change_percentage.h24:+.2f}%")

    # Transaction analysis
    h24_trans = pool.attributes.transactions.h24
    if h24_trans:
        print("\n24h Transaction Analysis:")
        total_txns = h24_trans.buys + h24_trans.sells
        buy_ratio = h24_trans.buys / total_txns * 100 if total_txns > 0 else 0
        print(f"Total Transactions: {total_txns}")
        print(f"Buy/Sell Ratio: {buy_ratio:.1f}% buys")
        print(f"Unique Traders: {h24_trans.buyers + h24_trans.sellers}")


async def main():
    client = GeckoTerminalClient()

    try:
        # Example 1: Search for a specific token
        print("🔍 Searching for ETH pools...")
        eth_pools = await client.search_pools(query="ETH", network="eth")
        for pool in eth_pools.data[:2]:  # Show top 2 results
            await analyze_pool(pool)

        # Example 2: Search across all networks
        print("\n🌐 Searching for USDC pools across all networks...")
        usdc_pools = await client.search_pools(query="USDC", page=1)
        for pool in usdc_pools.data[:2]:
            await analyze_pool(pool)

        # Example 3: Search for a specific pool address
        print("\n📍 Searching for a specific pool...")
        pool_address = (
            "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # USDC-ETH UniswapV3
        )
        specific_pool = await client.search_pools(query=pool_address, network="eth")
        if specific_pool.data:
            await analyze_pool(specific_pool.data[0])

    except Exception as e:
        print(f"❌ Error occurred: {e!s}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
