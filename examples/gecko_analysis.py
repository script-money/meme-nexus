import asyncio

from meme_nexus.clients.geckoterminal import GeckoTerminalClient


async def analyze_pool(pool):
    """Format and analyze single pool data"""
    print(f"\n{'='*40}")
    print(f"Pool Name: {pool.attributes.name}")
    print(f"{pool.relationships.network.data.id} {pool.relationships.dex.data.id}")
    print(f"Created At: {pool.attributes.pool_created_at.strftime('%Y-%m-%d %H:%M')}")

    # Price information
    print("\nPrice Trends:")
    print(f"Base Token: ${pool.attributes.base_token_price_usd:.6f}")
    print(f"Quote Token: ${pool.attributes.quote_token_price_usd:.2f}")
    print(f"24h Change: {pool.attributes.price_change_percentage.h24:.2f}%")

    # Trading data
    print("\nTrading Activity:")
    volume = pool.attributes.volume_usd.h24
    print(f"24h Volume: ${volume:,.2f}")
    print(f"Liquidity Reserve: ${pool.attributes.reserve_in_usd:,.2f}")

    # Real-time transaction statistics
    latest_trans = pool.attributes.transactions.get("m5")
    if latest_trans:
        print("\nLast 5 Minutes Transactions:")
        print(f"Buy Count: {latest_trans.buys}")
        print(f"Sell Count: {latest_trans.sells}")


async def main():
    client = GeckoTerminalClient()

    try:
        # Fetch new pools
        new_pools = await client.get_new_pools(page=2)
        print("🆕 New Pools Analysis")
        for pool in new_pools.data[:3]:  # Show top 3
            await analyze_pool(pool)

        # Fetch trending pools
        trending_pools = await client.get_trending_pools(duration="1h")
        print("\n🚀 Trending Pools Analysis")
        for pool in trending_pools.data[:3]:
            await analyze_pool(pool)

    except Exception as e:
        print(f"❌ Error occurred: {e!s}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
