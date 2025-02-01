import asyncio
import logging

from meme_nexus.clients.geckoterminal import GeckoTerminalClient

logging.basicConfig(level=logging.INFO)


async def analyze_token(token):
    """Format and analyze single token data"""
    print(f"\n{'='*40}")
    print(f"Token Name: {token.attributes.name} ({token.attributes.symbol})")
    print(f"Address: {token.attributes.address}")
    print(f"Decimals: {token.attributes.decimals}")

    if token.attributes.coingecko_coin_id:
        print(f"CoinGecko ID: {token.attributes.coingecko_coin_id}")

    if token.attributes.gt_score:
        print(f"GeckoTerminal Score: {token.attributes.gt_score:.2f}")

    # Social links
    print("\nSocial Links:")
    if token.attributes.discord_url:
        print(f"Discord: {token.attributes.discord_url}")
    if token.attributes.telegram_handle:
        print(f"Telegram: @{token.attributes.telegram_handle}")
    if token.attributes.twitter_handle:
        print(f"Twitter: @{token.attributes.twitter_handle}")

    # Categories and websites
    if token.attributes.categories:
        print("\nCategories:", ", ".join(token.attributes.categories))

    if token.attributes.websites:
        print("\nWebsites:", "\n".join(token.attributes.websites))


async def main():
    """Main function to demonstrate token info retrieval"""
    client = GeckoTerminalClient()

    # Example pool addresses for different networks
    examples = [
        # network, pool_address, description
        ("solana", "GL131vDDuhU4x4LffhBzgmSnp5m2QZ5G8RNJW33U1Rsr", "GTA 6 Coin Pool"),
        ("eth", "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852", "ETH/USDT Pool"),
    ]

    for network, pool_address, desc in examples:
        print(f"\nAnalyzing {desc}...")
        try:
            response = await client.get_pool_tokens(network, pool_address)
            print(f"Found {len(response.data)} tokens")
            for token in response.data:
                await analyze_token(token)

        except Exception as e:
            print(f"Error analyzing {desc}: {e}")
            if hasattr(e, "__cause__") and e.__cause__:
                print(f"Caused by: {e.__cause__}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
