# MemeNexus

MemeNexus is a Python library for interacting with meme coins. It provides tools for fetching token information, analyzing token performance, and monitoring trading activities.

## Installation

```bash
pip install memeNexus
```

## Quick Start

```python
import asyncio

from meme_nexus.clients import DexScreenerClient


async def main():
    # Initialize the client
    client = DexScreenerClient()

    # Search for a meme token
    token_address = "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin"
    response = await client.search_by_token_address(token_address)

    # Process and analyze the results
    for pair in response.pairs:
        print(f"Chain: {pair.chainId}")
        print(f"DEX: {pair.dexId}")
        print(f"Price: ${pair.priceUsd}")
        print(f"24h Volume: ${pair.volume.h24:,.2f}")
        if pair.liquidity:
            print(f"Liquidity: ${pair.liquidity.usd:,.2f}")
        if pair.info and pair.info.socials:
            print("Social Links:")
            for social in pair.info.socials:
                print(f"- {social.type}: {social.url}")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.