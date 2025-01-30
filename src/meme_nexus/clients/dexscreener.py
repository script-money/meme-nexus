import httpx
from meme_nexus.core import (
    DexScreenerResponse,
    Token,
    Pair,
    Txns,
    TimeframeTxns,
    TimeframeVolume,
    TimeframePriceChange,
    Liquidity,
    Info,
    Website,
    Social,
)
from meme_nexus.exceptions import APIError


class DexScreenerClient:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com"

    async def _make_request(self, endpoint, params=None):
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()  # Check HTTP status code
                return response.json()
            except httpx.HTTPError as e:
                raise APIError(f"DexScreener API request failed: {e}") from e

    async def search_by_token_address(self, token_address: str) -> DexScreenerResponse:
        """
        Search DexScreener by token address.

        Args:
            token_address: The token address to search for.

        Returns:
            DexScreenerResponse object containing the search results.
        """
        endpoint = "/latest/dex/search"
        params = {"q": token_address}
        data = await self._make_request(endpoint, params=params)

        # Convert JSON data to DexScreenerResponse object
        pairs = []
        for pair_data in data.get("pairs", []):  # Safely handle missing "pairs" key
            try:
                info_data = pair_data.get("info")
                info = None
                if info_data:
                    info = Info(
                        imageUrl=info_data.get("imageUrl"),
                        header=info_data.get("header"),
                        openGraph=info_data.get("openGraph"),
                        websites=[Website(**w) for w in info_data.get("websites", [])],
                        socials=[Social(**s) for s in info_data.get("socials", [])],
                    )

                pair = Pair(
                    chainId=pair_data["chainId"],
                    dexId=pair_data["dexId"],
                    url=pair_data["url"],
                    pairAddress=pair_data["pairAddress"],
                    labels=pair_data.get("labels", []),
                    baseToken=Token(**pair_data["baseToken"]),
                    quoteToken=Token(**pair_data["quoteToken"]),
                    priceNative=pair_data["priceNative"],
                    priceUsd=pair_data["priceUsd"],
                    txns=TimeframeTxns(
                        m5=Txns(**pair_data["txns"]["m5"]),
                        h1=Txns(**pair_data["txns"]["h1"]),
                        h6=Txns(**pair_data["txns"]["h6"]),
                        h24=Txns(**pair_data["txns"]["h24"]),
                    ),
                    volume=TimeframeVolume(
                        m5=pair_data["volume"]["m5"],
                        h1=pair_data["volume"]["h1"],
                        h6=pair_data["volume"]["h6"],
                        h24=pair_data["volume"]["h24"],
                    ),
                    priceChange=TimeframePriceChange(
                        m5=pair_data.get("priceChange", {}).get("m5"),
                        h1=pair_data.get("priceChange", {}).get("h1"),
                        h6=pair_data.get("priceChange", {}).get("h6"),
                        h24=pair_data.get("priceChange", {}).get("h24"),
                    ),
                    liquidity=Liquidity(
                        usd=pair_data["liquidity"]["usd"],
                        base=pair_data["liquidity"]["base"],
                        quote=pair_data["liquidity"]["quote"],
                    ),
                    fdv=pair_data["fdv"],
                    marketCap=pair_data["marketCap"],
                    pairCreatedAt=pair_data.get("pairCreatedAt"),
                    info=info,
                )
                pairs.append(pair)
            except (KeyError, TypeError) as e:
                # Handle incomplete data structure or type error
                print(f"Warning: Invalid or incomplete pair data: {e}")

        return DexScreenerResponse(
            schemaVersion=data.get("schemaVersion", "unknown"),  # Add default value to avoid error
            pairs=pairs,
        )
