import logging

import httpx

from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from meme_nexus.exceptions import APIError

logger = logging.getLogger(__name__)


class Website(BaseModel):
    label: str
    url: str


class Social(BaseModel):
    type: str
    url: str


class Info(BaseModel):
    imageUrl: str | None = None
    header: str | None = None
    openGraph: str | None = None
    websites: list[Website] = Field(default_factory=list)
    socials: list[Social] = Field(default_factory=list)


class Token(BaseModel):
    address: str
    name: str
    symbol: str


class Txns(BaseModel):
    buys: int = 0
    sells: int = 0

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in ["buys", "sells"]:
            if isinstance(values.get(field), str):
                values[field] = int(values[field]) if values[field] else 0
        return values


class TimeframeTxns(BaseModel):
    m5: Txns
    h1: Txns
    h6: Txns
    h24: Txns


class TimeframeVolume(BaseModel):
    m5: float = 0
    h1: float = 0
    h6: float = 0
    h24: float = 0

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in ["m5", "h1", "h6", "h24"]:
            if isinstance(values.get(field), str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class TimeframePriceChange(BaseModel):
    m5: float | None = None
    h1: float | None = None
    h6: float | None = None
    h24: float | None = None

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in ["m5", "h1", "h6", "h24"]:
            if isinstance(values.get(field), str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class Liquidity(BaseModel):
    usd: float
    base: float
    quote: float

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in ["usd", "base", "quote"]:
            if isinstance(values.get(field), str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class Pair(BaseModel):
    chainId: str
    dexId: str
    url: str
    pairAddress: str
    labels: list[str] = Field(default_factory=list)
    baseToken: Token
    quoteToken: Token
    priceNative: float
    priceUsd: float
    txns: TimeframeTxns
    volume: TimeframeVolume
    priceChange: TimeframePriceChange
    liquidity: Liquidity
    fdv: float | None = None
    marketCap: float | None = None
    pairCreatedAt: int | None = None
    info: Info | None = None

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        numeric_fields = ["priceNative", "priceUsd", "fdv", "marketCap"]
        for field in numeric_fields:
            if isinstance(values.get(field), str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class DexScreenerResponse(BaseModel):
    schemaVersion: str = "unknown"
    pairs: list[Pair] = Field(default_factory=list)


class DexScreenerClient:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30,
            headers={
                "User-Agent": "MemeNexus",
                "Accept": "application/json",
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def _make_request(self, endpoint: str, params: dict | None = None) -> dict:
        """
        Make a request to the DexScreener API with retry logic.

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            API response as dict

        Raises:
            APIError: If the request fails after all retries
        """
        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"DexScreener API request failed: {e}")
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

        try:
            return DexScreenerResponse.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse DexScreener response: {e}")
            raise APIError(f"Failed to parse DexScreener response: {e}") from e

    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
