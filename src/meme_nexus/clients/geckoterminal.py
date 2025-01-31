import logging

from datetime import datetime
from typing import Literal

import httpx

from pydantic import BaseModel, Field, ValidationError, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class PriceChangePercentage(BaseModel):
    m5: float
    h1: float
    h6: float
    h24: float

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in values:
            if isinstance(values[field], str):
                values[field] = float(values[field]) if values[field] else None
        return values


class TransactionWindow(BaseModel):
    buys: int
    sells: int
    buyers: int | None = None
    sellers: int | None = None


class VolumeUSD(BaseModel):
    m5: float
    h1: float
    h6: float
    h24: float

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in values:
            if isinstance(values[field], str):
                values[field] = float(values[field]) if values[field] else None
        return values


class PoolAttributes(BaseModel):
    name: str
    address: str
    base_token_price_usd: float = Field(alias="base_token_price_usd")
    quote_token_price_usd: float = Field(alias="quote_token_price_usd")
    base_token_price_native_currency: float = Field(
        alias="base_token_price_native_currency"
    )
    quote_token_price_native_currency: float = Field(
        alias="quote_token_price_native_currency"
    )
    base_token_price_quote_token: float = Field(alias="base_token_price_quote_token")
    quote_token_price_base_token: float = Field(alias="quote_token_price_base_token")
    reserve_in_usd: float = Field(alias="reserve_in_usd")
    pool_created_at: datetime
    fdv_usd: float | None = None
    market_cap_usd: float | None = None
    price_change_percentage: PriceChangePercentage
    transactions: dict[str, TransactionWindow]  # m5, h1 etc
    volume_usd: VolumeUSD

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        numeric_fields = {
            "base_token_price_usd",
            "quote_token_price_usd",
            "base_token_price_native_currency",
            "quote_token_price_native_currency",
            "base_token_price_quote_token",
            "quote_token_price_base_token",
            "reserve_in_usd",
            "fdv_usd",
            "market_cap_usd",
        }
        for field in numeric_fields:
            if field in values and isinstance(values[field], str):
                values[field] = float(values[field]) if values[field] else None
        return values


class RelationshipData(BaseModel):
    id: str
    type: Literal["token", "network", "dex"]


class RelationshipWrapper(BaseModel):
    data: RelationshipData


class PoolRelationships(BaseModel):
    base_token: RelationshipWrapper | None = None
    quote_token: RelationshipWrapper | None = None
    network: RelationshipWrapper | None = None
    dex: RelationshipWrapper | None = None


class PoolResponse(BaseModel):
    id: str
    type: Literal["pool"]
    attributes: PoolAttributes
    relationships: PoolRelationships


class PoolsResponse(BaseModel):
    data: list[PoolResponse]


class TokenAttributes(BaseModel):
    address: str
    name: str
    symbol: str
    decimals: int
    image_url: str | None = None
    coingecko_coin_id: str | None = None
    websites: list[str] = Field(default_factory=list)
    description: str | None = None
    gt_score: float | None = None
    discord_url: str | None = None
    telegram_handle: str | None = None
    twitter_handle: str | None = None
    categories: list[str] = Field(default_factory=list)
    gt_category_ids: list[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    id: str
    type: Literal["token"]
    attributes: TokenAttributes


class TokensResponse(BaseModel):
    data: list[TokenResponse]


class GeckoTerminalClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="https://api.geckoterminal.com/api/v2/",
            timeout=30,
            headers={
                "User-Agent": "MemeNexus",
                "Accept": "application/json",
            },
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def get_new_pools(
        self,
        page: int = Field(default=1, ge=1, le=10),
    ) -> PoolsResponse:
        """
        Fetch new pools across all networks

        Args:
            page: Pagination number (1-10)

        Returns:
            PoolsResponse: Parsed response data
        """
        params = {"page": str(page.default if hasattr(page, "default") else page)}
        response = await self.client.get("/networks/new_pools", params=params)
        response.raise_for_status()

        try:
            return PoolsResponse.model_validate(response.json())
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            raise ValueError("API response structure does not match") from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_trending_pools(
        self,
        page: int = Field(default=1, ge=1, le=10),
        duration: Literal["5m", "1h", "6h", "24h"] = "5m",
    ) -> PoolsResponse:
        """
        Fetch trending pools across all networks

        Args:
            page: Pagination number (1-10)
            duration: Time window for trending calculation

        Returns:
            PoolsResponse: Parsed response data
        """
        params = {
            "page": str(page.default if hasattr(page, "default") else page),
            "duration": duration,
        }

        response = await self.client.get("/networks/trending_pools", params=params)
        response.raise_for_status()
        try:
            return PoolsResponse.model_validate(response.json())
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            raise ValueError("API response structure does not match") from e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_pools(
        self,
        query: str,
        network: str | None = None,
        page: int = Field(default=1, ge=1, le=10),
    ) -> PoolsResponse:
        """
        Search for pools on a network

        Args:
            query: Search query - can be pool address, token address, or token symbol
            network: (Optional) Network ID from /networks list
            page: Pagination number (1-10)

        Returns:
            PoolsResponse: Parsed response data containing matching pools
        """
        params = {
            "query": query,
            "page": str(page.default if hasattr(page, "default") else page),
        }
        if network:
            params["network"] = network

        response = await self.client.get("/search/pools", params=params)
        response.raise_for_status()
        try:
            return PoolsResponse.model_validate(response.json())
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            raise ValueError("API response structure does not match") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def get_pool_tokens(
        self,
        network: str,
        pool_address: str,
    ) -> TokensResponse:
        """
        Get pool tokens info on a network

        Args:
            network: Network ID from /networks list (e.g., 'eth', 'solana')
            pool_address: Pool address

        Returns:
            TokensResponse: Parsed response containing token information
        """
        logger.info(f"Getting tokens for pool {pool_address} on {network}")

        response = await self.client.get(
            f"networks/{network}/pools/{pool_address}/info"
        )
        response.raise_for_status()

        data = response.json()
        return TokensResponse.model_validate(data)
