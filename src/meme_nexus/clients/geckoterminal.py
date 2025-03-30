import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx

from httpx import HTTPStatusError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..exceptions import (
    DataRangeError,
    InvalidParametersError,
)

logger = logging.getLogger(__name__)


class BaseGeckoModel(BaseModel):
    """Base model for all GeckoTerminal models with dict-like access."""

    def __getitem__(self, item):
        return getattr(self, item)

    def model_dump(self, **kwargs):
        """Override model_dump to handle datetime serialization."""
        kwargs.setdefault("mode", "json")
        return super().model_dump(**kwargs)


class PriceChangePercentage(BaseGeckoModel):
    m5: float
    h1: float
    h6: float
    h24: float

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in values:
            if isinstance(values[field], str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class TransactionWindow(BaseGeckoModel):
    buys: int
    sells: int
    buyers: int | None = None
    sellers: int | None = None


class VolumeUSD(BaseGeckoModel):
    m5: float
    h1: float
    h6: float
    h24: float

    @model_validator(mode="before")
    def parse_numeric_strings(cls, values):
        for field in values:
            if isinstance(values[field], str):
                values[field] = float(values[field]) if values[field] else 0
        return values


class TransactionsData(BaseGeckoModel):
    """A model that wraps transactions data with both dict and attribute access."""

    m5: TransactionWindow
    m15: TransactionWindow
    m30: TransactionWindow
    h1: TransactionWindow
    h24: TransactionWindow

    def __getitem__(self, key: str) -> TransactionWindow:
        return getattr(self, key)


class PoolAttributes(BaseGeckoModel):
    name: str
    address: str
    base_token_price_usd: float | None = Field(default=None)
    quote_token_price_usd: float | None = Field(default=None)
    base_token_price_native_currency: float | None = Field(default=None)
    quote_token_price_native_currency: float | None = Field(default=None)
    base_token_price_quote_token: float | None = Field(default=None)
    quote_token_price_base_token: float | None = Field(default=None)
    reserve_in_usd: float | None = Field(default=None)
    pool_created_at: datetime
    fdv_usd: float | None = Field(default=None)
    market_cap_usd: float | None = Field(default=None)
    price_change_percentage: PriceChangePercentage
    transactions: TransactionsData
    volume_usd: VolumeUSD

    # List of fields that are Optional Floats potentially coming as strings
    _optional_numeric_fields = [
        "base_token_price_usd",
        "quote_token_price_usd",
        "base_token_price_native_currency",
        "quote_token_price_native_currency",
        "base_token_price_quote_token",
        "quote_token_price_base_token",
        "reserve_in_usd",
        "fdv_usd",
        "market_cap_usd",
    ]

    @field_validator(*_optional_numeric_fields, mode="before")
    @classmethod
    def preprocess_optional_numeric_str(cls, v: Any) -> Any:
        """
        Pre-processes optional numeric fields before Pydantic's main validation.
        - Converts empty strings "" to None.
        - Passes other values (valid numeric strings, numbers, None) through
          for Pydantic to handle coercion and validation.
        """
        if v == "":
            return None
        return v


class RelationshipData(BaseGeckoModel):
    id: str
    type: Literal["token", "network", "dex"]


class RelationshipWrapper(BaseGeckoModel):
    data: RelationshipData


class PoolRelationships(BaseGeckoModel):
    base_token: RelationshipWrapper | None = None
    quote_token: RelationshipWrapper | None = None
    network: RelationshipWrapper | None = None
    dex: RelationshipWrapper | None = None


class PoolResponse(BaseGeckoModel):
    id: str
    type: Literal["pool"]
    attributes: PoolAttributes
    relationships: PoolRelationships


class PoolsResponse(BaseGeckoModel):
    data: list[PoolResponse]


class TokenAttributes(BaseGeckoModel):
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


class TokenResponse(BaseGeckoModel):
    id: str
    type: Literal["token"]
    attributes: TokenAttributes


class TokensResponse(BaseGeckoModel):
    data: list[TokenResponse]


class OHLCVAttributes(BaseGeckoModel):
    ohlcv_list: list[tuple[int, float, float, float, float, float]]

    def get_candles(self) -> list["Candle"]:
        """Convert raw OHLCV data to Candle objects and sort by timestamp ascending"""
        candles = [Candle.from_list(data) for data in self.ohlcv_list]
        return sorted(candles, key=lambda x: x.timestamp)


@dataclass(frozen=True, slots=True)
class Candle:
    """Represents a single OHLCV candle"""

    timestamp: int  # Unix timestamp in seconds
    open: float  # Opening price
    high: float  # Highest price
    low: float  # Lowest price
    close: float  # Closing price
    volume: float  # Trading volume

    @classmethod
    def from_list(cls, data: list | tuple) -> "Candle":
        """Create a Candle from a list or tuple of values"""
        return cls(*data)

    def to_list(self) -> list:
        """Convert to list format for serialization"""
        return [self.timestamp, self.open, self.high, self.low, self.close, self.volume]


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
            json_data = response.json()
            return PoolsResponse.model_validate(json_data)
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            logger.error(f"Raw JSON data: {json_data}")
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
            json_data = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e!s}")
            raise ValueError("Failed to parse API response as JSON") from e

        valid_pools: list[PoolResponse] = []
        raw_pool_list = json_data.get("data", [])

        for index, pool_data in enumerate(raw_pool_list):
            try:
                validated_pool = PoolResponse.model_validate(pool_data)
                valid_pools.append(validated_pool)
            except ValidationError as e:
                pool_id = pool_data.get("id", f"index_{index}")
                logger.warning(
                    f"Skipping pool '{pool_id}' due to validation error: {e}"
                )

        return PoolsResponse(data=valid_pools)

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
            json_data = response.json()
            return PoolsResponse.model_validate(json_data)
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            logger.error(f"Raw JSON data: {json_data}")
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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(HTTPStatusError),
    )
    async def get_ohlcv(
        self,
        network: str,
        pool_address: str,
        timeframe: Literal["minute", "hour", "day"],
        aggregate: int | None = 1,
        before_timestamp: int | None = None,
        limit: int | None = 100,
        currency: Literal["usd", "token"] = "usd",
        token: Literal["base", "quote"] | str = "base",
    ) -> list[Candle]:
        """
        Get OHLCV data for a pool

        Args:
            network: Network ID from /networks list (e.g., 'eth', 'solana')
            pool_address: Pool address
            timeframe: Time period for each candle (minute, hour, day)
            aggregate: Time period to aggregate (e.g., 15 for 15m candles)
                      Available values:
                      - day: 1
                      - hour: 1, 4, 12
                      - minute: 1, 5, 15
            before_timestamp: Return data before this timestamp (unix seconds)
            limit: Number of candles to return (default: 100, max: 1000)
            currency: Return prices in USD or token (default: usd)
            token: Return data for base or quote token (default: base)

        Returns:
            list[Candle]: List of OHLCV candles sorted by timestamp ascending

        Raises:
            InvalidParametersError: If the request parameters are invalid (400)
            DataRangeError: If requesting data beyond the allowed time range (401)
            httpx.HTTPError: For other HTTP errors
        """
        logger.info(
            f"Getting {timeframe} OHLCV for pool {pool_address} on {network}"
            f" (aggregate: {aggregate}, currency: {currency}, token: {token})"
        )

        # Check if before_timestamp is within allowed range (180 days)
        if before_timestamp is not None:
            max_age = datetime.now() - timedelta(days=180)
            if datetime.fromtimestamp(before_timestamp) < max_age:
                error_msg = (
                    "You can only access data from the past 180 days with Public API. "
                    "To access data beyond 180 days, please upgrade to the plan."
                )
                logger.error(error_msg)
                raise DataRangeError(error_msg)

        params = {}
        if aggregate is not None:
            params["aggregate"] = str(aggregate)
        if before_timestamp is not None:
            params["before_timestamp"] = str(before_timestamp)
        if limit is not None:
            params["limit"] = str(limit)
        if currency != "usd":
            params["currency"] = currency
        if token != "base":
            params["token"] = token

        response = await self.client.get(
            f"networks/{network}/pools/{pool_address}/ohlcv/{timeframe}",
            params=params,
        )

        if response.status_code == 400:
            error_msg = f"Invalid request parameters: {response.text}"
            logger.error(error_msg)
            raise InvalidParametersError(error_msg)
        elif response.status_code == 401:
            error_msg = f"API limit exceeded: {response.text}"
            logger.error(error_msg)
            raise DataRangeError(error_msg)

        # if '429 Too Many Requests', will retry by tenacity
        response.raise_for_status()
        response_json = response.json()

        try:
            attributes = OHLCVAttributes.model_validate(
                response_json["data"]["attributes"]
            )
            return attributes.get_candles()
        except ValidationError as e:
            logger.error(f"Model validation failed: {e!s}")
            logger.error(f"Raw JSON data: {response_json}")
            raise InvalidParametersError("API response structure does not match") from e

    async def get_ohlcv_history(
        self,
        network: str,
        pool_address: str,
        timeframe: Literal["minute", "hour", "day"],
        start_timestamp: int | None = None,
        aggregate: int | None = None,
        currency: Literal["usd", "token"] = "usd",
        token: Literal["base", "quote"] | str = "base",
    ) -> list[Candle]:
        """
        Get complete OHLCV history for a pool by making multiple requests

        Args:
            network: Network ID from /networks list
            pool_address: Pool address
            timeframe: Time period for each candle
            start_timestamp: Optional start timestamp to filter data
            aggregate: Time period to aggregate
            currency: Return prices in USD or token
            token: Return data for base or quote token

        Returns:
            list[Candle]: List of OHLCV candles sorted by timestamp ascending

        Note:
            Due to API limitations, only data from the past 180 days is available
            with the Public API. For older data, please upgrade to the Analyst plan.
        """
        all_ohlcv = []
        before_ts = None

        while True:
            logger.info(
                f"Fetching {timeframe} OHLCV for pool {pool_address} on {network}"
                f" (before: {before_ts})"
            )

            # Get next page of data
            ohlcv_data = await self.get_ohlcv(
                network=network,
                pool_address=pool_address,
                timeframe=timeframe,
                aggregate=aggregate,
                before_timestamp=before_ts,
                limit=1000,  # Maximum limit to reduce number of requests
                currency=currency,
                token=token,
            )

            if not ohlcv_data:  # Empty list returned
                break

            # Filter out data before start_timestamp if specified
            if start_timestamp is not None:
                ohlcv_data = [x for x in ohlcv_data if x.timestamp >= start_timestamp]

            all_ohlcv.extend(ohlcv_data)

            # Get oldest timestamp from current batch
            oldest_ts = min(x.timestamp for x in ohlcv_data)

            # If we got less than the requested limit, we've reached the end
            if len(ohlcv_data) < 1000:
                break

            # Stop if no progress to avoid infinite loop
            if before_ts and oldest_ts >= before_ts:
                break

            # Use oldest timestamp as before_timestamp for next request
            before_ts = oldest_ts

            # If we've reached the start_timestamp, we can stop
            if start_timestamp is not None and oldest_ts <= start_timestamp:
                break

        # Sort by timestamp ascending
        return sorted(all_ohlcv, key=lambda x: x.timestamp)

    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
