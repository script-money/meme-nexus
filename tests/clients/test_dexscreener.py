from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from meme_nexus.clients.dexscreener import DexScreenerClient
from meme_nexus.exceptions import APIError

# Test data - Full response with multiple pairs
FULL_RESPONSE = {
    "schemaVersion": "1.0.0",
    "pairs": [
        {
            "chainId": "solana",
            "dexId": "raydium",
            "url": "https://dexscreener.com/solana/ede4v78zjo54dfhcbwm8nmfrlwstvggbcjg8uwh5fsdv",
            "pairAddress": "EDE4v78Zjo54DfhcbWM8nmFrLWSTVGgBcjg8UWh5Fsdv",
            "labels": ["CLMM"],
            "baseToken": {
                "address": "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",
                "name": "PUMPKIN",
                "symbol": "PKIN",
            },
            "quoteToken": {
                "address": "So11111111111111111111111111111111111111112",
                "name": "Wrapped SOL",
                "symbol": "SOL",
            },
            "priceNative": "0.00004061",
            "priceUsd": "0.009307",
            "txns": {
                "m5": {"buys": 234, "sells": 146},
                "h1": {"buys": 4467, "sells": 3568},
                "h6": {"buys": 36947, "sells": 26559},
                "h24": {"buys": 37551, "sells": 26929},
            },
            "volume": {
                "h24": 42098991.96,
                "h6": 41723118.57,
                "h1": 5736842.83,
                "m5": 362557.94,
            },
            "priceChange": {"m5": -8.59, "h1": 26.01, "h6": 2474, "h24": 8849},
            "liquidity": {"usd": 2173910.21, "base": 124818625, "quote": 4415.8728},
            "fdv": 9307843,
            "marketCap": 9307843,
            "pairCreatedAt": 1738086997000,
            "info": {
                "imageUrl": "https://dd.dexscreener.com/ds-data/tokens/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin.png?key=e6eac1",
                "header": "https://dd.dexscreener.com/ds-data/tokens/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin/header.png?key=e6eac1",
                "openGraph": "https://cdn.dexscreener.com/token-images/og/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin?timestamp=1738113900000",
                "websites": [
                    {"label": "Website", "url": "https://pumpkin.fun/"},
                    {
                        "label": "Docs",
                        "url": "https://pumpkindotfun.gitbook.io/pumpkin-docs/",
                    },
                    {
                        "label": "Mirror(articles)",
                        "url": "https://mirror.xyz/0x6168f5Ea808A381af2D1947a2a076BA85A831922",
                    },
                    {
                        "label": "Dapp($PKIN preview)",
                        "url": "https://alpha.pumpkin.fun/token/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",
                    },
                ],
                "socials": [
                    {"type": "twitter", "url": "https://x.com/pumpkindotfun"},
                    {"type": "telegram", "url": "https://t.me/pumpkindotfun"},
                ],
            },
        },
        {
            "chainId": "solana",
            "dexId": "meteora",
            "url": "https://dexscreener.com/solana/9glwwru9g7qscqqp65zbhy6sykuyaytbebhqeh1sqz3p",
            "pairAddress": "9GLWwRU9g7qscQQp65ZBHy6syKuyaytbEbHQeh1sqZ3p",
            "labels": ["DLMM"],
            "baseToken": {
                "address": "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",
                "name": "PUMPKIN",
                "symbol": "PKIN",
            },
            "quoteToken": {
                "address": "So11111111111111111111111111111111111111112",
                "name": "Wrapped SOL",
                "symbol": "SOL",
            },
            "priceNative": "0.00003949",
            "priceUsd": "0.009051",
            "txns": {
                "m5": {"buys": 2, "sells": 28},
                "h1": {"buys": 531, "sells": 362},
                "h6": {"buys": 2994, "sells": 2378},
                "h24": {"buys": 2994, "sells": 2378},
            },
            "volume": {
                "h24": 3680292.18,
                "h6": 3680292.18,
                "h1": 711855.18,
                "m5": 27852.01,
            },
            "priceChange": {"m5": -9.68, "h1": 17.35, "h6": 271, "h24": 271},
            "liquidity": {"usd": 212128.68, "base": 14200458, "quote": 364.6971},
            "fdv": 9051828,
            "marketCap": 9051828,
            "pairCreatedAt": 1738096275000,
            "info": {
                "imageUrl": "https://dd.dexscreener.com/ds-data/tokens/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin.png?key=e6eac1",
                "header": "https://dd.dexscreener.com/ds-data/tokens/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin/header.png?key=e6eac1",
                "openGraph": "https://cdn.dexscreener.com/token-images/og/solana/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin?timestamp=1738113900000",
                "websites": [
                    {"label": "Website", "url": "https://pumpkin.fun/"},
                    {
                        "label": "Docs",
                        "url": "https://pumpkindotfun.gitbook.io/pumpkin-docs/",
                    },
                    {
                        "label": "Mirror(articles)",
                        "url": "https://mirror.xyz/0x6168f5Ea808A381af2D1947a2a076BA85A831922",
                    },
                    {
                        "label": "Dapp($PKIN preview)",
                        "url": "https://alpha.pumpkin.fun/token/2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin",
                    },
                ],
                "socials": [
                    {"type": "twitter", "url": "https://x.com/pumpkindotfun"},
                    {"type": "telegram", "url": "https://t.me/pumpkindotfun"},
                ],
            },
        },
    ],
}

# Test data - Invalid contract address response
INVALID_ADDRESS_RESPONSE = {"schemaVersion": "1.0.0", "pairs": []}

# Test data - Response missing info field
MISSING_INFO_RESPONSE = {
    "schemaVersion": "1.0.0",
    "pairs": [
        {
            "chainId": "solana",
            "dexId": "raydium",
            "url": "https://dexscreener.com/solana/9ozm7hv5angujvgedpqzziesnwzgacbt3ivwvmdrsb78",
            "pairAddress": "9ozm7hV5angUjvgEdPQzZiESNwzgaCBT3iVWvMdrSB78",
            "baseToken": {
                "address": "bf7BTmV7qUY1jiZA9FybL1tAswhsqhnYXMML8Frpump",
                "name": "UFC Coin",
                "symbol": "UFC",
            },
            "quoteToken": {
                "address": "So11111111111111111111111111111111111111112",
                "name": "Wrapped SOL",
                "symbol": "SOL",
            },
            "priceNative": "0.000003437",
            "priceUsd": "0.0008052",
            "txns": {
                "m5": {"buys": 438, "sells": 323},
                "h1": {"buys": 2765, "sells": 1871},
                "h6": {"buys": 2765, "sells": 1871},
                "h24": {"buys": 2765, "sells": 1871},
            },
            "volume": {
                "h24": 1923332.63,
                "h6": 1923332.63,
                "h1": 1923332.63,
                "m5": 366418.6,
            },
            "priceChange": {"m5": -4.94, "h1": 802, "h6": 802, "h24": 802},
            "liquidity": {"usd": 115446.83, "base": 71595332, "quote": 246.6727},
            "fdv": 805274,
            "marketCap": 805274,
            "pairCreatedAt": 1738200056000,
        }
    ],
}

# Test data - Token response from ICP chain
ICP_TOKEN_RESPONSE = {
    "schemaVersion": "1.0.0",
    "pairs": [
        {
            "chainId": "icp",
            "dexId": "icpswap",
            "url": "https://dexscreener.com/icp/wlv64-biaaa-aaaag-qcrlq-cai",
            "pairAddress": "wlv64-biaaa-aaaag-qcrlq-cai",
            "baseToken": {
                "address": "eayyd-iiaaa-aaaah-adtea-cai",
                "name": "Internet Doge",
                "symbol": "iDoge",
            },
            "quoteToken": {
                "address": "ryjl3-tyaaa-aaaaa-aaaba-cai",
                "name": "Internet Computer",
                "symbol": "ICP",
            },
            "priceNative": "0.1560",
            "priceUsd": "1.35",
            "txns": {
                "m5": {"buys": 0, "sells": 0},
                "h1": {"buys": 0, "sells": 0},
                "h6": {"buys": 5, "sells": 2},
                "h24": {"buys": 12, "sells": 9},
            },
            "volume": {
                "h24": 97.25,
                "h6": 30.82,
                "h1": 0,
                "m5": 0,
            },
            "priceChange": {"h6": 3.94, "h24": 0.15},
            "liquidity": {"usd": 987763.89, "base": 557228, "quote": 26501},
            "fdv": 1358675,
            "marketCap": 1358675,
            "info": {
                "imageUrl": "https://dd.dexscreener.com/ds-data/tokens/icp/eayyd-iiaaa-aaaah-adtea-cai.png?key=c52160",
                "header": "https://dd.dexscreener.com/ds-data/tokens/icp/eayyd-iiaaa-aaaah-adtea-cai/header.png?key=c52160",
                "openGraph": "https://cdn.dexscreener.com/token-images/og/icp/eayyd-iiaaa-aaaah-adtea-cai?timestamp=1738201500000",
                "websites": [
                    {"label": "Website", "url": "https://idoge.org"},
                    {"label": "DSCVR", "url": "https://dscvr.one/p/idoge"},
                ],
                "socials": [
                    {"type": "twitter", "url": "https://x.com/idoge_icp"},
                    {"type": "telegram", "url": "https://t.me/icpdoge"},
                ],
            },
        },
        {
            "chainId": "icp",
            "dexId": "icpswap",
            "url": "https://dexscreener.com/icp/mowbg-oqaaa-aaaag-qkicq-cai",
            "pairAddress": "mowbg-oqaaa-aaaag-qkicq-cai",
            "baseToken": {
                "address": "7pail-xaaaa-aaaas-aabmq-cai",
                "name": "BOB",
                "symbol": "BOB",
            },
            "quoteToken": {
                "address": "eayyd-iiaaa-aaaah-adtea-cai",
                "name": "Internet Doge",
                "symbol": "iDoge",
            },
            "priceNative": "0.6023",
            "priceUsd": "0.8184",
            "txns": {
                "m5": {"buys": 0, "sells": 0},
                "h1": {"buys": 0, "sells": 0},
                "h6": {"buys": 5, "sells": 4},
                "h24": {"buys": 16, "sells": 12},
            },
            "volume": {
                "h24": 130.12,
                "h6": 39.62,
                "h1": 0,
                "m5": 0,
            },
            "priceChange": {"h6": 6.38, "h24": 3.52},
            "liquidity": {"usd": 1221.41, "base": 652.4188, "quote": 505.9683},
            "fdv": 10311469,
            "marketCap": 10311469,
        },
    ],
}


@pytest.fixture
def client():
    return DexScreenerClient()


@pytest_asyncio.fixture
async def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock:
        mock_instance = AsyncMock()
        mock.return_value.__aenter__.return_value = mock_instance
        yield mock


@pytest.mark.asyncio
async def test_full_info_token(client, mock_httpx_client):
    """Test getting token with full information (multiple pairs)"""
    mock_response = MagicMock()
    mock_response.json.return_value = FULL_RESPONSE
    mock_response.raise_for_status.return_value = None

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address(
        "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin"
    )

    assert response.schemaVersion == "1.0.0"
    assert len(response.pairs) == 2  # Verify returned two pairs

    # Verify first pair (Raydium)
    pair1 = response.pairs[0]
    assert pair1.chainId == "solana"
    assert pair1.dexId == "raydium"
    assert pair1.baseToken.symbol == "PKIN"
    assert len(pair1.info.websites) == 4
    assert len(pair1.info.socials) == 2
    assert pair1.labels == ["CLMM"]
    assert float(pair1.priceUsd) == 0.009307

    # Verify second pair (Meteora)
    pair2 = response.pairs[1]
    assert pair2.chainId == "solana"
    assert pair2.dexId == "meteora"
    assert pair2.baseToken.symbol == "PKIN"
    assert pair2.labels == ["DLMM"]
    assert float(pair2.priceUsd) == 0.009051


@pytest.mark.asyncio
async def test_invalid_address(client, mock_httpx_client):
    """Test invalid contract address"""
    mock_response = MagicMock()
    mock_response.json.return_value = INVALID_ADDRESS_RESPONSE
    mock_response.raise_for_status.return_value = None

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address("0xinvalid")

    assert response.schemaVersion == "1.0.0"
    assert len(response.pairs) == 0


@pytest.mark.asyncio
async def test_api_error(client, mock_httpx_client):
    """Test API error"""
    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.side_effect = httpx.HTTPError("API Error")

    with pytest.raises(APIError):
        await client.search_by_token_address("0xabc")


@pytest.mark.asyncio
async def test_missing_info_token(client, mock_httpx_client):
    """Test token with missing info field"""
    mock_response = MagicMock()
    mock_response.json.return_value = MISSING_INFO_RESPONSE
    mock_response.raise_for_status.return_value = None

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address(
        "bf7BTmV7qUY1jiZA9FybL1tAswhsqhnYXMML8Frpump"
    )

    assert response.schemaVersion == "1.0.0"
    assert len(response.pairs) == 1

    pair = response.pairs[0]
    assert pair.chainId == "solana"
    assert pair.dexId == "raydium"
    assert pair.baseToken.symbol == "UFC"
    assert pair.info is None  # Verify info field is None
    assert float(pair.priceUsd) == 0.0008052
    assert pair.volume.h24 == 1923332.63
    assert pair.liquidity.usd == 115446.83
    assert pair.marketCap == 805274


@pytest.mark.asyncio
async def test_icp_token(client, mock_httpx_client):
    """Test token on ICP chain (multiple pairs with partial price change data)"""
    mock_response = MagicMock()
    mock_response.json.return_value = ICP_TOKEN_RESPONSE
    mock_response.raise_for_status.return_value = None

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address("eayyd-iiaaa-aaaah-adtea-cai")

    assert response.schemaVersion == "1.0.0"
    assert len(response.pairs) == 2

    # Verify first pair (iDoge/ICP)
    pair1 = response.pairs[0]
    assert pair1.chainId == "icp"
    assert pair1.dexId == "icpswap"
    assert pair1.baseToken.symbol == "iDoge"
    assert pair1.quoteToken.symbol == "ICP"
    assert float(pair1.priceUsd) == 1.35
    assert pair1.volume.h24 == 97.25
    assert pair1.volume.h1 == 0  # Verify zero transaction volume
    assert pair1.liquidity.usd == 987763.89
    assert len(pair1.info.websites) == 2
    assert len(pair1.info.socials) == 2

    # Verify second pair (BOB/iDoge)
    pair2 = response.pairs[1]
    assert pair2.chainId == "icp"
    assert pair2.dexId == "icpswap"
    assert pair2.baseToken.symbol == "BOB"
    assert pair2.quoteToken.symbol == "iDoge"
    assert float(pair2.priceUsd) == 0.8184
    assert pair2.volume.h24 == 130.12
    assert pair2.volume.h1 == 0
    assert pair2.liquidity.usd == 1221.41
    assert pair2.info is None  # Verify info field is None
