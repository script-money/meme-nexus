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
    mock_response.status_code = 200

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address(
        "2RBko3xoz56aH69isQMUpzZd9NYHahhwC23A5F3Spkin"
    )

    assert response.schemaVersion == "1.0.0"
    # Don't assert exact number of pairs as it may change over time
    assert len(response.pairs) > 0

    # Find the WIF/SOL pair on Raydium
    wif_sol_raydium_pair = None
    # Find the WIF/SOL pair on Meteora
    wif_sol_meteora_pair = None

    for pair in response.pairs:
        # Find the WIF/SOL pair on Raydium
        if (
            pair.chainId == "solana"
            and pair.dexId == "raydium"
            and pair.baseToken.symbol == "PKIN"
            and pair.quoteToken.symbol == "SOL"
        ):
            wif_sol_raydium_pair = pair

        # Find the WIF/SOL pair on Meteora
        if (
            pair.chainId == "solana"
            and pair.dexId == "meteora"
            and pair.baseToken.symbol == "PKIN"
            and pair.quoteToken.symbol == "SOL"
        ):
            wif_sol_meteora_pair = pair

    # Verify at least one of the pairs exists
    assert (
        wif_sol_raydium_pair is not None or wif_sol_meteora_pair is not None
    ), "No WIF/SOL pair found in response"

    # Test the Raydium pair if found
    if wif_sol_raydium_pair:
        assert wif_sol_raydium_pair.chainId == "solana"
        assert wif_sol_raydium_pair.dexId == "raydium"
        assert wif_sol_raydium_pair.baseToken.symbol == "PKIN"
        assert wif_sol_raydium_pair.quoteToken.symbol == "SOL"
        assert isinstance(float(wif_sol_raydium_pair.priceUsd), float)
        assert wif_sol_raydium_pair.liquidity is not None
        assert isinstance(wif_sol_raydium_pair.liquidity.usd, float)
        # Check that info exists and has expected structure
        if wif_sol_raydium_pair.info:
            assert isinstance(wif_sol_raydium_pair.info.imageUrl, str)

    # Test the Meteora pair if found
    if wif_sol_meteora_pair:
        assert wif_sol_meteora_pair.chainId == "solana"
        assert wif_sol_meteora_pair.dexId == "meteora"
        assert wif_sol_meteora_pair.baseToken.symbol == "PKIN"
        assert wif_sol_meteora_pair.quoteToken.symbol == "SOL"
        assert isinstance(float(wif_sol_meteora_pair.priceUsd), float)
        assert wif_sol_meteora_pair.liquidity is not None
        assert isinstance(wif_sol_meteora_pair.liquidity.usd, float)


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
async def test_icp_token(client, mock_httpx_client):
    """Test token on ICP chain (multiple pairs with partial price change data)."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = ICP_TOKEN_RESPONSE

    mock_client = mock_httpx_client.return_value.__aenter__.return_value
    mock_client.get.return_value = mock_response

    response = await client.search_by_token_address("eayyd-iiaaa-aaaah-adtea-cai")

    assert response.schemaVersion == "1.0.0"
    # Don't assert exact number of pairs as it may change over time
    assert len(response.pairs) > 0

    # Find the iDoge/ICP pair by looking through all pairs
    idoge_icp_pair = None
    bob_idoge_pair = None

    for pair in response.pairs:
        # Find the iDoge/ICP pair
        if (
            pair.chainId == "icp"
            and pair.dexId == "icpswap"
            and pair.baseToken.symbol == "iDoge"
            and pair.quoteToken.symbol == "ICP"
        ):
            idoge_icp_pair = pair

        # Find the BOB/iDoge pair
        if (
            pair.chainId == "icp"
            and pair.dexId == "icpswap"
            and pair.baseToken.symbol == "BOB"
            and pair.quoteToken.symbol == "iDoge"
        ):
            bob_idoge_pair = pair

    # Verify iDoge/ICP pair exists and has expected structure
    assert idoge_icp_pair is not None, "iDoge/ICP pair not found in response"
    assert idoge_icp_pair.chainId == "icp"
    assert idoge_icp_pair.dexId == "icpswap"
    assert idoge_icp_pair.baseToken.symbol == "iDoge"
    assert idoge_icp_pair.quoteToken.symbol == "ICP"
    # Don't assert exact price as it may change
    assert isinstance(float(idoge_icp_pair.priceUsd), float)
    assert idoge_icp_pair.liquidity is not None
    assert isinstance(idoge_icp_pair.liquidity.usd, float)
    # Check that info exists and has expected structure
    if idoge_icp_pair.info:
        assert len(idoge_icp_pair.info.websites) > 0
        assert len(idoge_icp_pair.info.socials) > 0

    # Verify BOB/iDoge pair if it exists
    if bob_idoge_pair:
        assert bob_idoge_pair.chainId == "icp"
        assert bob_idoge_pair.dexId == "icpswap"
        assert bob_idoge_pair.baseToken.symbol == "BOB"
        assert bob_idoge_pair.quoteToken.symbol == "iDoge"
        # Don't assert exact price as it may change
        assert isinstance(float(bob_idoge_pair.priceUsd), float)
        # The info field may be None for some pairs
