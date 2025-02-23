from datetime import UTC, datetime

from meme_nexus.clients.geckoterminal import (
    PoolsResponse,
)


def test_valid_full_response():
    """Test valid full response structure"""
    test_data = {
        "data": [
            {
                "id": "solana_GL131vDDuhU4x4LffhBzgmSnp5m2QZ5G8RNJW33U1Rsr",
                "type": "pool",
                "attributes": {
                    "base_token_price_usd": "0.000929392864780465",
                    "base_token_price_native_currency": "0.00000390195918356672",
                    "quote_token_price_usd": "238.16",
                    "quote_token_price_native_currency": "1.0",
                    "base_token_price_quote_token": "0.00000390",
                    "quote_token_price_base_token": "256282",
                    "address": "GL131vDDuhU4x4LffhBzgmSnp5m2QZ5G8RNJW33U1Rsr",
                    "name": "GTA 6 Coin / SOL",
                    "pool_created_at": "2025-01-31T02:12:03Z",
                    "fdv_usd": "192291",
                    "market_cap_usd": None,
                    "price_change_percentage": {
                        "m5": "0.24",
                        "h1": "0.24",
                        "h6": "0.24",
                        "h24": "0.24",
                    },
                    "transactions": {
                        "m5": {"buys": 21, "sells": 0, "buyers": 21, "sellers": 0},
                        "m15": {"buys": 21, "sells": 0, "buyers": 21, "sellers": 0},
                        "m30": {"buys": 21, "sells": 0, "buyers": 21, "sellers": 0},
                        "h1": {"buys": 21, "sells": 0, "buyers": 21, "sellers": 0},
                        "h24": {"buys": 21, "sells": 0, "buyers": 21, "sellers": 0},
                    },
                    "volume_usd": {
                        "m5": "430.581406",
                        "h1": "430.581406",
                        "h6": "430.581406",
                        "h24": "430.581406",
                    },
                    "reserve_in_usd": "357246.7066",
                },
                "relationships": {
                    "base_token": {
                        "data": {
                            "id": "solana_6WzjEvAUvWQxBAnc8s7F4DjLwssg8F1VQ4wUGbSyVnEQ",
                            "type": "token",
                        }
                    },
                    "quote_token": {
                        "data": {
                            "id": "solana_So11111111111111111111111111111111111111112",
                            "type": "token",
                        }
                    },
                    "network": {"data": {"id": "solana", "type": "network"}},
                    "dex": {"data": {"id": "raydium", "type": "dex"}},
                },
            },
        ]
    }

    parsed = PoolsResponse(**test_data)
    assert len(parsed.data) == 1
    pool = parsed.data[0]

    assert pool.attributes.base_token_price_usd == 0.000929392864780465
    assert isinstance(pool.attributes.pool_created_at, datetime)

    assert pool.relationships.dex.data.id == "raydium"
    assert pool.relationships.network.data.id == "solana"
    assert (
        pool.relationships.base_token.data.id
        == "solana_6WzjEvAUvWQxBAnc8s7F4DjLwssg8F1VQ4wUGbSyVnEQ"
    )
    assert (
        pool.relationships.quote_token.data.id
        == "solana_So11111111111111111111111111111111111111112"
    )


def test_none_fdv_response():
    """Test response with None fdv_usd value"""
    test_data = {
        "data": [
            {
                "id": "solana_6t7waBb41LrAVGHSWPeqPigjX1aMafKapHRJtR6n3wSE",
                "type": "pool",
                "attributes": {
                    "base_token_price_usd": 0.00528154070563412,
                    "base_token_price_native_currency": 3.01938568087539e-05,
                    "quote_token_price_usd": 171.03,
                    "quote_token_price_native_currency": 1.0,
                    "base_token_price_quote_token": 3.019e-05,
                    "quote_token_price_base_token": 33119.32,
                    "address": "6t7waBb41LrAVGHSWPeqPigjX1aMafKapHRJtR6n3wSE",
                    "name": "AKIDS / SOL",
                    "pool_created_at": "2025-02-18T08:40:13Z",
                    "fdv_usd": None,
                    "market_cap_usd": None,
                    "price_change_percentage": {
                        "m5": -0.58,
                        "h1": 11.01,
                        "h6": 35.75,
                        "h24": 56.53,
                    },
                    "transactions": {
                        "m5": {"buys": 15, "sells": 10, "buyers": 15, "sellers": 10},
                        "m15": {"buys": 78, "sells": 59, "buyers": 68, "sellers": 54},
                        "m30": {
                            "buys": 207,
                            "sells": 125,
                            "buyers": 151,
                            "sellers": 98,
                        },
                        "h1": {
                            "buys": 336,
                            "sells": 257,
                            "buyers": 211,
                            "sellers": 189,
                        },
                        "h24": {
                            "buys": 1726,
                            "sells": 1232,
                            "buyers": 870,
                            "sellers": 700,
                        },
                    },
                    "volume_usd": {
                        "m5": 5860.9944057729,
                        "h1": 154113.858216485,
                        "h6": 319179.5270969,
                        "h24": 476349.366234444,
                    },
                    "reserve_in_usd": 214239.0071,
                },
                "relationships": {
                    "base_token": {
                        "data": {
                            "id": "solana_2z5VijfstyHnDsvGwMExtvjTgTB1TT3JSf29PRe9MvNG",
                            "type": "token",
                        }
                    },
                    "quote_token": {
                        "data": {
                            "id": "solana_So11111111111111111111111111111111111111112",
                            "type": "token",
                        }
                    },
                    "network": {
                        "data": {
                            "id": "solana",
                            "type": "network",
                        }
                    },
                    "dex": {
                        "data": {
                            "id": "meteora",
                            "type": "dex",
                        }
                    },
                },
            },
        ]
    }

    # 测试解析
    parsed = PoolsResponse(**test_data)
    pool = parsed.data[0]

    # 验证关键字段
    assert pool.attributes.fdv_usd is None
    assert pool.attributes.market_cap_usd is None
    assert pool.attributes.base_token_price_usd == 0.00528154070563412
    assert pool.attributes.pool_created_at == datetime(
        2025, 2, 18, 8, 40, 13, tzinfo=UTC
    )
    assert pool.attributes.name == "AKIDS / SOL"

    # 验证价格变化百分比
    assert pool.attributes.price_change_percentage.m5 == -0.58
    assert pool.attributes.price_change_percentage.h24 == 56.53

    # 验证交易数据
    assert pool.attributes.transactions.m5.buys == 15
    assert pool.attributes.transactions.h24.sells == 1232

    # 验证关系数据
    assert pool.relationships.dex.data.id == "meteora"
    assert pool.relationships.network.data.id == "solana"
