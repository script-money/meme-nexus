from datetime import datetime

from meme_nexus.clients.geckoterminal import PoolsResponse


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
