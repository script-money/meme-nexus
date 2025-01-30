from dataclasses import dataclass


# DexScreenerResponse
@dataclass
class Token:
    address: str
    name: str
    symbol: str


@dataclass
class Txns:
    buys: int
    sells: int


@dataclass
class TimeframeTxns:
    m5: Txns
    h1: Txns
    h6: Txns
    h24: Txns


@dataclass
class TimeframeVolume:
    m5: float
    h1: float
    h6: float
    h24: float


@dataclass
class TimeframePriceChange:
    m5: float | None = None
    h1: float | None = None
    h6: float | None = None
    h24: float | None = None


@dataclass
class Liquidity:
    usd: float
    base: float
    quote: float


@dataclass
class Website:
    label: str
    url: str


@dataclass
class Social:
    type: str
    url: str


@dataclass
class Info:
    imageUrl: str | None = None
    header: str | None = None
    openGraph: str | None = None
    websites: list[Website] = None
    socials: list[Social] = None

    def __post_init__(self):
        if self.websites is None:
            self.websites = []
        if self.socials is None:
            self.socials = []


@dataclass
class Pair:
    chainId: str
    dexId: str
    url: str
    pairAddress: str
    labels: list[str]
    baseToken: Token
    quoteToken: Token
    priceNative: str
    priceUsd: str
    txns: TimeframeTxns
    volume: TimeframeVolume
    priceChange: TimeframePriceChange
    liquidity: Liquidity
    fdv: int
    marketCap: int
    pairCreatedAt: int | None = None
    info: Info | None = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


@dataclass
class DexScreenerResponse:
    schemaVersion: str
    pairs: list[Pair]
