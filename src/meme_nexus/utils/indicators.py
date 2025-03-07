import logging
import warnings

import numpy as np
import pandas as pd

from smartmoneyconcepts import smc

logger = logging.getLogger(__name__)


def calculate_swing_points(
    df: pd.DataFrame, swing_length: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calculate swing points from OHLC data.

    Args:
        df: DataFrame with OHLC data
        swing_length: Length parameter for swing point calculation

    Returns:
        tuple containing:
            - swings DataFrame with swing points
            - swing_highs Series (boolean mask)
            - swing_lows Series (boolean mask)
    """
    # Get the swings using SMC library
    swings = smc.swing_highs_lows(df, swing_length=swing_length)

    # If swings is not None and not empty, set the last swing_length points to NaN
    if swings is not None and not swings.empty:
        last_n_indices = (
            swings.index[-swing_length:] if len(swings) > swing_length else swings.index
        )
        swings.loc[last_n_indices, "HighLow"] = float("nan")

        first_n_indices = (
            swings.index[:swing_length] if len(swings) > swing_length else swings.index
        )
        swings.loc[first_n_indices, "HighLow"] = float("nan")

    # Create empty series for swing points
    swing_highs = pd.Series(False, index=df.index)
    swing_lows = pd.Series(False, index=df.index)

    if swings is not None:
        # Filter out invalid indices
        numeric_to_timestamp = dict(enumerate(df.index))
        valid_indices = [i for i in swings.index if i < len(numeric_to_timestamp)]
        swings_filtered = swings.loc[valid_indices]
        swings_filtered.index = swings_filtered.index.map(
            lambda x: numeric_to_timestamp[x]
        )

        # Now we can safely set the swing points
        swing_highs.loc[swings_filtered[swings_filtered["HighLow"] == 1.0].index] = True
        swing_lows.loc[swings_filtered[swings_filtered["HighLow"] == -1.0].index] = True

    return swings, swing_highs, swing_lows


def calculate_rainbow_indicator(
    ohlc: pd.DataFrame, ma_period: int = 100, atr_period: int = 200
) -> dict:
    """
    Calculate Rainbow indicator components.

    Rainbow indicator consists of five lines:
    1. Orange line (thick): MA of hl2 + 8 * RMA of ATR
    2. Yellow line (thin): MA of hl2 + 4 * RMA of ATR
    3. Green line (thin): MA of hl2
    4. Blue line (thin): MA of hl2 - 4 * RMA of ATR
    5. Cyan line (thick): MA of hl2 - 8 * RMA of ATR

    Also calculates trend states and change points.

    Args:
        df: DataFrame with OHLC data
        ma_period: Period for Moving Average calculation (default: 100)
        atr_period: Period for ATR calculation (default: 200)

    Returns:
        Dictionary with calculated indicators and signals
    """
    result = {}

    # Ensure sufficient data for calculations
    min_required_data = max(ma_period, atr_period)
    if len(ohlc) < min_required_data:
        warnings.warn(
            f"Need at least {min_required_data} data to calculate Rainbow indicator.",
            stacklevel=2,
        )
        return None

    ohlc["hl2"] = (ohlc["high"] + ohlc["low"]) / 2
    ohlc["ma"] = ohlc["hl2"].rolling(window=ma_period).mean()

    tr1 = ohlc["high"] - ohlc["low"]
    tr2 = abs(ohlc["high"] - ohlc["close"].shift(1))
    tr3 = abs(ohlc["low"] - ohlc["close"].shift(1))
    ohlc["tr"] = np.maximum(np.maximum(tr1, tr2), tr3)

    alpha = 1.0 / atr_period
    ohlc["atr_rma"] = ohlc["tr"].ewm(alpha=alpha, adjust=False).mean()

    ohlc["orange_line"] = ohlc["ma"] + 8 * ohlc["atr_rma"]
    ohlc["yellow_line"] = ohlc["ma"] + 4 * ohlc["atr_rma"]
    ohlc["green_line"] = ohlc["ma"]
    ohlc["blue_line"] = ohlc["ma"] - 4 * ohlc["atr_rma"]
    ohlc["cyan_line"] = ohlc["ma"] - 8 * ohlc["atr_rma"]

    ohlc["bull_trend"] = False
    ohlc["bear_trend"] = False
    ohlc["bull_start"] = False
    ohlc["bull_end"] = False
    ohlc["bear_start"] = False
    ohlc["bear_end"] = False

    for i in range(1, len(ohlc)):
        prev_bull = ohlc.iloc[i - 1]["bull_trend"]
        prev_bear = ohlc.iloc[i - 1]["bear_trend"]

        price_above_orange = ohlc.iloc[i]["close"] > ohlc.iloc[i]["orange_line"]
        price_below_green = ohlc.iloc[i]["close"] < ohlc.iloc[i]["green_line"]
        price_below_cyan = ohlc.iloc[i]["close"] < ohlc.iloc[i]["cyan_line"]
        price_above_green = ohlc.iloc[i]["close"] > ohlc.iloc[i]["green_line"]

        prev_price_above_orange = (
            ohlc.iloc[i - 1]["close"] > ohlc.iloc[i - 1]["orange_line"]
        )
        prev_price_below_green = (
            ohlc.iloc[i - 1]["close"] < ohlc.iloc[i - 1]["green_line"]
        )
        prev_price_below_cyan = (
            ohlc.iloc[i - 1]["close"] < ohlc.iloc[i - 1]["cyan_line"]
        )
        prev_price_above_green = (
            ohlc.iloc[i - 1]["close"] > ohlc.iloc[i - 1]["green_line"]
        )

        if not prev_bull and price_above_orange and not prev_price_above_orange:
            ohlc.iloc[i, ohlc.columns.get_loc("bull_trend")] = True
            ohlc.iloc[i, ohlc.columns.get_loc("bear_trend")] = False
            ohlc.iloc[i, ohlc.columns.get_loc("bull_start")] = True
        elif prev_bull and price_below_green and not prev_price_below_green:
            ohlc.iloc[i, ohlc.columns.get_loc("bull_trend")] = False
            ohlc.iloc[i, ohlc.columns.get_loc("bull_end")] = True
        elif prev_bull:
            ohlc.iloc[i, ohlc.columns.get_loc("bull_trend")] = True

        if not prev_bear and price_below_cyan and not prev_price_below_cyan:
            ohlc.iloc[i, ohlc.columns.get_loc("bear_trend")] = True
            ohlc.iloc[i, ohlc.columns.get_loc("bull_trend")] = False
            ohlc.iloc[i, ohlc.columns.get_loc("bear_start")] = True
        elif prev_bear and price_above_green and not prev_price_above_green:
            ohlc.iloc[i, ohlc.columns.get_loc("bear_trend")] = False
            ohlc.iloc[i, ohlc.columns.get_loc("bear_end")] = True
        elif prev_bear:
            ohlc.iloc[i, ohlc.columns.get_loc("bear_trend")] = True

    rainbow_cols = [
        "orange_line",
        "yellow_line",
        "green_line",
        "blue_line",
        "cyan_line",
        "bull_trend",
        "bear_trend",
        "bull_start",
        "bull_end",
        "bear_start",
        "bear_end",
    ]

    for col in rainbow_cols:
        result[col] = ohlc[col]

    return result


def calculate_order_blocks(df: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate order blocks from OHLC data and swing points.

    Args:
        df: DataFrame with OHLC data
        swings: DataFrame with swing points

    Returns:
        DataFrame with order blocks
    """

    # Calculate order blocks using SMC library
    order_blocks = smc.ob(df, swing_highs_lows=swings, close_mitigation=False)
    return order_blocks


def calculate_liquidity(
    df: pd.DataFrame, swings: pd.DataFrame, range_percent: float = 0.03
) -> pd.DataFrame:
    """
    Calculate liquidity from OHLC data and swing points.

    Args:
        df: DataFrame with OHLC data
        swings: DataFrame with swing points
        range_percent: Range percentage for liquidity calculation

    Returns:
        DataFrame with liquidity levels
    """

    # Calculate liquidity using SMC library
    liquidity = smc.liquidity(df, swing_highs_lows=swings, range_percent=range_percent)
    return liquidity


def calculate_fvg(df: pd.DataFrame, join_consecutive: bool = False) -> pd.DataFrame:
    """
    Calculate Fair Value Gaps (FVG) from OHLC data.

    Args:
        df: DataFrame with OHLC data
        join_consecutive: Whether to join consecutive FVGs

    Returns:
        DataFrame with FVGs
    """
    # Calculate FVGs using SMC library
    fvgs = smc.fvg(df, join_consecutive=join_consecutive)
    return fvgs


def calculate_bos_choch(
    df: pd.DataFrame, swings: pd.DataFrame, close_break: bool = True
) -> pd.DataFrame:
    """
    Calculate Break of Structure (BOS) and Change of Character (CHoCH).

    Args:
        df: DataFrame with OHLC data
        swings: DataFrame with swing points
        close_break: Whether to use close price for break detection

    Returns:
        DataFrame with BOS and CHoCH markers
    """
    # Calculate BOS and CHoCH using SMC library
    bos_choch = smc.bos_choch(df, swing_highs_lows=swings, close_break=close_break)
    return bos_choch


def calculate_all_indicators(df: pd.DataFrame, swing_length: int = 16) -> dict:
    """
    Calculate all technical indicators from OHLC data.

    Args:
        df: DataFrame with OHLC data
        swing_length: Length parameter for swing point calculation

    Returns:
        Dictionary with all calculated indicators
    """
    result = {}

    # Calculate swing points
    swings, swing_highs, swing_lows = calculate_swing_points(df, swing_length)
    result["swings"] = swings
    result["swing_highs"] = swing_highs
    result["swing_lows"] = swing_lows

    # Calculate rainbow indicator
    rainbow = calculate_rainbow_indicator(df)
    result.update(rainbow)

    # Calculate order blocks
    order_blocks = calculate_order_blocks(df, swings)
    result["order_blocks"] = order_blocks

    # Calculate liquidity
    liquidity = calculate_liquidity(df, swings)
    result["liquidity"] = liquidity

    # Calculate FVGs
    fvgs = calculate_fvg(df)
    result["fvgs"] = fvgs

    # Calculate BOS and CHoCH
    bos_choch = calculate_bos_choch(df, swings)
    result["bos_choch"] = bos_choch

    return result
