import warnings

import numpy as np
import pandas as pd

from smartmoneyconcepts import smc


def create_active_session_mask(index: pd.DatetimeIndex) -> pd.Series:
    """
    Creates a boolean mask to identify active trading sessions for time series data.

    This function identifies inactive trading periods (the "Killzone") based on
    predefined rules and returns a boolean mask. `True` values in the mask
    indicate that the corresponding timestamp is within an active session.

    The inactive period (Killzone) is defined as:
    1. Hours between 05:00 and 11:59 UTC.
    2. Weekends: After 05:00 UTC on Saturday, and all of Sunday.

    Note: This function assumes the input DatetimeIndex is in the UTC timezone.
          If your data uses a different timezone, please convert it to UTC before
          using this function. E.g., `df.index.tz_convert('UTC')`.

    Args:
        index: A pandas DatetimeIndex object.

    Returns:
        A boolean pandas Series, aligned with the input index.
        - True: Indicates an active session.
        - False: Indicates an inactive session (Killzone).
    """
    if isinstance(index, pd.DatetimeIndex):
        # These operations on a DatetimeIndex return NumPy arrays
        off_hours_mask = index.hour.isin([4, 5, 6, 7, 8, 9, 10, 11])
        weekend_mask = (
            ((index.weekday == 5) & (index.hour >= 5))
            | (index.weekday == 6)
            | ((index.weekday == 0) & (index.hour < 5))
        )

        exclusion_mask = off_hours_mask | weekend_mask
        active_session_mask_array = ~exclusion_mask

        return pd.Series(
            active_session_mask_array,
            index=index,
            name="active_mask",
            dtype=bool,
        )
    else:
        raise TypeError("Input must be a pandas DatetimeIndex.")


def calculate_swing_points(
    df: pd.DataFrame, swing_length: int, only_killzone: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calculate swing points from OHLC data.

    Args:
        df: DataFrame with OHLC data.
        swing_length: Length parameter for swing point calculation.
        only_killzone: If True, filters out swing points that are not in
                       active trading sessions (default: False).

    Returns:
        A tuple containing:
            - swings DataFrame: DataFrame with swing point information.
            - swing_highs Series: Boolean mask for swing highs.
            - swing_lows Series: Boolean mask for swing lows.
    """
    # Get the swings using SMC library
    swings = smc.swing_highs_lows(df, swing_length=swing_length)

    if swings is not None and not swings.empty:
        last_n_indices = (
            swings.index[-swing_length:] if len(swings) > swing_length else swings.index
        )
        swings.loc[last_n_indices, "HighLow"] = float("nan")
        first_n_indices = (
            swings.index[:swing_length] if len(swings) > swing_length else swings.index
        )
        swings.loc[first_n_indices, "HighLow"] = float("nan")

    swing_highs = pd.Series(False, index=df.index)
    swing_lows = pd.Series(False, index=df.index)

    if swings is not None:
        numeric_to_timestamp = dict(enumerate(df.index))
        valid_indices = [i for i in swings.index if i < len(numeric_to_timestamp)]
        swings_filtered = swings.loc[valid_indices]
        swings_filtered.index = swings_filtered.index.map(
            lambda x: numeric_to_timestamp[x]
        )
        swing_highs.loc[swings_filtered[swings_filtered["HighLow"] == 1.0].index] = True
        swing_lows.loc[swings_filtered[swings_filtered["HighLow"] == -1.0].index] = True

    # Apply killzone filter if requested
    if only_killzone:
        # 1. Call the standalone function to generate the active session mask
        active_mask = create_active_session_mask(df.index)

        # 2. Apply the mask to swing_highs and swing_lows
        #    Keep a swing point only if it's also True in the active_mask
        swing_highs &= active_mask
        swing_lows &= active_mask

        # 3. Apply the filter to the original swings DataFrame as well
        if swings is not None and not swings.empty and "swings_filtered" in locals():
            # Find all timestamps that are in the inactive period
            inactive_timestamps = df.index[~active_mask]

            # Map these timestamps back to the numeric indices
            timestamp_to_numeric = {ts: i for i, ts in numeric_to_timestamp.items()}
            inactive_numeric_indices = [
                timestamp_to_numeric[ts]
                for ts in swings_filtered.index
                if ts in inactive_timestamps
            ]

            # Set the HighLow value to NaN for swings in the inactive period
            if inactive_numeric_indices:
                swings.loc[inactive_numeric_indices, "HighLow"] = float("nan")

    return swings, swing_highs, swing_lows


def calculate_rainbow_indicator(
    ohlc: pd.DataFrame, ma_period: int = 100, atr_period: int = 200
) -> pd.DataFrame:
    """
    Calculate Rainbow indicator components.

    Rainbow indicator consists of five lines:
    1. Orange line (thick): MA of hl2 + 8 * RMA of ATR
    2. Yellow line (thin): MA of hl2 + 4 * RMA of ATR
    3. Green line (thin): MA of hl2
    4. Cyan line (thin): MA of hl2 - 4 * RMA of ATR
    5. Blue line (thick): MA of hl2 - 8 * RMA of ATR

    Also calculates trend state:
    - 1: Bull trend (price above orange line)
    - 0: Neutral (no clear trend)
    - -1: Bear trend (price below blue line)

    Args:
        ohlc: DataFrame with OHLC data
        ma_period: Period for Moving Average calculation (default: 100)
        atr_period: Period for ATR calculation (default: 200)

    Returns:
        DataFrame with calculated indicators and trend column
    """
    # Ensure sufficient data for calculations
    min_required_data = max(ma_period, atr_period)
    if len(ohlc) < min_required_data:
        warnings.warn(
            f"Need at least {min_required_data} data to calculate Rainbow indicator.",
            stacklevel=2,
        )
        return pd.DataFrame()

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
    ohlc["cyan_line"] = ohlc["ma"] - 4 * ohlc["atr_rma"]
    ohlc["blue_line"] = ohlc["ma"] - 8 * ohlc["atr_rma"]

    # Initialize trend column with 0 (neutral/oscillating)
    ohlc["trend"] = 0

    for i in range(1, len(ohlc)):
        prev_trend = ohlc.iloc[i - 1]["trend"]

        price_above_orange = ohlc.iloc[i]["close"] > ohlc.iloc[i]["orange_line"]
        price_below_green = ohlc.iloc[i]["close"] < ohlc.iloc[i]["green_line"]
        price_below_blue = ohlc.iloc[i]["close"] < ohlc.iloc[i]["blue_line"]
        price_above_green = ohlc.iloc[i]["close"] > ohlc.iloc[i]["green_line"]

        prev_price_above_orange = (
            ohlc.iloc[i - 1]["close"] > ohlc.iloc[i - 1]["orange_line"]
        )
        prev_price_below_green = (
            ohlc.iloc[i - 1]["close"] < ohlc.iloc[i - 1]["green_line"]
        )
        prev_price_below_blue = (
            ohlc.iloc[i - 1]["close"] < ohlc.iloc[i - 1]["blue_line"]
        )
        prev_price_above_green = (
            ohlc.iloc[i - 1]["close"] > ohlc.iloc[i - 1]["green_line"]
        )

        # Bull trend logic (trend = 1)
        if prev_trend != 1 and price_above_orange and not prev_price_above_orange:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = 1
        elif prev_trend == 1 and price_below_green and not prev_price_below_green:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = 0
        elif prev_trend == 1:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = 1

        # Bear trend logic (trend = -1)
        if prev_trend != -1 and price_below_blue and not prev_price_below_blue:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = -1
        elif prev_trend == -1 and price_above_green and not prev_price_above_green:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = 0
        elif prev_trend == -1:
            ohlc.iloc[i, ohlc.columns.get_loc("trend")] = -1

    # Select only the relevant columns for the result DataFrame
    rainbow_cols = [
        "orange_line",
        "yellow_line",
        "green_line",
        "cyan_line",
        "blue_line",
        "trend",
    ]

    # Return a new DataFrame with only the rainbow indicator columns
    result_df = ohlc[rainbow_cols].copy()

    return result_df


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


def calculate_fvg(
    df: pd.DataFrame, join_consecutive: bool = False, min_height_percent: float = 1.0
) -> pd.DataFrame:
    """
    Calculate Fair Value Gaps (FVG) from OHLC data.

    Args:
        df: DataFrame with OHLC data
        join_consecutive: Whether to join consecutive FVGs
        min_height_percent: Minimum height of FVG as percentage of price (default: 1.0)

    Returns:
        DataFrame with FVGs
    """
    # Calculate FVGs using SMC library
    fvgs = smc.fvg(df, join_consecutive=join_consecutive)

    # Filter FVGs by minimum height if we have any FVGs
    if fvgs is not None and not fvgs.empty:
        # Calculate height of each FVG
        fvgs["Height"] = abs(fvgs["Top"] - fvgs["Bottom"])

        # Calculate reference price (average of Top and Bottom)
        fvgs["RefPrice"] = (fvgs["Top"] + fvgs["Bottom"]) / 2

        # Calculate height as percentage of reference price
        fvgs["HeightPercent"] = (fvgs["Height"] / fvgs["RefPrice"]) * 100

        # Filter FVGs by minimum height percentage
        fvgs = fvgs[fvgs["HeightPercent"] >= min_height_percent]

        # Drop temporary columns
        fvgs = fvgs.drop(["Height", "RefPrice", "HeightPercent"], axis=1)

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


def calculate_all_indicators(
    df: pd.DataFrame, swing_length: int = 16, only_killzone: bool = False
) -> dict:
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
    swings, swing_highs, swing_lows = calculate_swing_points(
        df, swing_length, only_killzone
    )

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
    fvgs = calculate_fvg(df, join_consecutive=True, min_height_percent=2)
    result["fvgs"] = fvgs

    # Calculate BOS and CHoCH
    bos_choch = calculate_bos_choch(df, swings)
    result["bos_choch"] = bos_choch

    return result
