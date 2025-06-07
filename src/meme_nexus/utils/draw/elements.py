"""Individual chart element drawing functions."""

import mplfinance as mpf
import numpy as np
import pandas as pd

from ..format import format_number
from .config import get_color_scheme


def draw_volume(ohlc: pd.DataFrame) -> mpf.make_addplot:
    """Create volume overlay for the chart."""
    colors = get_color_scheme()

    bar_colors = [
        colors["green"] if close >= open else colors["red"]
        for open, close in zip(ohlc["open"], ohlc["close"], strict=True)
    ]

    return mpf.make_addplot(
        ohlc["volume"], type="bar", panel=1, color=bar_colors, alpha=0.75, width=0.8
    )


def draw_swing_points(
    ax,
    ohlc: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
    dark_mode: bool,
) -> list:
    """Draw swing point markers and labels."""
    # If ax is None, only return addplots for mplfinance
    if ax is None:
        return create_swing_point_plots(ohlc, swing_highs, swing_lows, dark_mode)

    # Legacy behavior: create both plots and labels
    addplots, dynamic_offset = create_swing_point_plots(
        ohlc, swing_highs, swing_lows, dark_mode
    )
    draw_swing_point_labels(
        ax, ohlc, swing_highs, swing_lows, dark_mode, dynamic_offset
    )
    return addplots, dynamic_offset


def create_swing_point_plots(
    ohlc: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
    dark_mode: bool,
) -> tuple:
    """Create swing point addplots for mplfinance."""
    colors = get_color_scheme()
    addplots = []

    price_range = ohlc["high"].max() - ohlc["low"].min()
    dynamic_offset_factor = 0.01
    dynamic_offset = price_range * dynamic_offset_factor

    # Create offset series
    high_offsets = pd.Series(
        [
            dynamic_offset if not pd.isna(high_value) else np.nan
            for high_value in swing_highs
        ],
        index=ohlc.index,
    )
    low_offsets = pd.Series(
        [
            dynamic_offset if not pd.isna(low_value) else np.nan
            for low_value in swing_lows
        ],
        index=ohlc.index,
    )

    # Create markers
    swing_high_markers = mpf.make_addplot(
        ohlc["high"].where(swing_highs) + high_offsets,
        type="scatter",
        marker="$⬇$",
        color=colors["light"] if dark_mode else colors["black"],
        markersize=6,
    )
    swing_low_markers = mpf.make_addplot(
        ohlc["low"].where(swing_lows) - low_offsets,
        type="scatter",
        marker="$⬆$",
        color=colors["light"] if dark_mode else colors["black"],
        markersize=6,
    )

    addplots.extend([swing_high_markers, swing_low_markers])
    return addplots, dynamic_offset


def draw_swing_point_labels(
    ax,
    ohlc: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
    dark_mode: bool,
    dynamic_offset: float,
):
    """Draw swing point price labels on the axes."""
    colors = get_color_scheme()

    # Add price labels
    for i, idx in enumerate(ohlc.index):
        if swing_highs[idx]:
            price = ohlc.loc[idx, "high"]
            ax.annotate(
                format_number(price, precision=4, is_format_k=False),
                (i, price + dynamic_offset),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color=colors["light"] if dark_mode else colors["black"],
            )
        if swing_lows[idx]:
            price = ohlc.loc[idx, "low"]
            ax.annotate(
                format_number(price, precision=4, is_format_k=False),
                (i, price - dynamic_offset),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6,
                color=colors["light"] if dark_mode else colors["black"],
            )


def draw_rainbow_indicator(ohlc: pd.DataFrame) -> list:
    """Draw rainbow indicator lines and trend signals."""
    colors = get_color_scheme()
    addplots = []

    # Add Rainbow lines
    rainbow_lines = [
        ("orange_line", colors["orange"]),
        ("yellow_line", colors["yellow"]),
        ("green_line", colors["lime_green"]),
        ("cyan_line", colors["cyan"]),
        ("blue_line", colors["blue"]),
    ]

    for col, color in rainbow_lines:
        line_plot = mpf.make_addplot(ohlc[col], color=color, width=0.5, panel=0)
        addplots.append(line_plot)

    # Add trend change markers
    bull_start = (ohlc["trend"] == 1) & (ohlc["trend"].shift(1) != 1)
    bull_end = (ohlc["trend"] == 0) & (ohlc["trend"].shift(1) == 1)
    bear_start = (ohlc["trend"] == -1) & (ohlc["trend"].shift(1) != -1)
    bear_end = (ohlc["trend"] == 0) & (ohlc["trend"].shift(1) == -1)

    # Add markers for trend changes
    if bull_start.any():
        bull_start_markers = mpf.make_addplot(
            ohlc["close"].where(bull_start),
            type="scatter",
            marker="^",
            markersize=60,
            color="lime",
            panel=0,
        )
        addplots.append(bull_start_markers)

    if bull_end.any():
        bull_end_markers = mpf.make_addplot(
            ohlc["close"].where(bull_end),
            type="scatter",
            marker="x",
            markersize=60,
            color="lime",
            panel=0,
        )
        addplots.append(bull_end_markers)

    if bear_start.any():
        bear_start_markers = mpf.make_addplot(
            ohlc["close"].where(bear_start),
            type="scatter",
            marker="v",
            markersize=60,
            color="red",
            panel=0,
        )
        addplots.append(bear_start_markers)

    if bear_end.any():
        bear_end_markers = mpf.make_addplot(
            ohlc["close"].where(bear_end),
            type="scatter",
            marker="x",
            markersize=60,
            color="red",
            panel=0,
        )
        addplots.append(bear_end_markers)

    return addplots


def draw_order_blocks(ax, ob: pd.DataFrame, swings: pd.DataFrame):
    """Draw order blocks on the chart."""
    colors = get_color_scheme()

    for idx, row in ob.iterrows():
        if pd.isna(row["Bottom"]) or pd.isna(row["Top"]):
            continue

        future_high = swings["Level"].iloc[idx + 1 :].max()
        future_low = swings["Level"].iloc[idx + 1 :].min()
        percentage_strength_width = round(row["Percentage"] / 10, 0)
        multiplier = 10

        if row["OB"] == 1.0 and future_low > row["Top"]:
            ax.fill_between(
                [idx, idx + percentage_strength_width * multiplier],
                row["Bottom"],
                row["Top"],
                color=colors["green"],
                alpha=0.3,
                zorder=4,
                edgecolor="none",
            )
            ax.text(
                idx + 1,
                (row["Top"] + row["Bottom"]) / 2,
                "OB",
                fontsize=4,
                color=colors["green"],
                alpha=0.8,
            )
        elif row["OB"] == -1.0 and future_high < row["Bottom"]:
            ax.fill_between(
                [idx, idx + percentage_strength_width * multiplier],
                row["Bottom"],
                row["Top"],
                color=colors["red"],
                alpha=0.3,
                zorder=4,
                edgecolor="none",
            )
            ax.text(
                idx + 1,
                (row["Bottom"] + row["Top"]) / 2,
                "OB",
                fontsize=4,
                color=colors["red"],
                alpha=0.8,
            )


def draw_liquidity(ax, liquidity: pd.DataFrame):
    """Draw liquidity levels on the chart."""
    colors = get_color_scheme()
    last_x = liquidity.index[-1]

    for _, row in liquidity.iterrows():
        alpha = 0.15
        if row["Swept"] == 0:
            row["Swept"] = last_x
            alpha = 0.75

        color = colors["green"] if row["Liquidity"] == 1.0 else colors["red"]
        label = "BSL" if row["Liquidity"] == 1.0 else "SSL"

        ax.hlines(
            y=row["Level"],
            xmin=_,
            xmax=row["Swept"],
            color=color,
            linewidth=1,
            alpha=alpha,
        )

        # Add label
        mid_x = (_ + row["Swept"]) / 2
        ax.text(
            mid_x,
            row["Level"],
            label,
            fontsize=4,
            ha="center",
            va=("bottom" if row["Liquidity"] == 1.0 else "top"),
            color=color,
            alpha=alpha,
        )


def draw_fvg(ax, fvgs: pd.DataFrame):
    """Draw Fair Value Gaps on the chart."""
    colors = get_color_scheme()

    for _, fvg in fvgs.iterrows():
        if (
            pd.notna(fvg["Top"])
            and pd.notna(fvg["Bottom"])
            and fvg["Top"] / fvg["Bottom"] > 1.01
        ):
            is_mitigated = fvg["MitigatedIndex"] != 0.0
            alpha = 0.15 if is_mitigated else 0.75
            color = colors["green"] if fvg["FVG"] == 1.0 else colors["red"]

            ax.vlines(
                x=_,
                ymin=fvg["Bottom"],
                ymax=fvg["Top"],
                colors=color,
                alpha=alpha,
                linewidth=4,
            )
            ax.text(
                _ + 1,
                (fvg["Top"] + fvg["Bottom"]) / 2,
                "FVG",
                fontsize=4,
                ha="left",
                va="center",
                color=color,
                alpha=alpha,
            )


def draw_bos_choch(ax, bos_choch: pd.DataFrame, ohlc: pd.DataFrame):
    """Draw Break of Structure and Change of Character."""
    colors = get_color_scheme()

    if bos_choch.empty:
        return

    valid_rows = bos_choch.dropna(subset=["BOS", "CHOCH"], how="all")

    for idx, row in valid_rows.iterrows():
        if pd.isna(row["BrokenIndex"]):
            continue

        row_index = int(row["BrokenIndex"])
        if row_index >= len(ohlc):
            continue

        # Draw CHOCH
        if not pd.isna(row["CHOCH"]):
            direction = "bullish" if row["CHOCH"] == 1.0 else "bearish"
            color = colors["green"] if direction == "bullish" else colors["red"]
            swing_index = idx

            if swing_index >= len(ohlc) or row_index >= len(ohlc):
                continue

            ax.hlines(
                y=row["Level"],
                xmin=swing_index,
                xmax=row_index,
                color=color,
                linewidth=0.5,
                linestyle="-",
            )

            ax.text(
                swing_index + (row_index - swing_index) / 2,
                row["Level"],
                "CHoCH",
                fontsize=4,
                ha="center",
                va=("bottom" if direction == "bullish" else "top"),
                color=color,
            )

        # Draw BOS
        if not pd.isna(row["BOS"]):
            direction = "bullish" if row["BOS"] == 1.0 else "bearish"
            color = colors["green"] if direction == "bullish" else colors["red"]
            swing_index = idx

            if swing_index >= len(ohlc) or row_index >= len(ohlc):
                continue

            ax.hlines(
                y=row["Level"],
                xmin=swing_index,
                xmax=row_index,
                color=color,
                linewidth=0.5,
                linestyle="-",
            )

            ax.text(
                swing_index + (row_index - swing_index) / 2,
                row["Level"],
                "BOS",
                fontsize=4,
                ha="center",
                va=("bottom" if direction == "bullish" else "top"),
                color=color,
            )


def draw_last_price(ax, ohlc: pd.DataFrame):
    """Draw last price marker on Y-axis."""
    colors = get_color_scheme()

    last_price = ohlc["close"].iloc[-1]
    prev_price = ohlc["close"].iloc[-2] if len(ohlc) > 1 else last_price
    price_color = colors["green"] if last_price >= prev_price else colors["red"]

    # Get current y-axis ticks and labels
    current_ticks = list(ax.get_yticks())
    current_labels = [
        format_number(tick, precision=4, is_format_k=False) for tick in current_ticks
    ]

    # Add the last price to ticks
    current_ticks.append(last_price)
    current_labels.append(format_number(last_price, precision=4, is_format_k=False))

    # Set new ticks and labels
    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)

    # Highlight the last price tick
    for i, tick in enumerate(current_ticks):
        if abs(tick - last_price) < 0.0001:  # Match the last price tick
            ax.get_yticklabels()[i].set_color(colors["white"])
            ax.get_yticklabels()[i].set_bbox(
                {
                    "boxstyle": "square,pad=0.4",
                    "mutation_aspect": 0.5,
                    "color": price_color,
                }
            )


def draw_highs_lows(ax, ohlc: pd.DataFrame, dark_mode: bool):
    """Draw all-time highs and lows."""
    colors = get_color_scheme()

    # Skip first and last candle for highs/lows
    all_time_highs = ohlc["high"].iloc[1:-1].max()
    all_time_highs_index = ohlc["high"].iloc[1:-1].idxmax()
    all_time_lows = ohlc["low"].iloc[1:-1].min()
    all_time_lows_index = ohlc["low"].iloc[1:-1].idxmin()

    all_time_highs_row_index = ohlc.index.get_loc(all_time_highs_index)
    all_time_lows_row_index = ohlc.index.get_loc(all_time_lows_index)

    ax.hlines(
        y=all_time_highs,
        xmin=all_time_highs_row_index,
        xmax=len(ohlc) + 1,
        color=colors["light"] if dark_mode else colors["black"],
        linewidth=0.2,
        alpha=0.5,
    )
    ax.hlines(
        y=all_time_lows,
        xmin=all_time_lows_row_index,
        xmax=len(ohlc) + 1,
        color=colors["light"] if dark_mode else colors["black"],
        linewidth=0.2,
        alpha=0.5,
    )
