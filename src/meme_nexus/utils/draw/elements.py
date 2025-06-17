"""Individual chart element drawing functions."""

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

from ...models import LiquidationHeatmapData
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
        ohlc["volume"], type="bar", panel=1, color=bar_colors, alpha=0.75, width=1.0
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


def draw_additional_charts(
    additional_charts: list[dict],
    ohlc: pd.DataFrame,
    is_draw_volume: bool,
    chart_type: str = "ratio",
) -> list:
    """
    Create addplots for additional charts (order depth, dominance charts, etc.).

    Args:
        additional_charts: List of chart configurations
        ohlc: OHLC DataFrame to align data with
        is_draw_volume: Whether volume panel is drawn
        chart_type: Type of chart ("ratio" or other types)
    """
    addplots = []

    for chart in additional_charts:
        series = chart.get("series", [])
        timestamps = chart.get("timestamps", [])

        # Create DataFrame for the additional chart
        chart_df = pd.DataFrame({"timestamp": timestamps, "value": series})

        if not chart_df.empty:
            # Convert timestamps to datetime index
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="s")
            chart_df = chart_df.set_index("timestamp")

            # Reindex to match main OHLC data (keep NaN values as NaN)
            chart_df = chart_df.reindex(ohlc.index)

            # Apply filtering and styling based on chart type
            chart_index = additional_charts.index(chart)
            panel_idx = (1 if is_draw_volume else 0) + 1 + chart_index

            if chart_type == "ratio":
                # Ratio charts: filter out values between -1 and 1, use histogram
                series = chart.get("series", [])
                has_positive = any(
                    val > 1 for val in series if val is not None and not np.isnan(val)
                )
                has_negative = any(
                    val < -1 for val in series if val is not None and not np.isnan(val)
                )

                if has_positive and not has_negative:
                    # Buy dominance chart - only show values > 1
                    chart_df["value"] = chart_df["value"].where(
                        chart_df["value"] > 1, np.nan
                    )
                    color = "green"
                elif has_negative and not has_positive:
                    # Sell dominance chart - only show values < -1
                    chart_df["value"] = chart_df["value"].where(
                        chart_df["value"] < -1, np.nan
                    )
                    color = "red"
                else:
                    # Mixed chart - filter both ranges
                    chart_df["value"] = chart_df["value"].where(
                        (chart_df["value"] > 1) | (chart_df["value"] < -1), np.nan
                    )
                    color = "steelblue"

                addplots.append(
                    mpf.make_addplot(
                        chart_df["value"],
                        type="bar",
                        color=color,
                        alpha=0.7,
                        panel=panel_idx,
                        ylabel="",
                    )
                )
            else:
                # Other chart types: use line chart with orange color
                addplots.append(
                    mpf.make_addplot(
                        chart_df["value"],
                        type="line",
                        color="#ff8d00",  # Orange color for other types
                        panel=panel_idx,
                        ylabel="",
                    )
                )

    return addplots


def configure_additional_chart_axes(
    additional_charts: list[dict],
    additional_axes: list,
    dark_mode: bool,
    chart_type: str = "ratio",
):
    """
    Configure Y-axis range and styling for additional charts.

    Args:
        additional_charts: List of chart configurations
        additional_axes: List of matplotlib axes for additional charts
        dark_mode: Whether to use dark mode styling
        chart_type: Type of chart ("ratio" or other types)
    """
    colors = get_color_scheme()

    for i, (chart, ax_additional) in enumerate(
        zip(additional_charts, additional_axes, strict=True)
    ):
        ax_additional.tick_params(axis="both", which="major", labelsize=6)
        ax_additional.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format_number(x))
        )
        ax_additional.set_ylabel("")

        if chart_type == "ratio":
            # Configure Y-axis range for ratio charts (dominance charts)
            series = chart.get("series", [])

            # Check if this is buy or sell chart based on data
            has_positive = any(
                val > 1 for val in series if val is not None and not np.isnan(val)
            )
            has_negative = any(
                val < -1 for val in series if val is not None and not np.isnan(val)
            )

            if has_positive and not has_negative:
                # Buy dominance chart - dynamic range based on max value
                valid_positives = [
                    val
                    for val in series
                    if val is not None and not np.isnan(val) and val > 1
                ]
                if valid_positives:
                    max_val = max(valid_positives)
                    # Set upper limit with some padding
                    if max_val <= 1.5:
                        y_max = 1.6
                        ticks = [1, 1.2, 1.4]
                        labels = ["±1", "1.2", "1.4"]
                    elif max_val <= 2.0:
                        y_max = 2.1
                        ticks = [1, 1.5, 2]
                        labels = ["±1", "1.5", "2"]
                    elif max_val <= 2.5:
                        y_max = 2.6
                        ticks = [1, 1.5, 2, 2.5]
                        labels = ["±1", "1.5", "2", "2.5"]
                    else:
                        y_max = max_val + 0.2
                        ticks = [1, 2, 3] if max_val > 2.5 else [1, 1.5, 2, 2.5]
                        labels = (
                            ["±1", "2", "3"]
                            if max_val > 2.5
                            else ["±1", "1.5", "2", "2.5"]
                        )

                    ax_additional.set_ylim(1, y_max)
                    ax_additional.set_yticks(ticks)
                    ax_additional.set_yticklabels(labels)

            elif has_negative and not has_positive:
                # Sell dominance chart - dynamic range based on min value
                valid_negatives = [
                    val
                    for val in series
                    if val is not None and not np.isnan(val) and val < -1
                ]
                if valid_negatives:
                    min_val = min(valid_negatives)
                    # Set lower limit with some padding
                    if min_val >= -1.5:
                        y_min = -1.6
                        ticks = [-1.4, -1.2, -1]
                        labels = ["-1.4", "-1.2", "±1"]
                    elif min_val >= -2.0:
                        y_min = -2.1
                        ticks = [-2, -1.5, -1]
                        labels = ["-2", "-1.5", "±1"]
                    elif min_val >= -2.5:
                        y_min = -2.6
                        ticks = [-2.5, -2, -1.5, -1]
                        labels = ["-2.5", "-2", "-1.5", "±1"]
                    else:
                        y_min = min_val - 0.2
                        ticks = [-3, -2, -1] if min_val < -2.5 else [-2.5, -2, -1.5, -1]
                        labels = (
                            ["-3", "-2", "±1"]
                            if min_val < -2.5
                            else ["-2.5", "-2", "-1.5", "±1"]
                        )

                    ax_additional.set_ylim(y_min, -1)
                    ax_additional.set_yticks(ticks)
                    ax_additional.set_yticklabels(labels)

            # Add title for additional chart (only for positive/buy charts)
            if has_positive and not has_negative:
                title = chart.get("title", f"Chart {i + 1}")
                ax_additional.text(
                    0.02,
                    0.8,
                    title,
                    transform=ax_additional.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6,
                    color=colors["white"] if dark_mode else colors["black"],
                )

            # Add zero line for reference (ratio charts)
            ax_additional.axhline(
                y=0, color=colors["light"], linestyle="--", linewidth=0.8, alpha=0.7
            )

        else:
            # Configure other chart types with auto-scaling
            series = chart.get("series", [])
            valid_values = [
                val for val in series if val is not None and not np.isnan(val)
            ]

            if valid_values:
                # Auto-scale Y-axis for other chart types
                min_val = min(valid_values)
                max_val = max(valid_values)
                value_range = max_val - min_val
                padding = value_range * 0.1  # 10% padding

                ax_additional.set_ylim(min_val - padding, max_val + padding)

            # Add title for other chart types
            title = chart.get("title", f"Chart {i + 1}")
            ax_additional.text(
                0.02,
                0.8,
                title,
                transform=ax_additional.transAxes,
                va="top",
                ha="left",
                fontsize=6,
                color=colors["white"] if dark_mode else colors["black"],
            )


def draw_liquidation_heatmap(
    ax,
    ohlc: pd.DataFrame,
    heatmap_data: LiquidationHeatmapData,
    dark_mode: bool = True,
):
    """
    Draw liquidation heatmap overlay as a 2D grid on the chart.

    This function creates a heat map visualization showing liquidation intensity
    across different price levels and time periods. The heatmap is drawn as a
    continuous 2D grid where each cell represents the liquidation volume at
    a specific price-time coordinate.

    Data Structure:
    The heatmap_data should contain:
    - data.liqHeatMap.data: Array of [time_idx, price_idx, liq_value] triplets
    - data.liqHeatMap.chartTimeArray: Array of timestamps in milliseconds
    - data.liqHeatMap.priceArray: Array of price levels
    - data.liqHeatMap.maxLiqValue: Maximum liquidation value for normalization

    Visualization:
    - Creates a 2D matrix mapping time indices (x-axis) to price indices (y-axis)
    - Colors range from dark purple (low liquidation) to bright green-yellow (high)
    - Uses bilinear interpolation for smooth color transitions
    - Positioned behind candlesticks but above background elements

    Args:
        ax: Matplotlib axis to draw on
        ohlc: OHLC dataframe with datetime index, used for coordinate mapping
        heatmap_data: LiquidationHeatmapData model containing liquidation heatmap data
        dark_mode: Whether to use dark mode color scheme (default: True)

    Example:
        The heatmap data can be fetched using:
        curl "https://api.coinank.com/api/liqMap/getLiqHeatMap?exchangeName=Binance&symbol=BTCUSDT&interval=1M"
    """
    # Extract heatmap components
    heat_data = heatmap_data.data  # [[time_idx, price_idx, liq_value], ...]
    chart_times = heatmap_data.chartTimeArray  # [timestamp1, timestamp2, ...]
    price_array = heatmap_data.priceArray  # [price1, price2, ...]
    max_liq_value = heatmap_data.maxLiqValue

    if not heat_data:
        return

    # Convert timestamps to seconds and create mapping to OHLC index
    chart_times_sec = [ts // 1000 for ts in chart_times]
    ohlc_times_sec = [int(ts.timestamp()) for ts in ohlc.index]

    # Create mapping from heatmap time index to OHLC dataframe index
    time_mapping = {}
    for heat_time_idx, heat_timestamp in enumerate(chart_times_sec):
        # Find closest matching timestamp in OHLC data
        # This works for any timeframe by finding the nearest candle
        closest_idx = min(
            range(len(ohlc_times_sec)),
            key=lambda i: abs(ohlc_times_sec[i] - heat_timestamp),
        )

        # Only map if the time difference is reasonable (within 2 hours)
        # This prevents mapping to very distant candles
        time_diff = abs(ohlc_times_sec[closest_idx] - heat_timestamp)
        if time_diff <= 7200:  # 2 hours in seconds
            # Adjust the mapping to center the heatmap cell on the candlestick
            time_mapping[heat_time_idx] = closest_idx

    # Get chart bounds to filter visible data
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    # Filter price array to visible range
    visible_price_indices = []
    visible_prices = []
    for i, price in enumerate(price_array):
        if y_min <= price <= y_max:
            visible_price_indices.append(i)
            visible_prices.append(price)

    if not visible_price_indices:
        return

    # Filter time indices to visible range
    visible_time_indices = []
    visible_chart_positions = []
    for time_idx, chart_pos in time_mapping.items():
        if x_min <= chart_pos <= x_max:
            visible_time_indices.append(time_idx)
            visible_chart_positions.append(chart_pos)

    if not visible_time_indices:
        return

    # Create 2D grid matrix for heatmap
    # Shape: (n_prices, n_times) - rows are prices (y-axis), cols are times (x-axis)
    n_prices = len(visible_price_indices)
    n_times = len(visible_time_indices)
    heatmap_matrix = np.zeros((n_prices, n_times))

    # Fill the matrix with liquidation values
    for time_idx_str, price_idx_str, liq_value_str in heat_data:
        time_idx = int(time_idx_str)
        price_idx = int(price_idx_str)
        liq_value = float(liq_value_str)

        # Skip if no liquidation value
        if liq_value <= 0:
            continue

        # Check if this time/price index is in our visible range
        if time_idx in visible_time_indices and price_idx in visible_price_indices:
            time_matrix_idx = visible_time_indices.index(time_idx)
            price_matrix_idx = visible_price_indices.index(price_idx)

            # Normalize liquidation value
            normalized_value = min(liq_value / max_liq_value, 1.0)
            heatmap_matrix[price_matrix_idx, time_matrix_idx] = normalized_value

    # Create colormap to match reference image
    if dark_mode:
        # Colors matching the reference: deep purple to cyan-green gradient
        colors_list = [
            "#15181e",  # Very dark (for zero/low values)
            "#1a0033",  # Dark purple
            "#330066",  # Purple
            "#4d0080",  # Medium purple
            "#6600cc",  # Bright purple
            "#0066cc",  # Purple-blue
            "#0080ff",  # Blue
            "#00ccff",  # Cyan
            "#00ff99",  # Cyan-green
            "#66ffcc",  # Light cyan-green
            "#ccff99",  # Bright green-yellow
        ]
        cmap = LinearSegmentedColormap.from_list("liquidation", colors_list, N=256)
    else:
        # Light mode: enhanced blue-to-red heat scale with better contrast
        # Optimized for clear distinction between long/short liquidations
        colors_list = [
            "#f8fcff",  # Almost transparent for zero values
            "#e6f3ff",  # Very light blue
            "#b3d9ff",  # Light blue
            "#80bfff",  # Medium blue
            "#4da6ff",  # Bright blue
            "#1a8cff",  # Strong blue
            "#0066cc",  # Deep blue
            "#004499",  # Navy blue
            "#002266",  # Very dark blue
            "#663300",  # Dark brown transition
            "#994400",  # Brown-orange
            "#cc5500",  # Orange
            "#ff6600",  # Bright orange
            "#ff8833",  # Yellow-orange
            "#ffaa66",  # Light orange
            "#ffcc99",  # Very light orange for maximum values
        ]
        cmap = LinearSegmentedColormap.from_list("liquidation", colors_list, N=256)

    # Set up coordinates for imshow
    # X coordinates: chart positions for times
    x_coords = np.array(visible_chart_positions)
    # Y coordinates: prices
    y_coords = np.array(visible_prices)

    # Create extent for imshow: [left, right, bottom, top]
    # Note: imshow expects extent as [x_min, x_max, y_min, y_max]
    if len(x_coords) > 1 and len(y_coords) > 1:
        x_step = (
            (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
            if len(x_coords) > 1
            else 1
        )
        y_step = (
            (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)
            if len(y_coords) > 1
            else 1
        )

        extent = [
            x_coords[0] - 0.5 * x_step,  # left - align to K-line center
            x_coords[-1] + 0.5 * x_step,  # right - align to K-line center
            y_coords[0] + 1.5 * y_step,  # bottom - align to price level center
            y_coords[-1] + 2.5 * y_step,  # top - align to price level center
        ]

        # Draw the heatmap using imshow
        # Note: imshow expects data as (rows, cols) where rows=y-axis, cols=x-axis
        # We need to flip the matrix vertically because imshow origin is top-left
        heatmap_flipped = np.flipud(heatmap_matrix)

        ax.imshow(
            heatmap_flipped,
            cmap=cmap,
            alpha=0.45,  # Reduced alpha for better candlestick visibility
            aspect="auto",
            extent=extent,
            zorder=0.8,  # Behind candlesticks but above background
            vmin=0,
            vmax=1,
            interpolation="nearest",  # Use nearest neighbor to avoid blurring alignment
        )
