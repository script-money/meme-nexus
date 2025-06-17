"""Main plotting function that orchestrates all chart elements."""

import base64
import os
import warnings

from datetime import datetime
from io import BytesIO
from typing import Any, Literal

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from PIL import Image

from ...models import LiquidationHeatmapData
from ..format import format_number, format_timeframe
from ..indicators import (
    calculate_bos_choch,
    calculate_fvg,
    calculate_liquidity,
    calculate_order_blocks,
    calculate_rainbow_indicator,
    calculate_swing_points,
    create_active_session_mask,
)
from .config import create_chart_style, get_color_scheme
from .elements import (
    configure_additional_chart_axes,
    create_swing_point_plots,
    draw_additional_charts,
    draw_bos_choch,
    draw_fvg,
    draw_highs_lows,
    draw_last_price,
    draw_liquidation_heatmap,
    draw_liquidity,
    draw_order_blocks,
    draw_rainbow_indicator,
    draw_swing_point_labels,
    draw_volume,
)


def plot_candlestick(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Literal["minute", "m", "hour", "h", "day", "d"],
    aggregate: int,
    limit: int | None = None,
    is_save_file=False,
    only_killzone=False,
    cache_dir: str = "./tmp/images/",
    is_draw_swingpoint=False,
    is_draw_orderblock=False,
    is_draw_liquidity=False,
    is_draw_fvg=False,
    is_draw_choch=False,
    is_draw_rainbow=False,
    is_draw_volume=True,
    is_draw_last_price=True,
    is_draw_liquidation_heatmap=False,
    liquidation_heatmap_data: LiquidationHeatmapData | dict | None = None,
    dark_mode=True,
    indicators: dict[str, Any] | None = None,
    additional_charts: list[dict[str, Any]] | None = None,
    zero_line_separator: float | None = None,
) -> tuple[str, str, str]:
    """
    Plot candlestick chart with various technical indicators and additional charts.

    Args:
        additional_charts: List of additional charts to display below main chart.
                          Each dict should contain:
                          - title: Chart title
                          - series: Data series (list of values)
                          - timestamps: Timestamps for data points
                          - type: Chart type ("Line", "histogram", "Candle")
        zero_line_separator: Value for positive/negative color separation.
                           If set, values >= separator are green, < separator are red.

    Returns:
        tuple: (file_path, mime_type, base64_string)
    """
    colors = get_color_scheme()

    # Check if index is DatetimeIndex, if not convert it
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="s")

    # Use the last 'limit' rows of data if limit is set
    ohlc = df.copy()
    if limit is not None:
        ohlc = ohlc.iloc[-limit:]

    # Create chart style
    style = create_chart_style(dark_mode)

    # Create marketcolor overrides for killzone filtering
    marketcolor_overrides = None
    if only_killzone:
        # Get active session mask
        active_mask = create_active_session_mask(ohlc.index)

        marketcolor_overrides = []
        for is_active in active_mask:
            if not is_active:
                # For inactive sessions (not in killzone), use transparent colors
                # Using RGBA tuples with reduced alpha (0.3 for 30% opacity)
                if dark_mode:
                    # For dark mode: transparent red for bearish, green for bullish
                    marketcolor_overrides.append(
                        mpf.make_marketcolors(
                            up=(0.0, 1.0, 0.0, 0.3),  # Transparent green (RGBA)
                            down=(1.0, 0.0, 0.0, 0.3),  # Transparent red (RGBA)
                            edge=(0.5, 0.5, 0.5, 0.3),  # Transparent gray edge (RGBA)
                            wick=(0.5, 0.5, 0.5, 0.3),  # Transparent gray wick (RGBA)
                        )
                    )
                else:
                    # For light mode: transparent colors
                    marketcolor_overrides.append(
                        mpf.make_marketcolors(
                            up=(0.0, 0.5, 0.0, 0.3),  # Transparent dark green (RGBA)
                            down=(1.0, 0.0, 0.0, 0.3),  # Transparent red (RGBA)
                            edge=(
                                0.25,
                                0.25,
                                0.25,
                                0.3,
                            ),  # Transparent dark gray edge (RGBA)
                            wick=(
                                0.25,
                                0.25,
                                0.25,
                                0.3,
                            ),  # Transparent dark gray wick (RGBA)
                        )
                    )
            else:
                # For active sessions, use default colors (None means no override)
                marketcolor_overrides.append(None)

    # Initialize addplots list
    addplots = []

    # Add volume if requested
    if is_draw_volume:
        volume_plot = draw_volume(ohlc)
        addplots.append(volume_plot)

    # Calculate swing points
    timeframe_formatted = format_timeframe(timeframe)
    base_length = {"m": 8, "h": 12, "d": 20}[timeframe_formatted]
    swing_length = max(6, round(base_length * (aggregate**0.5)))

    if indicators is None:
        swings, swing_highs, swing_lows = calculate_swing_points(
            ohlc, swing_length, only_killzone
        )
    else:
        swings = indicators.get("swings")
        swing_highs = indicators.get("swing_highs")
        swing_lows = indicators.get("swing_lows")

    # Handle Rainbow indicator
    if is_draw_rainbow:
        rainbow_cols = [
            "orange_line",
            "yellow_line",
            "green_line",
            "cyan_line",
            "blue_line",
            "trend",
        ]

        if indicators is not None and all(col in indicators for col in rainbow_cols):
            for col in rainbow_cols:
                ohlc[col] = indicators[col]
        else:
            rainbow_df = calculate_rainbow_indicator(ohlc)
            if not rainbow_df.empty:
                for col in rainbow_df.columns:
                    ohlc[col] = rainbow_df[col]
            else:
                warnings.warn(
                    "Insufficient data to calculate Rainbow indicator.",
                    stacklevel=2,
                )
                is_draw_rainbow = False

        if is_draw_rainbow:
            rainbow_plots = draw_rainbow_indicator(ohlc)
            addplots.extend(rainbow_plots)

    # Handle additional charts
    if additional_charts:
        additional_addplots = draw_additional_charts(
            additional_charts, ohlc, is_draw_volume, chart_type="ratio"
        )
        addplots.extend(additional_addplots)

    # Draw swing points
    dynamic_offset = None
    if is_draw_swingpoint:
        swing_plots, dynamic_offset = create_swing_point_plots(
            ohlc, swing_highs, swing_lows, dark_mode
        )
        addplots.extend(swing_plots)

    # Configure plot parameters
    plt.rcParams.update({"font.size": 8})

    # Calculate panel ratios based on volume and additional charts
    panel_ratios = [4]  # Main chart - increased from 3 to 4
    if is_draw_volume:
        panel_ratios.append(1)  # Volume panel
    if additional_charts:
        # Compress order depth charts - use 0.6 instead of 1
        panel_ratios.extend([0.6] * len(additional_charts))
    panel_ratios = tuple(panel_ratios)

    # Create the main plot
    plot_kwargs = {
        "data": ohlc,
        "type": "candle",
        "style": style,
        "addplot": addplots,
        "figsize": (10, 5),
        "datetime_format": "%m-%d %H:%M",
        "xrotation": 0,
        "returnfig": True,
        "panel_ratios": panel_ratios,
        "tight_layout": True,
        "scale_padding": {"left": 0, "right": 0},
        "ylabel": "",
        "warn_too_much_data": 10000,
        "volume": False,
    }

    # Only add marketcolor_overrides if it's not None
    if marketcolor_overrides is not None:
        plot_kwargs["marketcolor_overrides"] = marketcolor_overrides

    fig, ax_all = mpf.plot(**plot_kwargs)

    # Get axes - mplfinance creates 2 axes per panel (left and right y-axis)
    ax = ax_all[0]  # Main price chart
    ax_volume = ax_all[2] if is_draw_volume else None  # Volume chart (skip right axis)

    # Get additional chart axes - only take left y-axes (even indices)
    additional_axes = []
    if additional_charts:
        # Skip main (0,1) and volume (2,3) if present
        start_idx = 4 if is_draw_volume else 2
        # Take every second axis (left y-axis only)
        for i in range(len(additional_charts)):
            axis_idx = start_idx + (i * 2)  # Skip right y-axes
            if axis_idx < len(ax_all):
                additional_axes.append(ax_all[axis_idx])

    # Set y-axis limits
    y_min = ohlc["low"].min()
    y_max = ohlc["high"].max()
    price_range = y_max - y_min
    ax.set_ylim(y_min - price_range * 0.06, y_max + price_range * 0.06)

    # Configure axes
    ax.tick_params(axis="both", which="major", labelsize=6)

    # Double the Y-axis tick density
    from matplotlib.ticker import MaxNLocator

    current_ticks = len(ax.get_yticks())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=current_ticks * 2))

    if is_draw_volume and ax_volume is not None:
        ax_volume.tick_params(axis="both", which="major", labelsize=6)
        ax_volume.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format_number(x))
        )
        ax_volume.set_ylabel("")

    # Configure additional chart axes
    if additional_charts and additional_axes:
        configure_additional_chart_axes(
            additional_charts, additional_axes, dark_mode, chart_type="ratio"
        )

    # Add grid
    ax.grid(axis="x", linewidth=0.1, color=colors["light"])
    ax.grid(axis="y", linewidth=0.1, color=colors["light"])

    # Draw swing points labels
    if is_draw_swingpoint and dynamic_offset is not None:
        draw_swing_point_labels(
            ax, ohlc, swing_highs, swing_lows, dark_mode, dynamic_offset
        )

    # Draw last price
    if is_draw_last_price:
        draw_last_price(ax, ohlc)

    # Add title
    ax.text(
        0.5,
        1.05,
        f"{symbol} ({aggregate} {timeframe_formatted}) O:{ohlc['open'].iloc[-1]:.6g} H:{ohlc['high'].iloc[-1]:.6g} L:{ohlc['low'].iloc[-1]:.6g} C:{ohlc['close'].iloc[-1]:.6g}",  # noqa: E501
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=8,
    )

    # Add timestamp
    ax.text(
        1.02,
        1.05,
        f"Timestamp: {int(ohlc.index[-1].timestamp())}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        alpha=1,
    )

    # Configure x-axis
    num_ticks = min(10, len(ohlc))
    tick_indices = np.linspace(0, len(ohlc) - 1, num_ticks, dtype=int)
    if tick_indices[-1] != len(ohlc) - 1:
        tick_indices[-1] = len(ohlc) - 1

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [
            datetime.fromtimestamp(ohlc.index[i].timestamp()).strftime("%m-%d %H:%M")
            for i in tick_indices
        ],
        rotation=0,
    )

    # Draw all-time highs and lows
    draw_highs_lows(ax, ohlc, dark_mode)

    # Draw advanced indicators if swing points are available
    if swings is not None and (
        is_draw_orderblock or is_draw_liquidity or is_draw_fvg or is_draw_choch
    ):
        # Order blocks
        if is_draw_orderblock:
            if indicators is not None and "order_blocks" in indicators:
                ob = indicators["order_blocks"]
            else:
                ob = calculate_order_blocks(ohlc, swings)

            if not ob.empty:
                draw_order_blocks(ax, ob, swings)

        # Liquidity
        if is_draw_liquidity:
            if indicators is not None and "liquidity" in indicators:
                liquidity = indicators["liquidity"]
            else:
                liquidity = calculate_liquidity(ohlc, swings)

            draw_liquidity(ax, liquidity)

        # FVG
        if is_draw_fvg:
            if indicators is not None and "fvgs" in indicators:
                fvgs = indicators["fvgs"]
            else:
                fvgs = calculate_fvg(ohlc)

            draw_fvg(ax, fvgs)

        # BOS/CHOCH
        if is_draw_choch:
            if indicators is not None and "bos_choch" in indicators:
                bos_choch = indicators["bos_choch"]
            else:
                bos_choch = calculate_bos_choch(ohlc, swings)

            draw_bos_choch(ax, bos_choch, ohlc)

    # Draw liquidation heatmap
    if is_draw_liquidation_heatmap and liquidation_heatmap_data is not None:
        # Convert dict to LiquidationHeatmapData if needed (for backward compatibility)
        if isinstance(liquidation_heatmap_data, dict):
            heatmap_data = LiquidationHeatmapData.from_api_response(
                liquidation_heatmap_data
            )
        else:
            heatmap_data = liquidation_heatmap_data
        draw_liquidation_heatmap(ax, ohlc, heatmap_data, dark_mode)

    # Save the chart
    last_timestamp = datetime.fromtimestamp(ohlc.index[-1].timestamp())
    if "/" in symbol:
        symbol = symbol.split("/")[0]
    webp_filename = f"{symbol}-{last_timestamp}-{aggregate}{timeframe_formatted}.webp"

    # Create image buffer
    buffered = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buffered, dpi=300, bbox_inches="tight")

    # Convert to WebP
    img = Image.open(buffered)
    webp_buffer = BytesIO()
    img.save(webp_buffer, format="WEBP")
    base64_string = base64.b64encode(webp_buffer.getvalue()).decode("utf-8")
    mime_type = "image/webp"

    # Save file if requested
    file_path = os.path.join(os.path.abspath(cache_dir), webp_filename)
    if is_save_file:
        os.makedirs(cache_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(webp_buffer.getvalue())

    plt.close(fig)

    return file_path, mime_type, base64_string
