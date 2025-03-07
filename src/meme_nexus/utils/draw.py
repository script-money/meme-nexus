import base64
import logging
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

from .format import format_number, format_timeframe
from .indicators import (
    calculate_bos_choch,
    calculate_fvg,
    calculate_liquidity,
    calculate_order_blocks,
    calculate_rainbow_indicator,
    calculate_swing_points,
)

logger = logging.getLogger(__name__)


def plot_candlestick(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Literal["minute", "m", "hour", "h", "day", "d"],
    aggregate: int,
    limit: int | None = None,
    is_save_file=False,
    cache_dir: str = "./tmp/images/",
    is_draw_swingpoint=False,
    is_draw_orderblock=False,
    is_draw_liquidity=False,
    is_draw_fvg=False,
    is_draw_choch=False,
    is_draw_rainbow=False,
    dark_mode=True,
    indicators: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    green = "#0C8E76"
    red = "#F02F3C"
    dark = "#151821"
    gray = "#1A1D26"
    light = "#d1d4dc"
    white = "#ffffff"
    black = "#000000"

    # Check if index is DatetimeIndex, if not convert it
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="s")

    # Use the last 'limit' rows of data if limit is set
    ohlc = df.copy()
    if limit is not None:
        ohlc = ohlc.iloc[-limit:]

    # Set up the style for the chart
    market_colors = mpf.make_marketcolors(
        up=green,
        down=red,
        edge="inherit",
        wick="inherit",
        volume={"up": green, "down": red},
        alpha=0.95,
    )
    style = mpf.make_mpf_style(
        marketcolors=market_colors,
        figcolor=dark if dark_mode else white,
        facecolor=dark if dark_mode else white,
        edgecolor=gray if dark_mode else light,
        gridcolor=gray if dark_mode else white,
        gridstyle="-",
        y_on_right=True,
        rc={
            "axes.labelcolor": light if dark_mode else black,
            "xtick.color": light if dark_mode else black,
            "ytick.color": light if dark_mode else black,
            "text.color": light if dark_mode else black,
            "axes.titlecolor": light if dark_mode else black,
        },
    )

    # Create custom volume overlay
    colors = [
        green if close >= open else red
        for open, close in zip(ohlc["open"], ohlc["close"], strict=True)
    ]
    volume_overlay = mpf.make_addplot(
        ohlc["volume"], type="bar", panel=1, color=colors, alpha=0.75, width=0.8
    )

    timeframe = format_timeframe(timeframe)
    base_length = {"m": 8, "h": 12, "d": 20}[timeframe]

    # Calculate or use provided swing points
    swing_length = max(6, round(base_length * (aggregate**0.5)))

    if indicators is None:
        # Calculate indicators on-the-fly (backward compatibility)
        logger.info("No pre-calculated indicators provided, calculating on the fly")
        swing_data = calculate_swing_points(ohlc, swing_length)
        swings, swing_highs, swing_lows = swing_data
    else:
        # Use provided indicators
        logger.info("Using provided indicators")
        swings = indicators.get("swings")
        swing_highs = indicators.get("swing_highs")
        swing_lows = indicators.get("swing_lows")

    addplots = [volume_overlay]

    # Rainbow colors
    orange = "#b6420d"
    yellow = "#706b2b"
    lime_green = "#3a7d44"
    blue = "#0ea0a4"
    cyan = "#1b66af"

    # Calculate or use provided Rainbow indicator components if enabled
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

    if is_draw_rainbow:
        if indicators is not None and all(col in indicators for col in rainbow_cols):
            # Use provided rainbow indicators
            logger.info("Using provided rainbow indicators")
            for col in rainbow_cols:
                ohlc[col] = indicators[col]
        else:
            # Calculate rainbow indicator on-the-fly
            logger.info("Calculating rainbow indicators on the fly")
            rainbow_data = calculate_rainbow_indicator(ohlc)
            if rainbow_data:
                for col in rainbow_cols:
                    if col in rainbow_data:
                        ohlc[col] = rainbow_data[col]
            else:
                warnings.warn(
                    "Insufficient data to calculate Rainbow indicator.",
                    stacklevel=2,
                )
                is_draw_rainbow = False

        # Add Rainbow lines to the plot
        orange_line_plot = mpf.make_addplot(
            ohlc["orange_line"], color=orange, width=0.5, panel=0
        )
        yellow_line_plot = mpf.make_addplot(
            ohlc["yellow_line"], color=yellow, width=0.5, panel=0
        )
        green_line_plot = mpf.make_addplot(
            ohlc["green_line"], color=lime_green, width=0.5, panel=0
        )
        blue_line_plot = mpf.make_addplot(
            ohlc["blue_line"], color=blue, width=0.5, panel=0
        )
        cyan_line_plot = mpf.make_addplot(
            ohlc["cyan_line"], color=cyan, width=0.5, panel=0
        )

        addplots.extend(
            [
                orange_line_plot,
                yellow_line_plot,
                green_line_plot,
                blue_line_plot,
                cyan_line_plot,
            ]
        )

        # Add trend signals - ensure marker data is not empty
        if ohlc["bull_start"].any():
            bull_start_markers = mpf.make_addplot(
                ohlc["close"].where(ohlc["bull_start"]),
                type="scatter",
                marker="^",  # Up arrow
                markersize=60,
                color="lime",
                panel=0,
            )
            addplots.append(bull_start_markers)

        if ohlc["bull_end"].any():
            bull_end_markers = mpf.make_addplot(
                ohlc["close"].where(ohlc["bull_end"]),
                type="scatter",
                marker="x",  # X mark
                markersize=60,
                color="lime",
                panel=0,
            )
            addplots.append(bull_end_markers)

        if ohlc["bear_start"].any():
            bear_start_markers = mpf.make_addplot(
                ohlc["close"].where(ohlc["bear_start"]),
                type="scatter",
                marker="v",  # Down arrow
                markersize=60,
                color="red",
                panel=0,
            )
            addplots.append(bear_start_markers)

        if ohlc["bear_end"].any():
            bear_end_markers = mpf.make_addplot(
                ohlc["close"].where(ohlc["bear_end"]),
                type="scatter",
                marker="x",  # X mark
                markersize=60,
                color="red",
                panel=0,
            )
            addplots.append(bear_end_markers)

    if is_draw_swingpoint:
        price_range = ohlc["high"].max() - ohlc["low"].min()
        dynamic_offset_factor = 0.01
        dynamic_offset = price_range * dynamic_offset_factor

        # actual offset
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

        swing_high_markers = mpf.make_addplot(
            ohlc["high"].where(swing_highs) + high_offsets,
            type="scatter",
            marker="$⬇$",
            color=light if dark_mode else black,
            markersize=6,
        )
        swing_low_markers = mpf.make_addplot(
            ohlc["low"].where(swing_lows) - low_offsets,
            type="scatter",
            marker="$⬆$",
            color=light if dark_mode else black,
            markersize=6,
        )
        addplots.extend([swing_high_markers, swing_low_markers])

    plt.rcParams.update({"font.size": 8})

    fig, ax_all = mpf.plot(
        ohlc,
        type="candle",
        style=style,
        addplot=addplots,
        figsize=(10, 5),
        datetime_format="%m-%d %H:%M",
        xrotation=0,
        returnfig=True,
        panel_ratios=(3, 1),
        tight_layout=True,
        scale_padding={"left": 0, "right": 0},
        ylabel="",
        warn_too_much_data=10000,
    )

    # Get the main price subplot (usually the first one)
    ax = ax_all[0]
    ax_volume = ax_all[2]

    # Set more space for y axis
    y_min = ohlc["low"].min()
    y_max = ohlc["high"].max()
    price_range = y_max - y_min
    ax.set_ylim(y_min - price_range * 0.06, y_max + price_range * 0.06)

    # Reduce x and y label size
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax_volume.tick_params(axis="both", which="major", labelsize=6)

    # Format volume
    ax_volume.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: format_number(x))
    )

    # Add horizontal dotted lines for the main chart
    ax.grid(axis="x", linewidth=0.1, color=light)
    ax.grid(axis="y", linewidth=0.1, color=light)

    # Add swing points' price labels
    if is_draw_swingpoint:
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
                    color=light if dark_mode else black,
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
                    color=light if dark_mode else black,
                )

    # Add the title to the top left corner of the main chart
    ax.text(
        0.5,
        1.05,
        f"{symbol} ({aggregate} {timeframe}) O:{ohlc['open'].iloc[-1]:.6g} H:{ohlc['high'].iloc[-1]:.6g} L:{ohlc['low'].iloc[-1]:.6g} C:{ohlc['close'].iloc[-1]:.6g}",  # noqa: E501
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=8,
    )

    # Add Unix timestamp to the top right corner
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

    # Adjust x-axis ticks to show local time
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: (
                datetime.fromtimestamp(ohlc.index[int(x)].timestamp()).strftime(
                    "%m-%d %H:%M"
                )
                if x < len(ohlc)
                else ""
            )
        )
    )

    # Draw all time highs and lows
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
        color=light if dark_mode else black,
        linewidth=0.2,
        alpha=0.5,
    )
    ax.hlines(
        y=all_time_lows,
        xmin=all_time_lows_row_index,
        xmax=len(ohlc) + 1,
        color=light if dark_mode else black,
        linewidth=0.2,
        alpha=0.5,
    )

    # Draw order blocks & Liquidity
    if swings is not None and (
        is_draw_orderblock or is_draw_liquidity or is_draw_fvg or is_draw_choch
    ):
        # Get or calculate order blocks
        if indicators is not None and "order_blocks" in indicators:
            ob = indicators["order_blocks"]
            logger.info("Using provided order blocks")
        else:
            ob = calculate_order_blocks(ohlc, swings)
            logger.info("Calculating order blocks on the fly")

        if is_draw_orderblock and not ob.empty:
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
                        color=green,
                        alpha=0.3,
                        zorder=4,
                        edgecolor="none",
                    )
                    ax.text(
                        idx + 1,
                        (row["Top"] + row["Bottom"]) / 2,
                        "OB",
                        fontsize=4,
                        color=green,
                        alpha=0.8,
                    )
                elif row["OB"] == -1.0 and future_high < row["Bottom"]:
                    ax.fill_between(
                        [idx, idx + percentage_strength_width * multiplier],
                        row["Bottom"],
                        row["Top"],
                        color=red,
                        alpha=0.3,
                        zorder=4,
                        edgecolor="none",
                    )
                    ax.text(
                        idx + 1,
                        (row["Bottom"] + row["Top"]) / 2,
                        "OB",
                        fontsize=4,
                        color=red,
                        alpha=0.8,
                    )

        if is_draw_liquidity:
            # Get or calculate liquidity
            if indicators is not None and "liquidity" in indicators:
                liquidity = indicators["liquidity"]
                logger.info("Using provided liquidity data")
            else:
                liquidity = calculate_liquidity(ohlc, swings)
                logger.info("Calculating liquidity on the fly")
            last_x = liquidity.index[-1]
            for _, row in liquidity.iterrows():
                alpha = 0.1
                if row["Swept"] == 0:
                    row["Swept"] = last_x
                    alpha = 1.0
                color = green if row["Liquidity"] == -1.0 else red
                ax.hlines(
                    y=row["Level"],
                    xmin=_,
                    xmax=row["Swept"],
                    color=color,
                    linewidth=0.8,
                    alpha=alpha,
                )

        # Draw FVGs
        if is_draw_fvg:
            # Get or calculate FVGs
            if indicators is not None and "fvgs" in indicators:
                fvgs = indicators["fvgs"]
                logger.info("Using provided FVGs data")
            else:
                fvgs = calculate_fvg(ohlc)
                logger.info("Calculating FVGs on the fly")
            for _, fvg in fvgs.iterrows():
                if (
                    pd.notna(fvg["Top"])
                    and pd.notna(fvg["Bottom"])
                    and fvg["Top"] / fvg["Bottom"] > 1.01  # filter small FVGs
                ):
                    is_mitigated = fvg["MitigatedIndex"] != 0.0
                    alpha = 0.15 if is_mitigated else 0.75

                    if fvg["FVG"] == 1.0:
                        ax.vlines(
                            x=_,
                            ymin=fvg["Bottom"],
                            ymax=fvg["Top"],
                            colors=green,
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
                            color=green,
                            alpha=alpha,
                        )
                    elif fvg["FVG"] == -1.0:
                        ax.vlines(
                            x=_,
                            ymin=fvg["Bottom"],
                            ymax=fvg["Top"],
                            colors=red,
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
                            color=red,
                            alpha=alpha,
                        )

        # Draw CHOCH and BOS
        if is_draw_choch:
            # Get or calculate BOS/CHOCH data
            if indicators is not None and "bos_choch" in indicators:
                bos_choch = indicators["bos_choch"]
                logger.info("Using provided BOS/CHOCH data")
            else:
                bos_choch = calculate_bos_choch(ohlc, swings)
                logger.info("Calculating BOS/CHOCH on the fly")

            if not bos_choch.empty:
                valid_rows = bos_choch.dropna(subset=["BOS", "CHOCH"], how="all")

                for idx, row in valid_rows.iterrows():
                    if pd.isna(row["BrokenIndex"]):
                        continue

                    row_index = int(row["BrokenIndex"])

                    if row_index >= len(ohlc):
                        continue

                    if not pd.isna(row["CHOCH"]):
                        direction = "bullish" if row["CHOCH"] == 1.0 else "bearish"
                        color = green if direction == "bullish" else red

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

                    if not pd.isna(row["BOS"]):
                        direction = "bullish" if row["BOS"] == 1.0 else "bearish"
                        color = green if direction == "bullish" else red

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

    # Save the chart in WebP format
    last_timestamp = datetime.fromtimestamp(ohlc.index[-1].timestamp())  # local time
    if "/" in symbol:
        symbol = symbol.split("/")[0]
    webp_filename = f"{symbol}-{last_timestamp}-{aggregate}{timeframe}.webp"

    buffered = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buffered, dpi=300, bbox_inches="tight")

    img = Image.open(buffered)
    webp_buffer = BytesIO()
    img.save(webp_buffer, format="WEBP")
    base64_string = base64.b64encode(webp_buffer.getvalue()).decode("utf-8")
    mime_type = "image/webp"

    if is_save_file:
        os.makedirs(f"./{cache_dir}", exist_ok=True)
        webp_filepath = os.path.join(os.path.abspath(cache_dir), webp_filename)
        with open(webp_filepath, "wb") as f:
            f.write(webp_buffer.getvalue())

    plt.close(fig)

    return (
        os.path.join(os.path.abspath(cache_dir), webp_filename),
        mime_type,
        base64_string,
    )
