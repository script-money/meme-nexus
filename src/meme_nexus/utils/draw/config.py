"""Configuration and style settings for charts."""

from typing import TypedDict

import mplfinance as mpf


class ColorScheme(TypedDict):
    green: str
    red: str
    dark: str
    gray: str
    light: str
    white: str
    black: str
    orange: str
    yellow: str
    lime_green: str
    cyan: str
    blue: str


def get_color_scheme() -> ColorScheme:
    """Get the default color scheme for charts."""
    return ColorScheme(
        green="#0C8E76",
        red="#F02F3C",
        dark="#151821",
        gray="#1A1D26",
        light="#d1d4dc",
        white="#ffffff",
        black="#000000",
        orange="#b6420d",
        yellow="#706b2b",
        lime_green="#3a7d44",
        cyan="#0ea0a4",
        blue="#1b66af",
    )


def create_chart_style(dark_mode: bool = True) -> dict:
    """Create mplfinance style configuration."""
    colors = get_color_scheme()

    market_colors = mpf.make_marketcolors(
        up=colors["green"],
        down=colors["red"],
        edge="inherit",
        wick="inherit",
        volume={"up": colors["green"], "down": colors["red"]},
        alpha=0.95,
    )

    style = mpf.make_mpf_style(
        marketcolors=market_colors,
        figcolor=colors["dark"] if dark_mode else colors["white"],
        facecolor=colors["dark"] if dark_mode else colors["white"],
        edgecolor=colors["gray"] if dark_mode else colors["light"],
        gridcolor=colors["gray"] if dark_mode else colors["white"],
        gridstyle="-",
        y_on_right=True,
        rc={
            "axes.labelcolor": colors["light"] if dark_mode else colors["black"],
            "xtick.color": colors["light"] if dark_mode else colors["black"],
            "ytick.color": colors["light"] if dark_mode else colors["black"],
            "text.color": colors["light"] if dark_mode else colors["black"],
            "axes.titlecolor": colors["light"] if dark_mode else colors["black"],
        },
    )

    return style
