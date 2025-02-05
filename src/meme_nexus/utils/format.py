def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number with appropriate unit suffix (B, M, K) and precision.

    Args:
        value: The number to format
        precision: Number of decimal places to show

    Returns:
        Formatted string with appropriate unit suffix

    Examples:
        >>> format_number(1234)
        '1.2K'
        >>> format_number(1234567)
        '1.2M'
        >>> format_number(1234567890)
        '1.2B'
        >>> format_number(123)
        '123.0'
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000:
        formatted = f"{sign}{abs_value/1_000_000_000:.{precision}f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{sign}{abs_value/1_000_000:.{precision}f}M"
    elif abs_value >= 1_000:
        formatted = f"{sign}{abs_value/1_000:.{precision}f}K"
    else:
        formatted = f"{sign}{abs_value:.{precision}f}"

    if precision == 0:
        formatted = formatted.replace(".0", "")

    return formatted


def format_timeframe(timeframe: str) -> str:
    """Convert timeframe to a standardized short format.

    Args:
        timeframe: The timeframe strings ('minute', 'hour', 'day')

    Returns:
        Standardized timeframe format ('m', 'h', or 'd')

    Examples:
        >>> format_timeframe('minute')
        'm'
        >>> format_timeframe('h')
        'h'
        >>> format_timeframe('day')
        'd'
    """
    timeframe_map = {
        "minute": "m",
        "m": "m",
        "hour": "h",
        "h": "h",
        "day": "d",
        "d": "d",
    }
    return timeframe_map[timeframe.lower()]
