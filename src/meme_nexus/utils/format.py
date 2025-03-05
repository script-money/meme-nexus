def format_number(value: float, precision: int = 2, is_format_k: bool = True) -> str:
    """
    Format a number with appropriate unit suffix (T, B, M, K) and precision.

    Args:
        value: The number to format
        precision: Number of decimal places to show (non-negative integer)
        is_format_k: Whether to use 'K' for thousands

    Returns:
        Formatted string with appropriate unit suffix

    Examples:
        >>> format_number(1234)
        '1.23K'
        >>> format_number(1234567)
        '1.23M'
        >>> format_number(1234567890)
        '1.23B'
        >>> format_number(123)
        '123'
        >>> format_number(1234.5, precision=0)
        '1K'
    """
    if precision < 0:
        raise ValueError("precision must be a non-negative integer")

    # Special case for zero value
    if value == 0:
        return "0"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    # Define units with divisors, including the base case (no unit)
    units = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K" if is_format_k else ""),
        (1, ""),
    ]

    # Find the largest applicable divisor and unit
    for divisor, unit in units:
        if abs_value >= divisor:
            # When is_format_k=False and divisor=1000, don't divide the value
            if not is_format_k and divisor == 1000:
                # For decimal values, format with the minimum necessary precision
                if abs_value % 1 != 0:
                    # Convert to string with specified precision
                    formatted = f"{sign}{abs_value:.{precision}f}"
                    # Remove trailing zeros
                    formatted = formatted.rstrip("0").rstrip(
                        "." if formatted.endswith(".") else ""
                    )
                else:
                    formatted = f"{sign}{int(abs_value)}"
            else:
                # Calculate division result
                result = abs_value / divisor

                # Round according to precision
                if precision == 0:
                    # For precision=0, round directly to integer
                    rounded_result = round(result)
                    formatted = f"{sign}{rounded_result}{unit}"
                else:
                    # Handle cases where decimal is close to integer (1000.5 -> 1K)
                    rounded_to_precision = round(result, precision)
                    if rounded_to_precision == round(rounded_to_precision):
                        formatted = f"{sign}{int(rounded_to_precision)}{unit}"
                    else:
                        # Format with specified precision but remove trailing zeros
                        formatted = f"{sign}{rounded_to_precision:.{precision}f}{unit}"
                        # Remove trailing zeros and decimal point if present
                        if "." in formatted:
                            formatted = formatted.rstrip("0").rstrip(
                                "." if formatted.endswith(".") else ""
                            )

            return formatted

    # This code should not be reached, as at least the last unit (1, "") will match
    # But kept for code completeness
    return f"{sign}{abs_value}"


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
