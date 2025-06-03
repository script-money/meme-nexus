from decimal import ROUND_HALF_UP, Decimal


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

    # First round the float to a reasonable precision to eliminate floating point errors
    # Use a higher precision for intermediate calculations
    max_precision = max(precision + 2, 10)
    rounded_value = round(value, max_precision)

    # Use Decimal for precise arithmetic
    decimal_value = Decimal(str(rounded_value))
    abs_value = abs(decimal_value)
    sign = "-" if decimal_value < 0 else ""

    # Define units with divisors, including the base case (no unit)
    units = [
        (Decimal("1000000000000"), "T"),
        (Decimal("1000000000"), "B"),
        (Decimal("1000000"), "M"),
        (Decimal("1000"), "K" if is_format_k else ""),
        (Decimal("1"), ""),
    ]

    # Find the largest applicable divisor and unit
    for divisor, unit in units:
        if abs_value >= divisor:
            # When is_format_k=False and divisor=1000, don't divide the value
            if not is_format_k and divisor == Decimal("1000"):
                # For decimal values, format with the minimum necessary precision
                if abs_value % 1 != 0:
                    # Round to specified precision to avoid floating point errors
                    rounded_value = abs_value.quantize(
                        Decimal("0." + "0" * precision), rounding=ROUND_HALF_UP
                    )
                    formatted = f"{sign}{rounded_value}"
                    # Remove trailing zeros and decimal point if needed
                    if "." in formatted:
                        formatted = formatted.rstrip("0").rstrip(".")
                else:
                    formatted = f"{sign}{int(abs_value)}"
            else:
                # Calculate division result
                result = abs_value / divisor

                # Round according to precision
                if precision == 0:
                    # For precision=0, round directly to integer
                    rounded_result = int(
                        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    )
                    formatted = f"{sign}{rounded_result}{unit}"
                else:
                    # Round to specified precision
                    rounded_result = result.quantize(
                        Decimal("0." + "0" * precision), rounding=ROUND_HALF_UP
                    )

                    # Check if the rounded result is effectively an integer
                    if rounded_result % 1 == 0:
                        formatted = f"{sign}{int(rounded_result)}{unit}"
                    else:
                        formatted = f"{sign}{rounded_result}{unit}"
                        # Remove trailing zeros and decimal point if present
                        if "." in formatted and unit == "":
                            # Only strip trailing zeros for values without units
                            formatted = formatted.rstrip("0").rstrip(".")

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
