class APIError(Exception):
    """Base exception for API related errors."""

    pass


class GeckoTerminalError(APIError):
    """Base exception for GeckoTerminal API errors"""

    pass


class InvalidParametersError(GeckoTerminalError):
    """Raised when request parameters are invalid"""

    pass


class DataRangeError(GeckoTerminalError):
    """Raised when requesting data beyond the allowed time range"""

    pass
