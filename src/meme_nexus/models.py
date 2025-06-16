"""Pydantic models for data structures used in meme_nexus."""

from pydantic import BaseModel, Field


class LiquidationHeatmapData(BaseModel):
    """
    Model for liquidation heatmap data structure.

    This model defines the structure for liquidation heatmap data used in
    chart visualization, containing only the essential heatmap information
    without the full API response wrapper.

    Attributes:
        data: Array of [time_idx, price_idx, liq_value] triplets representing
              liquidation data points where:
              - time_idx: index in chartTimeArray
              - price_idx: index in priceArray
              - liq_value: liquidation volume at this point
        chartTimeArray: Array of timestamps in milliseconds
        priceArray: Array of price levels in ascending order
        maxLiqValue: Maximum liquidation value for normalization

    Example:
        ```python
        heatmap_data = LiquidationHeatmapData(
            data=[[0, 5, 1234.56], [1, 6, 2345.67]],
            chartTimeArray=[1647180800000, 1647184400000],
            priceArray=[45000.0, 45100.0, 45200.0],
            maxLiqValue=5000.0
        )
        ```
    """

    data: list[list[float]] = Field(
        description="Array of [time_idx, price_idx, liq_value] triplets"
    )
    chartTimeArray: list[int] = Field(description="Array of timestamps in milliseconds")
    priceArray: list[float] = Field(
        description="Array of price levels in ascending order"
    )
    maxLiqValue: float = Field(
        description="Maximum liquidation value for normalization"
    )

    class Config:
        """Pydantic configuration."""

        str_strip_whitespace = True
        validate_assignment = True

    @classmethod
    def from_api_response(cls, liq_heat_map: dict) -> "LiquidationHeatmapData":
        """
        Create LiquidationHeatmapData from full API response.

        Extracts the relevant heatmap data from the full CoinAnk API response
        structure which typically contains additional metadata.

        Args:
            api_response: Full API response dict with structure:
                         {'success': True, 'data': {'liqHeatMap': {...}}, ...}

        Returns:
            LiquidationHeatmapData instance with extracted data

        Raises:
            KeyError: If required fields are missing from API response
            ValueError: If API response structure is invalid
        """
        try:
            return cls(
                data=liq_heat_map["data"],
                chartTimeArray=liq_heat_map["chartTimeArray"],
                priceArray=liq_heat_map["priceArray"],
                maxLiqValue=liq_heat_map["maxLiqValue"],
            )
        except KeyError as e:
            raise KeyError(f"Missing required field in API response: {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid API response structure: {e}") from e
