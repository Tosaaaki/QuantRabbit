"""MACD divergence + RSI exhaustion reclaim scalp worker package."""

__all__ = ["scalp_macd_rsi_div_worker"]


def __getattr__(name: str):
    if name == "scalp_macd_rsi_div_worker":
        from .worker import scalp_macd_rsi_div_worker

        return scalp_macd_rsi_div_worker
    raise AttributeError(name)
