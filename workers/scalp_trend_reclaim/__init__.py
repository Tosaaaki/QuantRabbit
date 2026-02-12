"""TrendReclaimLong scalp worker package."""

__all__ = ["scalp_trend_reclaim_worker", "scalp_trend_reclaim_exit_worker"]


def __getattr__(name):
    if name == "scalp_trend_reclaim_worker":
        from .worker import scalp_trend_reclaim_worker

        return scalp_trend_reclaim_worker
    if name == "scalp_trend_reclaim_exit_worker":
        from .exit_worker import scalp_trend_reclaim_exit_worker

        return scalp_trend_reclaim_exit_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
