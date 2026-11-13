"""Baseline searches with shared result fields and counting rules."""

from .core import (
    SearchLimits,
    SearchResult,
    SearchStats,
    SearchStatus,
    astar,
    best_first_search,
    bfs,
    dijkstra,
    manhattan,
    weighted_astar,
)

__all__ = [
    "SearchLimits",
    "SearchResult",
    "SearchStats",
    "SearchStatus",
    "astar",
    "best_first_search",
    "bfs",
    "dijkstra",
    "manhattan",
    "weighted_astar",
]
