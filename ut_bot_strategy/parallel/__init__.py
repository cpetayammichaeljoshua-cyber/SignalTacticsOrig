"""
Parallel Processing Module for Trading Signal Bot

Provides enhanced parallel processing capabilities for:
- Batch processing of tasks with rate limiting
- Parallel market scanning of 100+ pairs
- Concurrent data fetching from multiple APIs
"""

from .parallel_processor import (
    ParallelBatchProcessor,
    ParallelMarketScanner,
    ParallelDataFetcher,
    MAX_CONCURRENT_SCANS,
    MAX_CONCURRENT_API_CALLS,
    BATCH_SIZE,
    SCAN_TIMEOUT_SECONDS,
)

__all__ = [
    'ParallelBatchProcessor',
    'ParallelMarketScanner',
    'ParallelDataFetcher',
    'MAX_CONCURRENT_SCANS',
    'MAX_CONCURRENT_API_CALLS',
    'BATCH_SIZE',
    'SCAN_TIMEOUT_SECONDS',
]
