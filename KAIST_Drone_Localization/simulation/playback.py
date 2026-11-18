"""Shared latest-frame policy for timestamped real-time replay."""

from bisect import bisect_right


def latest_due_index(timestamps, elapsed_s, last_index):
    """Return the newest due index, or None while waiting for the next capture."""
    index = bisect_right(timestamps, elapsed_s) - 1
    return index if index > last_index else None
