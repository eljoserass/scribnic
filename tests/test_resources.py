"""Resource aggregation semantics without relying on real process memory peaks."""

import os
import unittest
from unittest.mock import patch

from scribnic.resources import ResourceMonitor


class ResourceAggregationTests(unittest.TestCase):
    """Distinguish concurrent RSS sums from sums of unrelated process maxima."""

    def test_peak_sum_uses_values_from_each_sampling_pass(self) -> None:
        """A falling application RSS must not be added to a later runtime peak."""
        runtime_pid = os.getpid() + 100000
        monitor = ResourceMonitor("test", lambda: (runtime_pid,))
        values = {os.getpid(): 140, runtime_pid: 10}
        with patch.object(
            monitor, "_sample_process", side_effect=lambda pid: values[pid]
        ):
            monitor._sample()
            values.update({os.getpid(): 10, runtime_pid: 100})
            monitor._sample()
        self.assertEqual(monitor.summary()["peak_sampled_process_rss_sum_bytes"], 150)
