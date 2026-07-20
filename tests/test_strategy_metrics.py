import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import strategy


class StrategyMetricTests(unittest.TestCase):
    def test_momentum_change_is_computed_against_previous_day(self):
        s = pd.Series([100, 100, 100, 100, 100, 110, 120], dtype=float)
        current, previous, change = strategy.calc_momentum_change(s, lookback=2)

        self.assertGreater(current, previous)
        self.assertGreater(change, 0)

    def test_metric_value_is_rendered_without_percent_suffix(self):
        self.assertEqual(strategy.format_metric_value(1.23), "1.23")
        self.assertEqual(strategy.format_metric_value(-4.5), "-4.50")
        self.assertEqual(strategy.format_metric_value(0), "0.00")

    def test_flow_score_prefers_stronger_turnover_and_volume(self):
        strong = strategy.calc_flow_signal({'turnover': 2.5, 'amount': 35.0, 'volume_ratio': 12.0})
        weak = strategy.calc_flow_signal({'turnover': 0.2, 'amount': 4.0, 'volume_ratio': -2.0})

        self.assertGreater(strong, weak)


if __name__ == '__main__':
    unittest.main()
