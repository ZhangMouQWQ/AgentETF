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


if __name__ == '__main__':
    unittest.main()
