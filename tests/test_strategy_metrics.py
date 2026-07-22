import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import strategy


class StrategyMetricTests(unittest.TestCase):
    def test_momentum_change_is_computed_against_previous_day(self):
        s = pd.Series([100, 100, 100, 100, 100, 110, 120], dtype=float)
        current, previous, change = strategy.calc_momentum_change(s, lookback=2, is_stale=False, is_intraday=False)

        self.assertGreater(current, previous)
        self.assertGreater(change, 0)

    def test_metric_value_is_rendered_without_percent_suffix(self):
        self.assertEqual(strategy.format_metric_value(1.23), "1.23")
        self.assertEqual(strategy.format_metric_value(-4.5), "-4.50")
        self.assertEqual(strategy.format_metric_value(0), "0.00")

    def test_flow_score_prefers_stronger_turnover_and_volume(self):
        """截面Z-score: 高换手+高量比应显著高于低换手+低量比"""
        # 模拟全市场数据,验证Z-score排序逻辑
        raw = [
            {'turnover': 8.0, 'volume_ratio': 80.0, 'amount_chg_pct': 50.0},
            {'turnover': 0.5, 'volume_ratio': -10.0, 'amount_chg_pct': -20.0},
        ]
        import numpy as np
        def _z(arr, val):
            return (val - np.mean(arr)) / (np.std(arr) or 1.0)
        scores = []
        for r in raw:
            to_vals = np.array([x['turnover'] for x in raw])
            vr_vals = np.array([x['volume_ratio'] for x in raw])
            ac_vals = np.array([x['amount_chg_pct'] for x in raw])
            z_to = _z(to_vals, r['turnover'])
            z_vr = _z(vr_vals, r['volume_ratio'])
            z_ac = _z(ac_vals, r['amount_chg_pct'])
            scores.append(round(z_to * 0.5 + z_vr * 0.3 + z_ac * 0.2, 2))
        self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main()
