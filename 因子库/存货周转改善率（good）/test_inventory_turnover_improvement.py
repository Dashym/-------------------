import unittest

import pandas as pd

from inventory_turnover_improvement import build_turnover_improvement


class InventoryTurnoverImprovementTests(unittest.TestCase):
    def test_ttm_turnover_and_year_over_year_improvement(self):
        quarters = pd.period_range("2018Q1", "2020Q1", freq="Q-DEC")
        costs = [10.0] * len(quarters)
        inventories = [100.0] * len(quarters)
        current_index = quarters.get_loc(pd.Period("2020Q1", freq="Q-DEC"))
        costs[current_index - 3 : current_index + 1] = [20.0, 20.0, 20.0, 20.0]
        data = pd.DataFrame(
            {
                "股票代码": ["000001"] * len(quarters),
                "报告期": [q.end_time.normalize() for q in quarters],
                "首次公告日期": [
                    (q.end_time + pd.Timedelta(days=30)).normalize() for q in quarters
                ],
                "营业成本": costs,
                "存货": inventories,
            }
        )
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 3,
                "日期": pd.to_datetime(["2020-04-30", "2020-05-01", "2020-05-04"]),
            }
        )

        result = build_turnover_improvement(data, trading_days)
        row = result.loc[result["报告期"] == "2020-03-31"].iloc[0]

        self.assertEqual(row["基期报告期"], "2019-03-31")
        self.assertEqual(row["TTM营业成本"], 80.0)
        self.assertEqual(row["平均存货余额"], 100.0)
        self.assertEqual(row["当期存货周转率"], 0.8)
        self.assertEqual(row["基期存货周转率"], 0.4)
        self.assertAlmostEqual(row["存货周转改善率（TTM同比）"], 1.0)
        self.assertEqual(row["可用日期"], "2020-05-01")

    def test_missing_quarter_invalidates_ttm_window(self):
        quarters = list(pd.period_range("2018Q1", "2020Q1", freq="Q-DEC"))
        quarters.remove(pd.Period("2019Q3", freq="Q-DEC"))
        data = pd.DataFrame(
            {
                "股票代码": ["000001"] * len(quarters),
                "报告期": [q.end_time.normalize() for q in quarters],
                "首次公告日期": [
                    (q.end_time + pd.Timedelta(days=30)).normalize() for q in quarters
                ],
                "营业成本": [10.0] * len(quarters),
                "存货": [100.0] * len(quarters),
            }
        )
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "日期": pd.to_datetime(["2020-05-01"]),
            }
        )

        result = build_turnover_improvement(data, trading_days)

        self.assertFalse((result["报告期"] == "2020-03-31").any())

    def test_first_factor_requires_a_complete_base_ttm(self):
        quarters = pd.period_range("2019Q1", "2020Q4", freq="Q-DEC")
        data = pd.DataFrame(
            {
                "股票代码": ["000001"] * len(quarters),
                "报告期": [q.end_time.normalize() for q in quarters],
                "首次公告日期": [
                    (q.end_time + pd.Timedelta(days=30)).normalize() for q in quarters
                ],
                "营业成本": [10.0] * len(quarters),
                "存货": [100.0] * len(quarters),
            }
        )
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 4,
                "日期": pd.to_datetime(
                    ["2020-01-01", "2020-04-01", "2020-08-01", "2021-04-01"]
                ),
            }
        )

        result = build_turnover_improvement(data, trading_days)

        self.assertEqual(result["报告期"].tolist(), ["2020-12-31"])


if __name__ == "__main__":
    unittest.main()
