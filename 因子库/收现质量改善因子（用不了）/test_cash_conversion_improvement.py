import unittest

import pandas as pd

from cash_conversion_improvement import build_cash_conversion_improvement


def _financial_frame(quarters, revenues, cashflows, notice_offset_days=30):
    return pd.DataFrame(
        {
            "股票代码": ["000001"] * len(quarters),
            "报告期": [q.end_time.normalize() for q in quarters],
            "首次公告日期": [
                (q.end_time + pd.Timedelta(days=notice_offset_days)).normalize()
                for q in quarters
            ],
            "营业总收入": revenues,
            "经营活动产生的现金流量净额": cashflows,
        }
    )


class CashConversionImprovementTests(unittest.TestCase):
    def test_ttm_cash_conversion_and_year_over_year_improvement(self):
        quarters = pd.period_range("2019Q1", "2021Q1", freq="Q-DEC")
        revenues = [10.0] * len(quarters)
        cashflows = [5.0] * len(quarters)
        current_index = quarters.get_loc(pd.Period("2021Q1", freq="Q-DEC"))
        cashflows[current_index - 3 : current_index + 1] = [10.0, 10.0, 10.0, 10.0]
        data = _financial_frame(quarters, revenues, cashflows)
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 3,
                "日期": pd.to_datetime(["2021-04-30", "2021-05-03", "2021-05-04"]),
            }
        )

        result = build_cash_conversion_improvement(data, trading_days)
        row = result.loc[result["报告期"] == "2021-03-31"].iloc[0]

        self.assertEqual(row["基期报告期"], "2020-03-31")
        self.assertEqual(row["TTM营业总收入"], 40.0)
        self.assertEqual(row["TTM经营活动现金流量净额"], 40.0)
        self.assertAlmostEqual(row["当期收现质量"], 1.0)
        self.assertAlmostEqual(row["基期收现质量"], 0.5)
        self.assertAlmostEqual(row["收现质量改善因子"], 0.5)
        self.assertEqual(row["最晚披露日期"], "2021-04-30")
        self.assertEqual(row["可用日期"], "2021-05-03")

    def test_missing_quarter_invalidates_ttm_window(self):
        quarters = list(pd.period_range("2019Q1", "2021Q1", freq="Q-DEC"))
        quarters.remove(pd.Period("2019Q3", freq="Q-DEC"))
        data = _financial_frame(quarters, [10.0] * len(quarters), [5.0] * len(quarters))
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "日期": pd.to_datetime(["2021-05-03"]),
            }
        )

        result = build_cash_conversion_improvement(data, trading_days)

        # 基期 TTM（2019Q2~2020Q1）缺 2019Q3 → 无法计算同比改善
        self.assertFalse((result["报告期"] == "2021-03-31").any())

    def test_first_factor_requires_a_complete_base_ttm(self):
        quarters = pd.period_range("2019Q1", "2020Q4", freq="Q-DEC")
        data = _financial_frame(quarters, [10.0] * len(quarters), [5.0] * len(quarters))
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 4,
                "日期": pd.to_datetime(
                    ["2020-01-01", "2020-04-01", "2020-08-01", "2021-04-01"]
                ),
            }
        )

        result = build_cash_conversion_improvement(data, trading_days)

        self.assertEqual(result["报告期"].tolist(), ["2020-12-31"])

    def test_available_date_is_capped_by_no_lookahead_window(self):
        quarters = pd.period_range("2020Q1", "2024Q4", freq="Q-DEC")
        data = _financial_frame(quarters, [10.0] * len(quarters), [5.0] * len(quarters))
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 4,
                "日期": pd.to_datetime(
                    ["2024-10-31", "2024-11-01", "2024-11-04", "2025-02-03"]
                ),
            }
        )

        result = build_cash_conversion_improvement(data, trading_days)

        self.assertFalse(result.empty)
        self.assertTrue((result["可用日期"] <= "2024-11-01").all())
        self.assertEqual(result["可用日期"].max(), "2024-10-31")
        self.assertEqual(result["报告期"].max(), "2024-09-30")


if __name__ == "__main__":
    unittest.main()
