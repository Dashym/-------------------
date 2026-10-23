import unittest

import pandas as pd

from rd_expense_ratio import FACTOR_COLUMN, build_rd_expense_ratio


class RDExpenseRatioTests(unittest.TestCase):
    @staticmethod
    def make_inputs():
        periods = pd.period_range("2019Q1", "2020Q1", freq="Q-DEC")
        financial_rows = []
        for index, period in enumerate(periods):
            announcement = (period.end_time + pd.Timedelta(days=30)).normalize()
            financial_rows.append(
                {
                    "股票代码": "000001",
                    "报告期": period.end_time.normalize(),
                    "首次公告日期": announcement,
                    "研发费用": 10.0 + index,
                    "营业总收入": 100.0 + 10.0 * index,
                }
            )
        trading_days = pd.DataFrame(
            {
                "股票代码": ["000001"] * 20,
                "日期": pd.date_range("2019-04-01", periods=20, freq="30D"),
            }
        )
        return pd.DataFrame(financial_rows), trading_days

    def test_ttm_ratio_and_post_announcement_availability(self):
        financials, trading_days = self.make_inputs()

        result = build_rd_expense_ratio(financials, trading_days)

        self.assertEqual(len(result), 2)
        first = result.iloc[0]
        self.assertEqual(first["报告期"], "2019-12-31")
        self.assertAlmostEqual(first["TTM研发费用"], 46.0)
        self.assertAlmostEqual(first["TTM营业总收入"], 460.0)
        self.assertAlmostEqual(first[FACTOR_COLUMN], 0.1)
        self.assertGreater(
            pd.Timestamp(first["可用日期"]),
            pd.Timestamp(first["TTM最新所需财报公告日"]),
        )
        self.assertEqual(result.iloc[1]["报告期"], "2020-03-31")

    def test_rejects_missing_required_fields(self):
        financials, trading_days = self.make_inputs()

        with self.assertRaisesRegex(ValueError, "研发费用"):
            build_rd_expense_ratio(
                financials.drop(columns=["研发费用"]), trading_days
            )

    def test_does_not_calculate_if_a_quarter_is_missing(self):
        financials, trading_days = self.make_inputs()
        financials = financials.loc[
            financials["报告期"].dt.strftime("%Y-%m-%d") != "2019-06-30"
        ]

        result = build_rd_expense_ratio(financials, trading_days)

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
