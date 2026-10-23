import unittest

import pandas as pd

from momentum_ic import calculate_cross_sectional_ic, render_report


class MomentumICTests(unittest.TestCase):
    def test_calculates_ic_against_next_month_returns(self):
        tickers = [f"{code:06d}" for code in range(1, 13)]
        factor_values = list(range(12))
        current_prices = [100.0] * 12
        next_prices = [100.0 + value for value in factor_values]
        factor = pd.DataFrame(
            {
                "股票代码": tickers,
                "截面日期": ["2020-01-31"] * 12,
                "动量因子_cleaned": factor_values,
            }
        )
        quotes = pd.DataFrame(
            [
                {"股票代码": ticker, "日期": date, "收盘": price}
                for ticker, current, following in zip(tickers, current_prices, next_prices)
                for date, price in [("2020-01-31", current), ("2020-02-28", following)]
            ]
        )

        result = calculate_cross_sectional_ic(factor, quotes)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "Rank IC"], 1.0)
        self.assertAlmostEqual(result.loc[0, "Pearson IC"], 1.0)

    def test_missing_intermediate_month_does_not_bridge_return_horizon(self):
        tickers = [f"{code:06d}" for code in range(1, 13)]
        factor = pd.DataFrame(
            {
                "股票代码": tickers,
                "截面日期": ["2020-01-31"] * 12,
                "动量因子_cleaned": list(range(12)),
            }
        )
        quotes = pd.DataFrame(
            [
                {"股票代码": ticker, "日期": date, "收盘": price}
                for ticker in tickers
                for date, price in [("2020-01-31", 100.0), ("2020-03-31", 120.0)]
            ]
        )

        result = calculate_cross_sectional_ic(factor, quotes)

        self.assertTrue(result.empty)

    def test_report_includes_methodology_and_monthly_results(self):
        ic_results = pd.DataFrame(
            {
                "截面月份": ["2020-01"],
                "截面日期": ["2020-01-31"],
                "股票数": [12],
                "Rank IC": [0.5],
                "Pearson IC": [0.4],
            }
        )

        report = render_report(
            ic_results,
            factor_count=12,
            factor_dates=("2020-01-31", "2020-01-31"),
            quote_count=24,
            quote_dates=("2020-01-31", "2020-02-28"),
        )

        self.assertIn("下一自然月收益", report)
        self.assertIn("2020-01", report)
        self.assertIn("ICIR", report)


if __name__ == "__main__":
    unittest.main()
