import unittest

import numpy as np
import pandas as pd

from momentum_regression import (
    render_report,
    run_fama_macbeth_regression,
    two_sided_t_pvalue,
)


class MomentumRegressionTests(unittest.TestCase):
    def test_fama_macbeth_uses_next_month_adjusted_returns(self):
        tickers = [f"{code:06d}" for code in range(1, 13)]
        factor_values = np.linspace(-2, 2, len(tickers))
        factor = pd.DataFrame(
            {
                "股票代码": tickers,
                "截面日期": ["2020-01-31"] * len(tickers),
                "动量因子_cleaned": factor_values,
            }
        )
        quotes = pd.DataFrame(
            [
                {"股票代码": ticker, "日期": date, "收盘": price}
                for ticker, x in zip(tickers, factor_values)
                for date, price in [
                    ("2020-01-31", 100.0),
                    ("2020-02-29", 100.0 * (1 + 0.01 + 0.02 * x)),
                    ("2020-03-31", 100.0 * (1 + 0.01 + 0.02 * x) * (1 + 0.01 + 0.02 * x)),
                ]
            ]
        )
        second_factor = factor.copy()
        second_factor["截面日期"] = "2020-02-29"
        factor = pd.concat([factor, second_factor], ignore_index=True)

        result = run_fama_macbeth_regression(factor, quotes)

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc[0, "beta"], 0.02, places=10)
        self.assertAlmostEqual(result.loc[1, "beta"], 0.02, places=10)

    def test_missing_next_month_price_does_not_bridge_months(self):
        tickers = [f"{code:06d}" for code in range(1, 13)]
        factor = pd.DataFrame(
            {
                "股票代码": tickers,
                "截面日期": ["2020-01-31"] * len(tickers),
                "动量因子_cleaned": np.linspace(-1, 1, len(tickers)),
            }
        )
        quotes = pd.DataFrame(
            [
                {"股票代码": ticker, "日期": date, "收盘": price}
                for ticker in tickers
                for date, price in [("2020-01-31", 100.0), ("2020-03-31", 110.0)]
            ]
        )

        result = run_fama_macbeth_regression(factor, quotes)

        self.assertTrue(result.empty)

    def test_student_t_pvalue_known_values(self):
        self.assertAlmostEqual(two_sided_t_pvalue(0, 10), 1.0)
        self.assertAlmostEqual(two_sided_t_pvalue(2.2281388519649385, 10), 0.05, places=6)

    def test_regression_report_contains_required_results(self):
        regressions = pd.DataFrame(
            {
                "月份": ["2020-01", "2020-02"],
                "截面日期": ["2020-01-31", "2020-02-29"],
                "股票数": [12, 12],
                "alpha": [0.01, 0.01],
                "beta": [0.02, 0.02],
                "R-squared": [1.0, 1.0],
            }
        )

        report = render_report(
            regressions,
            factor_count=24,
            factor_dates=("2020-01-31", "2020-02-29"),
            quote_count=36,
            quote_dates=("2020-01-31", "2020-03-31"),
        )

        self.assertIn("Fama-MacBeth", report)
        self.assertIn("后复权", report)
        self.assertIn("动量因子 beta", report)
        self.assertIn("2020-02", report)


if __name__ == "__main__":
    unittest.main()
