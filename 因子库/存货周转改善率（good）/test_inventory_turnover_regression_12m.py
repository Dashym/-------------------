import unittest

import numpy as np
import pandas as pd

from inventory_turnover_regression_12m import (
    _coefficient_summary,
    _newey_west_mean_standard_error,
    calculate_monthly_regressions,
    render_report,
    two_sided_t_pvalue,
)


class InventoryTurnoverRegression12MTests(unittest.TestCase):
    def test_regression_targets_exact_twelve_month_cumulative_return(self):
        codes = [f"{index + 1:06d}" for index in range(12)]
        months = pd.period_range("2020-01", "2021-01", freq="M")
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                "存货周转改善率（TTM同比）": np.arange(12, dtype=float),
            }
        )
        market_rows = []
        adjusted_rows = []
        for index, code in enumerate(codes):
            for month in months:
                date = month.end_time.normalize().strftime("%Y-%m-%d")
                market_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 10.0,
                        "成交量": 1000.0,
                        "换手率": 10.0,
                    }
                )
                adjusted_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 100.0 * (1.01 + index * 0.001) ** month.ordinal,
                    }
                )

        result = calculate_monthly_regressions(
            factor, pd.DataFrame(market_rows), pd.DataFrame(adjusted_rows)
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "月份"], "2020-01")
        self.assertEqual(result.loc[0, "预测月份"], "2021-01")
        self.assertEqual(result.loc[0, "股票数"], 12)
        self.assertGreater(result.loc[0, "beta"], 0)

    def test_missing_intermediate_month_does_not_bridge_holding_period(self):
        codes = [f"{index + 1:06d}" for index in range(12)]
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                "存货周转改善率（TTM同比）": np.arange(12, dtype=float),
            }
        )
        market_rows = []
        adjusted_rows = []
        for code in codes:
            for date in ("2020-01-31", "2021-02-28"):
                market_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 10.0,
                        "成交量": 1000.0,
                        "换手率": 10.0,
                    }
                )
                adjusted_rows.append(
                    {"股票代码": code, "日期": date, "收盘": 100.0}
                )

        result = calculate_monthly_regressions(
            factor, pd.DataFrame(market_rows), pd.DataFrame(adjusted_rows)
        )

        self.assertTrue(result.empty)

    def test_student_t_pvalue(self):
        self.assertAlmostEqual(two_sided_t_pvalue(0.0, 10), 1.0)
        self.assertAlmostEqual(
            two_sided_t_pvalue(2.2281388519649385, 10), 0.05, places=6
        )

    def test_newey_west_standard_error_matches_bartlett_hac_formula(self):
        values = np.array([1.0, 2.0, 4.0, 3.0, 7.0, 6.0])
        lag_count = 2
        deviations = values - values.mean()
        long_run_variance = np.dot(deviations, deviations) / len(values)
        for lag in range(1, lag_count + 1):
            weight = 1 - lag / (lag_count + 1)
            autocovariance = np.dot(deviations[lag:], deviations[:-lag]) / len(values)
            long_run_variance += 2 * weight * autocovariance
        expected_se = np.sqrt(
            long_run_variance / len(values) * len(values) / (len(values) - 1)
        )

        self.assertAlmostEqual(
            _newey_west_mean_standard_error(values, max_lags=lag_count),
            expected_se,
        )

    def test_hac_summary_uses_adjusted_standard_error_for_t_and_p(self):
        values = np.array([-0.1, 0.2, 0.4, -0.2, 0.1, 0.3, -0.05, 0.2, 0.5, 0.1, 0.0, 0.3])
        summary = _coefficient_summary(values, max_lags=3)
        expected_t = values.mean() / _newey_west_mean_standard_error(
            values, max_lags=3
        )

        self.assertAlmostEqual(summary["t_stat"], expected_t)
        self.assertAlmostEqual(
            summary["p_value"],
            two_sided_t_pvalue(expected_t, len(values) - 1),
        )

    def test_newey_west_requires_more_observations_than_lags(self):
        with self.assertRaises(ValueError):
            _newey_west_mean_standard_error([1.0, 2.0], max_lags=2)

    def test_report_states_horizon_and_hac_adjustment(self):
        months = pd.period_range("2020-01", periods=12, freq="M")
        regressions = pd.DataFrame(
            {
                "月份": [str(month) for month in months],
                "截面日期": [
                    month.end_time.normalize().strftime("%Y-%m-%d")
                    for month in months
                ],
                "预测月份": [
                    str(month + 12) for month in months
                ],
                "股票数": [12] * len(months),
                "alpha": np.linspace(0.005, 0.02, len(months)),
                "beta": np.linspace(-0.02, 0.01, len(months)),
                "R-squared": np.linspace(0.1, 0.2, len(months)),
            }
        )

        report = render_report(
            regressions,
            factor_rows=144,
            factor_dates=("2020-01-15", "2020-02-15"),
            market_dates=("2019-01-01", "2021-02-28"),
            adjusted_dates=("2019-01-01", "2021-02-28"),
        )

        self.assertIn("未来 12 个月累计后复权收益", report)
        self.assertIn("Newey-West HAC", report)
        self.assertIn("最大滞后期数=11", report)
        self.assertIn("Fama-MacBeth", report)
        self.assertIn("2021-01", report)


if __name__ == "__main__":
    unittest.main()
