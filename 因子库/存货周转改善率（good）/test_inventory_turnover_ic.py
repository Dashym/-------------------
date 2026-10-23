import unittest

import numpy as np
import pandas as pd

from inventory_turnover_ic import calculate_monthly_ic, render_report


class InventoryTurnoverICTests(unittest.TestCase):
    @staticmethod
    def make_inputs(months=("2020-01-31", "2021-01-31")):
        codes = [f"{index + 1:06d}" for index in range(12)]
        factor_values = np.arange(12, dtype=float)
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                "存货周转改善率（TTM同比）": factor_values,
            }
        )
        market_rows = []
        adjusted_rows = []
        for index, code in enumerate(codes):
            for month_index, date in enumerate(months):
                market_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 10.0,
                        "成交量": (index + 1) * 1000,
                        "换手率": 10.0,
                    }
                )
                adjusted_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 100.0 * (1.0 + 0.01 * index) ** month_index,
                    }
                )
        return (
            factor,
            pd.DataFrame(market_rows),
            pd.DataFrame(adjusted_rows),
        )

    def test_uses_asof_month_end_factor_and_12_month_adjusted_return(self):
        factor, market, adjusted = self.make_inputs()

        result = calculate_monthly_ic(factor, market, adjusted)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "截面月份"], "2020-01")
        self.assertEqual(result.loc[0, "预测月份"], "2021-01")
        self.assertEqual(result.loc[0, "股票数"], 12)
        factor_values = np.arange(12, dtype=float)
        log_caps = np.log(np.arange(1, 13, dtype=float) * 10_000_000)
        design = np.column_stack([np.ones(12), log_caps])
        residuals = factor_values - design @ np.linalg.lstsq(
            design, factor_values, rcond=None
        )[0]
        standardized = (residuals - residuals.mean()) / residuals.std(ddof=0)
        returns = np.arange(12, dtype=float) * 0.01
        expected_rank_ic = pd.Series(standardized).rank().corr(
            pd.Series(returns).rank()
        )
        expected_pearson_ic = pd.Series(standardized).corr(pd.Series(returns))
        self.assertAlmostEqual(result.loc[0, "Rank IC"], expected_rank_ic)
        self.assertAlmostEqual(result.loc[0, "Pearson IC"], expected_pearson_ic)

    def test_missing_month_does_not_create_12_month_forward_return(self):
        factor, market, adjusted = self.make_inputs(
            months=("2020-01-31", "2021-02-28")
        )

        result = calculate_monthly_ic(factor, market, adjusted)

        self.assertTrue(result.empty)

    def test_fixed_12_month_horizon_uses_exact_calendar_months(self):
        codes = [f"{index + 1:06d}" for index in range(12)]
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                "存货周转改善率（TTM同比）": np.arange(12, dtype=float),
            }
        )
        months = pd.period_range("2020-01", "2021-01", freq="M")
        market_rows = []
        adjusted_rows = []
        for index, code in enumerate(codes):
            for month_index, month in enumerate(months):
                date = month.end_time.normalize().strftime("%Y-%m-%d")
                market_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 10.0,
                        "成交量": (index + 1) * 1000,
                        "换手率": 10.0,
                    }
                )
                adjusted_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 100.0 * (1.01 + index * 0.001) ** month_index,
                    }
                )
        market = pd.DataFrame(market_rows)
        adjusted = pd.DataFrame(adjusted_rows)

        result = calculate_monthly_ic(factor, market, adjusted)

        self.assertEqual(len(result), len(months) - 12)
        self.assertEqual(result.loc[0, "截面月份"], "2020-01")
        self.assertEqual(result.loc[0, "预测月份"], "2021-01")
        self.assertEqual(result.loc[0, "股票数"], 12)

    def test_report_documents_ic_methodology(self):
        result = pd.DataFrame(
            {
                "截面月份": ["2020-01"],
                "截面日期": ["2020-01-31"],
                "预测月份": ["2021-01"],
                "股票数": [12],
                "Rank IC": [0.5],
                "Pearson IC": [0.4],
            }
        )

        report = render_report(
            result,
            factor_rows=12,
            factor_dates=("2020-01-01", "2020-01-31"),
            market_dates=("2019-01-01", "2020-02-28"),
            adjusted_dates=("2019-01-01", "2021-01-31"),
        )

        self.assertIn("未来 12 个月累计后复权收益", report)
        self.assertIn("2021-01", report)
        self.assertIn("Rank IC", report)


if __name__ == "__main__":
    unittest.main()
