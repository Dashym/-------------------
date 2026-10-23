import unittest

import numpy as np
import pandas as pd

from rd_expense_ratio_ic_12m import calculate_monthly_ic, render_report as render_ic
from rd_expense_ratio_monotonicity_12m import (
    calculate_monthly_quintile_returns,
    render_report as render_monotonicity,
)
from rd_expense_ratio_regression_12m import (
    calculate_monthly_regressions,
    render_report as render_regression,
)
from rd_factor_test_common import (
    FACTOR_COLUMN,
    coefficient_summary,
    newey_west_mean_standard_error,
    prepare_monthly_cross_sections,
    two_sided_t_pvalue,
)


class RDExpenseRatioTestTests(unittest.TestCase):
    @staticmethod
    def make_inputs(month_count=25):
        codes = [f"{index + 1:06d}" for index in range(20)]
        months = pd.period_range("2020-01", periods=month_count, freq="M")
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                FACTOR_COLUMN: np.linspace(0.01, 0.10, len(codes)),
            }
        )
        market_rows = []
        adjusted_rows = []
        for stock_index, code in enumerate(codes):
            for month_index, month in enumerate(months):
                date = month.end_time.normalize().strftime("%Y-%m-%d")
                market_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 10.0,
                    }
                )
                adjusted_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 100.0
                        * (1.005 + stock_index * 0.0005) ** month_index,
                    }
                )
        return factor, pd.DataFrame(market_rows), pd.DataFrame(adjusted_rows)

    def test_forward_return_uses_exactly_twelve_calendar_months(self):
        factor, market, adjusted = self.make_inputs()

        sections = prepare_monthly_cross_sections(factor, market, adjusted)

        self.assertEqual(len(sections["截面月份"].unique()), 13)
        first_month = sections.loc[sections["截面月份"] == "2020-01"]
        self.assertEqual(first_month["预测月份"].iloc[0], "2021-01")
        self.assertEqual(len(first_month), 20)
        self.assertGreater(
            first_month["未来12个月收益"].corr(first_month["因子值_zscore"]), 0
        )

    def test_all_three_monthly_test_outputs_are_calculated(self):
        factor, market, adjusted = self.make_inputs()

        ic_results = calculate_monthly_ic(factor, market, adjusted)
        regression_results = calculate_monthly_regressions(factor, market, adjusted)
        quintile_results = calculate_monthly_quintile_returns(
            factor, market, adjusted
        )

        self.assertEqual(len(ic_results), 13)
        self.assertEqual(len(regression_results), 13)
        self.assertEqual(len(quintile_results), 13)
        self.assertTrue((regression_results["beta"] > 0).all())
        self.assertTrue((quintile_results["Q5收益"] > quintile_results["Q1收益"]).all())
        self.assertTrue((quintile_results["Q1股票数"] == 4).all())

        date_ranges = (
            ("2020-01-15", "2020-02-15"),
            ("2020-01-01", "2022-01-31"),
            ("2020-01-01", "2022-01-31"),
        )
        ic_report = render_ic(ic_results, len(factor), *date_ranges)
        regression_report = render_regression(
            regression_results, len(factor), *date_ranges
        )
        monotonicity_report = render_monotonicity(
            quintile_results, len(factor), *date_ranges
        )
        self.assertIn("未来 12 个月累计后复权收益", ic_report)
        self.assertIn("Newey-West HAC", regression_report)
        self.assertIn("滞后 11", monotonicity_report)

    def test_hac_summary_and_student_t_probability(self):
        values = np.array(
            [-0.1, 0.2, 0.4, -0.2, 0.1, 0.3, -0.05, 0.2, 0.5, 0.1, 0.0, 0.3]
        )
        summary = coefficient_summary(values, max_lags=3)
        self.assertAlmostEqual(
            summary["standard_error"],
            newey_west_mean_standard_error(values, max_lags=3),
        )
        self.assertAlmostEqual(
            two_sided_t_pvalue(2.2281388519649385, 10), 0.05, places=6
        )


if __name__ == "__main__":
    unittest.main()
