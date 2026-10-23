import unittest

import numpy as np
import pandas as pd

from gross_margin_stability_ic_12m import (
    calculate_monthly_ic,
    render_report as render_ic,
)
from gross_margin_stability_monotonicity_12m import (
    calculate_monthly_quintile_returns,
    render_report as render_monotonicity,
)
from gross_margin_stability_regression_12m import (
    calculate_monthly_regressions,
    render_report as render_regression,
)
from stability_test_common import (
    FACTOR_COLUMN,
    coefficient_summary,
    newey_west_mean_standard_error,
    prepare_monthly_cross_sections,
    two_sided_t_pvalue,
)


class StabilityFactorTests(unittest.TestCase):
    @staticmethod
    def make_inputs(month_count=28):
        stock_codes = [f"{index + 1:06d}" for index in range(20)]
        months = pd.period_range("2020-01", periods=month_count, freq="M")
        factor = pd.DataFrame(
            {
                "股票代码": stock_codes,
                "报告期": ["2020-03-31"] * len(stock_codes),
                "季度": ["2020Q1"] * len(stock_codes),
                "可用日期": ["2020-04-01"] * len(stock_codes),
                FACTOR_COLUMN: np.linspace(-1.0, 1.0, len(stock_codes)),
            }
        )
        market_rows = []
        adjusted_rows = []
        for stock_index, code in enumerate(stock_codes):
            for month_index, month in enumerate(months):
                date = month.end_time.normalize().strftime("%Y-%m-%d")
                market_rows.append({"股票代码": code, "日期": date, "收盘": 10.0})
                adjusted_rows.append(
                    {
                        "股票代码": code,
                        "日期": date,
                        "收盘": 100.0
                        * (1.005 + stock_index * 0.0005) ** month_index,
                    }
                )
        return factor, pd.DataFrame(market_rows), pd.DataFrame(adjusted_rows)

    def test_forward_returns_use_exact_twelve_month_calendar_shift(self):
        factor, market, adjusted = self.make_inputs()

        sections = prepare_monthly_cross_sections(factor, market, adjusted)

        self.assertEqual(len(sections["截面月份"].unique()), 13)
        first = sections.loc[sections["截面月份"] == "2020-04"]
        self.assertEqual(first["预测月份"].iloc[0], "2021-04")
        self.assertEqual(len(first), 20)

    def test_ic_regression_and_quintile_reports(self):
        factor, market, adjusted = self.make_inputs()

        ic = calculate_monthly_ic(factor, market, adjusted)
        regressions = calculate_monthly_regressions(factor, market, adjusted)
        quintiles = calculate_monthly_quintile_returns(factor, market, adjusted)

        self.assertEqual(len(ic), 13)
        self.assertEqual(len(regressions), 13)
        self.assertEqual(len(quintiles), 13)
        self.assertTrue((regressions["beta"] > 0).all())
        self.assertTrue((quintiles["Q5收益"] > quintiles["Q1收益"]).all())

        date_ranges = (
            ("2020-04-01", "2020-04-01"),
            ("2020-01-01", "2022-04-30"),
            ("2020-01-01", "2022-04-30"),
        )
        ic_text = render_ic(ic, len(factor), *date_ranges)
        regression_text = render_regression(regressions, len(factor), *date_ranges)
        monotonicity_text = render_monotonicity(
            quintiles, len(factor), *date_ranges
        )
        self.assertIn("未来 12 个月累计后复权收益", ic_text)
        self.assertIn("Newey-West HAC", regression_text)
        self.assertIn("滞后 11", monotonicity_text)

    def test_newey_west_and_t_distribution_helpers(self):
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
