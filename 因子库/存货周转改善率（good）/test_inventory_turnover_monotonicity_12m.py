import unittest

import numpy as np
import pandas as pd

from inventory_turnover_ic import FACTOR_COLUMN
from inventory_turnover_monotonicity_12m import (
    QUINTILE_COUNT,
    calculate_monthly_quintile_returns,
    render_report,
)


class InventoryTurnoverMonotonicity12MTests(unittest.TestCase):
    def test_builds_five_factor_sorted_groups_with_twelve_month_returns(self):
        codes = [f"{index + 1:06d}" for index in range(20)]
        months = pd.period_range("2020-01", "2021-01", freq="M")
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "可用日期": ["2020-01-15"] * len(codes),
                FACTOR_COLUMN: np.arange(20, dtype=float),
            }
        )
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
                        "成交量": 1000.0,
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

        result = calculate_monthly_quintile_returns(
            factor, pd.DataFrame(market_rows), pd.DataFrame(adjusted_rows)
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["截面月份"], "2020-01")
        self.assertEqual(row["预测月份"], "2021-01")
        self.assertEqual(row["股票数"], 20)
        self.assertEqual([row[f"Q{group}股票数"] for group in range(1, 6)], [4] * 5)
        self.assertTrue(
            all(
                row[f"Q{group}收益"] < row[f"Q{group + 1}收益"]
                for group in range(1, QUINTILE_COUNT)
            )
        )
        self.assertAlmostEqual(
            row["Q5-Q1收益"], row["Q5收益"] - row["Q1收益"]
        )

    def test_report_documents_five_groups_and_hac_spread(self):
        months = pd.period_range("2020-01", periods=12, freq="M")
        records = []
        for month in months:
            record = {
                "截面月份": str(month),
                "截面日期": month.end_time.normalize().strftime("%Y-%m-%d"),
                "预测月份": str(month + 12),
                "股票数": 50,
                "Q5-Q1收益": 0.05,
            }
            for group in range(1, 6):
                record[f"Q{group}股票数"] = 10
                record[f"Q{group}收益"] = group * 0.01
            records.append(record)
        results = pd.DataFrame(records)

        report = render_report(
            results,
            factor_rows=144,
            factor_dates=("2020-01-15", "2020-02-15"),
            market_dates=("2019-01-01", "2021-02-28"),
            adjusted_dates=("2019-01-01", "2021-02-28"),
        )

        self.assertIn("五分组单调性检验报告", report)
        self.assertIn("Newey-West HAC", report)
        self.assertIn("滞后 11", report)
        self.assertIn("Q5-Q1收益差：5.0000%", report)
        self.assertIn("分组平均收益随因子值严格递增", report)


if __name__ == "__main__":
    unittest.main()
