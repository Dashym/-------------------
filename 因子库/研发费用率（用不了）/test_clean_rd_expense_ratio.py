import unittest

import numpy as np
import pandas as pd

from clean_rd_expense_ratio import (
    FACTOR_COLUMN,
    STANDARDIZED_COLUMN,
    WINSORIZED_COLUMN,
    clean_rd_factor,
)


class CleanRDExpenseRatioTests(unittest.TestCase):
    @staticmethod
    def make_quotes(codes, dates):
        return pd.DataFrame(
            [
                {"股票代码": code, "日期": date}
                for code in codes
                for date in dates
            ]
        )

    @staticmethod
    def make_factor(codes, values, date="2020-04-01"):
        return pd.DataFrame(
            {
                "股票代码": codes,
                "报告期": ["2019-12-31"] * len(codes),
                "TTM最新所需财报公告日": ["2020-03-31"] * len(codes),
                "可用日期": [date] * len(codes),
                "TTM研发费用": [10.0] * len(codes),
                "TTM营业总收入": [100.0] * len(codes),
                FACTOR_COLUMN: values,
            }
        )

    def test_missing_values_are_excluded_and_valid_values_standardized(self):
        codes = [f"{index:06d}" for index in range(1, 5)]
        factor = self.make_factor(codes, [1.0, 2.0, np.nan, 4.0])
        quotes = self.make_quotes(codes, ["2020-04-01"])

        cleaned = clean_rd_factor(factor, quotes)

        self.assertEqual(len(cleaned), 3)
        self.assertNotIn("000003", cleaned["股票代码"].tolist())
        self.assertAlmostEqual(cleaned[STANDARDIZED_COLUMN].mean(), 0.0)
        self.assertAlmostEqual(cleaned[STANDARDIZED_COLUMN].std(ddof=0), 1.0)
        self.assertEqual(cleaned.attrs["cleaning_stats"]["missing_factor_rows"], 1)

    def test_mad_winsorization_precedes_cross_section_zscore(self):
        codes = [f"{index:06d}" for index in range(1, 7)]
        factor = self.make_factor(codes, [1, 2, 3, 4, 5, 100])
        quotes = self.make_quotes(codes, ["2020-04-01"])

        cleaned = clean_rd_factor(factor, quotes)

        expected_upper = 3.5 + 5 * 1.4826 * 1.5
        outlier = cleaned.loc[cleaned["股票代码"] == "000006"].iloc[0]
        self.assertAlmostEqual(outlier[WINSORIZED_COLUMN], expected_upper)
        self.assertAlmostEqual(cleaned[STANDARDIZED_COLUMN].mean(), 0.0)
        self.assertAlmostEqual(cleaned[STANDARDIZED_COLUMN].std(ddof=0), 1.0)

    def test_new_quarter_snapshot_carries_forward_known_factor_values(self):
        codes = [f"{index:06d}" for index in range(1, 6)]
        first = self.make_factor(codes[:4], [1.0, 2.0, 3.0, 4.0])
        second = self.make_factor(
            [codes[4]], [5.0], date="2020-07-01"
        ).assign(**{"报告期": "2020-03-31"})
        factor = pd.concat([first, second], ignore_index=True)
        quotes = self.make_quotes(codes, ["2020-04-01", "2020-07-01"])

        cleaned = clean_rd_factor(factor, quotes)

        self.assertEqual(len(cleaned), 5)
        latest = cleaned.loc[cleaned["可用日期"] == "2020-07-01"]
        self.assertEqual(latest["股票代码"].tolist(), ["000005"])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_new_zscore = (5.0 - values.mean()) / values.std(ddof=0)
        self.assertAlmostEqual(
            latest.iloc[0][STANDARDIZED_COLUMN], expected_new_zscore
        )

    def test_requires_a_valid_market_date_for_each_factor_update(self):
        codes = ["000001", "000002"]
        factor = self.make_factor(codes, [1.0, 2.0])
        quotes = self.make_quotes(codes, ["2020-04-02"])

        with self.assertRaisesRegex(ValueError, "不是行情交易日"):
            clean_rd_factor(factor, quotes)


if __name__ == "__main__":
    unittest.main()
