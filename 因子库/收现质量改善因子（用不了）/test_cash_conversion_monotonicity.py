import unittest

from cash_conversion_monotonicity import (
    _monotonicity_label,
    _summarize,
    calculate_monthly_quintile_returns,
)
from test_cash_conversion_ic import synthetic_dataset


class CashConversionMonotonicityTests(unittest.TestCase):
    def test_quintile_returns_increase_with_factor(self):
        unadjusted, adjusted, factor = synthetic_dataset(
            [("2024-09-30", "2024-10-01"), ("2024-10-31", "2024-11-01")]
        )

        results = calculate_monthly_quintile_returns(factor, unadjusted, adjusted)

        self.assertEqual(len(results), 2)
        for _, row in results.iterrows():
            self.assertEqual(row["股票数"], 20)
            for group in range(1, 5):
                self.assertLess(row[f"Q{group}收益"], row[f"Q{group + 1}收益"])
            self.assertGreater(row["Q5-Q1收益"], 0)

        summaries, spread = _summarize(results)
        self.assertGreater(spread["mean"], 0)
        self.assertEqual(spread["win_ratio"], 1.0)
        self.assertEqual(_monotonicity_label(summaries), "严格单调递增（因子越大 → T+1 收益越高）")

    def test_min_cross_section_filters_small_months(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])
        factor = factor.loc[factor["股票代码"] <= "000005"]

        with self.assertRaises(ValueError):
            calculate_monthly_quintile_returns(factor, unadjusted, adjusted)

    def test_quintile_count_must_be_at_least_two(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])

        with self.assertRaises(ValueError):
            calculate_monthly_quintile_returns(
                factor, unadjusted, adjusted, quintile_count=1
            )


if __name__ == "__main__":
    unittest.main()
