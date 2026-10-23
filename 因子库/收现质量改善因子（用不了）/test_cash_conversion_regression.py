import unittest

from cash_conversion_regression import (
    _fama_macbeth,
    calculate_monthly_regressions,
)
from test_cash_conversion_ic import synthetic_dataset


class CashConversionRegressionTests(unittest.TestCase):
    def test_fama_macbeth_beta_is_positive_when_returns_follow_factor(self):
        unadjusted, adjusted, factor = synthetic_dataset(
            [("2024-09-30", "2024-10-01", 1.0), ("2024-10-31", "2024-11-01", 1.5)]
        )

        regressions = calculate_monthly_regressions(factor, unadjusted, adjusted)

        self.assertEqual(len(regressions), 2)
        self.assertTrue((regressions["beta"] > 0).all())
        self.assertTrue((regressions["股票数"] == 20).all())

        summaries = _fama_macbeth(regressions)
        self.assertGreater(summaries["beta"]["mean"], 0)
        self.assertGreater(summaries["beta"]["standard_error"], 0)
        self.assertTrue(0.0 <= summaries["mean_r_squared"] <= 1.0)

    def test_min_cross_section_filters_small_months(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])
        factor = factor.loc[factor["股票代码"] <= "000005"]

        with self.assertRaises(ValueError):
            calculate_monthly_regressions(factor, unadjusted, adjusted)


if __name__ == "__main__":
    unittest.main()
