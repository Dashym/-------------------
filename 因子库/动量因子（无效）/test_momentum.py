import unittest

import pandas as pd

from momentum import calculate_momentum, clean_factor


class MomentumCalculationTests(unittest.TestCase):
    def test_first_complete_cross_section_and_12_1_return(self):
        months = pd.period_range("2019-01", "2020-01", freq="M")
        prices = list(range(100, 100 + len(months)))
        quotes = pd.DataFrame(
            {
                "股票代码": ["000001"] * len(months),
                "日期": [month.end_time.normalize() for month in months],
                "收盘": prices,
            }
        )

        result = calculate_momentum(quotes)

        self.assertEqual(result["截面日期"].min(), "2020-01-31")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "动量因子"], 111 / 100 - 1)

    def test_missing_prior_month_does_not_shift_the_lookback(self):
        months = list(pd.period_range("2019-01", "2020-01", freq="M"))
        quotes = pd.DataFrame(
            {
                "股票代码": ["000001"] * len(months),
                "日期": [month.end_time.normalize() for month in months],
                "收盘": list(range(100, 100 + len(months))),
            }
        )
        quotes = quotes.loc[quotes["日期"].dt.to_period("M") != pd.Period("2019-12")]

        result = calculate_momentum(quotes)

        self.assertFalse(result["截面日期"].str.startswith("2020-01").any())

    def test_cleaning_drops_missing_values_winsorizes_and_standardizes(self):
        values = list(range(10)) + [100, None]
        factor = pd.DataFrame(
            {
                "股票代码": [f"{code:06d}" for code in range(1, len(values) + 1)],
                "截面日期": ["2020-01-31"] * len(values),
                "动量因子": values,
            }
        )

        result = clean_factor(factor)

        self.assertEqual(len(result), 11)
        self.assertLess(result.loc[result["股票代码"] == "000011", "动量因子_winsor"].iloc[0], 100)
        self.assertAlmostEqual(result["动量因子_cleaned"].mean(), 0, places=12)
        self.assertAlmostEqual(result["动量因子_cleaned"].std(ddof=0), 1, places=12)

    def test_constant_cross_section_standardizes_to_zero(self):
        factor = pd.DataFrame(
            {
                "股票代码": ["000001", "000002"],
                "截面日期": ["2020-01-31", "2020-01-31"],
                "动量因子": [0.1, 0.1],
            }
        )

        result = clean_factor(factor)

        self.assertEqual(result["动量因子_cleaned"].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
