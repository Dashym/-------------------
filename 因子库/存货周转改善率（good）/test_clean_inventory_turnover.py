import unittest

import numpy as np
import pandas as pd

from clean_inventory_turnover import clean_inventory_factor


class InventoryTurnoverCleaningTests(unittest.TestCase):
    @staticmethod
    def make_inputs(factor_values, market_caps, available_date="2021-04-30"):
        codes = [f"{index + 1:06d}" for index in range(len(factor_values))]
        factor = pd.DataFrame(
            {
                "股票代码": codes,
                "报告期": ["2020-12-31"] * len(codes),
                "首次公告日期": ["2021-04-29"] * len(codes),
                "可用日期": [available_date] * len(codes),
                "基期报告期": ["2019-12-31"] * len(codes),
                "TTM营业成本": [100.0] * len(codes),
                "平均存货余额": [20.0] * len(codes),
                "当期存货周转率": [5.0] * len(codes),
                "基期存货周转率": [4.0] * len(codes),
                "存货周转改善率（TTM同比）": factor_values,
            }
        )
        quotes = pd.DataFrame(
            {
                "股票代码": codes,
                "日期": [available_date] * len(codes),
                "收盘": [10.0] * len(codes),
                "成交量": [cap * 0.1 / 10.0 for cap in market_caps],
                "换手率": [10.0] * len(codes),
            }
        )
        return factor, quotes

    def test_market_cap_neutralization_and_standardization(self):
        factor_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        market_caps = [100, 100, 200, 200, 400, 400]
        factor, quotes = self.make_inputs(factor_values, market_caps)

        result = clean_inventory_factor(factor, quotes)

        cleaned = result["存货周转改善率_cleaned"]
        self.assertAlmostEqual(cleaned.mean(), 0.0, places=12)
        self.assertAlmostEqual(cleaned.std(ddof=0), 1.0, places=12)
        self.assertAlmostEqual(
            result["存货周转改善率_neutral"].corr(result["ln流通市值"]),
            0.0,
            places=12,
        )

    def test_deduplicates_same_day_to_latest_report_and_drops_missing(self):
        factor, quotes = self.make_inputs(list(range(6)), [100, 110, 120, 130, 140, 150])
        duplicate = factor.iloc[[0]].copy()
        duplicate["报告期"] = "2021-03-31"
        duplicate["存货周转改善率（TTM同比）"] = 3.5
        factor = pd.concat([factor, duplicate], ignore_index=True)
        factor.loc[1, "存货周转改善率（TTM同比）"] = np.nan

        result = clean_inventory_factor(factor, quotes)

        row = result.loc[result["股票代码"] == "000001"].iloc[0]
        self.assertEqual(row["报告期"], "2021-03-31")
        self.assertEqual(row["存货周转改善率（TTM同比）"], 3.5)
        self.assertFalse((result["股票代码"] == "000002").any())
        self.assertEqual(result["股票代码"].nunique(), len(result))

    def test_mad_clips_extreme_value(self):
        factor, quotes = self.make_inputs(
            [0, 1, 2, 3, 4, 5, 100],
            [100, 110, 120, 130, 140, 150, 160],
        )

        result = clean_inventory_factor(factor, quotes)

        extreme = result.loc[result["股票代码"] == "000007"].iloc[0]
        self.assertLess(extreme["存货周转改善率_winsor"], 100)

    def test_later_disclosure_uses_point_in_time_cross_section(self):
        first_factor, first_quotes = self.make_inputs(
            [0, 1, 2, 3], [100, 200, 300, 400], "2021-04-30"
        )
        later_code, _ = self.make_inputs([10], [500], "2021-05-03")
        later_code["股票代码"] = "000005"
        later_code["报告期"] = "2021-03-31"
        all_quotes = pd.concat(
            [
                first_quotes,
                pd.DataFrame(
                    {
                        "股票代码": ["000001", "000002", "000003", "000004", "000005"],
                        "日期": ["2021-05-03"] * 5,
                        "收盘": [10.0] * 5,
                        "成交量": [10.0, 20.0, 30.0, 40.0, 50.0],
                        "换手率": [10.0] * 5,
                    }
                ),
            ],
            ignore_index=True,
        )
        factor = pd.concat([first_factor, later_code], ignore_index=True)

        result = clean_inventory_factor(factor, all_quotes)

        later_result = result.loc[
            (result["可用日期"] == "2021-05-03")
            & (result["股票代码"] == "000005")
        ].iloc[0]
        values = np.array([0.0, 1.0, 2.0, 3.0, 9.413])
        log_caps = np.log(np.array([100, 200, 300, 400, 500]) * 1000)
        design = np.column_stack([np.ones(5), log_caps])
        residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        expected = (residuals[-1] - residuals.mean()) / residuals.std(ddof=0)
        self.assertAlmostEqual(
            later_result["存货周转改善率_cleaned"], expected, places=12
        )


if __name__ == "__main__":
    unittest.main()
