import unittest

import numpy as np
import pandas as pd

from gross_margin_stability import (
    DRIFT_MIN_PAIRS,
    FACTOR_COLUMN,
    SAMPLE_END,
    WINDOW_QUARTERS,
    build_raw_gross_margin,
    build_stability_factor,
    clean_gross_margin_series,
)


class GrossMarginStabilityTests(unittest.TestCase):
    @staticmethod
    def financial_rows(stock_count=20, end="2024Q4"):
        periods = pd.period_range("2019Q1", end, freq="Q-DEC")
        rows = []
        for stock_index in range(stock_count):
            for index, period in enumerate(periods):
                margin = (
                    0.18
                    + stock_index * 0.006
                    + index * (0.0005 + stock_index * 0.00003)
                    + (index % 4) * stock_index * 0.0001
                )
                revenue = 1000 + stock_index * 10 + index
                announcement = (period.end_time + pd.Timedelta(days=35)).normalize()
                rows.append(
                    {
                        "股票代码": f"{stock_index + 1:06d}",
                        "报告期": period.end_time.normalize(),
                        "首次公告日期": announcement,
                        "营业总收入": revenue,
                        "营业成本": revenue * (1 - margin),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def make_calendar(start="2019-01-01", end="2025-12-31"):
        return pd.DataFrame({"日期": pd.bdate_range(start, end)})

    def test_raw_gross_margin_uses_revenue_and_cost_and_period_limit(self):
        financials = self.financial_rows(stock_count=2, end="2025Q1")
        raw = build_raw_gross_margin(financials)

        self.assertEqual(raw["季度"].min(), "2019Q1")
        self.assertEqual(raw["季度"].max(), str(SAMPLE_END))
        first = raw.iloc[0]
        expected_margin = (
            first["营业总收入"] - first["营业成本"]
        ) / first["营业总收入"]
        self.assertAlmostEqual(first["毛利率_raw"], expected_margin)

    def test_missing_quarter_is_linearly_interpolated_without_extrapolation(self):
        raw = build_raw_gross_margin(self.financial_rows(stock_count=3))
        target = (raw["股票代码"] == "000001") & (raw["季度"] == "2020Q2")
        before = raw.loc[
            (raw["股票代码"] == "000001") & (raw["季度"] == "2020Q1"),
            "毛利率_raw",
        ].iloc[0]
        after = raw.loc[
            (raw["股票代码"] == "000001") & (raw["季度"] == "2020Q3"),
            "毛利率_raw",
        ].iloc[0]
        raw.loc[target, "毛利率_raw"] = np.nan
        raw = raw.loc[
            ~((raw["股票代码"] == "000001") & (raw["季度"] == "2019Q1"))
        ].copy()

        cleaned = clean_gross_margin_series(raw)
        interpolated = cleaned.loc[
            (cleaned["股票代码"] == "000001") & (cleaned["季度"] == "2020Q2")
        ].iloc[0]
        self.assertTrue(interpolated["毛利率_插补标记"])
        self.assertAlmostEqual(interpolated["毛利率_插值"], (before + after) / 2)
        self.assertGreater(
            pd.Timestamp(interpolated["毛利率值可用公告日"]),
            pd.Timestamp(interpolated["首次公告日期"])
            if pd.notna(interpolated["首次公告日期"])
            else pd.Timestamp("1900-01-01"),
        )
        first_quarter = cleaned.loc[
            (cleaned["股票代码"] == "000001") & (cleaned["季度"] == "2019Q1")
        ].iloc[0]
        self.assertTrue(pd.isna(first_quarter["毛利率_插值"]))

    def test_factor_uses_twelve_quarter_window_and_industry_zscores(self):
        raw = build_raw_gross_margin(self.financial_rows())
        cleaned = clean_gross_margin_series(raw)
        factors = build_stability_factor(cleaned, self.make_calendar())

        expected_first = str(pd.Period("2019Q1", freq="Q-DEC") + WINDOW_QUARTERS - 1)
        self.assertEqual(factors["季度"].min(), expected_first)
        self.assertEqual(factors["季度"].max(), "2024Q4")
        first = factors.loc[factors["季度"] == expected_first]
        self.assertEqual(len(first), 20)
        for column in ("z_sigma", "z_cv", "z_drift"):
            self.assertAlmostEqual(first[column].mean(), 0.0, places=10)
            self.assertAlmostEqual(first[column].std(ddof=0), 1.0, places=10)
        expected_score = -(
            0.4 * first["z_sigma"]
            + 0.3 * first["z_cv"]
            + 0.3 * first["z_drift"]
        )
        np.testing.assert_allclose(first[FACTOR_COLUMN], expected_score)
        self.assertTrue(
            (
                pd.to_datetime(first["可用日期"])
                > pd.to_datetime(first["报告期"])
            ).all()
        )
        self.assertGreaterEqual(WINDOW_QUARTERS - 4, DRIFT_MIN_PAIRS)

    def test_future_quarters_do_not_change_prior_anchor_factor(self):
        financials = self.financial_rows(stock_count=20)
        raw = build_raw_gross_margin(financials)
        cleaned = clean_gross_margin_series(raw)
        calendar = self.make_calendar()
        baseline = build_stability_factor(cleaned, calendar)

        changed_financials = financials.copy()
        future = changed_financials["报告期"] > pd.Timestamp("2021-12-31")
        changed_financials.loc[future, "营业成本"] *= 0.5
        changed = build_stability_factor(
            clean_gross_margin_series(build_raw_gross_margin(changed_financials)),
            calendar,
        )
        baseline_values = baseline.loc[
            baseline["季度"] == "2021Q4", ["股票代码", FACTOR_COLUMN]
        ].sort_values("股票代码")
        changed_values = changed.loc[
            changed["季度"] == "2021Q4", ["股票代码", FACTOR_COLUMN]
        ].sort_values("股票代码")
        pd.testing.assert_frame_equal(
            baseline_values.reset_index(drop=True),
            changed_values.reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
