import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sales_receipt_ratio_improvement import (
    FACTOR_COLUMN,
    build_month_end_panel,
    build_ttm,
    describe_panel,
    load_crawl_cache,
    next_trading_day,
    normalize_codes,
    quarter_to_period_end,
    to_cross_section_file,
    to_detail_file,
    to_single_quarter,
)


# 2022Q1 ~ 2024Q4 共 12 个季度：TTM 自 2022Q4 起有效，同比再 shift(4)，故因子自 2023Q4 起有效
QUARTER_ENDS = [
    pd.Timestamp("2022-03-31"),
    pd.Timestamp("2022-06-30"),
    pd.Timestamp("2022-09-30"),
    pd.Timestamp("2022-12-31"),
    pd.Timestamp("2023-03-31"),
    pd.Timestamp("2023-06-30"),
    pd.Timestamp("2023-09-30"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2024-03-31"),
    pd.Timestamp("2024-06-30"),
    pd.Timestamp("2024-09-30"),
    pd.Timestamp("2024-12-31"),
]

REVENUE_SINGLE = 100.0
RECEIVED_SINGLE = [60.0] * 4 + [80.0] * 4 + [100.0] * 4


def synthetic_cache(notice_delay_days=30):
    """合成爬虫缓存：营业收入单季恒为 100；销售收现单季 60/80/100 → 收现比 0.6/0.8/1.0。

    于是「当期收现比 − 去年同期收现比」恒为 0.2，便于精确断言。
    """
    rows = []
    for index, period_end in enumerate(QUARTER_ENDS):
        position_in_year = index % 4 + 1
        received = RECEIVED_SINGLE[index]
        rows.append(
            {
                "股票代码": "000001",
                "报告期": period_end.strftime("%Y-%m-%d"),
                "营业总收入_累计": REVENUE_SINGLE * position_in_year,
                "销售收现_累计": received * position_in_year,
                "首次公告日期": (
                    period_end + pd.Timedelta(days=notice_delay_days)
                ).strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def synthetic_pool(listing_date="2015-01-05"):
    """合成股票池（申万二级成分股）。"""
    return pd.DataFrame({"股票代码": ["000001"], "上市日": [pd.Timestamp(listing_date)]})


class SalesReceiptRatioImprovementTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_path = Path(self.temp_dir.name) / "cache.csv"
        synthetic_cache().to_csv(self.cache_path, index=False, encoding="utf-8-sig")
        self.crawl = load_crawl_cache(self.cache_path)

    def _row(self, detail, period):
        matched = detail.loc[detail["报告期"] == pd.Timestamp(period)]
        self.assertEqual(len(matched), 1)
        return matched.iloc[0]

    def test_normalize_codes(self):
        raw = pd.Series(["000001.SZ", "600000", "1.0", " 300750.sz "])

        self.assertEqual(
            normalize_codes(raw).tolist(), ["000001", "600000", "000001", "300750"]
        )

    def test_load_crawl_cache_deduplicates_and_numbers_quarters(self):
        frame = synthetic_cache()
        duplicate = frame.iloc[[0]].copy()
        duplicate["销售收现_累计"] = 999.0
        combined = pd.concat([frame, duplicate], ignore_index=True)
        combined.to_csv(self.cache_path, index=False, encoding="utf-8-sig")

        loaded = load_crawl_cache(self.cache_path)

        self.assertEqual(len(loaded), len(frame))
        first = loaded.loc[loaded["报告期"] == pd.Timestamp("2022-03-31")].iloc[0]
        self.assertAlmostEqual(first["销售收现_累计"], 999.0)
        self.assertEqual(int(first["季度序号"]), 2022 * 4 + 1)
        self.assertTrue(loaded["季度序号"].is_monotonic_increasing)

    def test_to_single_quarter_restores_year_first_and_differences(self):
        index = [2023 * 4 + quarter for quarter in (1, 2, 3, 4)]
        frame = pd.DataFrame({"X": [10.0, 25.0, 45.0, 70.0]}, index=index)

        single = to_single_quarter(frame)

        self.assertEqual(single["X"].tolist(), [10.0, 15.0, 20.0, 25.0])

    def test_to_single_quarter_keeps_nan_when_previous_quarter_missing(self):
        # to_single_quarter 的输入是连续季度网格（缺季度已 reindex 为 NaN），与 build_ttm 一致
        index = [2023 * 4 + quarter for quarter in (1, 2, 3)]
        frame = pd.DataFrame({"X": [10.0, np.nan, 40.0]}, index=index)

        single = to_single_quarter(frame)

        self.assertAlmostEqual(single.loc[2023 * 4 + 1, "X"], 10.0)
        self.assertTrue(np.isnan(single.loc[2023 * 4 + 3, "X"]))

    def test_build_ttm_computes_ratio_and_year_on_year_improvement(self):
        detail = build_ttm(self.crawl)

        row = self._row(detail, "2024-12-31")
        self.assertAlmostEqual(row["营业收入TTM"], 400.0)
        self.assertAlmostEqual(row["销售收现TTM"], 400.0)
        self.assertAlmostEqual(row["销售收现比"], 1.0)
        self.assertAlmostEqual(row["去年同期销售收现比"], 0.8)
        self.assertAlmostEqual(row[FACTOR_COLUMN], 0.2, places=10)

    def test_build_ttm_keeps_constant_improvement_across_valid_quarters(self):
        detail = build_ttm(self.crawl)

        valid = detail.dropna(subset=[FACTOR_COLUMN])
        self.assertEqual(len(valid), 5)
        self.assertEqual(
            [str(period.date()) for period in valid["报告期"]],
            ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"],
        )
        self.assertTrue(np.allclose(valid[FACTOR_COLUMN], 0.2, atol=1e-12))

    def test_build_ttm_available_date_is_latest_notice_in_windows(self):
        detail = build_ttm(self.crawl)

        # 2024Q2：当期 TTM 窗口 2023Q3~2024Q2 的最晚公告日 = 2024-07-30
        self.assertEqual(
            self._row(detail, "2024-06-30")["可用日期"], pd.Timestamp("2024-07-30")
        )
        # 2023Q4：当期窗口 2023Q1~Q4 的最晚公告日 = 2024-01-30（跨年披露）
        self.assertEqual(
            self._row(detail, "2023-12-31")["可用日期"], pd.Timestamp("2024-01-30")
        )

    def test_build_ttm_postpones_availability_when_prior_year_notice_is_late(self):
        crawl = self.crawl.copy()
        late = crawl["报告期"] == pd.Timestamp("2023-09-30")
        crawl.loc[late, "首次公告日期"] = pd.Timestamp("2025-06-30")

        detail = build_ttm(crawl)

        # 2024Q2 的去年同期窗口含 2023Q3，其公告被推迟 → 可用日期必须顺延，不得提前使用
        self.assertEqual(
            self._row(detail, "2024-06-30")["可用日期"], pd.Timestamp("2025-06-30")
        )

    def test_next_trading_day_moves_strictly_forward_and_keeps_nat(self):
        days = np.array(
            ["2024-04-29", "2024-04-30", "2024-05-06"], dtype="datetime64[ns]"
        )
        notices = np.array(
            ["2024-04-30", "2024-05-04", "NaT", "2024-05-06"], dtype="datetime64[ns]"
        )

        shifted = next_trading_day(notices, days)

        self.assertEqual(shifted[0], np.datetime64("2024-05-06"))
        self.assertEqual(shifted[1], np.datetime64("2024-05-06"))
        self.assertTrue(np.isnat(shifted[2]))
        self.assertTrue(np.isnat(shifted[3]))

    def test_build_ttm_shifts_available_date_to_next_trading_day(self):
        trading_days = pd.date_range("2022-01-03", "2025-12-31", freq="B").values.astype(
            "datetime64[ns]"
        )

        detail = build_ttm(self.crawl, trading_days)

        # 公告日 2024-07-30 → 其后第一个交易日 2024-07-31
        self.assertEqual(
            self._row(detail, "2024-06-30")["可用日期"], pd.Timestamp("2024-07-31")
        )
        # 公告日 2024-01-30 → 其后第一个交易日 2024-01-31
        self.assertEqual(
            self._row(detail, "2023-12-31")["可用日期"], pd.Timestamp("2024-01-31")
        )

    def test_month_end_panel_only_uses_disclosed_reports(self):
        detail = build_ttm(self.crawl)
        dates = [
            pd.Timestamp("2024-04-30"),
            pd.Timestamp("2024-05-31"),
            pd.Timestamp("2024-08-30"),
        ]

        panel = build_month_end_panel(detail, synthetic_pool(), dates)

        # 2024-04-30 截面：2023Q4（公告 2024-01-30）与 2024Q1（公告 2024-04-30）均已公告
        # → 取最新的 2024Q1；公告日恰好等于截面日同样视为可用（仓库既有口径：公告日 ≤ 截面日）
        self.assertEqual(
            [str(period.date()) for period in panel["报告期"]],
            ["2024-03-31", "2024-03-31", "2024-06-30"],
        )
        self.assertEqual(panel["首次公告日期"].iloc[0], pd.Timestamp("2024-04-30"))
        self.assertEqual(panel["可用日期"].tolist(), dates)
        self.assertEqual(panel["截面月份"].tolist(), ["2024-04", "2024-05", "2024-08"])
        self.assertTrue(np.allclose(panel[FACTOR_COLUMN], 0.2, atol=1e-12))

    def test_month_end_panel_excludes_stocks_listed_after_cross_section(self):
        detail = build_ttm(self.crawl)
        dates = [pd.Timestamp("2024-04-30"), pd.Timestamp("2024-05-31")]

        panel = build_month_end_panel(detail, synthetic_pool("2024-05-10"), dates)

        self.assertEqual(panel["可用日期"].tolist(), [pd.Timestamp("2024-05-31")])

    def test_month_end_panel_raises_when_nothing_is_disclosed(self):
        detail = build_ttm(self.crawl)

        with self.assertRaises(ValueError):
            build_month_end_panel(detail, synthetic_pool(), [pd.Timestamp("2019-01-31")])

    def test_quarter_to_period_end(self):
        self.assertEqual(quarter_to_period_end(2024 * 4 + 1), pd.Timestamp("2024-03-31"))
        self.assertEqual(quarter_to_period_end(2024 * 4 + 2), pd.Timestamp("2024-06-30"))
        self.assertEqual(quarter_to_period_end(2023 * 4 + 4), pd.Timestamp("2023-12-31"))

    def test_output_tables_expose_expected_columns(self):
        detail = build_ttm(self.crawl)
        panel = build_month_end_panel(detail, synthetic_pool(), [pd.Timestamp("2024-05-31")])

        section = to_cross_section_file(panel)
        self.assertEqual(section["可用日期"].iloc[0], "2024-05-31")
        self.assertEqual(section["报告期"].iloc[0], "2024-03-31")
        self.assertEqual(section["首次公告日期"].iloc[0], "2024-04-30")
        self.assertEqual(
            section.columns.tolist(),
            [
                "股票代码",
                "可用日期",
                "报告期",
                "首次公告日期",
                "营业收入TTM",
                "销售收现TTM",
                "销售收现比",
                "去年同期销售收现比",
                FACTOR_COLUMN,
                "上市日",
                "截面月份",
            ],
        )

        audit = to_detail_file(detail)
        self.assertEqual(audit.columns[-1], FACTOR_COLUMN)
        self.assertEqual(audit["报告期"].iloc[0], "2022-03-31")
        self.assertTrue(audit["报告期"].is_monotonic_increasing)

    def test_describe_panel_reports_definition_and_outputs(self):
        detail = build_ttm(self.crawl)
        dates = [pd.Timestamp("2024-05-31")]
        panel = build_month_end_panel(detail, synthetic_pool(), dates)

        text = describe_panel(panel, detail, synthetic_pool(), dates)

        self.assertIn("销售收现比改善因子构建完成", text)
        self.assertIn("销售收现TTM / 营业收入TTM", text)
        self.assertIn("截面日数量：1 期", text)
        self.assertIn("销售收现比改善因子_raw.csv", text)
        self.assertIn("销售收现比改善因子_明细.csv", text)


if __name__ == "__main__":
    unittest.main()
