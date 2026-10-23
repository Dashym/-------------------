import unittest
from unittest import mock

import numpy as np
import pandas as pd

import cash_conversion_ic
from cash_conversion_ic import (
    FACTOR_COLUMN,
    _month_cross_section,
    _monthly_market_data,
    _next_day_returns,
    _pearson,
    _prepare_signals,
    _spearman,
    calculate_monthly_ic,
)


def synthetic_dataset(month_pairs):
    """构造 20 只股票的合成数据：因子值 = 序号，T+1 收益随因子单调递增。

    month_pairs : [(截面日, 收益实现日[, 收益幅度]), ...]，
                  截面日必须为当月最后一个交易日；幅度默认 1.0。
    """
    codes = [f"{index:06d}" for index in range(1, 21)]
    unadjusted_rows = []
    adjusted_rows = []
    factor_rows = []
    for pair in month_pairs:
        day, next_day = pair[0], pair[1]
        amplitude = pair[2] if len(pair) > 2 else 1.0
        for index, code in enumerate(codes, start=1):
            unadjusted_rows.append(
                {
                    "股票代码": code,
                    "日期": day,
                    "收盘": 10.0 if index <= 10 else 30.0,
                    "成交量": 1_000_000.0,
                    "换手率": 1.0,
                }
            )
            adjusted_rows.append({"股票代码": code, "日期": day, "收盘": 10.0})
            adjusted_rows.append(
                {
                    "股票代码": code,
                    "日期": next_day,
                    "收盘": 10.0 * (1.0 + amplitude * (-0.009 + 0.001 * index)),
                }
            )
            factor_rows.append(
                {
                    "股票代码": code,
                    "可用日期": day,
                    FACTOR_COLUMN: float(index),
                }
            )
    return (
        pd.DataFrame(unadjusted_rows),
        pd.DataFrame(adjusted_rows),
        pd.DataFrame(factor_rows),
    )


class CashConversionIcTests(unittest.TestCase):
    def test_next_day_returns_uses_following_trading_day(self):
        quotes = pd.DataFrame(
            {
                "股票代码": ["000001"] * 3,
                "日期": pd.to_datetime(["2024-10-30", "2024-11-01", "2024-11-04"]),
                "收盘": [10.0, 10.5, 9.45],
            }
        )

        result = _next_day_returns(quotes)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["收益实现日"].iloc[0], pd.Timestamp("2024-11-01"))
        self.assertAlmostEqual(result["次日收益"].iloc[0], 0.05)
        self.assertAlmostEqual(result["次日收益"].iloc[1], -0.1)

    def test_factor_must_be_available_before_month_end(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])
        factor["可用日期"] = pd.Timestamp("2024-11-01")

        with self.assertRaises(ValueError):
            calculate_monthly_ic(factor, unadjusted, adjusted)

    def test_return_realization_cap_excludes_november_snapshot(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-11-01", "2024-11-04")])
        signals = _prepare_signals(factor)
        market = _monthly_market_data(unadjusted)
        returns = _next_day_returns(adjusted)
        month = pd.Period("2024-11", freq="M")

        capped = _month_cross_section(signals, market, returns, month)
        self.assertTrue(capped.empty)

        with mock.patch.object(
            cash_conversion_ic, "MAX_RETURN_DATE", pd.Timestamp("2024-11-05")
        ):
            uncapped = _month_cross_section(signals, market, returns, month)
        self.assertEqual(len(uncapped), 20)
        self.assertEqual(uncapped["收益实现日"].max(), pd.Timestamp("2024-11-04"))

    def test_october_snapshot_is_kept_within_return_cap(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])

        results = calculate_monthly_ic(factor, unadjusted, adjusted)

        self.assertEqual(results["截面月份"].tolist(), ["2024-10"])
        self.assertEqual(results["截面日"].tolist(), ["2024-10-31"])
        self.assertEqual(results["收益实现日"].tolist(), ["2024-11-01"])
        self.assertTrue(
            (
                pd.to_datetime(results["最晚因子可用日期"])
                <= pd.to_datetime(results["截面日"])
            ).all()
        )

    def test_cross_section_is_standardized_and_ordered_by_return(self):
        unadjusted, adjusted, factor = synthetic_dataset([("2024-10-31", "2024-11-01")])
        signals = _prepare_signals(factor)
        market = _monthly_market_data(unadjusted)
        returns = _next_day_returns(adjusted)

        cross_section = _month_cross_section(
            signals, market, returns, pd.Period("2024-10", freq="M")
        )

        self.assertEqual(len(cross_section), 20)
        self.assertAlmostEqual(cross_section["因子清洗值"].mean(), 0.0, places=10)
        self.assertAlmostEqual(cross_section["因子清洗值"].std(ddof=0), 1.0, places=10)
        ordered = cross_section.sort_values("因子清洗值")
        self.assertLess(ordered["次日收益"].iloc[0], ordered["次日收益"].iloc[-1])

    def test_spearman_and_pearson(self):
        self.assertAlmostEqual(
            _spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0])), 1.0
        )
        self.assertAlmostEqual(
            _spearman(np.array([1.0, 2.0, 3.0]), np.array([9.0, 4.0, 1.0])), -1.0
        )
        self.assertAlmostEqual(
            _pearson(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])), 1.0
        )


if __name__ == "__main__":
    unittest.main()
