# -*- coding: utf-8 -*-
"""收现质量改善因子 单调性检验（月末截面五分位，T+1 后复权收益）
==================================================================
检验设定：
    1. 每月最后一个交易日为截面日 t；t 上使用"该日及之前已公告"的最新因子值；
    2. 因子形成日上限 2024-11-01、T+1 收益实现日上限 2024-11-02；
    3. 月末截面执行 MAD(5 倍) 去极值 → 对数流通市值回归中性化 → Z-score；
    4. 按清洗后因子升序分五组（Q1 最低、Q5 最高），组内等权平均 T+1 后复权收益；
    5. 多空组合 = Q5 − Q1，显著性用月度截面序列的 t 检验（收益不重叠，无需 HAC）；
       年化按 12 期/年（截面为月度）复利折算。

输出：收现质量改善因子_单调性结果.txt

运行环境：aksahre_demo/.venv（pandas / numpy）
    python cash_conversion_monotonicity.py
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cash_conversion_ic import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_COLUMN,
    FACTOR_FILE,
    MAX_FACTOR_ASOF,
    MAX_RETURN_DATE,
    MIN_CROSS_SECTION_SIZE,
    UNADJUSTED_QUOTE_FILE,
    _month_cross_section,
    _monthly_market_data,
    _next_day_returns,
    _prepare_signals,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "收现质量改善因子_单调性结果.txt"
QUINTILE_COUNT = 5
PERIODS_PER_YEAR = 12


def _norm_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _t_pvalue(t_stat, degrees):
    if degrees < 1 or not np.isfinite(t_stat):
        return float("nan")
    return 2.0 * (1.0 - _norm_cdf(abs(t_stat)))


def _significance(p_value):
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def calculate_monthly_quintile_returns(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
    quintile_count=QUINTILE_COUNT,
):
    """逐月按清洗后因子分组，计算各组 T+1 后复权收益。"""
    if quintile_count < 2:
        raise ValueError("分组数必须至少为 2")

    signals = _prepare_signals(factor)
    market = _monthly_market_data(unadjusted_quotes)
    returns = _next_day_returns(adjusted_quotes)

    results = []
    for month in sorted(market["月份"].unique()):
        cross_section = _month_cross_section(
            signals, market, returns, month, min_cross_section
        )
        if cross_section.empty:
            continue
        ordered = cross_section.sort_values(["因子清洗值", "股票代码"])
        stock_count = len(ordered)
        if stock_count < quintile_count:
            continue

        targets = ordered["次日收益"].to_numpy(dtype="float64")
        row = {
            "截面月份": str(month),
            "截面日": ordered["截面日"].max().strftime("%Y-%m-%d"),
            "收益实现日": ordered["收益实现日"].max().strftime("%Y-%m-%d"),
            "股票数": stock_count,
        }
        for group in range(1, quintile_count + 1):
            start = int(stock_count * (group - 1) / quintile_count)
            end = int(stock_count * group / quintile_count)
            row[f"Q{group}收益"] = float(targets[start:end].mean())
        row["Q5-Q1收益"] = row[f"Q{quintile_count}收益"] - row["Q1收益"]
        results.append(row)

    if not results:
        raise ValueError("没有满足最小样本数要求的有效月度截面")
    return pd.DataFrame(results)


def _summarize(results, quintile_count=QUINTILE_COUNT):
    summaries = {}
    for group in range(1, quintile_count + 1):
        values = results[f"Q{group}收益"].to_numpy(dtype="float64")
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        standard_error = std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
        t_stat = mean / standard_error if standard_error > 0 else float("nan")
        summaries[group] = {
            "mean": mean,
            "std": std,
            "standard_error": standard_error,
            "t_stat": t_stat,
            "p_value": _t_pvalue(t_stat, len(values) - 1),
            "annualized": (1.0 + mean) ** PERIODS_PER_YEAR - 1.0,
        }

    spread_values = results["Q5-Q1收益"].to_numpy(dtype="float64")
    spread_mean = float(spread_values.mean())
    spread_std = float(spread_values.std(ddof=1))
    spread_se = (
        spread_std / np.sqrt(len(spread_values)) if len(spread_values) > 1 else float("nan")
    )
    spread_t = spread_mean / spread_se if spread_se > 0 else float("nan")
    spread = {
        "mean": spread_mean,
        "std": spread_std,
        "standard_error": spread_se,
        "t_stat": spread_t,
        "p_value": _t_pvalue(spread_t, len(spread_values) - 1),
        "annualized": (1.0 + spread_mean) ** PERIODS_PER_YEAR - 1.0,
        "win_ratio": float((spread_values > 0).mean()),
    }
    return summaries, spread


def _monotonicity_label(summaries, quintile_count=QUINTILE_COUNT):
    means = [summaries[group]["mean"] for group in range(1, quintile_count + 1)]
    increasing = all(means[index] <= means[index + 1] for index in range(quintile_count - 1))
    decreasing = all(means[index] >= means[index + 1] for index in range(quintile_count - 1))
    if increasing:
        return "严格单调递增（因子越大 → T+1 收益越高）"
    if decreasing:
        return "严格单调递减（因子越大 → T+1 收益越低）"
    up_steps = sum(
        1 for index in range(quintile_count - 1) if means[index] < means[index + 1]
    )
    return f"非严格单调（递增步数 {up_steps}/{quintile_count - 1}）"


def render_report(results, summaries, spread, factor_rows, factor_dates, market_dates, adjusted_dates):
    lines = [
        "=" * 78,
        "收现质量改善因子 单调性检验报告（五分位，T+1 后复权收益）",
        "=" * 78,
        "",
        "检验设定：",
        "  每月最后一个交易日为截面日，使用截至该日已公告的最新因子值（不使用未来财报）",
        f"  因子形成日上限：{MAX_FACTOR_ASOF:%Y-%m-%d}；T+1 收益实现日上限：{MAX_RETURN_DATE:%Y-%m-%d}",
        "  月末截面执行 MAD(5 倍) 去极值、对数流通市值回归中性化，中性化残差 Z-score 用于排序",
        "  按清洗后因子升序分五组：Q1 最低 ~ Q5 最高，组内等权平均 T+1 后复权收益",
        "  收益 = 后复权收盘(t+1) / 后复权收盘(t) - 1（|收益| > 10.5% 视为涨跌停不可成交，已剔除）",
        "  显著性：月度截面收益序列的 t 检验（收益不重叠，无需 Newey-West 调整）",
        f"  年化按 {PERIODS_PER_YEAR} 期/年复利折算（截面为月度）",
        "",
        "数据概况：",
        f"  原始因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（用于流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（用于收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；{results['截面月份'].iloc[0]} ~ {results['截面月份'].iloc[-1]}",
        f"  截面日范围：{results['截面日'].iloc[0]} ~ {results['截面日'].iloc[-1]}",
        f"  收益实现日范围：{results['收益实现日'].iloc[0]} ~ {results['收益实现日'].iloc[-1]}",
        f"  平均有效股票数：{results['股票数'].mean():.1f}，范围 {results['股票数'].min()} ~ {results['股票数'].max()}",
        "",
        "各分组 T+1 收益（组内等权，再对月度截面取均值）：",
        f"{'分组':<10}{'平均收益':>14}{'标准差':>14}{'年化收益':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 76,
    ]
    for group in range(1, QUINTILE_COUNT + 1):
        summary = summaries[group]
        lines.append(
            f"{'Q' + str(group):<10}{summary['mean']:>14.4%}{summary['std']:>14.4%}"
            f"{summary['annualized']:>14.2%}{summary['t_stat']:>12.4f}{summary['p_value']:>12.4f}"
        )
    lines.extend(
        [
            "-" * 76,
            f"{'Q5-Q1':<10}{spread['mean']:>14.4%}{spread['std']:>14.4%}"
            f"{spread['annualized']:>14.2%}{spread['t_stat']:>12.4f}{spread['p_value']:>12.4f} "
            f"{_significance(spread['p_value'])}",
            "",
            f"多空胜率（Q5-Q1 > 0 的月份占比）：{spread['win_ratio']:.1%}",
            f"单调性判断：{_monotonicity_label(summaries)}",
            f"结论：Q5-Q1 平均 {spread['mean']:.4%}，t={spread['t_stat']:.4f}，p={spread['p_value']:.4f}"
            f"{_significance(spread['p_value'])}；样本内 "
            + ("存在显著多空收益。" if np.isfinite(spread["p_value"]) and spread["p_value"] < 0.1 else "多空收益不显著。"),
            "",
            "逐月分组结果（T+1 后复权收益）：",
            f"{'因子月份':<12}{'截面日':<14}{'收益实现日':<14}{'样本数':>8}"
            + "".join(f"{'Q' + str(group):>10}" for group in range(1, QUINTILE_COUNT + 1))
            + f"{'Q5-Q1':>12}",
            "-" * (48 + 10 * QUINTILE_COUNT + 12),
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"{row['截面月份']:<12}{row['截面日']:<14}{row['收益实现日']:<14}{row['股票数']:>8d}"
            + "".join(
                f"{row[f'Q{group}收益']:>10.2%}" for group in range(1, QUINTILE_COUNT + 1)
            )
            + f"{row['Q5-Q1收益']:>12.2%}"
        )
    lines.extend(["", "=" * 78, ""])
    return "\n".join(lines)


def main():
    for input_file in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    results = calculate_monthly_quintile_returns(factor, unadjusted, adjusted)
    summaries, spread = _summarize(results)
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")
    report = render_report(
        results,
        summaries,
        spread,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        (market_dates.min().strftime("%Y-%m-%d"), market_dates.max().strftime("%Y-%m-%d")),
        (
            adjusted_dates.min().strftime("%Y-%m-%d"),
            adjusted_dates.max().strftime("%Y-%m-%d"),
        ),
    )
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("单调性检验报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
