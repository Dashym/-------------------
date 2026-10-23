# -*- coding: utf-8 -*-
"""收现质量改善因子 Fama-MacBeth 回归检验（月末截面，T+1 后复权收益）
======================================================================
检验设定：
    1. 每月最后一个交易日为截面日 t；t 上使用"该日及之前已公告"的最新因子值；
    2. 因子形成日上限 2024-11-01、T+1 收益实现日上限 2024-11-02；
    3. 自变量 X = 月末截面清洗后因子（MAD(5 倍) 去极值 → 对数流通市值中性化 → Z-score）；
    4. 逐月横截面 OLS：y(i,t) = alpha(t) + beta(t) * X(i,t) + error(i,t)，
       y 为 T+1 后复权收益；
    5. 第二阶段：alpha / beta 的时间序列均值，标准误 = 序列标准差 / sqrt(截面数)；
       收益不重叠，无需 Newey-West 调整；p 值采用正态近似（同仓库 PB / 规模因子口径）。

输出：收现质量改善因子_回归结果.txt

运行环境：aksahre_demo/.venv（pandas / numpy）
    python cash_conversion_regression.py
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cash_conversion_ic import (
    ADJUSTED_QUOTE_FILE,
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
OUTPUT_FILE = SCRIPT_DIR / "收现质量改善因子_回归结果.txt"


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


def calculate_monthly_regressions(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
    """逐月横截面 OLS：T+1 收益 ~ 清洗后因子。"""
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

        target = cross_section["次日收益"].to_numpy(dtype="float64")
        regressor = cross_section["因子清洗值"].to_numpy(dtype="float64")
        design = np.column_stack([np.ones(len(regressor)), regressor])
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        predicted = design @ coefficients
        residual_sum = ((target - predicted) ** 2).sum()
        total_sum = ((target - target.mean()) ** 2).sum()
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0

        results.append(
            {
                "截面月份": str(month),
                "截面日": cross_section["截面日"].max().strftime("%Y-%m-%d"),
                "收益实现日": cross_section["收益实现日"].max().strftime("%Y-%m-%d"),
                "股票数": len(cross_section),
                "alpha": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "R-squared": float(r_squared),
            }
        )

    if not results:
        raise ValueError("没有满足最小样本数要求的有效月度截面")
    return pd.DataFrame(results)


def _fama_macbeth(regressions):
    summaries = {}
    for column in ("alpha", "beta"):
        values = regressions[column].to_numpy(dtype="float64")
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        standard_error = std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
        t_stat = mean / standard_error if standard_error > 0 else float("nan")
        summaries[column] = {
            "mean": mean,
            "std": std,
            "standard_error": standard_error,
            "t_stat": t_stat,
            "p_value": _t_pvalue(t_stat, len(values) - 1),
        }
    summaries["mean_r_squared"] = float(regressions["R-squared"].mean())
    return summaries


def render_report(regressions, summaries, factor_rows, factor_dates, market_dates, adjusted_dates):
    alpha = summaries["alpha"]
    beta = summaries["beta"]
    if beta["mean"] > 0:
        direction = "因子值越大，T+1 收益越高（正向）"
    elif beta["mean"] < 0:
        direction = "因子值越大，T+1 收益越低（反向）"
    else:
        direction = "无方向性"

    lines = [
        "=" * 78,
        "收现质量改善因子 Fama-MacBeth 回归检验报告（T+1 后复权收益）",
        "=" * 78,
        "",
        "检验设定：",
        "  每月最后一个交易日为截面日，使用截至该日已公告的最新因子值（不使用未来财报）",
        f"  因子形成日上限：{MAX_FACTOR_ASOF:%Y-%m-%d}；T+1 收益实现日上限：{MAX_RETURN_DATE:%Y-%m-%d}",
        "  自变量 X：月末截面清洗后因子（MAD(5 倍) 截尾 → 对数流通市值中性化 → Z-score）",
        "  因变量 y：T+1 后复权收益 = 后复权收盘(t+1) / 后复权收盘(t) - 1",
        "  每月截面 OLS：y(i,t) = alpha(t) + beta(t) * X(i,t) + error(i,t)",
        "  第二阶段：系数时间序列均值；标准误 = 序列标准差 / sqrt(截面数)（收益不重叠，无 HAC 调整）",
        "  p 值：正态近似；*** p<0.01，** p<0.05，* p<0.1",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "",
        "数据概况：",
        f"  原始因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（用于流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（用于收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效回归截面：{len(regressions)} 期，{regressions['截面月份'].iloc[0]} ~ {regressions['截面月份'].iloc[-1]}",
        f"  截面日范围：{regressions['截面日'].iloc[0]} ~ {regressions['截面日'].iloc[-1]}",
        f"  收益实现日范围：{regressions['收益实现日'].iloc[0]} ~ {regressions['收益实现日'].iloc[-1]}",
        f"  总观测数：{int(regressions['股票数'].sum()):,}",
        f"  平均有效股票数/截面：{regressions['股票数'].mean():.1f}，范围 {regressions['股票数'].min()} ~ {regressions['股票数'].max()}",
        f"  平均 R-squared：{summaries['mean_r_squared']:.4f}",
        "",
        f"{'变量':<20}{'均值系数':>14}{'标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 74,
        f"{'const':<20}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}"
        f"{alpha['t_stat']:>12.4f}{alpha['p_value']:>12.4f} {_significance(alpha['p_value'])}",
        f"{'收现质量改善因子':<20}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}"
        f"{beta['t_stat']:>12.4f}{beta['p_value']:>12.4f} {_significance(beta['p_value'])}",
        "-" * 74,
        "",
        f"beta 方向：{direction}；beta={beta['mean']:.6f}"
        f"（因子每提高 1 个截面标准差，T+1 收益平均变化 {beta['mean'] * 100:.4f} 个百分点）",
        f"Fama-MacBeth t={beta['t_stat']:.4f}，p={beta['p_value']:.4f}",
        f"结论：{direction}，样本内 "
        + (
            "因子对截面收益有统计显著的解释力。"
            if np.isfinite(beta["p_value"]) and beta["p_value"] < 0.1
            else "因子对截面收益的解释力不显著（p>0.1）。"
        ),
        "",
        "逐月回归结果：",
        f"{'因子月份':<12}{'截面日':<14}{'收益实现日':<14}{'样本数':>8}{'alpha':>14}{'beta':>14}{'R-squared':>14}",
        "-" * 90,
    ]
    lines.extend(
        f"{row['截面月份']:<12}{row['截面日']:<14}{row['收益实现日']:<14}{row['股票数']:>8d}"
        f"{row['alpha']:>14.6f}{row['beta']:>14.6f}{row['R-squared']:>14.4f}"
        for _, row in regressions.iterrows()
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
    regressions = calculate_monthly_regressions(factor, unadjusted, adjusted)
    summaries = _fama_macbeth(regressions)
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")
    report = render_report(
        regressions,
        summaries,
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
    log.info("Fama-MacBeth 回归报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
