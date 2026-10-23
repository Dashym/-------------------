import logging
import sys

import numpy as np
import pandas as pd

from rd_factor_test_common import (
    FACTOR_COLUMN,
    FACTOR_FILE,
    FORWARD_HORIZON_MONTHS,
    HAC_MAX_LAGS,
    MIN_CROSS_SECTION_SIZE,
    SCRIPT_DIR,
    UNADJUSTED_QUOTE_FILE,
    ADJUSTED_QUOTE_FILE,
    coefficient_summary,
    load_inputs,
    prepare_monthly_cross_sections,
    report_dates,
    significance,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "研发费用率_回归结果_未来12个月.txt"


def calculate_monthly_regressions(factor, unadjusted_quotes, adjusted_quotes):
    cross_sections = prepare_monthly_cross_sections(
        factor, unadjusted_quotes, adjusted_quotes
    )
    results = []
    for month, section in cross_sections.groupby("截面月份", sort=True):
        x = section["因子值_zscore"].to_numpy(dtype=float)
        y = section["未来12个月收益"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        if np.linalg.matrix_rank(design) < 2:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        total_sum_squares = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            1 - float(np.dot(residuals, residuals)) / total_sum_squares
            if total_sum_squares > 0
            else 0.0
        )
        results.append(
            {
                "月份": month,
                "截面日期": section["截面日期"].iloc[0],
                "预测月份": section["预测月份"].iloc[0],
                "股票数": len(section),
                "alpha": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "R-squared": r_squared,
            }
        )
    return pd.DataFrame(
        results,
        columns=[
            "月份",
            "截面日期",
            "预测月份",
            "股票数",
            "alpha",
            "beta",
            "R-squared",
        ],
    )


def render_report(regressions, factor_rows, factor_dates, market_dates, adjusted_dates):
    if len(regressions) <= HAC_MAX_LAGS:
        raise ValueError(
            f"有效回归截面需多于 HAC 滞后期数 {HAC_MAX_LAGS}，当前为 {len(regressions)} 期"
        )
    alpha = coefficient_summary(regressions["alpha"])
    beta = coefficient_summary(regressions["beta"])
    p_value = beta["p_value"]
    if p_value < 0.01:
        conclusion = "因子回归系数在 1% 水平显著。"
    elif p_value < 0.05:
        conclusion = "因子回归系数在 5% 水平显著。"
    elif p_value < 0.1:
        conclusion = "因子回归系数在 10% 水平显著，统计证据较弱。"
    else:
        conclusion = "因子回归系数未达到 10% 显著性水平。"

    lines = [
        "=" * 72,
        "研发费用率（TTM）Fama-MacBeth 回归报告",
        "=" * 72,
        "",
        "回归设定：",
        "  因变量 y：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  自变量 X：月末时点已公告的最新研发费用率；月末截面 MAD(5倍) 去极值并 Z-score",
        "  不做市值中性化，沿用清洗因子设定",
        "  每月截面 OLS：y(i,t+12) = alpha(t) + beta(t) * X(i,t) + error(i,t)",
        "  汇总：Fama-MacBeth，报告逐月系数的时间序列均值",
        f"  第二阶段标准误：Newey-West HAC，Bartlett 核，最大滞后期数={HAC_MAX_LAGS}，含 n/(n-1) 有限样本修正",
        "  p 值采用双侧 Student-t(df=有效截面数-1)",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "",
        "数据概况：",
        f"  清洗后因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效回归截面：{len(regressions)} 期，{regressions['月份'].iloc[0]} ~ {regressions['月份'].iloc[-1]}",
        f"  总观测数：{int(regressions['股票数'].sum()):,}",
        f"  平均有效股票数/截面：{regressions['股票数'].mean():.1f}",
        f"  平均 R-squared：{regressions['R-squared'].mean():.4f}",
        "",
        f"{'变量':<18}{'均值系数':>14}{'HAC标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 70,
        f"{'const':<18}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}"
        f"{alpha['t_stat']:>12.4f}{alpha['p_value']:>12.4f} {significance(alpha['p_value'])}",
        f"{'研发费用率 beta':<18}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}"
        f"{beta['t_stat']:>12.4f}{beta['p_value']:>12.4f} {significance(beta['p_value'])}",
        "-" * 70,
        "显著性标记：*** p<0.01，** p<0.05，* p<0.1",
        f"Fama-MacBeth beta={beta['mean']:.6f}，t={beta['t_stat']:.4f}，p={beta['p_value']:.4f}",
        f"结论：{conclusion}",
        "",
        "逐月回归结果：",
        f"{'月份':<12}{'预测月份':<12}{'样本数':>8}{'alpha':>14}{'beta':>14}{'R-squared':>14}",
        "-" * 74,
    ]
    lines.extend(
        f"{row['月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}"
        f"{row['alpha']:>14.6f}{row['beta']:>14.6f}{row['R-squared']:>14.4f}"
        for _, row in regressions.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        regressions = calculate_monthly_regressions(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(regressions, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError, ArithmeticError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("未来 %s 个月 Fama-MacBeth 回归报告已输出：%s", FORWARD_HORIZON_MONTHS, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
