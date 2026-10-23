import logging
import sys

import numpy as np
import pandas as pd

from technical_barrier_test_common import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_FILE,
    FORWARD_HORIZON_MONTHS,
    HAC_MAX_LAGS,
    MIN_CROSS_SECTION_SIZE,
    SCRIPT_DIR,
    UNADJUSTED_QUOTE_FILE,
    coefficient_summary,
    load_inputs,
    prepare_monthly_cross_sections,
    report_dates,
    significance,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "技术壁垒升级_回归结果_未来12个月.txt"


def calculate_monthly_regressions(factor, unadjusted, adjusted):
    sections = prepare_monthly_cross_sections(factor, unadjusted, adjusted)
    results = []
    for month, section in sections.groupby("截面月份", sort=True):
        x = section["因子值_zscore"].to_numpy(dtype=float)
        y = section["未来12个月收益"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        if np.linalg.matrix_rank(design) < 2:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        total_sum_squares = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - float(np.dot(residuals, residuals)) / total_sum_squares if total_sum_squares > 0 else 0.0
        results.append(
            {
                "月份": month,
                "预测月份": section["预测月份"].iloc[0],
                "截面日期": section["截面日期"].iloc[0],
                "股票数": len(section),
                "alpha": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "R-squared": r_squared,
            }
        )
    return pd.DataFrame(results)


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if len(results) <= HAC_MAX_LAGS:
        raise ValueError(f"有效回归截面需多于 HAC 滞后期数 {HAC_MAX_LAGS}，当前为 {len(results)} 期")
    alpha = coefficient_summary(results["alpha"])
    beta = coefficient_summary(results["beta"])
    if beta["p_value"] < 0.01:
        conclusion = "在 1% 水平显著"
    elif beta["p_value"] < 0.05:
        conclusion = "在 5% 水平显著"
    elif beta["p_value"] < 0.1:
        conclusion = "在 10% 水平显著，证据较弱"
    else:
        conclusion = "未达到 10% 显著性水平"
    lines = [
        "=" * 72,
        "技术壁垒升级因子 Fama-MacBeth 回归报告",
        "=" * 72,
        "",
        "回归设定：",
        "  因变量：未来 12 个月累计后复权收益 = t+12 月末价格 / t 月末价格 - 1",
        "  自变量：月末时点可用的技术壁垒升级因子；截面 MAD(5倍) 截尾并 Z-score",
        "  财务升级Z-score = 0.5 * TTM毛利率同比变化标准化 + 0.5 * TTM营收同比增长率标准化",
        "  因子值 = 财务升级Z-score × 高壁垒标签；只在高壁垒标签=1的样本内标准化，其他股票取 0",
        "  财报信号可用日 = max(首次公告日, 报告期截止日 + 15 天)",
        "  每月 OLS：y(i,t+12) = alpha(t) + beta(t) * X(i,t) + error(i,t)",
        "  第二阶段 Newey-West HAC，Bartlett 核，最大滞后 11，含 n/(n-1) 有限样本修正",
        "  p 值采用双侧 Student-t(df=有效截面数-1)",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效回归截面：{len(results)} 期；{results['月份'].iloc[0]} ~ {results['月份'].iloc[-1]}",
        f"  总观测数：{int(results['股票数'].sum()):,}；平均 R-squared：{results['R-squared'].mean():.4f}",
        "",
        f"{'变量':<20}{'均值系数':>14}{'HAC标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 72,
        f"{'const':<20}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}{alpha['t_stat']:>12.4f}{alpha['p_value']:>12.4f} {significance(alpha['p_value'])}",
        f"{'技术壁垒升级 beta':<20}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}{beta['t_stat']:>12.4f}{beta['p_value']:>12.4f} {significance(beta['p_value'])}",
        "-" * 72,
        f"Fama-MacBeth beta={beta['mean']:.6f}，t={beta['t_stat']:.4f}，p={beta['p_value']:.4f}",
        f"结论：因子回归系数{conclusion}。",
        "",
        "逐月回归结果：",
        f"{'月份':<12}{'预测月份':<12}{'样本数':>8}{'alpha':>14}{'beta':>14}{'R-squared':>14}",
        "-" * 74,
    ]
    lines.extend(
        f"{row['月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}{row['alpha']:>14.6f}{row['beta']:>14.6f}{row['R-squared']:>14.4f}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        results = calculate_monthly_regressions(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(results, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError, ArithmeticError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("未来 %s 个月回归报告已输出：%s", FORWARD_HORIZON_MONTHS, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
