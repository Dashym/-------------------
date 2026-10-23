# -*- coding: utf-8 -*-
"""研发-毛利转化因子 Logit 检验（方向哑变量 vs 未来 1 年收益方向）。

兜底策略：放弃 OLS，改用 Logit 分类模型检验方向哑变量（0/1）与未来 1 年
收益方向的关系。采用 Fama-MacBeth 式框架：每季度截面做一次 Logit 回归，
对哑变量系数的时间序列取均值，并用 Newey-West HAC（滞后 3）标准误检验显著性。
"""

import logging
import sys

import numpy as np
import pandas as pd

from rd_gm_test_common import (
    DUMMY_COLUMN,
    HAC_MAX_LAGS,
    MIN_CROSS_SECTION_SIZE,
    RETURN_COLUMN,
    SCRIPT_DIR,
    coefficient_summary,
    load_inputs,
    prepare_quarterly_cross_sections,
    report_dates,
    significance,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "研发毛利转化因子_Logit结果.txt"


def fit_logit(design, y, max_iterations=100, tolerance=1e-9):
    """二分类 Logit 回归（极大似然，牛顿-拉夫逊迭代）。

    design : (n, k) 设计矩阵（首列为截距）
    y      : (n,) 0/1 标签
    返回系数 beta (k,)。若 Hessian 奇异或迭代发散则返回 NaN 向量。
    """
    n, k = design.shape
    beta = np.zeros(k)
    for _ in range(max_iterations):
        z = design @ beta
        probabilities = 1.0 / (1.0 + np.exp(-z))
        probabilities = np.clip(probabilities, 1e-10, 1.0 - 1e-10)
        gradient = design.T @ (y - probabilities)
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)
        if not np.all(np.isfinite(step)):
            return np.full(k, np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < tolerance:
            return beta
    return beta


def calculate_quarterly_logit(factor, unadjusted, adjusted):
    sections = prepare_quarterly_cross_sections(factor, unadjusted, adjusted)
    results = []
    for quarter, section in sections.groupby("截面季度", sort=True):
        x = section[DUMMY_COLUMN].to_numpy(dtype=float)
        y = (section[RETURN_COLUMN] > 0).astype(float).to_numpy()
        if len(np.unique(x)) < 2:
            continue  # 哑变量截面内无变化，无法估计
        if len(np.unique(y)) < 2:
            continue  # 收益方向全 0 或全 1，无法估计
        design = np.column_stack([np.ones(len(x)), x])
        beta = fit_logit(design, y)
        if not np.all(np.isfinite(beta)):
            continue
        positives = int(x.sum())
        results.append(
            {
                "截面季度": quarter,
                "截面日期": section["截面日期"].iloc[0],
                "预测季度": section["预测季度"].iloc[0],
                "股票数": len(section),
                "信号股票数": positives,
                "信号占比": positives / len(section),
                "alpha": float(beta[0]),
                "beta": float(beta[1]),
            }
        )
    return pd.DataFrame(results)


def render_report(regressions, factor_rows, factor_dates, market_dates, adjusted_dates):
    if regressions.empty:
        raise ValueError("没有达到最小样本数要求的有效季度 Logit 截面")

    alpha = coefficient_summary(regressions["alpha"])
    beta = coefficient_summary(regressions["beta"])
    p_value = beta["p_value"]
    if p_value < 0.01:
        conclusion = "方向哑变量系数在 1% 水平显著，信号对未来 1 年正收益方向有显著预测力。"
    elif p_value < 0.05:
        conclusion = "方向哑变量系数在 5% 水平显著。"
    elif p_value < 0.1:
        conclusion = "方向哑变量系数在 10% 水平显著，统计证据较弱。"
    else:
        conclusion = "方向哑变量系数未达到 10% 显著性水平。"

    lines = [
        "=" * 72,
        "研发-毛利转化因子 Logit 检验报告",
        "=" * 72,
        "",
        "检验设定：",
        "  模型：Logit（二分类极大似然），替代 OLS",
        "  因变量 y：未来 1 年收益方向 = 1（未来 1 年收益 > 0，行业中性化后）else 0",
        "  自变量 x：方向哑变量（研发强度_t-8 同比上升 且 毛利率_t 同比上升 -> 1）",
        "  每季度截面 Logit：P(y=1|x) = sigmoid(alpha + beta * x)",
        "  汇总：Fama-MacBeth，报告逐季系数的时间序列均值",
        f"  标准误：Newey-West HAC，Bartlett 核，最大滞后 {HAC_MAX_LAGS}",
        "  p 值采用双侧 Student-t(df=有效截面数-1)",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效 Logit 截面：{len(regressions)} 期，{regressions['截面季度'].iloc[0]} ~ {regressions['截面季度'].iloc[-1]}",
        f"  平均有效股票数：{regressions['股票数'].mean():.1f}",
        f"  平均信号占比：{regressions['信号占比'].mean():.2%}",
        "",
        f"{'变量':<18}{'均值系数':>14}{'HAC标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 70,
        f"{'const':<18}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}"
        f"{alpha['t_stat']:>12.4f}{alpha['p_value']:>12.4f} {significance(alpha['p_value'])}",
        f"{'方向哑变量 beta':<18}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}"
        f"{beta['t_stat']:>12.4f}{beta['p_value']:>12.4f} {significance(beta['p_value'])}",
        "-" * 70,
        "显著性标记：*** p<0.01，** p<0.05，* p<0.1",
        f"Fama-MacBeth beta={beta['mean']:.6f}，t={beta['t_stat']:.4f}，p={beta['p_value']:.4f}",
        f"结论：{conclusion}",
        "",
        "逐季 Logit 回归结果：",
        f"{'截面季度':<12}{'预测季度':<12}{'样本数':>8}{'信号数':>8}{'alpha':>14}{'beta':>14}",
        "-" * 66,
    ]
    lines.extend(
        f"{row['截面季度']:<12}{row['预测季度']:<12}{row['股票数']:>8d}{row['信号股票数']:>8d}"
        f"{row['alpha']:>14.6f}{row['beta']:>14.6f}"
        for _, row in regressions.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        regressions = calculate_quarterly_logit(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(regressions, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError, ArithmeticError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("Logit 报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
