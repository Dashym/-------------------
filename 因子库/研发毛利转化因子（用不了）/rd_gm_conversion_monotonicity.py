# -*- coding: utf-8 -*-
"""研发-毛利转化因子五分组单调性检验（季度截面，未来 1 年收益）。"""

import logging
import sys

import numpy as np
import pandas as pd

from rd_gm_test_common import (
    FACTOR_COLUMN,
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
OUTPUT_FILE = SCRIPT_DIR / "研发毛利转化因子_单调性结果.txt"
QUINTILE_COUNT = 5


def calculate_quarterly_quintile_returns(factor, unadjusted, adjusted):
    sections = prepare_quarterly_cross_sections(
        factor, unadjusted, adjusted,
        min_cross_section=max(MIN_CROSS_SECTION_SIZE, QUINTILE_COUNT),
    )
    results = []
    for quarter, section in sections.groupby("截面季度", sort=True):
        ranked = section[FACTOR_COLUMN].rank(method="average")
        groups = np.ceil(ranked.to_numpy() * QUINTILE_COUNT / len(section)).astype(int)
        section = section.copy()
        section["分组"] = np.clip(groups, 1, QUINTILE_COUNT)
        if section["分组"].nunique() != QUINTILE_COUNT:
            continue
        row = {
            "截面季度": quarter,
            "截面日期": section["截面日期"].iloc[0],
            "预测季度": section["预测季度"].iloc[0],
            "股票数": len(section),
        }
        for group in range(1, QUINTILE_COUNT + 1):
            group_data = section.loc[section["分组"] == group]
            row[f"Q{group}股票数"] = len(group_data)
            row[f"Q{group}收益"] = group_data[RETURN_COLUMN].mean()
        row["Q5-Q1收益"] = row["Q5收益"] - row["Q1收益"]
        results.append(row)

    columns = ["截面季度", "截面日期", "预测季度", "股票数"]
    for group in range(1, QUINTILE_COUNT + 1):
        columns.extend([f"Q{group}股票数", f"Q{group}收益"])
    columns.append("Q5-Q1收益")
    return pd.DataFrame(results, columns=columns)


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if results.empty:
        raise ValueError("没有达到最小样本数要求的有效季度分组截面")

    group_summaries = {
        group: coefficient_summary(results[f"Q{group}收益"])
        for group in range(1, QUINTILE_COUNT + 1)
    }
    spread = coefficient_summary(results["Q5-Q1收益"])
    group_means = [group_summaries[group]["mean"] for group in range(1, 6)]
    increasing = all(left < right for left, right in zip(group_means, group_means[1:]))
    decreasing = all(left > right for left, right in zip(group_means, group_means[1:]))
    if increasing:
        monotonicity = "分组平均收益随因子值严格递增"
    elif decreasing:
        monotonicity = "分组平均收益随因子值严格递减"
    else:
        monotonicity = "分组平均收益未呈现严格单调变化"

    lines = [
        "=" * 76,
        "研发-毛利转化因子五分组单调性检验报告",
        "=" * 76,
        "",
        "检验设定：",
        "  截面日 = 季度末最后一个交易日，按基础连续因子值升序分为五组（Q1最低，Q5最高）",
        "  各组收益为组内等权平均未来 1 年累计后复权收益",
        "  收益 = t+4 季度末后复权收盘 / t 季度末后复权收盘 - 1",
        "  收益率已按申万三级行业中位数去均值，并与因子值做 1%/99% 双缩尾",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}；每组至少包含 1 只股票",
        "  组收益与 Q5-Q1 收益差采用 Newey-West HAC 标准误（未来 1 年收益重叠）",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效季度截面：{len(results)} 期；{results['截面季度'].iloc[0]} ~ {results['截面季度'].iloc[-1]}",
        f"  平均有效股票数：{results['股票数'].mean():.1f}",
        "",
        "各组未来 1 年收益（对季度组收益时间序列取均值）：",
        f"{'分组':<10}{'平均收益':>14}{'HAC标准误':>14}{'t统计量':>14}{'p值':>12}",
        "-" * 64,
    ]
    for group in range(1, QUINTILE_COUNT + 1):
        summary = group_summaries[group]
        lines.append(
            f"{'Q' + str(group):<10}{summary['mean']:>14.4%}"
            f"{summary['standard_error']:>14.4%}{summary['t_stat']:>14.4f}"
            f"{summary['p_value']:>12.4f}"
        )
    lines.extend(
        [
            "-" * 64,
            f"Q5-Q1收益差：{spread['mean']:.4%}，HAC标准误={spread['standard_error']:.4%}，"
            f"t={spread['t_stat']:.4f}，p={spread['p_value']:.4f} {significance(spread['p_value'])}",
            f"Q5 多头：平均收益={group_summaries[5]['mean']:.4%}，"
            f"t={group_summaries[5]['t_stat']:.4f}，p={group_summaries[5]['p_value']:.4f} "
            f"{significance(group_summaries[5]['p_value'])}",
            f"通过标准：Q5-Q1 多空收益差显著（t>2）且 Q5 多头显著。",
            f"单调性判断：{monotonicity}。",
            "",
            "逐季分组结果：",
            f"{'截面季度':<12}{'预测季度':<12}{'样本数':>8}"
            f"{'Q1':>12}{'Q2':>12}{'Q3':>12}{'Q4':>12}{'Q5':>12}{'Q5-Q1':>12}",
            "-" * 92,
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"{row['截面季度']:<12}{row['预测季度']:<12}{row['股票数']:>8d}"
            + "".join(f"{row[f'Q{group}收益']:>12.2%}" for group in range(1, 6))
            + f"{row['Q5-Q1收益']:>12.2%}"
        )
    lines.extend(["", "=" * 76, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        results = calculate_quarterly_quintile_returns(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(results, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError, ArithmeticError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("单调性报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
