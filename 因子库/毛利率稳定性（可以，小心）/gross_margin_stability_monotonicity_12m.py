import logging
import sys

import numpy as np
import pandas as pd

from stability_test_common import (
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
OUTPUT_FILE = SCRIPT_DIR / "毛利率稳定性_单调性检验_未来12个月.txt"
QUINTILE_COUNT = 5


def calculate_monthly_quintile_returns(factor, unadjusted, adjusted):
    sections = prepare_monthly_cross_sections(
        factor,
        unadjusted,
        adjusted,
        min_cross_section=max(MIN_CROSS_SECTION_SIZE, QUINTILE_COUNT),
    )
    results = []
    for month, section in sections.groupby("截面月份", sort=True):
        ranks = section["因子值_zscore"].rank(method="average")
        section = section.copy()
        section["分组"] = np.clip(
            np.ceil(ranks.to_numpy() * QUINTILE_COUNT / len(section)).astype(int),
            1,
            QUINTILE_COUNT,
        )
        if section["分组"].nunique() != QUINTILE_COUNT:
            continue
        row = {
            "因子月份": month,
            "预测月份": section["预测月份"].iloc[0],
            "样本数": len(section),
        }
        for group in range(1, QUINTILE_COUNT + 1):
            group_data = section.loc[section["分组"] == group]
            row[f"Q{group}样本数"] = len(group_data)
            row[f"Q{group}收益"] = group_data["未来12个月收益"].mean()
        row["Q5-Q1收益"] = row["Q5收益"] - row["Q1收益"]
        results.append(row)
    return pd.DataFrame(results)


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if len(results) <= HAC_MAX_LAGS:
        raise ValueError(
            f"有效分组截面需多于 HAC 滞后期数 {HAC_MAX_LAGS}，当前为 {len(results)} 期"
        )
    summaries = {
        group: coefficient_summary(results[f"Q{group}收益"])
        for group in range(1, QUINTILE_COUNT + 1)
    }
    spread = coefficient_summary(results["Q5-Q1收益"])
    means = [summaries[group]["mean"] for group in range(1, 6)]
    increasing = all(left < right for left, right in zip(means, means[1:]))
    decreasing = all(left > right for left, right in zip(means, means[1:]))
    monotonicity = (
        "平均收益严格递增"
        if increasing
        else "平均收益严格递减"
        if decreasing
        else "平均收益未呈严格单调变化"
    )
    lines = [
        "=" * 76,
        "毛利率稳定性因子五分组单调性检验报告",
        "=" * 76,
        "",
        "检验设定：",
        "  月末使用已可用的最新毛利率稳定性得分，按得分升序分为五组（Q1低稳定性，Q5高稳定性）",
        "  因子 = -(0.4*z_sigma + 0.3*z_cv + 0.3*z_drift)，三个子指标均按同行季度 Z-score",
        "  子指标基于最近 12 季度序列；原始毛利率按同行季度 1%/99% 缩尾并在线性插值后计算",
        "  行业截面为财报数据覆盖的汽车零部件股票，因子在所需公告后的首个交易日可用",
        "  月末截面 MAD(5倍) 截尾并 Z-score；因子已在汽车零部件同行内标准化，不另做市值中性化",
        "  各组收益为组内等权平均未来 12 个月累计后复权收益",
        "  Q5-Q1 表示高稳定性组减低稳定性组",
        f"  每期最少股票数：{MIN_CROSS_SECTION_SIZE}；各组至少 1 只",
        f"  组收益及 Q5-Q1 月度序列采用 Newey-West HAC（Bartlett 核，滞后 {HAC_MAX_LAGS}）",
        "  p 值采用双侧 Student-t(df=有效截面数-1)",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；{results['因子月份'].iloc[0]} ~ {results['因子月份'].iloc[-1]}",
        f"  不复权行情范围：{market_dates[0]} ~ {market_dates[1]}",
        "",
        f"{'分组':<10}{'平均收益':>14}{'HAC标准误':>14}{'t统计量':>14}{'p值':>12}",
        "-" * 64,
    ]
    for group in range(1, 6):
        summary = summaries[group]
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
            f"单调性判断：{monotonicity}。",
            "",
            "逐月分组结果：",
            f"{'月份':<12}{'预测月份':<12}{'样本数':>8}"
            f"{'Q1':>12}{'Q2':>12}{'Q3':>12}{'Q4':>12}{'Q5':>12}{'Q5-Q1':>12}",
            "-" * 92,
        ]
    )
    lines.extend(
        f"{row['因子月份']:<12}{row['预测月份']:<12}{row['样本数']:>8d}"
        + "".join(f"{row[f'Q{group}收益']:>12.2%}" for group in range(1, 6))
        + f"{row['Q5-Q1收益']:>12.2%}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 76, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        results = calculate_monthly_quintile_returns(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(results, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError, ArithmeticError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("未来 %s 个月单调性报告已输出：%s", FORWARD_HORIZON_MONTHS, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
