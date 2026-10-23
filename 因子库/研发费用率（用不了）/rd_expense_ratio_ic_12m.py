import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rd_factor_test_common import (
    FACTOR_COLUMN,
    FACTOR_FILE,
    FORWARD_HORIZON_MONTHS,
    MIN_CROSS_SECTION_SIZE,
    SCRIPT_DIR,
    UNADJUSTED_QUOTE_FILE,
    ADJUSTED_QUOTE_FILE,
    load_inputs,
    prepare_monthly_cross_sections,
    report_dates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "研发费用率_IC_未来12个月.txt"


def calculate_monthly_ic(factor, unadjusted_quotes, adjusted_quotes):
    cross_sections = prepare_monthly_cross_sections(
        factor, unadjusted_quotes, adjusted_quotes
    )
    results = []
    for month, section in cross_sections.groupby("截面月份", sort=True):
        rank_ic = section["因子值_zscore"].rank(method="average").corr(
            section["未来12个月收益"].rank(method="average")
        )
        pearson_ic = section["因子值_zscore"].corr(section["未来12个月收益"])
        results.append(
            {
                "截面月份": month,
                "截面日期": section["截面日期"].iloc[0],
                "预测月份": section["预测月份"].iloc[0],
                "股票数": len(section),
                "Rank IC": rank_ic,
                "Pearson IC": pearson_ic,
            }
        )
    return pd.DataFrame(
        results,
        columns=[
            "截面月份",
            "截面日期",
            "预测月份",
            "股票数",
            "Rank IC",
            "Pearson IC",
        ],
    )


def _statistics(values):
    values = values.dropna()
    if values.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive_ratio": float("nan"),
            "t_stat": float("nan"),
        }
    mean = values.mean()
    standard_deviation = values.std(ddof=1) if len(values) > 1 else float("nan")
    standard_error = standard_deviation / np.sqrt(len(values))
    return {
        "mean": mean,
        "std": standard_deviation,
        "ir": mean / standard_deviation
        if np.isfinite(standard_deviation) and standard_deviation > 0
        else float("nan"),
        "positive_ratio": (values > 0).mean(),
        "t_stat": mean / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else float("nan"),
    }


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if results.empty:
        raise ValueError("没有达到最小样本数要求的有效月度 IC 截面")
    rank = _statistics(results["Rank IC"])
    pearson = _statistics(results["Pearson IC"])
    lines = [
        "=" * 72,
        "研发费用率（TTM）IC 检验报告",
        "=" * 72,
        "",
        "检测口径：",
        "  使用截至月末已公告的最新研发费用率数据；月末截面重新执行 MAD(5倍) 去极值和 Z-score",
        "  不做市值中性化，保持因子清洗时的设定",
        "  收益：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  Rank IC：Spearman 秩相关；Pearson IC：线性相关",
        f"  最少有效股票数/截面：{MIN_CROSS_SECTION_SIZE}",
        "  IR：月度 IC 均值 / 月度 IC 样本标准差（ddof=1）",
        "  t-stat 为常规均值检验，未做 Newey-West 自相关调整",
        "",
        "数据概况：",
        f"  清洗后因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；{results['截面月份'].iloc[0]} ~ {results['截面月份'].iloc[-1]}",
        f"  平均有效股票数：{results['股票数'].mean():.1f}，范围 {results['股票数'].min()} ~ {results['股票数'].max()}",
        "",
        f"{'指标':<16}{'Rank IC':>14}{'Pearson IC':>16}",
        "-" * 46,
        f"{'IC 均值':<16}{rank['mean']:>14.4f}{pearson['mean']:>16.4f}",
        f"{'IC 标准差':<16}{rank['std']:>14.4f}{pearson['std']:>16.4f}",
        f"{'ICIR':<16}{rank['ir']:>14.4f}{pearson['ir']:>16.4f}",
        f"{'IC > 0 占比':<16}{rank['positive_ratio']:>13.2%}{pearson['positive_ratio']:>15.2%}",
        f"{'t-stat':<16}{rank['t_stat']:>14.4f}{pearson['t_stat']:>16.4f}",
        "",
        f"综合提示：样本内 Rank IC 均值为 {rank['mean']:.4f}，ICIR 为 {rank['ir']:.4f}；请结合样本长度及回测判断，不代表未来表现。",
        "",
        "逐月结果：",
        f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 58,
    ]
    lines.extend(
        f"{row['截面月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}"
        f"{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        results = calculate_monthly_ic(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(results, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("未来 %s 个月 IC 报告已输出：%s", FORWARD_HORIZON_MONTHS, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
