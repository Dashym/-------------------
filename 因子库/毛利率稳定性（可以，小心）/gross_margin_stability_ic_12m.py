import logging
import sys

import numpy as np
import pandas as pd

from stability_test_common import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_FILE,
    FORWARD_HORIZON_MONTHS,
    MIN_CROSS_SECTION_SIZE,
    SCRIPT_DIR,
    UNADJUSTED_QUOTE_FILE,
    load_inputs,
    prepare_monthly_cross_sections,
    report_dates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "毛利率稳定性_IC_未来12个月.txt"


def calculate_monthly_ic(factor, unadjusted, adjusted):
    sections = prepare_monthly_cross_sections(factor, unadjusted, adjusted)
    results = []
    for month, section in sections.groupby("截面月份", sort=True):
        results.append(
            {
                "因子月份": month,
                "预测月份": section["预测月份"].iloc[0],
                "截面日期": section["截面日期"].iloc[0],
                "样本数": len(section),
                "Rank IC": section["因子值_zscore"]
                .rank(method="average")
                .corr(section["未来12个月收益"].rank(method="average")),
                "Pearson IC": section["因子值_zscore"].corr(
                    section["未来12个月收益"]
                ),
            }
        )
    return pd.DataFrame(results)


def _summary(values):
    values = values.dropna()
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else float("nan")
    se = std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "ir": mean / std if np.isfinite(std) and std > 0 else float("nan"),
        "positive": (values > 0).mean(),
        "t": mean / se if np.isfinite(se) and se > 0 else float("nan"),
    }


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if results.empty:
        raise ValueError("没有达到最小样本数要求的有效月度 IC 截面")
    rank = _summary(results["Rank IC"])
    pearson = _summary(results["Pearson IC"])
    lines = [
        "=" * 72,
        "毛利率稳定性因子 IC 检验报告",
        "=" * 72,
        "",
        "检测口径：",
        "  月末使用当时已可用的最新毛利率稳定性得分",
        "  因子为行业内 z_sigma、z_cv、z_drift 按 0.4/0.3/0.3 加权后的负值",
        "  子指标来自最近 12 季度的毛利率样本标准差、变异系数及至少 5 对同比绝对变化均值",
        "  原始毛利率=(营业收入-营业成本)/营业收入；季度序列线性插值，按同行季度 1%/99% 缩尾",
        "  行业截面为财报数据覆盖的汽车零部件股票；同行数据未齐的估值延后至所需财报公告后生效",
        "  月末截面再按 MAD(5倍) 截尾并 Z-score；不做市值中性化",
        "  收益：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  Rank IC 为 Spearman 秩相关；Pearson IC 为线性相关",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "  IC 的 t-stat 为常规均值检验，未对重叠收益导致的序列相关做 HAC 调整",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；{results['因子月份'].iloc[0]} ~ {results['因子月份'].iloc[-1]}",
        f"  平均有效股票数：{results['样本数'].mean():.1f}，范围 {results['样本数'].min()} ~ {results['样本数'].max()}",
        "",
        f"{'指标':<16}{'Rank IC':>14}{'Pearson IC':>16}",
        "-" * 46,
        f"{'IC 均值':<16}{rank['mean']:>14.4f}{pearson['mean']:>16.4f}",
        f"{'IC 标准差':<16}{rank['std']:>14.4f}{pearson['std']:>16.4f}",
        f"{'ICIR':<16}{rank['ir']:>14.4f}{pearson['ir']:>16.4f}",
        f"{'IC > 0 占比':<16}{rank['positive']:>13.2%}{pearson['positive']:>15.2%}",
        f"{'常规 t-stat':<16}{rank['t']:>14.4f}{pearson['t']:>16.4f}",
        "",
        f"Rank IC 均值={rank['mean']:.4f}，ICIR={rank['ir']:.4f}；请结合样本长度及其他检验判断，不代表未来表现。",
        "",
        "逐月结果：",
        f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 58,
    ]
    lines.extend(
        f"{row['因子月份']:<12}{row['预测月份']:<12}{row['样本数']:>8d}"
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
