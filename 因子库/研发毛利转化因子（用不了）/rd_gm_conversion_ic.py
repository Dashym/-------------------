# -*- coding: utf-8 -*-
"""研发-毛利转化因子 IC 检验（季度截面，未来 1 年收益）。"""

import logging
import sys

import numpy as np
import pandas as pd

from rd_gm_test_common import (
    FACTOR_COLUMN,
    MIN_CROSS_SECTION_SIZE,
    RETURN_COLUMN,
    SCRIPT_DIR,
    load_inputs,
    prepare_quarterly_cross_sections,
    report_dates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
OUTPUT_FILE = SCRIPT_DIR / "研发毛利转化因子_IC结果.txt"


def calculate_quarterly_ic(factor, unadjusted, adjusted):
    sections = prepare_quarterly_cross_sections(factor, unadjusted, adjusted)
    results = []
    for quarter, section in sections.groupby("截面季度", sort=True):
        results.append(
            {
                "截面季度": quarter,
                "截面日期": section["截面日期"].iloc[0],
                "预测季度": section["预测季度"].iloc[0],
                "股票数": len(section),
                "Rank IC": section[FACTOR_COLUMN].rank(method="average").corr(
                    section[RETURN_COLUMN].rank(method="average")
                ),
                "Pearson IC": section[FACTOR_COLUMN].corr(section[RETURN_COLUMN]),
            }
        )
    return pd.DataFrame(results)


def _summary(values):
    values = values.dropna()
    if values.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive": float("nan"),
            "t": float("nan"),
        }
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
        raise ValueError("没有达到最小样本数要求的有效季度 IC 截面")
    rank = _summary(results["Rank IC"])
    pearson = _summary(results["Pearson IC"])
    lines = [
        "=" * 72,
        "研发-毛利转化因子 IC 检验报告",
        "=" * 72,
        "",
        "检测口径：",
        "  因子：基础连续因子（研发强度百分位 × 毛利率百分位，行业中性化并 1%/99% 缩尾）",
        "  截面日 = 季度末最后一个交易日，使用截至该日已公告的最新因子值（不使用未来财报）",
        "  收益：未来 1 年（4 个季度）累计后复权收益 = t+4 季度末后复权收盘 / t 季度末后复权收盘 - 1",
        "  收益率已按申万三级行业中位数去均值，并与因子值做 1%/99% 双缩尾",
        "  Rank IC：Spearman 秩相关；Pearson IC：线性相关",
        f"  最少有效股票数/截面：{MIN_CROSS_SECTION_SIZE}",
        "  IC 的 t-stat 为常规均值检验",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效季度截面：{len(results)} 期；{results['截面季度'].iloc[0]} ~ {results['截面季度'].iloc[-1]}",
        f"  平均有效股票数：{results['股票数'].mean():.1f}，范围 {results['股票数'].min()} ~ {results['股票数'].max()}",
        "",
        f"{'指标':<16}{'Rank IC':>14}{'Pearson IC':>16}",
        "-" * 46,
        f"{'IC 均值':<16}{rank['mean']:>14.4f}{pearson['mean']:>16.4f}",
        f"{'IC 标准差':<16}{rank['std']:>14.4f}{pearson['std']:>16.4f}",
        f"{'ICIR':<16}{rank['ir']:>14.4f}{pearson['ir']:>16.4f}",
        f"{'IC > 0 占比':<16}{rank['positive']:>13.2%}{pearson['positive']:>15.2%}",
        f"{'t-stat':<16}{rank['t']:>14.4f}{pearson['t']:>16.4f}",
        "",
        f"Rank IC 均值={rank['mean']:.4f}，ICIR={rank['ir']:.4f}（目标 > 0.5）；",
        "请结合样本长度及分组/Logit 检验判断，不代表未来表现。",
        "",
        "逐季结果：",
        f"{'截面季度':<12}{'预测季度':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 58,
    ]
    lines.extend(
        f"{row['截面季度']:<12}{row['预测季度']:<12}{row['股票数']:>8d}"
        f"{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    try:
        factor, unadjusted, adjusted = load_inputs()
        results = calculate_quarterly_ic(factor, unadjusted, adjusted)
        dates = report_dates(factor, unadjusted, adjusted)
        report = render_report(results, len(factor), *dates)
        OUTPUT_FILE.write_text(report, encoding="utf-8")
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 1
    print(report, end="")
    log.info("IC 报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
