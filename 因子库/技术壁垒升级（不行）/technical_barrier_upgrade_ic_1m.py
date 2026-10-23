import logging
import sys

import numpy as np
import pandas as pd

from technical_barrier_test_common import (
    MIN_CROSS_SECTION_SIZE,
    SCRIPT_DIR,
    load_inputs,
    prepare_monthly_cross_sections,
    report_dates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
HORIZON_MONTHS = 1
RETURN_COLUMN = "未来1个月收益"
OUTPUT_FILE = SCRIPT_DIR / "技术壁垒升级_IC_未来1个月.txt"


def calculate_monthly_ic(factor, unadjusted, adjusted):
    sections = prepare_monthly_cross_sections(
        factor,
        unadjusted,
        adjusted,
        horizon_months=HORIZON_MONTHS,
    )
    sections = sections.loc[
        sections["截面月份"].between("2019-01", "2024-12")
    ]
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
                .corr(section[RETURN_COLUMN].rank(method="average")),
                "Pearson IC": section["因子值_zscore"].corr(
                    section[RETURN_COLUMN]
                ),
            }
        )
    return pd.DataFrame(results)


def _summary(values):
    values = values.dropna()
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else float("nan")
    standard_error = std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "ir": mean / std if np.isfinite(std) and std > 0 else float("nan"),
        "positive": (values > 0).mean() if len(values) else float("nan"),
        "t": mean / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else float("nan"),
    }


def _format_summary(label, results):
    rank = _summary(results["Rank IC"])
    pearson = _summary(results["Pearson IC"])
    return [
        label,
        f"  有效月份：{rank['count']}",
        f"  Rank IC：均值={rank['mean']:.4f}，标准差={rank['std']:.4f}，"
        f"ICIR={rank['ir']:.4f}，正值占比={rank['positive']:.2%}，t={rank['t']:.4f}",
        f"  Pearson IC：均值={pearson['mean']:.4f}，标准差={pearson['std']:.4f}，"
        f"ICIR={pearson['ir']:.4f}，正值占比={pearson['positive']:.2%}，t={pearson['t']:.4f}",
    ]


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if results.empty:
        raise ValueError("没有达到最小样本数要求的有效月度 IC 截面")
    periods = [
        ("2020-2022 子样本", results["因子月份"].between("2020-01", "2022-12")),
        ("2023-2024 子样本", results["因子月份"].between("2023-01", "2024-12")),
    ]
    lines = [
        "=" * 76,
        "技术壁垒升级因子 IC 检验报告（未来 1 个月）",
        "=" * 76,
        "",
        "检测口径：",
        "  未来 1 个月累计后复权收益 = 下一月末后复权收盘 / 本月末后复权收盘 - 1",
        "  因子值 = (财务升级Z-score × 高壁垒标签)，并仅在高壁垒标签=1的样本内做截面 Z-score",
        "  高壁垒标签=0的股票因子值固定为 0；财务升级Z-score由毛利率同比变化和营收同比增长率等权合成",
        "  财报信号可用日 = max(首次公告日, 报告期截止日 + 15 天)，月末仅使用可用日不晚于截面日的数据",
        "  TTM 营收和营业成本仅使用信号可用日已到达的连续四个季度数据",
        "  月末截面因子再按 MAD(5倍) 截尾并 Z-score，减少状态依赖分析受极端值影响",
        "  独立子样本：2020-01 至 2022-12、2023-01 至 2024-12；依据因子月份划分",
        "  注意：因 2019 年才开始有财报数据且需要同比 TTM，因子实际记录始于 2021-03；早期月份不会补造数据",
        "  Rank IC 为 Spearman 秩相关；Pearson IC 为线性相关",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "  IC 均值 t-stat 为常规均值检验，未对收益序列相关做 HAC 调整",
        "",
        "全样本统计：",
    ]
    lines.extend(_format_summary(    "全样本（报告期内所有有效截面）", results))
    for label, mask in periods:
        subset = results.loc[mask]
        lines.append("")
        lines.extend(_format_summary(label, subset))
    lines.extend(
        [
            "",
            "数据概况：",
            f"  因子记录：{factor_rows:,} 条；因子文件可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
            f"  不复权行情：{market_dates[0]} ~ {market_dates[1]}",
            f"  后复权行情：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
            f"  检验截面：{len(results)} 期；{results['因子月份'].iloc[0]} ~ {results['因子月份'].iloc[-1]}",
            f"  2020-2022 子样本可用 IC：{results['因子月份'].between('2020-01', '2022-12').sum()} 期；"
            f"2023-2024 子样本可用 IC：{results['因子月份'].between('2023-01', '2024-12').sum()} 期",
            f"  平均有效股票数：{results['样本数'].mean():.1f}，范围 "
            f"{results['样本数'].min()} ~ {results['样本数'].max()}",
            "",
            "逐月结果：",
            f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
            "-" * 58,
        ]
    )
    lines.extend(
        f"{row['因子月份']:<12}{row['预测月份']:<12}{row['样本数']:>8d}"
        f"{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 76, ""])
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
    log.info("未来 %s 个月 IC 报告已输出：%s", HORIZON_MONTHS, OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
