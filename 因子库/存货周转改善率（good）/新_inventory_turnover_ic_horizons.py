# -*- coding: utf-8 -*-
"""存货周转改善率 IC 检验 · 多持有期对比（样本内，形成期 ≤ 2024-03）。"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_turnover_ic import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_COLUMN,
    FACTOR_FILE,
    MIN_CROSS_SECTION_SIZE,
    UNADJUSTED_QUOTE_FILE,
    _monthly_market_data,
    _normalize_codes,
    _statistics,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
IN_SAMPLE_END = pd.Period("2024-03", freq="M")
HORIZONS = [1, 3, 6, 12]
OUTPUT_FILE = SCRIPT_DIR / "新_存货周转改善率_IC_多期限对比.txt"


def calculate_ic_for_horizon(factor, unadjusted_quotes, adjusted_quotes, horizon):
    required = {"股票代码", "可用日期", FACTOR_COLUMN}
    missing = required.difference(factor.columns)
    if missing:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing))}")

    signals = factor.loc[:, ["股票代码", "可用日期", FACTOR_COLUMN]].copy()
    signals["股票代码"] = _normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(
        signals["可用日期"], errors="raise"
    ).dt.normalize()
    signals[FACTOR_COLUMN] = pd.to_numeric(signals[FACTOR_COLUMN], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", FACTOR_COLUMN]
    )
    signals = signals.sort_values(["股票代码", "可用日期"])
    signals = signals.drop_duplicates(["股票代码", "可用日期"], keep="last")

    market = _monthly_market_data(unadjusted_quotes)
    adjusted = _monthly_market_data(adjusted_quotes, adjusted=True)
    market = market.rename(columns={"日期": "市值日期"})
    adjusted_prices = adjusted.pivot(index="月份", columns="股票代码", values="收盘")
    all_months = pd.period_range(
        adjusted_prices.index.min(), adjusted_prices.index.max(), freq="M"
    )
    adjusted_prices = adjusted_prices.reindex(all_months)
    forward_returns = adjusted_prices.shift(-horizon).div(adjusted_prices).sub(1)
    forward_returns.index.name = "月份"
    forward_returns = (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="持有期收益")
        .dropna(subset=["持有期收益"])
    )

    market_months = sorted(market["月份"].unique())
    results = []
    for month in market_months:
        if month > IN_SAMPLE_END:
            continue
        month_end = market.loc[market["月份"] == month, "市值日期"].max()
        as_of = signals.loc[signals["可用日期"] <= month_end]
        if as_of.empty:
            continue
        as_of = (
            as_of.sort_values(["股票代码", "可用日期"])
            .drop_duplicates("股票代码", keep="last")
        )
        current_market = market.loc[market["月份"] == month]
        current_returns = forward_returns.loc[forward_returns["月份"] == month]
        cross_section = as_of.merge(
            current_market[["股票代码", "月份", "ln流通市值"]],
            on="股票代码",
            how="inner",
            validate="one_to_one",
        ).merge(
            current_returns[["股票代码", "月份", "持有期收益"]],
            on=["股票代码", "月份"],
            how="inner",
            validate="one_to_one",
        )
        cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[FACTOR_COLUMN, "ln流通市值", "持有期收益"]
        )
        if len(cross_section) < MIN_CROSS_SECTION_SIZE:
            continue

        values = cross_section[FACTOR_COLUMN]
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            values = values.clip(
                lower=median - 5 * 1.4826 * mad,
                upper=median + 5 * 1.4826 * mad,
            )
        design = np.column_stack(
            [np.ones(len(cross_section)), cross_section["ln流通市值"].to_numpy(dtype=float)]
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, values.to_numpy(dtype=float), rcond=None
        )
        residuals = values.to_numpy(dtype=float) - design @ coefficients
        residual_std = residuals.std(ddof=0)
        if residual_std == 0 or not np.isfinite(residual_std):
            continue
        standardized = (residuals - residuals.mean()) / residual_std
        returns = cross_section["持有期收益"].to_numpy(dtype=float)
        rank_ic = pd.Series(standardized).rank(method="average").corr(
            pd.Series(returns).rank(method="average")
        )
        pearson_ic = pd.Series(standardized).corr(pd.Series(returns))
        results.append(
            {
                "截面月份": str(month),
                "预测月份": str(month + horizon),
                "股票数": len(cross_section),
                "Rank IC": rank_ic,
                "Pearson IC": pearson_ic,
            }
        )
    return pd.DataFrame(results)


def _judge(rank_icir):
    magnitude = abs(rank_icir)
    if magnitude < 0.1:
        return "无效"
    if magnitude < 0.3:
        return "弱有效"
    return "有效"


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

    lines = [
        "=" * 104,
        "存货周转改善率 IC 检验 · 多持有期对比（样本内）",
        "=" * 104,
        "",
        "口径说明：",
        "  样本内截止：形成期 ≤ 2024-03（未来收益最远到 2025-03，不碰 2025-04 起的样本外）",
        "  月末截面使用截至该月最后交易日已公告的最新因子值（不使用未来财报）",
        "  对月末截面重新执行 MAD(5倍) 去极值、对数流通市值回归中性化及 Z-score",
        "  收益：未来 N 个月累计后复权收益 = t+N 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  Rank IC：Spearman 秩相关；Pearson IC：线性相关；t-stat 未做 HAC 调整",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "",
        "持有期     有效截面数     RankIC均值     RankICIR    Rank t-stat    PearsonIC均值    PearsonICIR    Pearson t-stat    综合判断",
        "-" * 104,
    ]

    for horizon in HORIZONS:
        results = calculate_ic_for_horizon(factor, unadjusted, adjusted, horizon)
        rank = _statistics(results["Rank IC"])
        pearson = _statistics(results["Pearson IC"])
        judge = _judge(rank["ir"])
        lines.append(
            f"{horizon}个月      {len(results):>6d}     "
            f"{rank['mean']:>9.4f}     {rank['ir']:>8.4f}     {rank['t_stat']:>10.4f}     "
            f"{pearson['mean']:>13.4f}     {pearson['ir']:>10.4f}     {pearson['t_stat']:>13.4f}     {judge:>8}"
        )

    lines.extend(
        [
            "-" * 104,
            "",
            "综合判断标准：按 Rank ICIR 绝对值分档（<0.1 无效；0.1~0.3 弱有效；≥0.3 有效）。",
            "",
            "=" * 104,
        ]
    )

    report = "\n".join(lines)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    log.info("多持有期 IC 对比报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
