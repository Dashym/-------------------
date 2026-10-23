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
)
from inventory_turnover_regression_12m import (
    HAC_MAX_LAGS,
    _coefficient_summary,
    _significance,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "存货周转改善率_单调性检验_未来12个月.txt"
FORWARD_HORIZON_MONTHS = 12
QUINTILE_COUNT = 5


def calculate_monthly_quintile_returns(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
    quintile_count=QUINTILE_COUNT,
):
    if quintile_count < 2:
        raise ValueError("分组数必须至少为 2")
    required = {"股票代码", "可用日期", FACTOR_COLUMN}
    missing_columns = required.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")

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
    prices = adjusted.pivot(index="月份", columns="股票代码", values="收盘")
    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)
    forward_returns = prices.shift(-FORWARD_HORIZON_MONTHS).div(prices).sub(1)
    forward_returns.index.name = "月份"
    forward_returns = (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="未来12个月收益")
        .dropna(subset=["未来12个月收益"])
    )

    results = []
    for month in sorted(market["月份"].unique()):
        month_end = market.loc[market["月份"] == month, "日期"].max()
        as_of = signals.loc[signals["可用日期"] <= month_end]
        if as_of.empty:
            continue
        as_of = (
            as_of.sort_values(["股票代码", "可用日期"])
            .drop_duplicates("股票代码", keep="last")
        )
        cross_section = as_of.merge(
            market.loc[
                market["月份"] == month,
                ["股票代码", "月份", "ln流通市值"],
            ],
            on="股票代码",
            how="inner",
            validate="one_to_one",
        ).merge(
            forward_returns.loc[forward_returns["月份"] == month],
            on=["股票代码", "月份"],
            how="inner",
            validate="one_to_one",
        )
        cross_section = (
            cross_section.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[FACTOR_COLUMN, "ln流通市值", "未来12个月收益"])
            .sort_values("股票代码")
        )
        if len(cross_section) < max(min_cross_section, quintile_count):
            continue

        values = cross_section[FACTOR_COLUMN]
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            values = values.clip(
                lower=median - 5 * 1.4826 * mad,
                upper=median + 5 * 1.4826 * mad,
            )

        size = cross_section["ln流通市值"].to_numpy(dtype=float)
        neutral_design = np.column_stack([np.ones(len(cross_section)), size])
        neutral_coefficients, _, _, _ = np.linalg.lstsq(
            neutral_design, values.to_numpy(dtype=float), rcond=None
        )
        residuals = values.to_numpy(dtype=float) - neutral_design @ neutral_coefficients
        residual_std = residuals.std(ddof=0)
        if residual_std == 0 or not np.isfinite(residual_std):
            continue

        cross_section["分组"] = np.ceil(
            pd.Series(residuals).rank(method="average").to_numpy()
            * quintile_count
            / len(cross_section)
        ).astype(int)
        cross_section["分组"] = cross_section["分组"].clip(
            lower=1, upper=quintile_count
        )
        group_counts = cross_section.groupby("分组").size()
        if len(group_counts) != quintile_count:
            continue

        row = {
            "截面月份": str(month),
            "截面日期": month_end.strftime("%Y-%m-%d"),
            "预测月份": str(month + FORWARD_HORIZON_MONTHS),
            "股票数": len(cross_section),
        }
        for group in range(1, quintile_count + 1):
            group_data = cross_section.loc[cross_section["分组"] == group]
            row[f"Q{group}股票数"] = len(group_data)
            row[f"Q{group}收益"] = group_data["未来12个月收益"].mean()
        row["Q5-Q1收益"] = row["Q5收益"] - row["Q1收益"]
        results.append(row)

    columns = ["截面月份", "截面日期", "预测月份", "股票数"]
    for group in range(1, quintile_count + 1):
        columns.extend([f"Q{group}股票数", f"Q{group}收益"])
    columns.append("Q5-Q1收益")
    return pd.DataFrame(results, columns=columns)


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    if len(results) <= HAC_MAX_LAGS:
        raise ValueError(
            f"有效分组截面需多于 HAC 滞后期数 {HAC_MAX_LAGS}，当前为 {len(results)} 期"
        )

    group_summaries = {
        group: _coefficient_summary(results[f"Q{group}收益"])
        for group in range(1, QUINTILE_COUNT + 1)
    }
    spread = _coefficient_summary(results["Q5-Q1收益"])
    group_means = [group_summaries[group]["mean"] for group in range(1, 6)]
    increasing = all(
        left < right for left, right in zip(group_means, group_means[1:])
    )
    decreasing = all(
        left > right for left, right in zip(group_means, group_means[1:])
    )
    if increasing:
        monotonicity = "分组平均收益随因子值严格递增"
    elif decreasing:
        monotonicity = "分组平均收益随因子值严格递减"
    else:
        monotonicity = "分组平均收益未呈现严格单调变化"

    lines = [
        "=" * 76,
        "存货周转改善率（TTM同比）五分组单调性检验报告",
        "=" * 76,
        "",
        "检验设定：",
        "  每月末使用当时已公告的最新因子值；按因子值升序分为五组（Q1最低，Q5最高）",
        "  月末截面执行 MAD(5倍) 去极值、对数流通市值回归中性化；中性化残差用于排序",
        "  各组收益：组内等权平均未来 12 个月累计后复权收益",
        "  收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}；每组必须至少包含 1 只股票",
        f"  显著性检验：月度组收益均值及 Q5-Q1 收益差采用 Newey-West HAC，Bartlett 核，滞后 {HAC_MAX_LAGS}",
        "  t 统计量的 p 值采用双侧 Student-t，df=有效月度截面数-1",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；{results['截面月份'].iloc[0]} ~ {results['截面月份'].iloc[-1]}",
        f"  平均有效股票数：{results['股票数'].mean():.1f}",
        "",
        "各组未来 12 个月收益（组间等权；对月度组收益时间序列取均值）：",
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
            f"t={spread['t_stat']:.4f}，p={spread['p_value']:.4f} "
            f"{_significance(spread['p_value'])}",
            f"单调性判断：{monotonicity}。",
            "",
            "逐月分组结果（收益为未来12个月累计收益）：",
            f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}"
            f"{'Q1':>12}{'Q2':>12}{'Q3':>12}{'Q4':>12}{'Q5':>12}{'Q5-Q1':>12}",
            "-" * 92,
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"{row['截面月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}"
            + "".join(f"{row[f'Q{group}收益']:>12.2%}" for group in range(1, 6))
            + f"{row['Q5-Q1收益']:>12.2%}"
        )
    lines.extend(["", "=" * 76, ""])
    return "\n".join(lines)


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
    results = calculate_monthly_quintile_returns(factor, unadjusted, adjusted)
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")
    report = render_report(
        results,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        (market_dates.min().strftime("%Y-%m-%d"), market_dates.max().strftime("%Y-%m-%d")),
        (
            adjusted_dates.min().strftime("%Y-%m-%d"),
            adjusted_dates.max().strftime("%Y-%m-%d"),
        ),
    )
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("未来 12 个月单调性报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
