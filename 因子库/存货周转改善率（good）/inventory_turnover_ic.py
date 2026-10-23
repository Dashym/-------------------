import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "存货周转改善率_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
OUTPUT_FILE = SCRIPT_DIR / "存货周转改善率_IC_未来12个月.txt"
FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MIN_CROSS_SECTION_SIZE = 10
FORWARD_HORIZON_MONTHS = 12


def _normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def _monthly_market_data(quotes, adjusted=False):
    required = {"股票代码", "日期", "收盘"}
    if not adjusted:
        required.update({"成交量", "换手率"})
    missing_columns = required.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    columns = ["股票代码", "日期", "收盘"]
    if not adjusted:
        columns.extend(["成交量", "换手率"])
    data = quotes.loc[:, columns].copy()
    data["股票代码"] = _normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    data["月份"] = data["日期"].dt.to_period("M")

    if not adjusted:
        valid_turnover = data.loc[data["换手率"] > 0, "换手率"]
        if valid_turnover.empty:
            raise ValueError("行情中没有有效换手率，无法估算流通市值")
        turnover_scale = 10000 if valid_turnover.max() > 1.5 else 100
        data["流通股本估算"] = (
            data["成交量"] / data["换手率"] * turnover_scale
        )
        data["ln流通市值"] = np.log(
            (data["收盘"] * data["流通股本估算"]).where(
                (data["收盘"] * data["流通股本估算"]) > 0
            )
        )
        data.loc[~np.isfinite(data["ln流通市值"]), "ln流通市值"] = np.nan

    month_end = (
        data.sort_values(["股票代码", "日期"])
        .groupby(["股票代码", "月份"], as_index=False)
        .tail(1)
        .copy()
    )
    return month_end


def calculate_monthly_ic(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
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
    signals = signals.drop_duplicates(
        ["股票代码", "可用日期"], keep="last"
    )

    market = _monthly_market_data(unadjusted_quotes)
    adjusted = _monthly_market_data(adjusted_quotes, adjusted=True)
    market = market.rename(columns={"日期": "市值日期"})
    adjusted_prices = adjusted.pivot(
        index="月份", columns="股票代码", values="收盘"
    )
    all_months = pd.period_range(
        adjusted_prices.index.min(), adjusted_prices.index.max(), freq="M"
    )
    adjusted_prices = adjusted_prices.reindex(all_months)
    forward_returns = (
        adjusted_prices.shift(-12).div(adjusted_prices).sub(1)
    )
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
        if len(cross_section) < min_cross_section:
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
            [
                np.ones(len(cross_section)),
                cross_section["ln流通市值"].to_numpy(dtype=float),
            ]
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
                "截面日期": month_end.strftime("%Y-%m-%d"),
                "预测月份": str(month + FORWARD_HORIZON_MONTHS),
                "股票数": len(cross_section),
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
    cumulative = results["Rank IC"].cumsum()

    lines = [
        "=" * 72,
        "存货周转改善率（TTM同比）IC 检验报告",
        "=" * 72,
        "",
        "检测口径：",
        "  月末截面使用截至该月最后交易日已公告的最新因子值（不使用未来财报）",
        "  对月末截面重新执行 MAD(5倍) 去极值、对数流通市值回归中性化及 Z-score",
        "  收益：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  Rank IC：因子与未来 12 个月累计收益的 Spearman 秩相关；Pearson IC：线性相关",
        f"  最少有效股票数/截面：{MIN_CROSS_SECTION_SIZE}",
        "  IR：月度 IC 均值 / 月度 IC 样本标准差（ddof=1）",
        "  t-stat 未做 Newey-West 等时间序列自相关调整",
        "",
        "数据概况：",
        f"  原始因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（用于流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（用于收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；因子月份 {results['截面月份'].iloc[0]} ~ {results['截面月份'].iloc[-1]}",
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
        f"累计 Rank IC：{cumulative.iloc[0]:.4f} -> {cumulative.iloc[-1]:.4f}",
        f"综合提示：样本内 Rank IC {'为正' if rank['mean'] > 0 else '为负'}，均值 {rank['mean']:.4f}，ICIR {rank['ir']:.4f}；请结合样本长度及后续分层回测判断，不代表未来表现。",
        "",
        "逐月结果：",
        f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 58,
    ]
    lines.extend(
        f"{row['截面月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in results.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
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
    results = calculate_monthly_ic(factor, unadjusted, adjusted)
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
    log.info("IC 检验报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
