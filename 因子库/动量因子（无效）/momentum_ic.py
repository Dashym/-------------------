import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "动量因子_cleaned.csv"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_前复权.csv"
OUTPUT_FILE = SCRIPT_DIR / "动量因子_IC结果.txt"
FACTOR_VALUE_COLUMN = "动量因子_cleaned"
FACTOR_REQUIRED_COLUMNS = {"股票代码", "截面日期", FACTOR_VALUE_COLUMN}
QUOTE_REQUIRED_COLUMNS = {"股票代码", "日期", "收盘"}
MIN_STOCKS_PER_CROSS_SECTION = 10


def prepare_forward_returns(quotes):
    missing_columns = QUOTE_REQUIRED_COLUMNS.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = data["股票代码"].astype(str).str.strip().str.zfill(6)
    data["日期"] = pd.to_datetime(data["日期"], errors="raise")
    data["收盘"] = pd.to_numeric(data["收盘"], errors="raise")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    data["月份"] = data["日期"].dt.to_period("M")

    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    if data.empty:
        raise ValueError("行情数据中没有有效收盘价")

    month_end = (
        data.sort_values(["股票代码", "日期"])
        .groupby(["月份", "股票代码"], as_index=False)
        .tail(1)
    )
    prices = month_end.pivot(index="月份", columns="股票代码", values="收盘")
    months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(months)
    forward_returns = prices.shift(-1).div(prices).sub(1)
    forward_returns.index.name = "月份"

    return (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="下一月收益")
        .dropna(subset=["下一月收益"])
    )


def calculate_cross_sectional_ic(factor, quotes, min_stocks=MIN_STOCKS_PER_CROSS_SECTION):
    missing_columns = FACTOR_REQUIRED_COLUMNS.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = factor.loc[:, ["股票代码", "截面日期", FACTOR_VALUE_COLUMN]].copy()
    data["股票代码"] = data["股票代码"].astype(str).str.strip().str.zfill(6)
    data["截面日期"] = pd.to_datetime(data["截面日期"], errors="raise")
    data["月份"] = data["截面日期"].dt.to_period("M")
    data[FACTOR_VALUE_COLUMN] = pd.to_numeric(data[FACTOR_VALUE_COLUMN], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", FACTOR_VALUE_COLUMN]
    )

    if data.duplicated(["月份", "股票代码"]).any():
        raise ValueError("因子数据存在重复的月份和股票代码记录")

    forward_returns = prepare_forward_returns(quotes)
    merged = data.merge(forward_returns, on=["月份", "股票代码"], how="inner")

    rows = []
    for month, cross_section in merged.groupby("月份", sort=True):
        if len(cross_section) < min_stocks:
            continue
        factor_values = cross_section[FACTOR_VALUE_COLUMN]
        returns = cross_section["下一月收益"]
        factor_ranks = factor_values.rank(method="average")
        return_ranks = returns.rank(method="average")
        rows.append(
            {
                "截面月份": str(month),
                "截面日期": cross_section["截面日期"].max().strftime("%Y-%m-%d"),
                "股票数": len(cross_section),
                "Rank IC": factor_ranks.corr(return_ranks),
                "Pearson IC": factor_values.corr(returns, method="pearson"),
            }
        )

    return pd.DataFrame(
        rows, columns=["截面月份", "截面日期", "股票数", "Rank IC", "Pearson IC"]
    )


def _ic_statistics(ic_values):
    values = ic_values.dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive_ratio": float("nan"),
            "abs_mean": float("nan"),
            "t_stat": float("nan"),
            "nw_t_stat": float("nan"),
            "nw_lag": 0,
        }

    mean = values.mean()
    standard_deviation = values.std(ddof=1) if len(values) > 1 else float("nan")
    standard_error = standard_deviation / np.sqrt(len(values))
    nw_t_stat_val, _nw_se, nw_lag_val = _newey_west_t_stat(values.values)
    return {
        "count": len(values),
        "mean": mean,
        "std": standard_deviation,
        "ir": mean / standard_deviation
        if np.isfinite(standard_deviation) and standard_deviation > 0
        else float("nan"),
        "positive_ratio": (values > 0).mean(),
        "abs_mean": values.abs().mean(),
        "t_stat": mean / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else float("nan"),
        "nw_t_stat": nw_t_stat_val,
        "nw_lag": nw_lag_val,
    }


def _newey_west_t_stat(values, max_lags=None):
    T = len(values)
    if T < 3:
        return float("nan"), float("nan"), 0
    if max_lags is None:
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    max_lags = min(max_lags, T - 1)
    demeaned = values - values.mean()
    gamma = np.zeros(max_lags + 1)
    for j in range(max_lags + 1):
        gamma[j] = np.sum(demeaned[:T - j] * demeaned[j:]) / T
    weights = 1 - np.arange(1, max_lags + 1) / (max_lags + 1)
    nw_variance = gamma[0] + 2 * np.sum(weights * gamma[1:])
    nw_variance = max(nw_variance, 1e-15)
    nw_se = np.sqrt(nw_variance / T)
    nw_t_stat = values.mean() / nw_se if nw_se > 0 else float("nan")
    return nw_t_stat, nw_se, max_lags


def render_report(ic_results, factor_count, factor_dates, quote_count, quote_dates):
    if ic_results.empty:
        raise ValueError("没有达到最小股票数要求的有效 IC 截面")

    rank = _ic_statistics(ic_results["Rank IC"])
    pearson = _ic_statistics(ic_results["Pearson IC"])
    rank_cumulative = ic_results["Rank IC"].fillna(0).cumsum()
    if abs(rank["mean"]) >= 0.05 and abs(rank["ir"]) >= 0.5:
        conclusion = "Rank IC 显示较强预测关系，可进一步做分层回测验证。"
    elif abs(rank["mean"]) >= 0.03 and abs(rank["ir"]) >= 0.3:
        conclusion = "Rank IC 显示一定预测关系，建议结合分层回测评估稳健性。"
    elif abs(rank["mean"]) >= 0.02:
        conclusion = "Rank IC 信号偏弱，建议结合其他指标或分层回测继续评估。"
    else:
        conclusion = "样本内 Rank IC 较弱；暂不能据此认为因子有稳定预测能力。"

    lines = [
        "=" * 72,
        "动量因子 IC / IR 检测结果",
        "=" * 72,
        "",
        "检测口径：",
        "  因子：动量因子_cleaned（逐月截面去极值并 Z-score 后）",
        "  收益：前复权月末收盘价的下一自然月收益 = 下月月末收盘 / 本月月末收盘 - 1",
        "  IC：每个截面分别计算 Spearman Rank IC 和 Pearson IC",
        f"  截面最少有效股票数：{MIN_STOCKS_PER_CROSS_SECTION}",
        "  IR：各期 IC 均值 / 各期 IC 样本标准差（ddof=1）",
        "  t-stat：各期 IC 均值 / IC 标准误",
        "  NW t-stat：Newey-West 自相关稳健 t 统计量（Bartlett 核，自动滞后阶数）",
        "  注意：没有下一月末行情的最后一个因子截面不会进入统计",
        "",
        "数据概况：",
        f"  清洗因子记录：{factor_count:,} 条，日期范围 {factor_dates[0]} ~ {factor_dates[-1]}",
        f"  前复权行情记录：{quote_count:,} 条，日期范围 {quote_dates[0]} ~ {quote_dates[-1]}",
        f"  有效预测截面：{len(ic_results)} 期，范围 {ic_results['截面月份'].iloc[0]} ~ {ic_results['截面月份'].iloc[-1]}",
        f"  平均有效股票数/截面：{ic_results['股票数'].mean():.1f}，范围 {ic_results['股票数'].min()} ~ {ic_results['股票数'].max()}",
        "",
        f"{'指标':<18}{'Rank IC':>14}{'Pearson IC':>16}",
        "-" * 48,
        f"{'IC 均值':<18}{rank['mean']:>14.4f}{pearson['mean']:>16.4f}",
        f"{'IC 标准差':<18}{rank['std']:>14.4f}{pearson['std']:>16.4f}",
        f"{'ICIR':<18}{rank['ir']:>14.4f}{pearson['ir']:>16.4f}",
        f"{'IC > 0 占比':<18}{rank['positive_ratio']:>13.2%}{pearson['positive_ratio']:>15.2%}",
        f"{'abs(IC) 均值':<18}{rank['abs_mean']:>14.4f}{pearson['abs_mean']:>16.4f}",
        f"{'t-stat':<18}{rank['t_stat']:>14.4f}{pearson['t_stat']:>16.4f}",
        f"{'NW t-stat':<18}{rank['nw_t_stat']:>14.4f}{pearson['nw_t_stat']:>16.4f}",
        f"{'NW 滞后阶数':<18}{rank['nw_lag']:>14d}{pearson['nw_lag']:>16d}",
        "",
        f"累计 Rank IC（逐期求和）：{rank_cumulative.iloc[0]:.4f} -> {rank_cumulative.iloc[-1]:.4f}",
        f"累计 Rank IC 范围：[{rank_cumulative.min():.4f}, {rank_cumulative.max():.4f}]",
        f"综合判断：{conclusion}",
        "",
        "逐月截面结果：",
        f"{'月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 46,
    ]
    lines.extend(
        f"{row['截面月份']:<12}{row['股票数']:>8d}{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in ic_results.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    for input_file in (FACTOR_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    quotes = pd.read_csv(QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    ic_results = calculate_cross_sectional_ic(factor, quotes)
    factor_dates = pd.to_datetime(factor["截面日期"], errors="raise")
    quote_dates = pd.to_datetime(quotes["日期"], errors="raise")
    report = render_report(
        ic_results,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        len(quotes),
        (quote_dates.min().strftime("%Y-%m-%d"), quote_dates.max().strftime("%Y-%m-%d")),
    )

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("IC/IR 文本报告已输出: %s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())