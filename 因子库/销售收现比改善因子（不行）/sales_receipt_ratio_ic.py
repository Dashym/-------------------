# -*- coding: utf-8 -*-
"""销售收现比改善因子 IC / IR 检验（月末截面，T+1 后复权收益）
================================================================
检验设定：
    1. 每月最后一个交易日为截面日 t；t 上使用"该日及之前已公告"的最新因子值
       （财报可用日期 = 公告日之后第一个交易日，故不使用未披露财报）；
    2. 因子形成日上限 2024-11-01、T+1 收益实现日上限 2024-11-02，严禁穿越；
    3. 收益 = 后复权收盘(t+1) / 后复权收盘(t) − 1（T+1 持有，单日）；
    4. 每个月末截面重新执行 MAD(5 倍) 去极值 → 对数流通市值回归中性化 → Z-score；
    5. IC：因子清洗值与 T+1 收益的 Spearman（Rank IC）/ Pearson 相关；
    6. IR = 月度 IC 均值 / 月度 IC 样本标准差（ddof=1）；t 统计量为 IC 均值的 t 检验。

输出：销售收现比改善因子_IC结果.txt

运行环境：aksahre_demo/.venv（pandas / numpy）
    python sales_receipt_ratio_ic.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "销售收现比改善因子_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
OUTPUT_FILE = SCRIPT_DIR / "销售收现比改善因子_IC结果.txt"

FACTOR_COLUMN = "销售收现比改善因子"
MIN_CROSS_SECTION_SIZE = 10
MAX_FACTOR_ASOF = pd.Timestamp("2024-11-01")
MAX_RETURN_DATE = pd.Timestamp("2024-11-02")
LIMIT_MOVE_THRESHOLD = 0.105


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
    """取每月最后一个交易日行情；不复权数据用于估算对数流通市值。"""
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
        data["流通股本估算"] = data["成交量"] / data["换手率"] * turnover_scale
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


def _prepare_signals(factor):
    """因子长表 → 可用日期级信号（同一股票同一可用日期只保留一条）。"""
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
    signals = signals.sort_values(["股票代码", "可用日期"]).drop_duplicates(
        ["股票代码", "可用日期"], keep="last"
    )
    return signals.reset_index(drop=True)


def _next_day_returns(adjusted_quotes):
    """后复权收盘价 → T+1 次日收益（含收益实现日）。"""
    required = {"股票代码", "日期", "收盘"}
    missing_columns = required.difference(adjusted_quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = adjusted_quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = _normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")

    data = data.sort_values(["股票代码", "日期"])
    grouped = data.groupby("股票代码", sort=False)
    data["收益实现日"] = grouped["日期"].shift(-1)
    data["次日收益"] = grouped["收盘"].shift(-1) / data["收盘"] - 1
    data = data.dropna(subset=["收益实现日", "次日收益"])
    return data[["股票代码", "日期", "收益实现日", "次日收益"]]


def _clean_cross_section(cross_section):
    """截面清洗：MAD(5 倍) 去极值 → 对数流通市值回归中性化 → Z-score。"""
    values = cross_section[FACTOR_COLUMN].to_numpy(dtype="float64")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0 and np.isfinite(mad):
        values = np.clip(values, median - 5 * 1.4826 * mad, median + 5 * 1.4826 * mad)

    design = np.column_stack(
        [np.ones(len(values)), cross_section["ln流通市值"].to_numpy(dtype="float64")]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    residual_std = residuals.std(ddof=0)
    if residual_std == 0 or not np.isfinite(residual_std):
        cleaned = np.zeros(len(residuals))
    else:
        cleaned = (residuals - residuals.mean()) / residual_std

    cross_section = cross_section.copy()
    cross_section["因子截尾值"] = values
    cross_section["因子清洗值"] = cleaned
    return cross_section


def _month_cross_section(
    signals,
    market,
    returns,
    month,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
    limit_threshold=LIMIT_MOVE_THRESHOLD,
):
    """构造某个月末截面：点时因子 + 对数流通市值 + T+1 收益，并做截面清洗。"""
    month_market = market.loc[market["月份"] == month]
    if month_market.empty:
        return pd.DataFrame()
    month_end = month_market["日期"].max()
    if month_end > MAX_FACTOR_ASOF:
        return pd.DataFrame()

    as_of = signals.loc[signals["可用日期"] <= month_end]
    if as_of.empty:
        return pd.DataFrame()
    as_of = (
        as_of.sort_values(["股票代码", "可用日期"])
        .drop_duplicates("股票代码", keep="last")
        .rename(columns={"可用日期": "因子可用日期"})
    )

    cross_section = as_of.merge(
        month_market[["股票代码", "ln流通市值"]],
        on="股票代码",
        how="inner",
        validate="one_to_one",
    ).merge(
        returns.loc[
            returns["日期"] == month_end, ["股票代码", "收益实现日", "次日收益"]
        ],
        on="股票代码",
        how="inner",
        validate="one_to_one",
    )
    cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[FACTOR_COLUMN, "ln流通市值", "收益实现日", "次日收益"]
    )
    cross_section = cross_section.loc[cross_section["收益实现日"] <= MAX_RETURN_DATE]
    cross_section = cross_section.loc[
        cross_section["次日收益"].abs() <= limit_threshold
    ]
    if len(cross_section) < min_cross_section:
        return pd.DataFrame()

    cross_section = cross_section.copy()
    cross_section["月份"] = month
    cross_section["截面日"] = month_end
    cross_section = cross_section.sort_values("股票代码").reset_index(drop=True)
    return _clean_cross_section(cross_section)


def _pearson(first, second):
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.sqrt(
        (first_centered ** 2).sum() * (second_centered ** 2).sum()
    )
    if denominator == 0:
        return float("nan")
    return float((first_centered * second_centered).sum() / denominator)


def _spearman(first, second):
    first_ranks = np.argsort(np.argsort(first)).astype("float64") + 1
    second_ranks = np.argsort(np.argsort(second)).astype("float64") + 1
    return _pearson(first_ranks, second_ranks)


def _statistics(values):
    values = pd.Series(values).dropna().to_numpy(dtype="float64")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    ir = mean / std if std > 0 else float("nan")
    if std > 0 and len(values) > 1:
        t_stat = mean / (std / np.sqrt(len(values)))
    else:
        t_stat = float("nan")
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "positive_ratio": float((values > 0).mean()),
        "t_stat": t_stat,
    }


def calculate_monthly_ic(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
    """逐月计算 Rank IC / Pearson IC（T+1 后复权收益）。"""
    signals = _prepare_signals(factor)
    market = _monthly_market_data(unadjusted_quotes)
    returns = _next_day_returns(adjusted_quotes)

    results = []
    for month in sorted(market["月份"].unique()):
        cross_section = _month_cross_section(
            signals, market, returns, month, min_cross_section
        )
        if cross_section.empty:
            continue
        values = cross_section["因子清洗值"].to_numpy(dtype="float64")
        target = cross_section["次日收益"].to_numpy(dtype="float64")
        results.append(
            {
                "截面月份": str(month),
                "截面日": cross_section["截面日"].max().strftime("%Y-%m-%d"),
                "收益实现日": cross_section["收益实现日"].max().strftime("%Y-%m-%d"),
                "最晚因子可用日期": cross_section["因子可用日期"]
                .max()
                .strftime("%Y-%m-%d"),
                "股票数": len(cross_section),
                "Rank IC": _spearman(values, target),
                "Pearson IC": _pearson(values, target),
            }
        )

    if not results:
        raise ValueError("没有满足最小样本数要求的有效月度截面")
    return pd.DataFrame(results)


def render_report(results, factor_rows, factor_dates, market_dates, adjusted_dates):
    rank = _statistics(results["Rank IC"])
    pearson = _statistics(results["Pearson IC"])
    cumulative = results["Rank IC"].cumsum()

    lines = [
        "=" * 76,
        "销售收现比改善因子 IC 检验报告",
        "=" * 76,
        "",
        "检测口径：",
        "  月末截面（每月最后一个交易日）使用截至该日已公告的最新因子值，不使用未来财报",
        f"  因子形成日上限：{MAX_FACTOR_ASOF:%Y-%m-%d}；T+1 收益实现日上限：{MAX_RETURN_DATE:%Y-%m-%d}",
        "  收益：T+1 持有，后复权收益 = 后复权收盘(t+1) / 后复权收盘(t) - 1",
        "  对月末截面重新执行 MAD(5 倍) 去极值、对数流通市值回归中性化及 Z-score",
        f"  涨跌停近似剔除：|T+1 收益| > {LIMIT_MOVE_THRESHOLD:.1%}（视为不可成交）",
        "  Rank IC：因子清洗值与 T+1 收益的 Spearman 秩相关；Pearson IC：线性相关",
        f"  最少有效股票数/截面：{MIN_CROSS_SECTION_SIZE}",
        "  IR：月度 IC 均值 / 月度 IC 样本标准差（ddof=1）；t 统计量为 IC 均值的 t 检验",
        "",
        "数据概况：",
        f"  原始因子：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（用于流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（用于收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效月度截面：{len(results)} 期；因子月份 {results['截面月份'].iloc[0]} ~ {results['截面月份'].iloc[-1]}",
        f"  截面日范围：{results['截面日'].iloc[0]} ~ {results['截面日'].iloc[-1]}",
        f"  收益实现日范围：{results['收益实现日'].iloc[0]} ~ {results['收益实现日'].iloc[-1]}",
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
        f"综合提示：样本内 Rank IC 均值 {rank['mean']:.4f}，ICIR {rank['ir']:.4f}；"
        "请结合样本长度及分层回测判断，不代表未来表现。",
        "",
        "逐月结果：",
        f"{'因子月份':<12}{'截面日':<14}{'收益实现日':<14}{'最晚因子可用日':<16}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}",
        "-" * 92,
    ]
    lines.extend(
        f"{row['截面月份']:<12}{row['截面日']:<14}{row['收益实现日']:<14}"
        f"{row['最晚因子可用日期']:<16}{row['股票数']:>8d}{row['Rank IC']:>12.4f}{row['Pearson IC']:>14.4f}"
        for _, row in results.iterrows()
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
