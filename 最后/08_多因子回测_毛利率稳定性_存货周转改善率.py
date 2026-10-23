# -*- coding: utf-8 -*-
"""毛利率稳定性 + 存货周转改善率（取反）等权合成多因子回测。

策略设定
--------
- 因子 1：毛利率稳定性（Fama-MacBeth beta=+0.0316，t=3.96，p=0.0004，方向为正，越大越好）
- 因子 2：存货周转改善率（Fama-MacBeth beta=-0.0162，t=-2.75，p=0.0085，方向为负，取反后越大越好）
- 合成：两因子各自截面 MAD(5x) 截尾 + Z-score（存货周转改善率额外做对数流通市值中性化），
  等权打分合成 = 0.5 * z(毛利率稳定性) - 0.5 * z(存货周转改善率)
- 回测区间（形成期）：2022-05 ~ 2025-03，每月末调仓
- 选股：复合因子排名前 20%，等权买入
- 持有期：12 个月（重叠持仓，月末调仓，每月形成新队列）
- 基准：等权全样本（样本内全部可交易股票，月度调仓）
- 收益口径：后复权月末收盘，t+1 月末收盘 / t 月末收盘 - 1
- 说明：未计交易成本、无涨跌停不可成交剔除；夏普比率以无风险利率 0 计算。
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "因子库"

GM_FACTOR_FILE = LIB_DIR / "毛利率稳定性（可以，小心）" / "毛利率稳定性_raw.csv"
INV_FACTOR_FILE = LIB_DIR / "存货周转改善率（good）" / "存货周转改善率_raw.csv"
UNADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_不复权_去重排序_剔除异常_格式统一.csv"
)
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)

GM_FACTOR_COLUMN = "毛利率稳定性"
INV_FACTOR_COLUMN = "存货周转改善率（TTM同比）"

MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
MIN_CROSS_SECTION_SIZE = 20
TOP_FRACTION = 0.20
HOLDING_MONTHS = 12
FORMATION_START = pd.Period("2022-05", freq="M")
FORMATION_END = pd.Period("2025-03", freq="M")
GM_WEIGHT = 0.5
INV_WEIGHT = 0.5

OUTPUT_REPORT = BASE_DIR / "多因子回测_结果.txt"
OUTPUT_MONTHLY = BASE_DIR / "多因子回测_月度收益与净值.csv"


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def _monthly_quotes(quotes, need_market_cap):
    columns = ["股票代码", "日期", "收盘"]
    if need_market_cap:
        columns += ["成交量", "换手率"]
    missing = set(columns) - set(quotes.columns)
    if missing:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing))}")

    data = quotes.loc[:, columns].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    data["月份"] = data["日期"].dt.to_period("M")

    if need_market_cap:
        valid_turnover = data.loc[data["换手率"] > 0, "换手率"]
        if valid_turnover.empty:
            raise ValueError("行情中没有有效的正换手率，无法估算流通市值")
        turnover_scale = 10000 if valid_turnover.max() > 1.5 else 100
        shares = pd.Series(np.nan, index=data.index, dtype=float)
        mask = data["换手率"] > 0
        shares[mask] = data.loc[mask, "成交量"] / data.loc[mask, "换手率"] * turnover_scale
        data["流通股本估算"] = shares
        market_value = data["收盘"] * data["流通股本估算"]
        data["ln流通市值"] = np.log(market_value.where(market_value > 0))
        data.loc[~np.isfinite(data["ln流通市值"]), "ln流通市值"] = np.nan

    return (
        data.sort_values(["股票代码", "日期"])
        .groupby(["股票代码", "月份"], as_index=False)
        .tail(1)
        .copy()
    )


def _latest_signals(signals, month_end):
    as_of = signals.loc[signals["可用日期"] <= month_end]
    if as_of.empty:
        return pd.DataFrame(columns=signals.columns)
    return (
        as_of.sort_values(["股票代码", "可用日期"])
        .drop_duplicates("股票代码", keep="last")
        .set_index("股票代码")
    )


def _winsorize(values):
    values = values.astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    if mad > 0 and np.isfinite(mad):
        values = values.clip(
            lower=median - MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad,
            upper=median + MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad,
        )
    return values


def _zscore(values):
    values = values.astype(float)
    standard_deviation = values.std(ddof=0)
    if standard_deviation == 0 or not np.isfinite(standard_deviation):
        return None
    return (values - values.mean()) / standard_deviation


def _winsorize_zscore(values):
    return _zscore(_winsorize(values))


def _neutralize_zscore(values, ln_market_cap):
    values = _winsorize(values)
    frame = pd.DataFrame(
        {"value": values, "ln_mcap": ln_market_cap.astype(float)}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return None
    design = np.column_stack([np.ones(len(frame)), frame["ln_mcap"].to_numpy(dtype=float)])
    coefficients, _, _, _ = np.linalg.lstsq(
        design, frame["value"].to_numpy(dtype=float), rcond=None
    )
    residuals = frame["value"].to_numpy(dtype=float) - design @ coefficients
    standard_deviation = residuals.std(ddof=0)
    if standard_deviation == 0 or not np.isfinite(standard_deviation):
        return None
    return pd.Series(
        (residuals - residuals.mean()) / standard_deviation, index=frame.index
    )


def _load_factor(path, factor_column):
    required = {"股票代码", "可用日期", factor_column}
    factor = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    missing = required - set(factor.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少必要字段: {', '.join(sorted(missing))}")
    signals = factor.loc[:, ["股票代码", "可用日期", factor_column]].copy()
    signals["股票代码"] = normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(signals["可用日期"], errors="raise").dt.normalize()
    signals[factor_column] = pd.to_numeric(signals[factor_column], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", factor_column]
    )
    signals = signals.sort_values(["股票代码", "可用日期"])
    signals = signals.drop_duplicates(["股票代码", "可用日期"], keep="last")
    return signals


def build_cross_sections(gm_signals, inv_signals, ln_mcap, monthly_months):
    """逐月构建复合因子截面并确定选股。

    返回 dict: {month: {"selected": [...], "universe": [...], "n": int, "n_select": int}}
    """
    cross_sections = {}
    formation_months = [m for m in monthly_months if FORMATION_START <= m <= FORMATION_END]

    for month in formation_months:
        month_end = pd.Timestamp(month.end_time).normalize()

        gm_latest = _latest_signals(gm_signals, month_end)
        inv_latest = _latest_signals(inv_signals, month_end)
        common = gm_latest.index.intersection(inv_latest.index)
        if len(common) < MIN_CROSS_SECTION_SIZE:
            continue

        mcap_row = ln_mcap.loc[month].reindex(common) if month in ln_mcap.index else None
        if mcap_row is None:
            continue

        z_gm = _winsorize_zscore(gm_latest.loc[common, GM_FACTOR_COLUMN])
        if z_gm is None:
            continue
        z_inv = _neutralize_zscore(inv_latest.loc[common, INV_FACTOR_COLUMN], mcap_row)
        if z_inv is None:
            continue

        both = z_gm.index.intersection(z_inv.index)
        if len(both) < MIN_CROSS_SECTION_SIZE:
            continue

        composite = GM_WEIGHT * z_gm.loc[both] - INV_WEIGHT * z_inv.loc[both]
        composite = composite.sort_values(ascending=False)
        n_select = max(1, int(np.floor(len(composite) * TOP_FRACTION)))
        selected = list(composite.index[:n_select])

        cross_sections[month] = {
            "selected": selected,
            "universe": list(both),
            "n": len(both),
            "n_select": n_select,
        }
    return cross_sections


def _performance_summary(monthly_series):
    monthly = monthly_series.dropna()
    if monthly.empty:
        return {}
    nav = (1 + monthly).cumprod()
    total_return = nav.iloc[-1] - 1
    n_months = len(monthly)
    annualized_return = (1 + total_return) ** (12 / n_months) - 1
    annualized_vol = monthly.std(ddof=1) * np.sqrt(12)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "nav": nav,
        "n_months": n_months,
    }


def main():
    for path in (GM_FACTOR_FILE, INV_FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not path.is_file():
            log.error("输入文件不存在: %s", path)
            return 1

    log.info("加载因子与行情数据 ...")
    gm_signals = _load_factor(GM_FACTOR_FILE, GM_FACTOR_COLUMN)
    inv_signals = _load_factor(INV_FACTOR_FILE, INV_FACTOR_COLUMN)

    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE,
        usecols=["股票代码", "日期", "收盘", "成交量", "换手率"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE,
        usecols=["股票代码", "日期", "收盘"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )

    market_month = _monthly_quotes(unadjusted, need_market_cap=True)
    adj_month = _monthly_quotes(adjusted, need_market_cap=False)

    prices = adj_month.pivot(index="月份", columns="股票代码", values="收盘")
    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)
    monthly_ret = prices.pct_change()

    ln_mcap = market_month.pivot(index="月份", columns="股票代码", values="ln流通市值")
    ln_mcap = ln_mcap.reindex(all_months)

    cross_sections = build_cross_sections(gm_signals, inv_signals, ln_mcap, all_months)
    if not cross_sections:
        log.error("没有达到最小样本数要求的有效月度截面")
        return 1

    formation_months = sorted(cross_sections.keys())
    full_sample = sorted({code for item in cross_sections.values() for code in item["universe"]})

    cohort_returns = {}
    for month in formation_months:
        selected = cross_sections[month]["selected"]
        start = month + 1
        end = month + HOLDING_MONTHS
        window = monthly_ret.loc[start:end, selected]
        cohort_returns[month] = window.mean(axis=1)

    eval_start = formation_months[0] + 1
    eval_end = formation_months[-1] + HOLDING_MONTHS
    eval_months = pd.period_range(eval_start, eval_end, freq="M")

    strategy_ret = {}
    benchmark_ret = {}
    active_counts = {}
    for month in eval_months:
        month_returns = monthly_ret.loc[month]
        benchmark_ret[month] = float(month_returns.reindex(full_sample).mean())

        values = []
        for form in formation_months:
            if (form + 1) <= month <= (form + HOLDING_MONTHS):
                value = cohort_returns[form].get(month)
                if value is not None and np.isfinite(value):
                    values.append(value)
        active_counts[month] = len(values)
        if values:
            strategy_ret[month] = float(np.mean(values))

    strategy_series = pd.Series(strategy_ret).sort_index()
    benchmark_series = pd.Series(benchmark_ret).sort_index()
    excess_series = strategy_series - benchmark_series

    strategy_stats = _performance_summary(strategy_series)
    benchmark_stats = _performance_summary(benchmark_series)

    monthly_frame = pd.DataFrame(
        {
            "月份": [str(m) for m in eval_months],
            "策略月收益": [strategy_series.get(m, np.nan) for m in eval_months],
            "基准月收益": [benchmark_series.get(m, np.nan) for m in eval_months],
            "超额收益": [excess_series.get(m, np.nan) for m in eval_months],
            "活跃队列数": [active_counts.get(m, 0) for m in eval_months],
        }
    )
    monthly_frame["策略净值"] = (1 + monthly_frame["策略月收益"].fillna(0)).cumprod()
    monthly_frame["基准净值"] = (1 + monthly_frame["基准月收益"].fillna(0)).cumprod()
    monthly_frame.to_csv(OUTPUT_MONTHLY, index=False, encoding="utf-8-sig")

    excess_mean = excess_series.mean()
    excess_std = excess_series.std(ddof=1)
    information_ratio = excess_mean / excess_std * np.sqrt(12) if excess_std > 0 else float("nan")
    tracking_error = excess_std * np.sqrt(12)
    win_rate = (excess_series > 0).mean()
    excess_total = (1 + excess_series).prod() - 1
    excess_annualized = (1 + excess_total) ** (12 / len(excess_series)) - 1

    selected_counts = [cross_sections[m]["n_select"] for m in formation_months]
    universe_counts = [cross_sections[m]["n"] for m in formation_months]

    lines = [
        "=" * 76,
        "毛利率稳定性 + 存货周转改善率（取反）等权合成多因子回测报告",
        "=" * 76,
        "",
        "策略设定：",
        "  因子1 毛利率稳定性：Fama-MacBeth beta=+0.0316（t=3.96, p=0.0004），方向为正",
        "  因子2 存货周转改善率：Fama-MacBeth beta=-0.0162（t=-2.75, p=0.0085），方向为负 → 取反",
        "  合成：两因子截面 MAD(5x) 截尾 + Z-score；存货周转改善率另做对数流通市值中性化",
        "  等权打分：复合得分 = 0.5*z(毛利率稳定性) - 0.5*z(存货周转改善率)",
        "  回测（形成期）：2022-05 ~ 2025-03，每月末调仓",
        "  选股：复合因子排名前 20%，等权买入",
        "  持有期：12 个月（重叠持仓，月末形成新队列）",
        "  基准：等权全样本（样本内全部可交易股票，月度调仓）",
        "  收益：后复权月末收盘，t+1 月末收盘 / t 月末收盘 - 1",
        "  说明：未计交易成本、无涨跌停剔除；夏普以无风险利率 0 计算",
        "",
        "数据概况：",
        f"  毛利率稳定性因子：{len(gm_signals):,} 条，可用日期 {gm_signals['可用日期'].min().strftime('%Y-%m-%d')} ~ {gm_signals['可用日期'].max().strftime('%Y-%m-%d')}",
        f"  存货周转改善率因子：{len(inv_signals):,} 条，可用日期 {inv_signals['可用日期'].min().strftime('%Y-%m-%d')} ~ {inv_signals['可用日期'].max().strftime('%Y-%m-%d')}",
        f"  后复权行情：{prices.index.min()} ~ {prices.index.max()}",
        f"  有效形成截面：{len(formation_months)} 期（{formation_months[0]} ~ {formation_months[-1]}）",
        f"  截面股票数：{min(universe_counts)} ~ {max(universe_counts)}，均值 {np.mean(universe_counts):.1f}",
        f"  每期选股数（前20%）：{min(selected_counts)} ~ {max(selected_counts)}，均值 {np.mean(selected_counts):.1f}",
        f"  全样本股票数：{len(full_sample)}",
        "",
        "绩效对比（收益期 {} ~ {}，共 {} 个月）：".format(
            eval_months[0], eval_months[-1], len(eval_months)
        ),
        f"{'指标':<14}{'策略':>16}{'基准':>16}{'超额':>16}",
        "-" * 62,
        f"{'累计收益':<14}{strategy_stats['total_return']:>16.2%}{benchmark_stats['total_return']:>16.2%}{excess_total:>16.2%}",
        f"{'年化收益':<14}{strategy_stats['annualized_return']:>16.2%}{benchmark_stats['annualized_return']:>16.2%}{excess_annualized:>16.2%}",
        f"{'年化波动率':<14}{strategy_stats['annualized_vol']:>16.2%}{benchmark_stats['annualized_vol']:>16.2%}{tracking_error:>16.2%}",
        f"{'夏普比率':<14}{strategy_stats['sharpe']:>16.2f}{benchmark_stats['sharpe']:>16.2f}{information_ratio:>16.2f}",
        f"{'最大回撤':<14}{strategy_stats['max_drawdown']:>16.2%}{benchmark_stats['max_drawdown']:>16.2%}{'':>16}",
        "",
        f"月度超额收益均值：{excess_mean:.4%}；月度胜率：{win_rate:.2%}",
        f"信息比率（年化）：{information_ratio:.2f}；跟踪误差（年化）：{tracking_error:.2%}",
        "",
        "逐月结果：",
        f"{'月份':<10}{'策略收益':>12}{'基准收益':>12}{'超额收益':>12}{'活跃队列':>10}",
        "-" * 56,
    ]
    for _, row in monthly_frame.iterrows():
        lines.append(
            f"{row['月份']:<10}{row['策略月收益']:>12.2%}{row['基准月收益']:>12.2%}"
            f"{row['超额收益']:>12.2%}{row['活跃队列数']:>10d}"
        )
    lines.append("")
    lines.append("注：首尾各若干月活跃队列不足 12 个，属于 12 个月持有期重叠组合的自然爬坡/收尾。")
    lines.append("最后形成期 2025-03 持有至 2026-03，完整 12 个月，行情数据完全覆盖。")
    lines.append("=" * 76)
    report = "\n".join(lines)

    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    log.info("报告已输出：%s", OUTPUT_REPORT)
    log.info("月度收益与净值已输出：%s", OUTPUT_MONTHLY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
