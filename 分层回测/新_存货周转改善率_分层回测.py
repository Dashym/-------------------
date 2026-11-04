# -*- coding: utf-8 -*-
"""存货周转改善率 单因子分层回测（样本内 + 样本外）

样本内：形成期 2022-05 ~ 2024-02，12个月重叠持有（滚动调仓）
样本外：2025-03月末单期建仓，2025-04 ~ 2026-03 月度追踪（单期持有）

处理流程：5×1.4826 倍 MAD 截尾 → ln(流通市值) 中性化 → Z-score
存货周转改善率为负向因子（改善率越高，未来收益越低）—— 多头 Q1，空头 Q5
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "aksahre_demo"
FACTOR_FILE = (
    SCRIPT_DIR.parent / "因子库" / "存货周转改善率（good）" / "存货周转改善率_raw.csv"
)
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)

FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
MIN_CROSS_SECTION_SIZE = 10
HOLDING_MONTHS = 12
N_GROUPS = 5

IN_SAMPLE_START = pd.Period("2022-05", freq="M")
IN_SAMPLE_END = pd.Period("2024-02", freq="M")
OOS_FORMATION = pd.Period("2025-03", freq="M")
OOS_EVAL_MONTHS = 12
POSITIVE_FACTOR = False

OUTPUT_FILE = SCRIPT_DIR / "新_存货周转改善率_分层回测.txt"
OUTPUT_PNG_INSAMPLE = SCRIPT_DIR / "新_存货周转改善率_分层回测_样本内.png"
OUTPUT_PNG_OOS = SCRIPT_DIR / "新_存货周转改善率_分层回测_样本外.png"


def _normalize_codes(values):
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
    data["股票代码"] = _normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    data["月份"] = data["日期"].dt.to_period("M")
    if need_market_cap:
        valid_turnover = data.loc[data["换手率"] > 0, "换手率"]
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


def _load_factor(path, factor_column):
    factor = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    signals = factor.loc[:, ["股票代码", "可用日期", factor_column]].copy()
    signals["股票代码"] = _normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(signals["可用日期"], errors="raise").dt.normalize()
    signals[factor_column] = pd.to_numeric(signals[factor_column], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", factor_column]
    )
    return signals.sort_values(["股票代码", "可用日期"]).drop_duplicates(
        ["股票代码", "可用日期"], keep="last"
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
    return pd.Series((residuals - residuals.mean()) / standard_deviation, index=frame.index)


def build_sections(signals, ln_mcap, months):
    sections = {}
    for month in months:
        month_end = pd.Timestamp(month.end_time).normalize()
        latest = _latest_signals(signals, month_end)
        if len(latest) < MIN_CROSS_SECTION_SIZE:
            continue
        if month not in ln_mcap.index:
            continue
        mcap_row = ln_mcap.loc[month].reindex(latest.index)
        z = _neutralize_zscore(latest[FACTOR_COLUMN], mcap_row)
        if z is None or len(z) < MIN_CROSS_SECTION_SIZE:
            continue
        sections[month] = z
    return sections


def _group_labels(values, n_groups):
    try:
        return pd.qcut(values, n_groups, labels=False).astype(int)
    except Exception:
        labels = np.floor(values.rank(method="first") / (len(values) + 1) * n_groups)
        return np.clip(labels.astype(int), 0, n_groups - 1)


def layered_backtest(sections, monthly_ret):
    group_monthly = {g: {} for g in range(N_GROUPS)}
    for month, z in sections.items():
        labels = _group_labels(z.astype(float), N_GROUPS)
        for g in range(N_GROUPS):
            codes = z.index[labels == g]
            if len(codes) == 0:
                continue
            window = monthly_ret.loc[month + 1: month + HOLDING_MONTHS].reindex(columns=codes)
            for m, v in window.mean(axis=1).items():
                if np.isfinite(v):
                    group_monthly[g].setdefault(m, []).append(v)
    return {
        g: pd.Series(
            {m: float(np.mean(vals)) for m, vals in group_monthly[g].items()}
        ).sort_index()
        for g in range(N_GROUPS)
    }


def oos_layered_backtest(z_scores, monthly_ret, formation_month, eval_months):
    labels = _group_labels(z_scores.astype(float), N_GROUPS)
    groups = {}
    for g in range(N_GROUPS):
        codes = z_scores.index[labels == g]
        if len(codes) == 0:
            groups[g] = pd.Series(dtype=float)
            continue
        window = monthly_ret.loc[
            formation_month + 1: formation_month + eval_months
        ].reindex(columns=codes)
        groups[g] = window.mean(axis=1)
    return groups


def _perf(series):
    s = series.dropna()
    if s.empty:
        return {"total": float("nan"), "ann": float("nan"), "vol": float("nan"),
                "sharpe": float("nan"), "mdd": float("nan"), "nav": pd.Series(dtype=float)}
    nav = (1 + s).cumprod()
    total = nav.iloc[-1] - 1
    n = len(s)
    ann = (1 + total) ** (12 / n) - 1
    vol = s.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    mdd = (nav / nav.cummax() - 1).min()
    return {"total": total, "ann": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd, "nav": nav}


def _group_table(perf_dict):
    lines = [f"{'分组':<8}{'累计收益':>12}{'年化收益':>12}{'年化波动':>12}{'夏普比':>10}{'最大回撤':>12}"]
    lines.append("-" * 66)
    for g in range(N_GROUPS):
        p = perf_dict[g]
        lines.append(
            f"{'Q'+str(g+1):<8}{p['total']:>12.2%}{p['ann']:>12.2%}{p['vol']:>12.2%}"
            f"{p['sharpe']:>10.2f}{p['mdd']:>12.2%}"
        )
    return lines


def _ls_text(p):
    return f"累计 {p['total']:.2%} | 年化 {p['ann']:.2%} | 夏普 {p['sharpe']:.2f} | 最大回撤 {p['mdd']:.2%}"


def _long_short(groups):
    first = groups[0].dropna()
    last = groups[N_GROUPS - 1].dropna()
    if first.empty or last.empty:
        return pd.Series(dtype=float)
    idx = first.index.union(last.index)
    if POSITIVE_FACTOR:
        return last.reindex(idx) - first.reindex(idx)
    return first.reindex(idx) - last.reindex(idx)


def _bench_series_from_sections(sections, monthly_ret):
    codes = set()
    months = []
    for month, z in sections.items():
        codes.update(z.index)
        months.append(month)
    if not months:
        return pd.Series(dtype=float)
    codes = sorted(codes)
    months = sorted(set(months))
    eval_months = pd.period_range(months[0] + 1, months[-1] + HOLDING_MONTHS, freq="M")
    return pd.Series(
        {m: float(monthly_ret.loc[m].reindex(codes).mean()) for m in eval_months}
    ).sort_index()


def _fmt_range(sections):
    if not sections:
        return "无有效截面"
    ms = sorted(sections.keys())
    return f"{ms[0]} ~ {ms[-1]}（{len(ms)} 期）"


def _monthly_table(groups, bench_series, eval_months):
    lines = [
        f"{'月份':<12}" + "".join(f"{'Q'+str(g+1):>12}" for g in range(N_GROUPS)) + f"{'基准':>12}",
        "-" * (12 + 12 * (N_GROUPS + 1)),
    ]
    for month in eval_months:
        row = f"{str(month):<12}"
        for g in range(N_GROUPS):
            val = groups[g].get(month, np.nan)
            row += f"{val:>12.2%}" if np.isfinite(val) else f"{'nan':>12}"
        bval = bench_series.get(month, np.nan) if bench_series is not None else np.nan
        row += f"{bval:>12.2%}" if np.isfinite(bval) else f"{'nan':>12}"
        lines.append(row)
    return lines


def _render_insample(groups, perf, ls_perf, bench_perf):
    lines = [
        "=" * 74,
        f"【样本内】分层回测  ——  形成期 {IN_SAMPLE_START} ~ {IN_SAMPLE_END}",
        "=" * 74,
        "",
        f"因子处理：5×1.4826 倍 MAD 截尾 → ln(流通市值) 中性化 → Z-score",
        f"因子方向：负向因子（改善率越高，未来收益越低），多头 Q1 / 空头 Q5",
        "",
    ]
    p = bench_perf
    if np.isfinite(p.get("total", np.nan)):
        lines.append(
            f"基准（等权全样本）：累计 {p['total']:.2%} | 年化 {p['ann']:.2%} | "
            f"夏普 {p['sharpe']:.2f} | 最大回撤 {p['mdd']:.2%}"
        )
    lines += [""]
    lines += _group_table(perf)
    lines += ["", f"多空组合（Q1-Q5）：{_ls_text(ls_perf)}", ""]
    return lines


def _render_oos(groups, perf, ls_perf, bench_perf, bench_series, formation_count):
    lines = [
        "=" * 74,
        "【样本外】分层回测  ——  单期建仓 + 月度追踪",
        "=" * 74,
        "",
        f"因子处理：5×1.4826 倍 MAD 截尾 → ln(流通市值) 中性化 → Z-score",
        f"建仓截面：{OOS_FORMATION} 月末，因子值 ≤ 月末的最新增益",
        f"评估区间：{OOS_FORMATION+1} ~ {OOS_FORMATION+OOS_EVAL_MONTHS}",
        f"建仓截面股票数：{formation_count}",
        f"因子方向：负向因子（改善率越高，未来收益越低），多头 Q1 / 空头 Q5",
        "",
    ]
    lines += _group_table(perf)
    lines += ["", f"多空组合（Q1-Q5）：{_ls_text(ls_perf)}", ""]
    lines += ["逐月收益（组内等权）："]
    eval_months = pd.period_range(
        OOS_FORMATION + 1, OOS_FORMATION + OOS_EVAL_MONTHS, freq="M"
    )
    lines += _monthly_table(groups, bench_series, eval_months)
    lines += [""]
    return lines


def _plot_insample(groups, perf, bench_perf, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        colors = ["#2c3e50", "#16a085", "#f1c40f", "#e67e22", "#c0392b"]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        def to_dt(series):
            return pd.DatetimeIndex(series.index.to_timestamp())

        for g in range(N_GROUPS):
            nav = perf[g]["nav"].dropna()
            if nav.empty:
                continue
            ax.plot(to_dt(nav), nav, color=colors[g], lw=1.6, label=f"Q{g+1}")
        bench_nav = bench_perf["nav"].dropna()
        if not bench_nav.empty:
            ax.plot(to_dt(bench_nav), bench_nav, color="gray", ls="--", lw=1.8, label="基准")
        ax.set_title(f"存货周转改善率 样本内分层净值（形成期 {IN_SAMPLE_START} ~ {IN_SAMPLE_END}）", fontsize=12)
        ax.set_ylabel("净值（起点=1）")
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print("已生成:", out_path.name)
    except Exception as exc:
        print("图表生成失败:", exc)


def _plot_oos(groups, perf, bench_perf, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        colors = ["#2c3e50", "#16a085", "#f1c40f", "#e67e22", "#c0392b"]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        def to_dt(series):
            return pd.DatetimeIndex(series.index.to_timestamp())

        for g in range(N_GROUPS):
            nav = perf[g]["nav"].dropna()
            if nav.empty:
                continue
            ax.plot(to_dt(nav), nav, color=colors[g], lw=1.6, label=f"Q{g+1}")
        bench_nav = bench_perf["nav"].dropna()
        if not bench_nav.empty:
            ax.plot(to_dt(bench_nav), bench_nav, color="gray", ls="--", lw=1.8, label="基准")
        ax.set_title("存货周转改善率 样本外分层净值（建仓 2025-04，追踪至 2026-03）", fontsize=12)
        ax.set_ylabel("净值（起点=1）")
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print("已生成:", out_path.name)
    except Exception as exc:
        print("图表生成失败:", exc)


def main():
    for path in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not path.is_file():
            print(f"输入文件不存在: {path}")
            return 1

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    signals = _load_factor(FACTOR_FILE, FACTOR_COLUMN)
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE,
        usecols=["股票代码", "日期", "收盘", "成交量", "换手率"],
        dtype={"股票代码": str}, encoding="utf-8-sig",
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE,
        usecols=["股票代码", "日期", "收盘"],
        dtype={"股票代码": str}, encoding="utf-8-sig",
    )

    market_month = _monthly_quotes(unadjusted, need_market_cap=True)
    adj_month = _monthly_quotes(adjusted, need_market_cap=False)
    prices = adj_month.pivot(index="月份", columns="股票代码", values="收盘")
    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)
    monthly_ret = prices.pct_change()
    ln_mcap = market_month.pivot(index="月份", columns="股票代码", values="ln流通市值")
    ln_mcap = ln_mcap.reindex(all_months)

    last_full_month = all_months[-1] - HOLDING_MONTHS
    months = [m for m in all_months if m <= last_full_month]

    all_sections = build_sections(signals, ln_mcap, months)

    insample_sections = {m: z for m, z in all_sections.items() if IN_SAMPLE_START <= m <= IN_SAMPLE_END}

    print(f"样本内截面: {len(insample_sections)} 期")
    in_groups = layered_backtest(insample_sections, monthly_ret)
    in_perf = {g: _perf(in_groups[g]) for g in range(N_GROUPS)}
    ls_in = _long_short(in_groups)
    ls_in_perf = _perf(ls_in)
    bench_in = _bench_series_from_sections(insample_sections, monthly_ret)
    bench_in_perf = _perf(bench_in)

    oos_month_end = pd.Timestamp(OOS_FORMATION.end_time).normalize()
    oos_latest = _latest_signals(signals, oos_month_end)
    print(f"样本外建仓截面: {len(oos_latest)} 只股票")

    if OOS_FORMATION in ln_mcap.index:
        mcap_row = ln_mcap.loc[OOS_FORMATION].reindex(oos_latest.index)
        oos_z = _neutralize_zscore(oos_latest[FACTOR_COLUMN], mcap_row)
    else:
        oos_z = None
        print("警告: 样本外建仓月无流通市值数据")

    if oos_z is not None and len(oos_z) >= MIN_CROSS_SECTION_SIZE:
        oos_groups = oos_layered_backtest(oos_z, monthly_ret, OOS_FORMATION, OOS_EVAL_MONTHS)
        oos_perf = {g: _perf(oos_groups[g]) for g in range(N_GROUPS)}
        ls_oos = _long_short(oos_groups)
        ls_oos_perf = _perf(ls_oos)
        all_oos_codes = oos_z.index.tolist()
        oos_bench = pd.Series({
            m: float(monthly_ret.loc[m].reindex(all_oos_codes).mean())
            for m in pd.period_range(OOS_FORMATION + 1, OOS_FORMATION + OOS_EVAL_MONTHS, freq="M")
        }).sort_index()
        oos_bench_perf = _perf(oos_bench)
        oos_count = len(oos_z)
    else:
        oos_groups = {}
        oos_perf = {}
        ls_oos_perf = _perf(pd.Series(dtype=float))
        oos_bench_perf = _perf(pd.Series(dtype=float))
        oos_count = 0

    report = [
        "=" * 74,
        "存货周转改善率（TTM同比） 分层回测报告",
        "=" * 74,
        "",
        "回测口径：",
        f"  因子文件：因子库/存货周转改善率（good）/存货周转改善率_raw.csv",
        "  后复权行情（收益）：A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv",
        "  月末截面按「可用日期 ≤ 月末」取最新因子值，排序分 5 组（Q1=最低分 ~ Q5=最高分），组内等权",
        "  因子预处理：5×1.4826 倍 MAD 截尾 → ln(流通市值) 中性化 → Z-score",
        f"  样本内：形成期 {IN_SAMPLE_START} ~ {IN_SAMPLE_END}，12 个月重叠持有（月末调仓）",
        f"  样本外：{OOS_FORMATION} 月末单期建仓，{OOS_FORMATION+1} ~ {OOS_FORMATION+OOS_EVAL_MONTHS} 月度追踪",
        "  未计交易成本、无涨跌停剔除；夏普以无风险利率 0 计算",
        "",
    ]
    report += _render_insample(in_groups, in_perf, ls_in_perf, bench_in_perf)
    report += ["", ""]
    report += _render_oos(oos_groups, oos_perf, ls_oos_perf, oos_bench_perf, oos_bench, oos_count)
    report += ["", "=" * 74]

    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    try:
        _plot_insample(in_groups, in_perf, bench_in_perf, OUTPUT_PNG_INSAMPLE)
        _plot_oos(oos_groups, oos_perf, oos_bench_perf, OUTPUT_PNG_OOS)
    except Exception as exc:
        print("图表生成失败:", exc)

    print(f"\n已输出: {OUTPUT_FILE.name} | {OUTPUT_PNG_INSAMPLE.name} | {OUTPUT_PNG_OOS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())