# -*- coding: utf-8 -*-
"""因子分层回测对比：毛利率稳定性 vs 存货周转改善率（因子库口径）。

以因子库 IC 检验口径为基准：
  - 因子文件：因子库 raw（毛利率稳定性_raw.csv / 存货周转改善率_raw.csv）
  - 不复权行情（流通市值）：A股日线_20190101-20260331_不复权.csv（根目录）
  - 后复权行情（收益）：clean_temp\...后复权_去重排序_剔除异常_格式统一.csv
  - 月末截面 asof 取「可用日期 <= 月末」的最新因子值
  - 存货周转改善率：MAD(5x) 截尾 -> ln流通市值中性化 -> Z-score
  - 毛利率稳定性：MAD(5x) 截尾 -> Z-score（不做市值中性化）
  - 分 5 组、组内等权、12 个月重叠持有
  - 样本内形成期 <= 2024-12；样本外形成期 >= 2025-01
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\Heyuanhang\Desktop\aksahre_demo")
LIB_DIR = BASE_DIR.parent / "因子库"
GM_FACTOR_FILE = LIB_DIR / "毛利率稳定性（可以，小心）" / "毛利率稳定性_raw.csv"
INV_FACTOR_FILE = LIB_DIR / "存货周转改善率（good）" / "存货周转改善率_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
DST = Path(r"C:\Users\Heyuanhang\Desktop\分层回测")

GM_FACTOR_COLUMN = "毛利率稳定性"
INV_FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
MIN_CROSS_SECTION_SIZE = 10
HOLDING_MONTHS = 12
N_GROUPS = 5

IN_SAMPLE_END = pd.Period("2024-12", freq="M")
OOS_START = pd.Period("2025-01", freq="M")

OUTPUT_REPORT = DST / "因子分层回测对比.txt"
OUTPUT_PNG_INSAMPLE = DST / "因子分层回测_净值图_样本内.png"
OUTPUT_PNG_OOS = DST / "因子分层回测_净值图_样本外.png"


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


def _load_factor(path, factor_column):
    factor = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    signals = factor.loc[:, ["股票代码", "可用日期", factor_column]].copy()
    signals["股票代码"] = _normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(signals["可用日期"], errors="raise").dt.normalize()
    signals[factor_column] = pd.to_numeric(signals[factor_column], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", factor_column]
    )
    signals = signals.sort_values(["股票代码", "可用日期"])
    signals = signals.drop_duplicates(["股票代码", "可用日期"], keep="last")
    return signals


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
    design = np.column_stack(
        [np.ones(len(frame)), frame["ln_mcap"].to_numpy(dtype=float)]
    )
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


def build_sections(signals, factor_column, ln_mcap, months, neutralize):
    sections = {}
    for month in months:
        month_end = pd.Timestamp(month.end_time).normalize()
        latest = _latest_signals(signals, month_end)
        if len(latest) < MIN_CROSS_SECTION_SIZE:
            continue
        if neutralize:
            if month not in ln_mcap.index:
                continue
            mcap_row = ln_mcap.loc[month].reindex(latest.index)
            z = _neutralize_zscore(latest[factor_column], mcap_row)
        else:
            z = _winsorize_zscore(latest[factor_column])
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


def layered_backtest(sections, monthly_ret, n_groups=N_GROUPS):
    group_monthly = {g: {} for g in range(n_groups)}
    for month, z in sections.items():
        labels = _group_labels(z.astype(float), n_groups)
        for g in range(n_groups):
            codes = z.index[labels == g]
            if len(codes) == 0:
                continue
            window = monthly_ret.loc[month + 1: month + HOLDING_MONTHS].reindex(columns=codes)
            for m, v in window.mean(axis=1).items():
                if np.isfinite(v):
                    group_monthly[g].setdefault(m, []).append(v)
    groups = {
        g: pd.Series(
            {m: float(np.mean(vals)) for m, vals in group_monthly[g].items()}
        ).sort_index()
        for g in range(n_groups)
    }
    return groups


def _perf(series):
    s = series.dropna()
    if s.empty:
        return {
            "total": float("nan"), "ann": float("nan"), "vol": float("nan"),
            "sharpe": float("nan"), "mdd": float("nan"),
            "nav": pd.Series(dtype=float),
        }
    nav = (1 + s).cumprod()
    total = nav.iloc[-1] - 1
    n = len(s)
    ann = (1 + total) ** (12 / n) - 1
    vol = s.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    mdd = (nav / nav.cummax() - 1).min()
    return {"total": total, "ann": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd, "nav": nav}


def _group_table(perf_dict):
    lines = [f"{'分组':<8}{'累计收益':>12}{'年化收益':>12}{'夏普':>10}{'最大回撤':>12}"]
    lines.append("-" * 54)
    for g in range(N_GROUPS):
        p = perf_dict[g]
        lines.append(
            f"{'Q'+str(g+1):<8}{p['total']:>12.2%}{p['ann']:>12.2%}{p['sharpe']:>10.2f}{p['mdd']:>12.2%}"
        )
    return lines


def _ls_text(p):
    return f"累计 {p['total']:.2%} | 年化 {p['ann']:.2%} | 夏普 {p['sharpe']:.2f} | 回撤 {p['mdd']:.2%}"


def _long_short(groups, positive):
    first = groups[0].dropna()
    last = groups[4].dropna()
    if first.empty or last.empty:
        return pd.Series(dtype=float)
    idx = first.index.union(last.index)
    if positive:
        return last.reindex(idx) - first.reindex(idx)
    return first.reindex(idx) - last.reindex(idx)


def _bench_series(sections_by_factor, monthly_ret):
    codes = set()
    months = []
    for sections in sections_by_factor.values():
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


def _render_section(title, gm_groups, gm_perf, ls_gm, inv_groups, inv_perf, ls_inv, bench_perf):
    lines = [title, ""]
    if np.isfinite(bench_perf["total"]):
        lines.append(
            "  基准：等权全样本（累计 %.2f%% / 年化 %.2f%% / 夏普 %.2f）"
            % (bench_perf["total"] * 100, bench_perf["ann"] * 100, bench_perf["sharpe"])
        )
    else:
        lines.append("  基准：等权全样本（无有效数据）")
    lines += ["", "一、毛利率稳定性（正向因子，越高越稳定；多头 Q5、空头 Q1）", ""]
    lines += _group_table(gm_perf)
    lines += ["", f"多空组合（Q5-Q1）：{_ls_text(ls_gm)}", ""]
    lines += ["二、存货周转改善率（负向因子，改善率越高未来收益越低；多头 Q1、空头 Q5）", ""]
    lines += _group_table(inv_perf)
    lines += ["", f"多空组合（Q1-Q5）：{_ls_text(ls_inv)}", ""]
    lines += [
        "三、对比结论",
        "",
        f"  1) 毛利率稳定性：Q1 累计 {gm_perf[0]['total']:.2%}，Q5 累计 {gm_perf[4]['total']:.2%}，"
        f"多空(Q5-Q1)累计 {ls_gm['total']:.2%}、夏普 {ls_gm['sharpe']:.2f} —— 正向单调性"
        f"{'成立' if ls_gm['total'] > 0 else '不成立'}。",
        f"  2) 存货周转改善率：Q1 累计 {inv_perf[0]['total']:.2%}，Q5 累计 {inv_perf[4]['total']:.2%}，"
        f"多空(Q1-Q5)累计 {ls_inv['total']:.2%}、夏普 {ls_inv['sharpe']:.2f} —— 负向单调性"
        f"{'成立' if ls_inv['total'] > 0 else '不成立'}。",
        "  3) 两因子多空收益对比：" + (
            f"毛利率稳定性({ls_gm['total']:.2%}) > 存货周转改善率({ls_inv['total']:.2%})"
            if ls_gm["total"] >= ls_inv["total"]
            else f"存货周转改善率({ls_inv['total']:.2%}) > 毛利率稳定性({ls_gm['total']:.2%})"
        ),
        "",
    ]
    return lines


def _plot_section(gm_groups, gm_perf, inv_groups, inv_perf, bench_perf, suptitle, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2c3e50", "#16a085", "#f1c40f", "#e67e22", "#c0392b"]

    def to_dt(series):
        return pd.DatetimeIndex(series.index.to_timestamp())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)
    for ax, groups, perf_dict, name in (
        (axes[0], gm_groups, gm_perf, "毛利率稳定性（Q1 因子值最低 → Q5 最高）"),
        (axes[1], inv_groups, inv_perf, "存货周转改善率（Q1 因子值最低 → Q5 最高）"),
    ):
        for g in range(N_GROUPS):
            nav = perf_dict[g]["nav"].dropna()
            if nav.empty:
                continue
            ax.plot(to_dt(nav), nav, color=colors[g], lw=1.6, label=f"Q{g+1}")
        bench_nav = bench_perf["nav"].dropna()
        if not bench_nav.empty:
            ax.plot(to_dt(bench_nav), bench_nav, color="gray", ls="--", lw=1.8, label="基准")
        ax.set_title(name, fontsize=12)
        ax.set_ylabel("净值（起点=1）")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle(suptitle, fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    print("已生成:", out_path.name)


def main():
    for path in (GM_FACTOR_FILE, INV_FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not path.is_file():
            print("输入文件不存在:", path)
            return 1

    DST.mkdir(parents=True, exist_ok=True)

    gm_signals = _load_factor(GM_FACTOR_FILE, GM_FACTOR_COLUMN)
    inv_signals = _load_factor(INV_FACTOR_FILE, INV_FACTOR_COLUMN)
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

    gm_sections = build_sections(gm_signals, GM_FACTOR_COLUMN, ln_mcap, months, neutralize=False)
    inv_sections = build_sections(inv_signals, INV_FACTOR_COLUMN, ln_mcap, months, neutralize=True)

    def split(sections):
        insample = {m: z for m, z in sections.items() if m <= IN_SAMPLE_END}
        oos = {m: z for m, z in sections.items() if m >= OOS_START}
        return insample, oos

    gm_in, gm_oos = split(gm_sections)
    inv_in, inv_oos = split(inv_sections)

    gm_in_groups = layered_backtest(gm_in, monthly_ret)
    inv_in_groups = layered_backtest(inv_in, monthly_ret)
    gm_in_perf = {g: _perf(gm_in_groups[g]) for g in range(N_GROUPS)}
    inv_in_perf = {g: _perf(inv_in_groups[g]) for g in range(N_GROUPS)}
    ls_gm_in = _long_short(gm_in_groups, positive=True)
    ls_inv_in = _long_short(inv_in_groups, positive=False)
    ls_gm_in_perf = _perf(ls_gm_in)
    ls_inv_in_perf = _perf(ls_inv_in)
    bench_in = _bench_series({"gm": gm_in, "inv": inv_in}, monthly_ret)
    bench_in_perf = _perf(bench_in)

    gm_oos_groups = layered_backtest(gm_oos, monthly_ret)
    inv_oos_groups = layered_backtest(inv_oos, monthly_ret)
    gm_oos_perf = {g: _perf(gm_oos_groups[g]) for g in range(N_GROUPS)}
    inv_oos_perf = {g: _perf(inv_oos_groups[g]) for g in range(N_GROUPS)}
    ls_gm_oos = _long_short(gm_oos_groups, positive=True)
    ls_inv_oos = _long_short(inv_oos_groups, positive=False)
    ls_gm_oos_perf = _perf(ls_gm_oos)
    ls_inv_oos_perf = _perf(ls_inv_oos)
    bench_oos = _bench_series({"gm": gm_oos, "inv": inv_oos}, monthly_ret)
    bench_oos_perf = _perf(bench_oos)

    report = [
        "=" * 74,
        "因子分层回测对比报告（毛利率稳定性 vs 存货周转改善率）",
        "=" * 74,
        "",
        "回测口径（以因子库 IC 检验为基准）：",
        "  因子文件：毛利率稳定性_raw.csv（列「毛利率稳定性」）、存货周转改善率_raw.csv（列「存货周转改善率（TTM同比）」）",
        "  不复权行情（流通市值）：A股日线_20190101-20260331_不复权.csv",
        "  后复权行情（收益）：clean_temp\\A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv",
        "  月末截面按「可用日期 <= 月末」取最新因子值，再排序分 5 组（Q1=最低分 ~ Q5=最高分），组内等权",
        "  存货周转改善率：MAD(5x) 截尾 + 对数流通市值中性化 + Z-score",
        "  毛利率稳定性：MAD(5x) 截尾 + Z-score（不做市值中性化）",
        "  持有 12 个月（重叠持仓、月末调仓）；未计交易成本、无涨跌停剔除；夏普以无风险利率 0 计算",
        "  样本内形成期 <= 2024-12；样本外形成期 >= 2025-01",
        "",
    ]
    report += _render_section(
        f"【样本内】形成期（毛利率 {_fmt_range(gm_in)}；存货 {_fmt_range(inv_in)}）",
        gm_in_groups, gm_in_perf, ls_gm_in_perf,
        inv_in_groups, inv_in_perf, ls_inv_in_perf, bench_in_perf,
    )
    report += ["", "=" * 74, ""]
    report += _render_section(
        f"【样本外】形成期（毛利率 {_fmt_range(gm_oos)}；存货 {_fmt_range(inv_oos)}）",
        gm_oos_groups, gm_oos_perf, ls_gm_oos_perf,
        inv_oos_groups, inv_oos_perf, ls_inv_oos_perf, bench_oos_perf,
    )
    report += ["", "=" * 74]

    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    try:
        _plot_section(gm_in_groups, gm_in_perf, inv_in_groups, inv_in_perf, bench_in_perf,
                      "样本内（形成期 ≤ 2024-12）分层净值", OUTPUT_PNG_INSAMPLE)
        _plot_section(gm_oos_groups, gm_oos_perf, inv_oos_groups, inv_oos_perf, bench_oos_perf,
                      "样本外（形成期 ≥ 2025-01）分层净值", OUTPUT_PNG_OOS)
    except Exception as exc:
        print("图表生成失败:", exc)

    print("已输出:", OUTPUT_REPORT.name, "|", OUTPUT_PNG_INSAMPLE.name, "|", OUTPUT_PNG_OOS.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())



