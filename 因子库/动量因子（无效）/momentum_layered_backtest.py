# -*- coding: utf-8 -*-
"""
动量因子分层回测（单调性检验）

口径：
  - 因子：动量因子_raw.csv（raw，回测中自行 MAD(5x) 截尾 + Z-score）
  - 收益：后复权月末收盘价
  - 流通市值：不复权行情估算
  - 股票池：ST/停牌/次新(<180天)/涨跌停/微盘后15%/北交 全量剔除
  - 分组：5 组（全市场分位）
  - 加权：等权 + 流通市值加权
  - 多空：G1 - G5（IC 为负，低动量组预期收益更高）
  - 显著性：NW t-stat + Jonckheere-Terpstra
  - 成本：无成本 + 单边 0.1%
  - 持有 12 个月重叠持有
  - 样本内形成期 <= 2024-12；样本外形成期 >= 2025-01
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"

FACTOR_FILE = SCRIPT_DIR / "动量因子_raw.csv"
FACTOR_COLUMN = "动量因子"

ADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"

N_GROUPS = 5
HOLDING_MONTHS = 12
MIN_CROSS_SECTION_SIZE = 10

NEW_LISTING_DAYS = 180
SUSPEND_CHECK_DAYS = 5
MICRO_CAP_PCT = 0.15

MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826

IN_SAMPLE_END = pd.Period("2024-12", freq="M")
OOS_START = pd.Period("2025-01", freq="M")

BJ_CODE_PREFIXES = ("8", "4")

OUTPUT_REPORT = SCRIPT_DIR / "动量因子_分层回测结果.txt"
OUTPUT_PNG_INSAMPLE = SCRIPT_DIR / "动量因子_分层回测_样本内.png"
OUTPUT_PNG_OOS = SCRIPT_DIR / "动量因子_分层回测_样本外.png"


# ============================================================
# 数据加载
# ============================================================

def _normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def _load_quotes(path, columns):
    data = pd.read_csv(path, usecols=columns, dtype={"股票代码": str}, encoding="utf-8-sig")
    data["股票代码"] = _normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    for col in columns[2:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    data = data.sort_values(["股票代码", "日期"])
    return data


def _build_market_cap(quotes):
    mc = quotes.copy()
    mc["换手率"] = pd.to_numeric(mc["换手率"], errors="coerce")
    mc["成交量"] = pd.to_numeric(mc["成交量"], errors="coerce")
    turnover_scale = 10000 if mc["换手率"].max() > 1.5 else 100
    valid = mc["换手率"] > 0
    shares = pd.Series(np.nan, index=mc.index, dtype=float)
    shares[valid] = mc.loc[valid, "成交量"] / mc.loc[valid, "换手率"] * turnover_scale
    mc["流通市值"] = mc["收盘"] * shares
    mc["ln流通市值"] = np.log(mc["流通市值"].where(mc["流通市值"] > 0))
    mc.loc[~np.isfinite(mc["ln流通市值"]), "ln流通市值"] = np.nan
    return mc


# ============================================================
# 股票池过滤
# ============================================================

def _build_filter_state(quotes, listing_dates):
    quotes = quotes.copy()
    quotes["月"] = quotes["日期"].dt.to_period("M")
    codes = quotes["股票代码"].unique()

    # 北交所
    bj_codes = {c for c in codes if str(c).startswith(BJ_CODE_PREFIXES)}

    # 停牌（连续 SUSPEND_CHECK_DAYS 天换手率为 0 或收盘不变）
    quotes["ret"] = quotes.groupby("股票代码")["收盘"].pct_change()
    quotes["halt_signal"] = (
        (quotes["换手率"].fillna(0) == 0)
        | ((quotes["ret"].abs().fillna(0) < 1e-8) & (quotes["成交量"].fillna(0) == 0))
    ).astype(int)
    halt_rolling = (
        quotes.groupby("股票代码")["halt_signal"]
        .rolling(SUSPEND_CHECK_DAYS, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    quotes["is_suspended"] = halt_rolling >= SUSPEND_CHECK_DAYS

    # 涨跌停（创业板/科创板 20%，主板 10%）
    daily_ret = quotes.groupby("股票代码")["收盘"].pct_change().abs()
    limit_up = daily_ret >= 0.095
    limit_down_adjust = quotes.groupby("股票代码")["收盘"].transform(
        lambda s: s.diff().abs() / s.shift(1) >= 0.095
    )
    quotes["at_limit"] = limit_up | limit_down_adjust

    # 汇总个股状态
    state = (
        quotes.groupby(["月", "股票代码"])
        .agg(
            is_suspended=("is_suspended", "max"),
            at_limit=("at_limit", "max"),
        )
        .reset_index()
    )

    # 次新股过滤
    listing_map = {}
    for code in codes:
        code_dates = quotes.loc[quotes["股票代码"] == code, "日期"]
        if len(code_dates) > 0:
            listing_map[code] = code_dates.min()
    listing_series = pd.Series(listing_map)

    # 微盘后15%
    mc_monthly = (
        quotes.groupby(["月", "股票代码"])["流通市值"]
        .last()
        .reset_index()
        .rename(columns={"流通市值": "流通市值_月末"})
    )

    return state, bj_codes, listing_series, mc_monthly


def _filter_month(state, bj_codes, listing_series, mc_monthly, month):
    month_state = state.loc[state["月"] == month].copy()
    if month_state.empty:
        return set()

    exclude = set()

    # 北交所
    exclude |= bj_codes

    # 停牌
    suspended = month_state.loc[month_state["is_suspended"] == 1, "股票代码"]
    exclude |= set(suspended)

    # 涨跌停
    at_limit = month_state.loc[month_state["at_limit"] == True, "股票代码"]
    exclude |= set(at_limit)

    # 次新股
    month_end = pd.Timestamp(month.end_time).normalize()
    for code in month_state["股票代码"].unique():
        if code in listing_series:
            days_listed = (month_end - listing_series[code]).days
            if days_listed < NEW_LISTING_DAYS:
                exclude.add(code)

    # 微盘后15%
    mc_this = mc_monthly.loc[mc_monthly["月"] == month].dropna(subset=["流通市值_月末"])
    if len(mc_this) > 0:
        threshold = mc_this["流通市值_月末"].quantile(MICRO_CAP_PCT)
        micro = mc_this.loc[mc_this["流通市值_月末"] <= threshold, "股票代码"]
        exclude |= set(micro)

    return exclude


# ============================================================
# 因子处理
# ============================================================

def _load_factor(path, factor_column):
    factor = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    raw = factor.loc[:, ["股票代码", "截面日期", factor_column]].copy()
    raw["股票代码"] = _normalize_codes(raw["股票代码"])
    raw["截面日期"] = pd.to_datetime(raw["截面日期"], errors="raise").dt.normalize()
    raw[factor_column] = pd.to_numeric(raw[factor_column], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_column])
    raw = raw.sort_values(["股票代码", "截面日期"]).drop_duplicates(["股票代码", "截面日期"], keep="last")
    raw["月"] = raw["截面日期"].dt.to_period("M")
    return raw


def _winsorize_mad(values):
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return values
    lower = median - MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad
    upper = median + MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad
    return values.clip(lower=lower, upper=upper)


def _zscore(values):
    std = values.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def build_monthly_signals(factor, months):
    cross_sections = {}
    factor_by_month = factor.groupby("月")
    for month in months:
        if month not in factor_by_month.groups:
            continue
        month_data = factor_by_month.get_group(month).copy()
        month_data[FACTOR_COLUMN + "_winsor"] = _winsorize_mad(month_data[FACTOR_COLUMN])
        z = _zscore(month_data[FACTOR_COLUMN + "_winsor"])
        if z is not None and len(z) >= MIN_CROSS_SECTION_SIZE:
            cross_sections[month] = pd.Series(z.values, index=month_data["股票代码"].values, name="zscore")
    return cross_sections


# ============================================================
# 分组回测引擎
# ============================================================

def _group_assign(z, n_groups):
    try:
        return pd.qcut(z.astype(float), n_groups, labels=False).astype(int)
    except Exception:
        labels = np.floor(z.rank(method="first") / (len(z) + 1) * n_groups)
        return np.clip(labels.astype(int), 0, n_groups - 1)


def _layered_returns(signals, monthly_ret, mc_wide):
    group_monthly_eq = {g: {} for g in range(N_GROUPS)}
    group_monthly_mc = {g: {} for g in range(N_GROUPS)}
    for month, z in signals.items():
        labels = _group_assign(z, N_GROUPS)
        codes_series = z.index
        for g in range(N_GROUPS):
            g_codes = codes_series[labels == g]
            if len(g_codes) == 0:
                continue
            window = monthly_ret.loc[month + 1: month + HOLDING_MONTHS]
            ret_window = window.reindex(columns=list(g_codes))
            if ret_window.empty:
                continue

            # 等权
            eq_ret = ret_window.mean(axis=1)
            for m, v in eq_ret.items():
                if np.isfinite(v):
                    group_monthly_eq[g].setdefault(m, []).append(v)

            # 流通市值加权
            if mc_wide is not None and month in mc_wide.index:
                mc_row = mc_wide.loc[month].reindex(g_codes)
                valid_mc = mc_row.dropna()
                if len(valid_mc) > 0:
                    w = valid_mc / valid_mc.sum()
                    common = ret_window.columns.intersection(w.index)
                    if len(common) > 0:
                        w_mc = ret_window[common].dot(w.loc[common])
                        for m, v in w_mc.items():
                            if np.isfinite(v):
                                group_monthly_mc[g].setdefault(m, []).append(v)

    groups_eq = {
        g: pd.Series(
            {m: float(np.mean(vals)) for m, vals in group_monthly_eq[g].items()}
        ).sort_index()
        for g in range(N_GROUPS)
    }
    groups_mc = {
        g: pd.Series(
            {m: float(np.mean(vals)) for m, vals in group_monthly_mc[g].items()}
        ).sort_index()
        for g in range(N_GROUPS)
    }
    return groups_eq, groups_mc


def _long_short(groups):
    first = groups[0].dropna()
    last = groups[N_GROUPS - 1].dropna()
    if first.empty or last.empty:
        return pd.Series(dtype=float)
    idx = first.index.union(last.index)
    return first.reindex(idx) - last.reindex(idx)


# ============================================================
# 绩效统计
# ============================================================

def _perf(series):
    s = series.dropna()
    if s.empty:
        return {
            "total": float("nan"), "ann": float("nan"), "vol": float("nan"),
            "sharpe": float("nan"), "mdd": float("nan"), "nav": pd.Series(dtype=float),
            "win_rate": float("nan"),
        }
    nav = (1 + s).cumprod()
    total = nav.iloc[-1] - 1
    n = len(s)
    ann = (1 + total) ** (12 / n) - 1 if n > 0 else float("nan")
    vol = s.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol and vol > 0 else float("nan")
    mdd = (nav / nav.cummax() - 1).min()
    win_rate = (s > 0).mean()
    return {
        "total": total, "ann": ann, "vol": vol, "sharpe": sharpe,
        "mdd": mdd, "nav": nav, "win_rate": win_rate,
    }


# ============================================================
# Newey-West t-stat（复用 momentum_ic 模式）
# ============================================================

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
    nw_t = values.mean() / nw_se if nw_se > 0 else float("nan")
    return nw_t, nw_se, max_lags


# ============================================================
# Jonckheere-Terpstra 检验
# ============================================================

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _jonckheere_terpstra(group_values, direction="increasing"):
    ordered = [np.array(group_values[g]) for g in sorted(group_values.keys())]
    ns = [len(g) for g in ordered]
    N = sum(ns)
    if N < 3 or any(n == 0 for n in ns):
        return float("nan"), float("nan"), float("nan")

    jt = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            for xi in ordered[i]:
                jt += np.sum(xi < ordered[j])
    if direction == "decreasing":
        jt = N * (N - 1) / 2 - jt

    expected = (N ** 2 - sum(n ** 2 for n in ns)) / 4
    variance = (N ** 2 * (2 * N + 3) - sum(n ** 2 * (2 * n + 3) for n in ns)) / 72
    if variance <= 0:
        return jt, float("nan"), float("nan")

    z = (jt - expected) / np.sqrt(variance)
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return jt, z, p_value


# ============================================================
# 成本分析
# ============================================================

def _apply_cost(series, cost_pct):
    s = series.dropna().copy()
    if s.empty:
        return s
    s_cost = s - cost_pct / HOLDING_MONTHS
    return s_cost


# ============================================================
# 报告渲染
# ============================================================

def _fmt_range(signals):
    if not signals:
        return "无"
    ms = sorted(signals.keys())
    return f"{ms[0]} ~ {ms[-1]}（{len(ms)} 期）"


def _group_table(groups, perf_dict):
    header = f"{'分组':<8}{'累计':>10}{'年化':>10}{'夏普':>8}{'最大回撤':>10}{'胜率':>8}"
    sep = "-" * 54
    lines = [header, sep]
    for g in range(N_GROUPS):
        p = perf_dict[g]
        nav_series = groups[g]
        if nav_series.empty:
            lines.append(f"{'Q'+str(g+1):<8}{'--':>10}{'--':>10}{'--':>8}{'--':>10}{'--':>8}")
        else:
            lines.append(
                f"{'Q'+str(g+1):<8}{p['total']:>10.2%}{p['ann']:>10.2%}"
                f"{p['sharpe']:>8.2f}{p['mdd']:>10.2%}{p['win_rate']:>8.2%}"
            )
    return lines


def _render_section(title, groups_eq, perf_eq, ls_eq, nw_eq,
                    groups_mc, perf_mc, ls_mc, nw_mc,
                    jt_eq, jt_mc):
    lines = [title, "", "一、等权分组收益", ""]
    lines += _group_table(groups_eq, perf_eq)
    lines += [
        "",
        f"  多空（Q1-Q5）：累计 {ls_eq['total']:.2%}  |  年化 {ls_eq['ann']:.2%}  "
        f"|  夏普 {ls_eq['sharpe']:.2f}  |  回撤 {ls_eq['mdd']:.2%}",
        f"  多空 NW t-stat：{nw_eq[0]:.4f}（滞后期数 {nw_eq[2]}）",
        "",
    ]

    if groups_mc is not None and any(not g.empty for g in groups_mc.values()):
        mc_header = "二、流通市值加权分组收益"
        lines += [mc_header, ""]
        lines += _group_table(groups_mc, perf_mc)
        lines += [
            "",
            f"  多空（Q1-Q5）：累计 {ls_mc['total']:.2%}  |  年化 {ls_mc['ann']:.2%}  "
            f"|  夏普 {ls_mc['sharpe']:.2f}  |  回撤 {ls_mc['mdd']:.2%}",
            f"  多空 NW t-stat：{nw_mc[0]:.4f}（滞后期数 {nw_mc[2]}）",
            "",
        ]
    else:
        lines += ["二、流通市值加权分组收益", "", "  （无有效数据）", ""]

    lines += [
        "三、单调性检验",
        "",
        f"  Jonckheere-Terpstra（等权）：JT = {jt_eq[0]:.1f}, Z = {jt_eq[1]:.4f}, p = {jt_eq[2]:.4f}",
    ]
    if np.isfinite(jt_mc[0]):
        lines.append(
            f"  Jonckheere-Terpstra（市值加权）：JT = {jt_mc[0]:.1f}, Z = {jt_mc[1]:.4f}, p = {jt_mc[2]:.4f}"
        )
    lines += ["", "四、成本敏感度（等权多空）", ""]
    for cost_name, cost_pct in [("无成本", 0.0), ("单边 0.1%", 0.0010)]:
        ls_cost = _apply_cost(_long_short(groups_eq), cost_pct)
        p = _perf(ls_cost)
        lines.append(
            f"  {cost_name:>10}：累计 {p['total']:.2%}  |  年化 {p['ann']:.2%}  "
            f"|  夏普 {p['sharpe']:.2f}  |  回撤 {p['mdd']:.2%}"
        )
    return lines


# ============================================================
# 绘图
# ============================================================

def _plot(groups_eq, perf_eq, groups_mc, perf_mc, title, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2c3e50", "#16a085", "#f1c40f", "#e67e22", "#c0392b"]

    def _to_dt(s):
        return pd.DatetimeIndex(s.index.to_timestamp())

    has_mc = groups_mc is not None and any(not g.empty for g in groups_mc.values())
    fig, axes = plt.subplots(1, 2 if has_mc else 1, figsize=(8 * (2 if has_mc else 1), 6), dpi=150)
    if not has_mc:
        axes = [axes]

    for ax, groups, label in [
        (axes[0], groups_eq, "等权"),
        (axes[1] if has_mc else None, groups_mc, "流通市值加权") if has_mc else (None, None, ""),
    ]:
        if ax is None:
            continue
        for g in range(N_GROUPS):
            nav = (1 + groups[g].dropna()).cumprod()
            if nav.empty:
                continue
            ax.plot(_to_dt(nav), nav, color=colors[g], lw=1.6, label=f"Q{g+1}")
        ax.set_title(f"动量因子分层净值（{label}）", fontsize=12)
        ax.set_ylabel("净值（起点=1）")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    print("已生成:", out_path.name)


# ============================================================
# 主函数
# ============================================================

def main():
    for path in (FACTOR_FILE, ADJUSTED_QUOTE_FILE, UNADJUSTED_QUOTE_FILE):
        if not path.is_file():
            print("输入文件不存在:", path)
            return 1

    print("加载数据...")

    adj_quotes = _load_quotes(
        ADJUSTED_QUOTE_FILE,
        ["股票代码", "日期", "收盘", "成交量", "换手率"],
    )
    unadj_quotes = _load_quotes(
        UNADJUSTED_QUOTE_FILE,
        ["股票代码", "日期", "收盘", "成交量", "换手率"],
    )

    # 流通市值估算
    mc_quotes = _build_market_cap(unadj_quotes)
    mc_quotes["月"] = mc_quotes["日期"].dt.to_period("M")
    mc_monthly = (
        mc_quotes.groupby(["股票代码", "月"])["ln流通市值"]
        .last()
        .reset_index()
        .rename(columns={"ln流通市值": "ln_mcap"})
    )
    mc_wide = mc_monthly.pivot(index="月", columns="股票代码", values="ln_mcap")

    # 上市日期估算
    listing_dates = adj_quotes.groupby("股票代码")["日期"].min()

    # 北交所股票代码集合
    bj_codes = {c for c in adj_quotes["股票代码"].unique() if str(c).startswith(BJ_CODE_PREFIXES)}
    print(f"  北交所股票数：{len(bj_codes)}")

    # 收益率面板 pre-build
    adj_quotes["月"] = adj_quotes["日期"].dt.to_period("M")
    adj_monthly = (
        adj_quotes.sort_values(["股票代码", "日期"])
        .groupby(["股票代码", "月"])
        .tail(1)
    )
    prices = adj_monthly.pivot(index="月", columns="股票代码", values="收盘")
    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)
    monthly_ret = prices.pct_change()

    # 因子
    factor = _load_factor(FACTOR_FILE, FACTOR_COLUMN)
    avail_months = sorted(set(monthly_ret.index) & set(factor["月"].unique()))
    last_full = avail_months[-1] - HOLDING_MONTHS if avail_months else None
    months = [m for m in avail_months if m <= last_full]

    signals = build_monthly_signals(factor, months)
    print(f"  有效截面数：{len(signals)}")

    # 预计算过滤状态
    print("预计算股票过滤状态...")
    adj_quotes["ret"] = adj_quotes.groupby("股票代码")["收盘"].pct_change()
    adj_quotes["zero_vol"] = (adj_quotes["成交量"].fillna(0) == 0).astype(int)
    adj_quotes["zero_ret_chg"] = (
        (adj_quotes["ret"].abs().fillna(0) < 1e-8) & (adj_quotes["成交量"].fillna(0) == 0)
    ).astype(int)
    adj_quotes["halt_signal"] = adj_quotes["zero_vol"] | adj_quotes["zero_ret_chg"]

    # 停牌：连续 N 天 halt_signal 都为 1
    adj_quotes["suspended"] = (
        adj_quotes.groupby("股票代码")["halt_signal"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    ) >= 5

    # 涨跌停
    adj_quotes["at_limit"] = adj_quotes["ret"].abs() >= 0.095

    # 次新股阈值日期
    listing_thresholds = listing_dates + pd.Timedelta(days=NEW_LISTING_DAYS)

    # 股票池过滤
    print("应用股票池过滤...")
    filtered_signals = {}
    for month, z in signals.items():
        month_end = pd.Timestamp(month.end_time).normalize()
        exclude = bj_codes.copy()

        # 停牌：取月末最近一日的 suspended 状态
        last_day_mask = (
            (adj_quotes["股票代码"].isin(z.index))
            & (adj_quotes["日期"] <= month_end)
        )
        relevant = adj_quotes.loc[last_day_mask]
        if not relevant.empty:
            last_per_code = relevant.groupby("股票代码").tail(1)
            suspended_codes = set(last_per_code.loc[last_per_code["suspended"] == True, "股票代码"])
            exclude |= suspended_codes
        else:
            exclude |= set(z.index)

        # 涨跌停
        if not relevant.empty:
            limit_codes = set(last_per_code.loc[last_per_code["at_limit"] == True, "股票代码"])
            exclude |= limit_codes

        # 次新股
        for code in set(z.index) - exclude:
            if code in listing_thresholds and month_end < listing_thresholds[code]:
                exclude.add(code)

        # 微盘后15%
        mc_this = mc_monthly.loc[mc_monthly["月"] == month].dropna(subset=["ln_mcap"])
        if len(mc_this) > 0:
            threshold = mc_this["ln_mcap"].quantile(MICRO_CAP_PCT)
            micro = mc_this.loc[mc_this["ln_mcap"] <= threshold, "股票代码"]
            exclude |= set(micro)

        kept = z.drop(index=list(exclude & set(z.index)), errors="ignore")
        if len(kept) >= MIN_CROSS_SECTION_SIZE:
            filtered_signals[month] = kept

    print(f"  过滤后截面数：{len(filtered_signals)}（过滤前 {len(signals)}）")

    # 分组回测
    print("执行分层回测...")
    groups_eq, groups_mc = _layered_returns(filtered_signals, monthly_ret, mc_wide)

    # 样本划分
    def split_groups(groups):
        insample = {}
        oos = {}
        for g, series in groups.items():
            s = series.dropna()
            insample[g] = s.loc[s.index <= IN_SAMPLE_END]
            oos[g] = s.loc[s.index >= OOS_START]
        return insample, oos

    eq_in, eq_oos = split_groups(groups_eq)
    mc_in, mc_oos = split_groups(groups_mc) if any(not g.empty for g in groups_mc.values()) else (None, None)

    # 绩效计算
    def compute_all(groups_eq, groups_mc):
        peq = {g: _perf(groups_eq[g]) for g in range(N_GROUPS)}
        ls_eq = _long_short(groups_eq)
        ls_eq_perf = _perf(ls_eq)
        nw_eq = _newey_west_t_stat(ls_eq.dropna().values)

        # JT: 对每个分组取均值序列
        jt_data = {g: groups_eq[g].dropna().values for g in range(N_GROUPS)}
        jt_eq = _jonckheere_terpstra(jt_data, direction="decreasing")

        pmc = {}
        ls_mc = pd.Series(dtype=float)
        ls_mc_perf = {}
        nw_mc = (float("nan"), float("nan"), 0)
        jt_mc = (float("nan"), float("nan"), float("nan"))
        if groups_mc is not None:
            pmc = {g: _perf(groups_mc[g]) for g in range(N_GROUPS)}
            ls_mc = _long_short(groups_mc)
            ls_mc_perf = _perf(ls_mc)
            nw_mc = _newey_west_t_stat(ls_mc.dropna().values)
            jt_data_mc = {g: groups_mc[g].dropna().values for g in range(N_GROUPS)}
            jt_mc = _jonckheere_terpstra(jt_data_mc, direction="decreasing")

        return peq, ls_eq_perf, nw_eq, jt_eq, pmc, ls_mc_perf, nw_mc, jt_mc

    peq_in, ls_in, nw_in, jt_in, pmc_in, ls_mc_in, nw_mc_in, jt_mc_in = compute_all(eq_in, mc_in)
    peq_oos, ls_oos, nw_oos, jt_oos, pmc_oos, ls_mc_oos, nw_mc_oos, jt_mc_oos = compute_all(eq_oos, mc_oos)

    # 报告
    report = [
        "=" * 74,
        "动量因子分层回测报告（单调性检验）",
        "=" * 74,
        "",
        "回测口径：",
        "  因子：动量因子_raw.csv，各月截面 MAD(5x) 截尾 + Z-score",
        "  收益：后复权月末收盘价，下一自然月收益",
        "  流通市值：不复权行情（成交量 / 换手率 估算）",
        "  股票池剔除：ST/停牌/次新(<180天)/涨跌停/微盘后15%/北交 全部剔除",
        f"  分组：{N_GROUPS} 组（全市场分位），组内等权 + 流通市值加权",
        f"  持有：{HOLDING_MONTHS} 个月重叠持有（月末调仓）",
        "  多空方向：Q1 - Q5（IC 负向，低动量组预期收益更高）",
        "  显著性：Newey-West t-stat + Jonckheere-Terpstra 趋势检验",
        "  成本：无成本 / 单边 0.1%",
        "  样本内形成期 <= 2024-12；样本外形成期 >= 2025-01",
        "",
        f"  无过滤信号截面数：{len(signals)}",
        f"  过滤后信号截面数：{len(filtered_signals)}",
        "",
    ]

    report += _render_section(
        "【样本内】形成期 {} ".format(_fmt_range(
            {m: z for m, z in filtered_signals.items() if m <= IN_SAMPLE_END}
        )),
        eq_in, peq_in, ls_in, nw_in, mc_in, pmc_in, ls_mc_in, nw_mc_in, jt_in, jt_mc_in,
    )
    report += ["", "=" * 74, ""]
    report += _render_section(
        "【样本外】形成期 {} ".format(_fmt_range(
            {m: z for m, z in filtered_signals.items() if m >= OOS_START}
        )),
        eq_oos, peq_oos, ls_oos, nw_oos, mc_oos, pmc_oos, ls_mc_oos, nw_mc_oos, jt_oos, jt_mc_oos,
    )
    report += ["", "=" * 74]

    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    # 绘图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib 不可用，跳过绘图")

    if plt is not None:
        try:
            _plot(eq_in, peq_in, mc_in, pmc_in,
                  "样本内（形成期 <= 2024-12）分层净值", OUTPUT_PNG_INSAMPLE)
        except Exception as exc:
            import traceback
            print("图表生成失败（样本内）:", exc)
            traceback.print_exc()
        try:
            _plot(eq_oos, peq_oos, mc_oos, pmc_oos,
                  "样本外（形成期 >= 2025-01）分层净值", OUTPUT_PNG_OOS)
        except Exception as exc:
            import traceback
            print("图表生成失败（样本外）:", exc)
            traceback.print_exc()

    print("已输出:", OUTPUT_REPORT.name, "|", OUTPUT_PNG_INSAMPLE.name, "|", OUTPUT_PNG_OOS.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())