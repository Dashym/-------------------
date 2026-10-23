# -*- coding: utf-8 -*-
"""Fama-MacBeth 回归加权多因子模型：VIF + 系数估计 + 固定权重回测（样本内/样本外）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(r"C:\Users\Heyuanhang\Desktop\aksahre_demo")
LIB_DIR = BASE_DIR.parent / "因子库"
GM_FACTOR_FILE = LIB_DIR / "毛利率稳定性（可以，小心）" / "毛利率稳定性_raw.csv"
INV_FACTOR_FILE = LIB_DIR / "存货周转改善率（good）" / "存货周转改善率_raw.csv"
UNADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp" / "A股日线_20190101-20260331_不复权_去重排序_剔除异常_格式统一.csv"
)
ADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp" / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
DST = Path(r"C:\Users\Heyuanhang\Desktop\综合比较打分法和回归法的回测结果")

GM_FACTOR_COLUMN = "毛利率稳定性"
INV_FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
MIN_CROSS_SECTION_SIZE = 20
TOP_FRACTION = 0.20
HOLDING_MONTHS = 12
HAC_MAX_LAGS = 11

TRAIN_START = pd.Period("2022-05", freq="M")
TRAIN_END = pd.Period("2024-12", freq="M")
OOS_START = pd.Period("2025-03", freq="M")
OOS_END = pd.Period("2026-03", freq="M")

OUTPUT_REPORT = DST / "FamaMacBeth回归_结果.txt"
OUTPUT_INS = DST / "回归模型_样本内回测结果.txt"
OUTPUT_OOS = DST / "回归模型_样本外回测结果.txt"
OUTPUT_CSV = DST / "回归模型_月度收益与净值.csv"
OUTPUT_PNG = DST / "回归模型_净值回撤图.png"
OUTPUT_EW_INS = DST / "打分模型_样本内回测结果.txt"
OUTPUT_EW_OOS = DST / "打分模型_样本外回测结果.txt"


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


def newwey_west_mean_se(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    count = len(values)
    residuals = values - values.mean()
    long_run_variance = float(np.dot(residuals, residuals) / count)
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)
        covariance = float(np.dot(residuals[lag:], residuals[:-lag]) / count)
        long_run_variance += 2 * weight * covariance
    variance_of_mean = long_run_variance / count * count / (count - 1)
    if variance_of_mean < 0 and np.isclose(variance_of_mean, 0):
        variance_of_mean = 0.0
    if variance_of_mean < 0:
        raise ArithmeticError("Newey-West HAC 方差估计为负")
    return float(np.sqrt(variance_of_mean))


def build_cross_sections(gm_signals, inv_signals, ln_mcap, prices, months):
    sections = {}
    for month in months:
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
        frame = pd.DataFrame({"z_gm": z_gm.loc[both], "z_inv": z_inv.loc[both]})
        if (month + HOLDING_MONTHS) in prices.index:
            p0 = prices.loc[month].reindex(both)
            p1 = prices.loc[month + HOLDING_MONTHS].reindex(both)
            frame["fwd_ret"] = (p1 / p0 - 1).astype(float)
        sections[month] = frame
    return sections


def compute_vif(sections, months):
    pooled = pd.concat(
        [sections[m][["z_gm", "z_inv"]] for m in months], ignore_index=True
    ).dropna()
    corr = float(pooled["z_gm"].corr(pooled["z_inv"]))
    r2 = corr ** 2
    vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return {"n": len(pooled), "corr": corr, "vif_gm": vif, "vif_inv": vif}


def fama_macbeth(sections, months):
    rows = []
    for month in months:
        frame = sections[month].dropna(subset=["z_gm", "z_inv", "fwd_ret"])
        if len(frame) < MIN_CROSS_SECTION_SIZE:
            continue
        y = frame["fwd_ret"].to_numpy(dtype=float)
        design = np.column_stack(
            [
                np.ones(len(frame)),
                frame["z_gm"].to_numpy(dtype=float),
                frame["z_inv"].to_numpy(dtype=float),
            ]
        )
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        rows.append(
            {
                "月份": str(month),
                "样本数": len(frame),
                "alpha": coef[0],
                "beta_gm": coef[1],
                "beta_inv": coef[2],
            }
        )
    df = pd.DataFrame(rows)
    summary = {}
    for key, label in [("alpha", "const"), ("beta_gm", "毛利率稳定性"), ("beta_inv", "存货周转改善率")]:
        values = df[key].to_numpy(dtype=float)
        mean = float(values.mean())
        se = newwey_west_mean_se(values, HAC_MAX_LAGS)
        t = mean / se if np.isfinite(se) and se > 0 else float("nan")
        p = float(2 * stats.t.sf(abs(t), df=len(values) - 1)) if np.isfinite(t) else float("nan")
        summary[key] = {"label": label, "mean": mean, "se": se, "t": t, "p": p}
    return summary, df


def backtest(sections, formation_months, w_gm, w_inv, monthly_ret, full_sample):
    cohort_returns = {}
    for month in formation_months:
        frame = sections[month]
        composite = w_gm * frame["z_gm"] + w_inv * frame["z_inv"]
        composite = composite.sort_values(ascending=False)
        n_select = max(1, int(np.floor(len(composite) * TOP_FRACTION)))
        selected = list(composite.index[:n_select])
        start = month + 1
        end = month + HOLDING_MONTHS
        window = monthly_ret.loc[start:end, selected]
        cohort_returns[month] = window.mean(axis=1)

    eval_start = formation_months[0] + 1
    eval_end = min(formation_months[-1] + HOLDING_MONTHS, monthly_ret.index.max())
    eval_months = pd.period_range(eval_start, eval_end, freq="M")

    strategy = {}
    for month in eval_months:
        values = []
        for form in formation_months:
            if (form + 1) <= month <= (form + HOLDING_MONTHS):
                value = cohort_returns[form].get(month)
                if value is not None and np.isfinite(value):
                    values.append(value)
        if values:
            strategy[month] = float(np.mean(values))

    benchmark = {}
    for month in eval_months:
        benchmark[month] = float(monthly_ret.loc[month].reindex(full_sample).mean())

    return pd.Series(strategy).sort_index(), pd.Series(benchmark).sort_index()


def performance_summary(strategy_series, benchmark_series):
    strategy = strategy_series.dropna()
    benchmark = benchmark_series.reindex(strategy.index)
    if strategy.empty:
        return {}
    nav = (1 + strategy).cumprod()
    bnav = (1 + benchmark).cumprod()
    total = nav.iloc[-1] - 1
    btotal = bnav.iloc[-1] - 1
    n = len(strategy)
    ann = (1 + total) ** (12 / n) - 1
    bann = (1 + btotal) ** (12 / n) - 1
    vol = strategy.std(ddof=1) * np.sqrt(12)
    bvol = benchmark.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    bsharpe = bann / bvol if bvol > 0 else float("nan")
    maxdd = (nav / nav.cummax() - 1).min()
    bmaxdd = (bnav / bnav.cummax() - 1).min()
    excess = strategy - benchmark
    win_rate = (excess > 0).mean()
    excess_std = excess.std(ddof=1)
    ir = excess.mean() / excess_std * np.sqrt(12) if excess_std > 0 else float("nan")
    return {
        "n": n,
        "total": total, "btotal": btotal,
        "ann": ann, "bann": bann,
        "vol": vol, "bvol": bvol,
        "sharpe": sharpe, "bsharpe": bsharpe,
        "maxdd": maxdd, "bmaxdd": bmaxdd,
        "win_rate": win_rate, "ir": ir,
        "nav": nav, "bnav": bnav,
        "excess": excess,
    }


def _fmt_pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "nan"


def main():
    for path in (GM_FACTOR_FILE, INV_FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not path.is_file():
            print("输入文件不存在:", path)
            return 1

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

    all_formation_months = [m for m in all_months if TRAIN_START <= m <= OOS_END]
    sections = build_cross_sections(gm_signals, inv_signals, ln_mcap, prices, all_formation_months)
    if not sections:
        print("没有达到最小样本数要求的有效截面")
        return 1

    train_months = [m for m in all_formation_months if TRAIN_START <= m <= TRAIN_END]
    oos_months = [m for m in all_formation_months if OOS_START <= m <= OOS_END]
    train_months = [m for m in train_months if m in sections]
    oos_months = [m for m in oos_months if m in sections]

    full_sample = sorted({c for m in sections for c in sections[m].index})

    # 一、VIF
    vif = compute_vif(sections, train_months)
    # 二、Fama-MacBeth
    fm_summary, fm_df = fama_macbeth(sections, train_months)
    beta_gm = fm_summary["beta_gm"]["mean"]
    beta_inv = fm_summary["beta_inv"]["mean"]

    # 三、回测
    ins_strategy, ins_bench = backtest(
        sections, train_months, beta_gm, beta_inv, monthly_ret, full_sample
    )
    oos_strategy, oos_bench = backtest(
        sections, oos_months, beta_gm, beta_inv, monthly_ret, full_sample
    )
    # 等权模型（0.5/0.5，存货取反）同一样本外机制重算
    ew_strategy, ew_bench = backtest(
        sections, oos_months, 0.5, -0.5, monthly_ret, full_sample
    )
    ew_ins_strategy, ew_ins_bench = backtest(
        sections, train_months, 0.5, -0.5, monthly_ret, full_sample
    )

    ins_m = performance_summary(ins_strategy, ins_bench)
    oos_m = performance_summary(oos_strategy, oos_bench)
    ew_m = performance_summary(ew_strategy, ew_bench)
    ew_ins_m = performance_summary(ew_ins_strategy, ew_ins_bench)

    def detail_frame(strategy_series, benchmark_series, m):
        idx = strategy_series.index
        bench = benchmark_series.reindex(idx)
        return pd.DataFrame({
            "月份": [str(x) for x in idx],
            "策略月收益": strategy_series.to_numpy(),
            "基准月收益": bench.to_numpy(),
            "超额收益": (strategy_series - bench).to_numpy(),
            "策略净值": m["nav"].to_numpy(),
            "基准净值": m["bnav"].to_numpy(),
        })

    def render_bt(path, title, desc_lines, m, detail, extra=None):
        lines = ["=" * 74, title, "=" * 74, ""]
        lines += desc_lines
        lines.append("")
        lines.append(f"区间：{detail['月份'].iloc[0]} ~ {detail['月份'].iloc[-1]}（共 {m['n']} 个月）")
        lines.append("")
        lines.append("绩效指标表：")
        lines.append(f"{'指标':<14}{'策略':>16}{'基准':>16}")
        lines.append("-" * 46)
        lines.append(f"{'累计收益':<14}{m['total']:>16.2%}{m['btotal']:>16.2%}")
        lines.append(f"{'年化收益':<14}{m['ann']:>16.2%}{m['bann']:>16.2%}")
        lines.append(f"{'年化波动率':<14}{m['vol']:>16.2%}{m['bvol']:>16.2%}")
        lines.append(f"{'夏普比率':<14}{m['sharpe']:>16.2f}{m['bsharpe']:>16.2f}")
        lines.append(f"{'最大回撤':<14}{m['maxdd']:>16.2%}{m['bmaxdd']:>16.2%}")
        lines.append(f"{'月度胜率':<14}{m['win_rate']:>16.2%}{'（策略跑赢基准占比）':>16}")
        lines.append(f"{'信息比率':<14}{m['ir']:>16.2f}{'（年化）':>16}")
        lines.append("")
        if extra:
            lines += extra
            lines.append("")
        lines.append("逐月收益与净值明细（区间内净值从 1.0 起复权）：")
        lines.append(f"{'月份':<10}{'策略月收益':>12}{'基准月收益':>12}{'超额收益':>12}{'策略净值':>12}{'基准净值':>12}")
        lines.append("-" * 70)
        for _, row in detail.iterrows():
            lines.append(
                f"{row['月份']:<10}{row['策略月收益']:>12.2%}{row['基准月收益']:>12.2%}"
                f"{row['超额收益']:>12.2%}{row['策略净值']:>12.4f}{row['基准净值']:>12.4f}"
            )
        lines.append("=" * 74)
        path.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))

    # ---------- FamaMacBeth 报告 ----------
    fm_lines = [
        "=" * 74,
        "多因子 Fama-MacBeth 回归报告（回归加权模型）",
        "=" * 74,
        "",
        "回归设定：",
        "  因子1 毛利率稳定性；因子2 存货周转改善率（TTM同比）",
        "  预处理：截面 MAD(5x) 截尾 + Z-score；存货周转改善率另做对数流通市值中性化",
        "  因变量：未来 12 个月累计后复权收益 = t+12 月末收盘 / t 月末收盘 - 1",
        "  截面 OLS：y = alpha + beta1*z_gm + beta2*z_inv（不加市值/BP 等控制变量）",
        "  第二阶段：系数时间序列均值 + Newey-West HAC（Bartlett，最大滞后 11）",
        f"  训练集：{train_months[0]} ~ {train_months[-1]}（共 {len(train_months)} 个月）",
        f"  样本外：{oos_months[0]} ~ {oos_months[-1]}（共 {len(oos_months)} 个月）",
        "",
        "一、多重共线性 VIF 检验（训练集截面池）：",
        f"  截面池样本数：{vif['n']:,}",
        f"  z_gm 与 z_inv 相关系数：{vif['corr']:.4f}",
        f"  VIF（毛利率稳定性）：{vif['vif_gm']:.4f}",
        f"  VIF（存货周转改善率）：{vif['vif_inv']:.4f}",
        f"  结论：{'存在严重多重共线性（VIF>10）' if vif['vif_gm'] > 10 else 'VIF 远小于 10，无严重多重共线性'}",
        "",
        "二、Fama-MacBeth 回归结果：",
        f"{'变量':<14}{'平均系数':>12}{'HAC标准误':>12}{'t统计量':>10}{'p值':>10}",
        "-" * 58,
    ]
    for key in ["alpha", "beta_gm", "beta_inv"]:
        s = fm_summary[key]
        fm_lines.append(f"{s['label']:<14}{s['mean']:>12.4f}{s['se']:>12.4f}{s['t']:>10.2f}{s['p']:>10.4f}")
    fm_lines.append("")
    fm_lines.append(f"固定权重：复合得分 = {beta_gm:.4f} * z_gm + {beta_inv:.4f} * z_inv")
    fm_lines.append("")
    fm_lines.append("逐月回归明细（训练集）：")
    fm_lines.append(f"{'月份':<10}{'样本数':>8}{'alpha':>12}{'beta_gm':>12}{'beta_inv':>12}")
    fm_lines.append("-" * 54)
    for _, row in fm_df.iterrows():
        fm_lines.append(
            f"{row['月份']:<10}{row['样本数']:>8d}{row['alpha']:>12.4f}{row['beta_gm']:>12.4f}{row['beta_inv']:>12.4f}"
        )
    fm_lines.append("=" * 74)
    OUTPUT_REPORT.write_text("\n".join(fm_lines), encoding="utf-8")
    print("\n".join(fm_lines))

    common_desc = [
        "  因子1 毛利率稳定性；因子2 存货周转改善率（方向由回归系数符号决定）",
        "  预处理：截面 MAD(5x) 截尾 + Z-score；存货周转改善率另做对数流通市值中性化",
        "  选股：复合因子排名前 20%，等权买入；持有期 12 个月（月末调仓，重叠持仓）",
        "  基准：等权全样本（月度调仓）",
        "  收益：后复权月末收盘，t+1 月末收盘 / t 月末收盘 - 1",
        "  未计交易成本、无涨跌停剔除；夏普/信息比率以无风险利率 0 计算",
    ]

    ins_desc = common_desc + [
        f"  权重：训练集 Fama-MacBeth 估计（beta_gm={beta_gm:.4f}, beta_inv={beta_inv:.4f}）",
    ]
    ins_detail = detail_frame(ins_strategy, ins_bench, ins_m)
    print("\n\n==================== 样本内回测 ====================")
    render_bt(OUTPUT_INS, "回归模型 样本内回测结果（训练集）", ins_desc, ins_m, ins_detail)

    ew_ins_desc = common_desc + [
        "  权重：等权打分（复合得分 = 0.5*z_gm - 0.5*z_inv，存货取反）",
    ]
    ew_ins_detail = detail_frame(ew_ins_strategy, ew_ins_bench, ew_ins_m)
    render_bt(OUTPUT_EW_INS, "打分模型 样本内回测结果（等权 0.5/-0.5）", ew_ins_desc, ew_ins_m, ew_ins_detail)

    oos_desc = common_desc + [
        f"  权重：训练集估计后固定不变（beta_gm={beta_gm:.4f}, beta_inv={beta_inv:.4f}）",
        "  注：因子权重与选股规则沿用训练集估计，样本外未作调整",
    ]
    oos_detail = detail_frame(oos_strategy, oos_bench, oos_m)
    extra = [
        "与等权打分模型横向对比（同一样本外区间、同一回测机制）：",
        f"{'模型':<16}{'累计收益':>14}{'年化收益':>14}{'夏普比率':>12}{'最大回撤':>12}",
        "-" * 62,
        f"{'回归加权模型':<16}{oos_m['total']:>14.2%}{oos_m['ann']:>14.2%}{oos_m['sharpe']:>12.2f}{oos_m['maxdd']:>12.2%}",
        f"{'等权打分模型':<16}{ew_m['total']:>14.2%}{ew_m['ann']:>14.2%}{ew_m['sharpe']:>12.2f}{ew_m['maxdd']:>12.2%}",
        f"{'基准(等权全样本)':<16}{oos_m['btotal']:>14.2%}{oos_m['bann']:>14.2%}{oos_m['bsharpe']:>12.2f}{oos_m['bmaxdd']:>12.2%}",
    ]
    print("\n\n==================== 样本外回测 ====================")
    render_bt(OUTPUT_OOS, "回归模型 样本外回测结果", oos_desc, oos_m, oos_detail, extra=extra)

    ew_oos_desc = common_desc + [
        "  权重：等权打分（复合得分 = 0.5*z_gm - 0.5*z_inv，存货取反）",
    ]
    ew_detail = detail_frame(ew_strategy, ew_bench, ew_m)
    render_bt(OUTPUT_EW_OOS, "打分模型 样本外回测结果（等权 0.5/-0.5）", ew_oos_desc, ew_m, ew_detail)

    csv_rows = []
    for name, series, bench, m in [
        ("样本内", ins_strategy, ins_bench, ins_m),
        ("样本外", oos_strategy, oos_bench, oos_m),
    ]:
        bench_aligned = bench.reindex(series.index)
        for i, month in enumerate(series.index):
            csv_rows.append({
                "区间": name,
                "月份": str(month),
                "策略月收益": series.iloc[i],
                "基准月收益": bench_aligned.iloc[i],
                "超额收益": m["excess"].iloc[i],
                "策略净值": m["nav"].iloc[i],
                "基准净值": m["bnav"].iloc[i],
            })
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("已输出:", OUTPUT_REPORT.name, "|", OUTPUT_INS.name, "|", OUTPUT_OOS.name, "|", OUTPUT_EW_INS.name, "|", OUTPUT_EW_OOS.name, "|", OUTPUT_CSV.name)

    # ---------- 净值回撤图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        def to_dt(series):
            return pd.DatetimeIndex(series.index.to_timestamp())

        def drawdown(nav):
            return nav / nav.cummax() - 1

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), dpi=150)
        boundary = pd.Timestamp("2025-03-01")

        ax1.plot(to_dt(ins_strategy), ins_m["nav"], label="回归模型·样本内", color="#c0392b", lw=2)
        ax1.plot(to_dt(oos_strategy), oos_m["nav"], label="回归模型·样本外", color="#e67e22", lw=2)
        ax1.plot(to_dt(ew_strategy), ew_m["nav"], label="等权模型·样本外", color="#8e44ad", lw=2)
        ax1.plot(to_dt(ins_strategy), ins_m["bnav"], label="基准(全样本)", color="#2980b9", lw=2)
        ax1.axvline(boundary, color="gray", linestyle="--", linewidth=1.5)
        ax1.set_title("回归加权模型：净值曲线（训练/样本外分界 2025-03）")
        ax1.set_ylabel("净值（各自区间起点=1）")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(to_dt(ins_strategy), drawdown(ins_m["nav"]), label="回归模型·样本内", color="#c0392b", lw=2)
        ax2.plot(to_dt(oos_strategy), drawdown(oos_m["nav"]), label="回归模型·样本外", color="#e67e22", lw=2)
        ax2.plot(to_dt(ew_strategy), drawdown(ew_m["nav"]), label="等权模型·样本外", color="#8e44ad", lw=2)
        ax2.plot(to_dt(ins_strategy), drawdown(ins_m["bnav"]), label="基准(全样本)", color="#2980b9", lw=2)
        ax2.axvline(boundary, color="gray", linestyle="--", linewidth=1.5)
        ax2.set_title("回撤曲线")
        ax2.set_ylabel("回撤")
        ax2.legend(loc="lower left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUT_PNG)
        plt.close(fig)
        print("已生成:", OUTPUT_PNG.name)
    except Exception as exc:
        print("图表生成失败（不影响其余输出）:", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())






