# -*- coding: utf-8 -*-
"""毛利率稳定性 vs 存货周转改善率 多重共线性检验。

因子值口径（与分层回测一致）：
  - 毛利率稳定性：MAD(5x) 截尾 + Z-score（不做市值中性化）
  - 存货周转改善率：MAD(5x) 截尾 + 对数流通市值中性化 + Z-score
  - 截面：每月末，asof 取「可用日期 <= 月末」最新值
  - 样本：当月两因子都非空的股票（交集，缺失剔除不填充）
  - 区间：2022-05 ~ 2024-12（32 个月）
  - 相关系数：Pearson + Spearman，逐月计算得到时间序列
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\Heyuanhang\Desktop\aksahre_demo")
LIB_DIR = BASE_DIR.parent / "因子库"
GM_FACTOR_FILE = LIB_DIR / "毛利率稳定性（可以，小心）" / "毛利率稳定性_raw.csv"
INV_FACTOR_FILE = LIB_DIR / "存货周转改善率（good）" / "存货周转改善率_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
DST = Path(r"C:\Users\Heyuanhang\Desktop\多重共线性")

GM_FACTOR_COLUMN = "毛利率稳定性"
INV_FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
MIN_CROSS_SECTION_SIZE = 10

START = pd.Period("2022-05", freq="M")
END = pd.Period("2024-12", freq="M")

OUTPUT_REPORT = DST / "多重共线性检验结果.txt"
OUTPUT_PNG = DST / "相关系数时间序列.png"


def _normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def _monthly_quotes(quotes):
    columns = ["股票代码", "日期", "收盘", "成交量", "换手率"]
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


def _summarize(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "min": float(s.min()),
        "max": float(s.max()),
        "count": int(len(s)),
    }


def _level(mean_abs):
    if not np.isfinite(mean_abs):
        return "无法判断"
    if mean_abs < 0.3:
        return "无共线性（|r| < 0.3）"
    if mean_abs <= 0.7:
        return "弱到中等共线性（0.3 ≤ |r| ≤ 0.7）"
    return "强共线性（|r| > 0.7，需注意）"


def _dist_levels(series):
    s = pd.Series(series).dropna().abs()
    if s.empty:
        return (0, 0, 0)
    return (
        int((s < 0.3).sum()),
        int(((s >= 0.3) & (s <= 0.7)).sum()),
        int((s > 0.7).sum()),
    )


def main():
    for path in (GM_FACTOR_FILE, INV_FACTOR_FILE, UNADJUSTED_QUOTE_FILE):
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
    market_month = _monthly_quotes(unadjusted)
    ln_mcap = market_month.pivot(index="月份", columns="股票代码", values="ln流通市值")
    all_months = pd.period_range(ln_mcap.index.min(), ln_mcap.index.max(), freq="M")
    ln_mcap = ln_mcap.reindex(all_months)

    months = [m for m in all_months if START <= m <= END]

    records = []
    for month in months:
        month_end = pd.Timestamp(month.end_time).normalize()
        gm_latest = _latest_signals(gm_signals, month_end)
        inv_latest = _latest_signals(inv_signals, month_end)
        if len(gm_latest) < MIN_CROSS_SECTION_SIZE or len(inv_latest) < MIN_CROSS_SECTION_SIZE:
            continue
        z_gm = _winsorize_zscore(gm_latest[GM_FACTOR_COLUMN])
        if z_gm is None:
            continue
        if month not in ln_mcap.index:
            continue
        mcap_row = ln_mcap.loc[month].reindex(inv_latest.index)
        z_inv = _neutralize_zscore(inv_latest[INV_FACTOR_COLUMN], mcap_row)
        if z_inv is None:
            continue
        common = z_gm.index.intersection(z_inv.index)
        if len(common) < MIN_CROSS_SECTION_SIZE:
            continue
        g = z_gm.loc[common]
        iv = z_inv.loc[common]
        records.append({
            "月份": str(month),
            "股票数": int(len(common)),
            "Pearson": float(g.corr(iv)),
            "Spearman": float(g.corr(iv, method="spearman")),
        })

    if not records:
        print("检验区间内没有有效截面")
        return 1

    results = pd.DataFrame(records)
    pearson_stat = _summarize(results["Pearson"])
    spearman_stat = _summarize(results["Spearman"])

    mean_pearson = pearson_stat["mean"]
    vif = 1.0 / (1.0 - mean_pearson ** 2) if abs(mean_pearson) < 1.0 else float("inf")
    p_low, p_mid, p_high = _dist_levels(results["Pearson"])
    s_low, s_mid, s_high = _dist_levels(results["Spearman"])

    def _vif_level(v):
        if not np.isfinite(v):
            return "无法判断"
        if v < 5:
            return "无共线性（VIF < 5）"
        if v < 10:
            return "中等共线性（5 ≤ VIF < 10）"
        return "强共线性（VIF ≥ 10）"

    lines = [
        "=" * 74,
        "毛利率稳定性 vs 存货周转改善率 多重共线性检验报告",
        "=" * 74,
        "",
        "检验口径：",
        "  因子值版本：毛利率稳定性 = MAD(5x)截尾 + Z-score（不做市值中性化）",
        "              存货周转改善率 = MAD(5x)截尾 + 对数流通市值中性化 + Z-score",
        "  截面时间点：每月末，asof 取「可用日期 <= 月末」最新因子值",
        "  样本范围：当月两因子都非空的全部股票（缺失剔除、不填充）",
        f"  检验区间：{START} ~ {END}（有效截面 {pearson_stat['count']} 期）",
        "  相关系数：Pearson（线性）+ Spearman（秩相关，看单调性稳定性）",
        "  计算频率：每月末截面计算一次，得到相关系数时间序列",
        "  判断阈值：|r| < 0.3 无共线性；0.3 ≤ |r| ≤ 0.7 弱到中等；|r| > 0.7 强共线性",
        "",
        "数据概况：",
        f"  毛利率稳定性因子：{len(gm_signals):,} 条，可用日期 {gm_signals['可用日期'].min().strftime('%Y-%m-%d')} ~ {gm_signals['可用日期'].max().strftime('%Y-%m-%d')}",
        f"  存货周转改善率因子：{len(inv_signals):,} 条，可用日期 {inv_signals['可用日期'].min().strftime('%Y-%m-%d')} ~ {inv_signals['可用日期'].max().strftime('%Y-%m-%d')}",
        f"  有效截面：{pearson_stat['count']} 期（{results['月份'].iloc[0]} ~ {results['月份'].iloc[-1]}）",
        f"  平均有效股票数：{results['股票数'].mean():.1f}，范围 {results['股票数'].min()} ~ {results['股票数'].max()}",
        "",
        "一、相关系数时间序列汇总：",
        f"{'指标':<12}{'Pearson':>14}{'Spearman':>14}",
        "-" * 40,
        f"{'均值':<12}{pearson_stat['mean']:>14.4f}{spearman_stat['mean']:>14.4f}",
        f"{'标准差':<12}{pearson_stat['std']:>14.4f}{spearman_stat['std']:>14.4f}",
        f"{'最小值':<12}{pearson_stat['min']:>14.4f}{spearman_stat['min']:>14.4f}",
        f"{'最大值':<12}{pearson_stat['max']:>14.4f}{spearman_stat['max']:>14.4f}",
        "",
        "二、逐月明细：",
        f"{'月份':<10}{'股票数':>8}{'Pearson':>12}{'Spearman':>12}",
        "-" * 42,
    ]
    for _, row in results.iterrows():
        lines.append(
            f"{row['月份']:<10}{row['股票数']:>8d}{row['Pearson']:>12.4f}{row['Spearman']:>12.4f}"
        )

    lines += [
        "",
        "三、共线性判断（按 |r| 均值）：",
        f"  Pearson 均值 |r| = {abs(mean_pearson):.4f} → {_level(abs(mean_pearson))}",
        f"  Spearman 均值 |ρ| = {abs(spearman_stat['mean']):.4f} → {_level(abs(spearman_stat['mean']))}",
        f"  逐月 |r| 分布（Pearson）：<0.3 共 {p_low} 期；0.3~0.7 共 {p_mid} 期；>0.7 共 {p_high} 期",
        f"  逐月 |ρ| 分布（Spearman）：<0.3 共 {s_low} 期；0.3~0.7 共 {s_mid} 期；>0.7 共 {s_high} 期",
        "",
        "四、VIF 换算（两因子，VIF = 1/(1-r²)，用 Pearson 均值）：",
        f"  VIF = {vif:.4f} → {_vif_level(vif)}",
        "",
        "五、结论：",
    ]
    if abs(mean_pearson) < 0.3:
        lines.append("  两因子间不存在明显共线性（|r| 均值 < 0.3），可同时纳入多因子模型。")
    elif abs(mean_pearson) <= 0.7:
        lines.append("  两因子间存在弱到中等共线性（0.3 ≤ |r| 均值 ≤ 0.7），同时纳入时需注意信息重叠。")
    else:
        lines.append("  两因子间存在强共线性（|r| 均值 > 0.7），建议只保留其一或做正交化处理。")
    lines += ["", "=" * 74]

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        x = pd.DatetimeIndex(
            [pd.Period(m, freq="M").to_timestamp() for m in results["月份"]]
        )
        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
        ax.plot(x, results["Pearson"], marker="o", lw=1.6, label="Pearson r")
        ax.plot(x, results["Spearman"], marker="s", lw=1.6, label="Spearman ρ")
        ax.axhline(0.3, color="orange", ls="--", lw=1, alpha=0.8, label="±0.3 阈值")
        ax.axhline(-0.3, color="orange", ls="--", lw=1, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("毛利率稳定性 vs 存货周转改善率 相关系数时间序列（2022-05 ~ 2024-12）")
        ax.set_xlabel("月份")
        ax.set_ylabel("相关系数")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUT_PNG)
        plt.close(fig)
        print("已生成:", OUTPUT_PNG.name)
    except Exception as exc:
        print("图表生成失败:", exc)

    print("已输出:", OUTPUT_REPORT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())


