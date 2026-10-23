from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(r"C:\Users\Heyuanhang\Desktop\aksahre_demo")
FINANCIAL_FILE = BASE_DIR / "processed_data" / "财报_清洗完成.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_后复权.csv"
INDUSTRY_HISTORY_FILE = Path(r"C:\Users\Heyuanhang\Desktop\因子库\申万_汽车零部件_全量历史变动.csv")
OUTPUT_FILE = SCRIPT_DIR / "技术壁垒升级_月末截面.csv"
REPORTING_LAG_DAYS = 15


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def load_industry_history():
    df = pd.read_csv(INDUSTRY_HISTORY_FILE, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("行业历史分类文件为空")
    cols = {
        "股票代码": "股票代码",
        "三级行业名称": "三级行业名称",
        "纳入日": "纳入日",
        "更新日": "更新日",
    }
    selected = df.loc[:, list(cols.keys())].rename(columns=cols).copy()
    selected["股票代码"] = normalize_codes(selected["股票代码"])
    selected["纳入日"] = pd.to_datetime(selected["纳入日"], errors="coerce")
    selected["更新日"] = pd.to_datetime(selected["更新日"], errors="coerce")
    selected["三级行业名称"] = selected["三级行业名称"].fillna("").astype(str)
    selected["高壁垒标签"] = (
        (
            selected["三级行业名称"].str.contains("底盘", na=False)
            & selected["三级行业名称"].str.contains("发动机", na=False)
        )
        | (
            selected["三级行业名称"].str.contains("电子", na=False)
            & selected["三级行业名称"].str.contains("电气", na=False)
        )
    ).astype(int)
    return selected.sort_values(["股票代码", "纳入日"]).reset_index(drop=True)


def get_high_barrier_label(industry_history, month_end):
    month_end = pd.Timestamp(month_end).normalize()
    label_df = industry_history.loc[
        (industry_history["纳入日"] <= month_end)
        & ((industry_history["更新日"].isna()) | (industry_history["更新日"] >= month_end))
    ].copy()
    if label_df.empty:
        return 0
    label_df = label_df.sort_values(["股票代码", "纳入日"], ascending=[True, False])
    latest = label_df.drop_duplicates(subset=["股票代码"], keep="first")
    return latest.set_index("股票代码")["高壁垒标签"].to_dict()


def compute_ttm_metrics(financials):
    required = {"股票代码", "报告期", "首次公告日期", "营业总收入", "营业成本"}
    missing = required.difference(financials.columns)
    if missing:
        raise ValueError(f"财报数据缺少字段: {sorted(missing)}")

    data = financials.loc[:, ["股票代码", "报告期", "首次公告日期", "营业总收入", "营业成本"]].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="coerce")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data["营业总收入"] = pd.to_numeric(data["营业总收入"], errors="coerce")
    data["营业成本"] = pd.to_numeric(data["营业成本"], errors="coerce")
    data = data.dropna(subset=["股票代码", "报告期", "首次公告日期", "营业总收入", "营业成本"]).copy()
    data["信号可用日期"] = pd.concat(
        [
            data["首次公告日期"].dt.normalize(),
            (data["报告期"].dt.normalize() + pd.Timedelta(days=REPORTING_LAG_DAYS)),
        ],
        axis=1,
    ).max(axis=1)
    data = data.sort_values(["股票代码", "报告期"]).reset_index(drop=True)
    return data


def month_end_cross_sections(financials, industry_history):
    data = compute_ttm_metrics(financials)
    if data.empty:
        raise ValueError("财报数据为空")

    month_ends = pd.date_range("2019-01-31", "2024-12-31", freq="M")
    rows = []
    for month_end in month_ends:
        eligible = data.loc[data["信号可用日期"] <= month_end].copy()
        if eligible.empty:
            continue
        latest_by_stock = (
            eligible.sort_values(["股票代码", "信号可用日期", "报告期"], ascending=[True, True, True])
            .drop_duplicates("股票代码", keep="last")
            .copy()
        )
        latest_by_stock["季度"] = latest_by_stock["报告期"].dt.to_period("Q-DEC")
        latest_by_stock["股票代码"] = latest_by_stock["股票代码"].astype(str)
        latest_by_stock["高壁垒标签"] = latest_by_stock["股票代码"].map(get_high_barrier_label(industry_history, month_end))
        latest_by_stock["高壁垒标签"] = latest_by_stock["高壁垒标签"].fillna(0).astype(int)

        summary = []
        for code, g in latest_by_stock.groupby("股票代码", sort=True):
            q = g["季度"].iloc[0]
            window = pd.period_range(q - 3, q, freq="Q-DEC")
            available_financials = data.loc[
                (data["股票代码"] == code) & (data["信号可用日期"] <= month_end)
            ].copy()
            ttm = available_financials.loc[
                available_financials["报告期"].dt.to_period("Q-DEC").isin(window)
            ].copy()
            if (
                len(ttm) != 4
                or set(ttm["报告期"].dt.to_period("Q-DEC")) != set(window)
            ):
                continue
            ttm = ttm.sort_values("报告期")
            rev = ttm["营业总收入"].sum()
            cost = ttm["营业成本"].sum()
            if rev <= 0 or not np.isfinite(rev):
                continue
            gm_ttm = (rev - cost) / rev
            prior_window = pd.period_range(q - 7, q - 4, freq="Q-DEC")
            prior = available_financials.loc[
                available_financials["报告期"].dt.to_period("Q-DEC").isin(prior_window)
            ].copy()
            if (
                len(prior) != 4
                or set(prior["报告期"].dt.to_period("Q-DEC")) != set(prior_window)
            ):
                continue
            prior_ttm_rev = prior["营业总收入"].sum()
            prior_ttm_cost = prior["营业成本"].sum()
            if prior_ttm_rev <= 0 or not np.isfinite(prior_ttm_rev):
                continue
            gm_prior = (prior_ttm_rev - prior_ttm_cost) / prior_ttm_rev
            yoy_change = gm_ttm - gm_prior
            rev_growth = (rev - prior_ttm_rev) / abs(prior_ttm_rev)
            summary.append({
                "股票代码": code,
                "月末日期": month_end.strftime("%Y-%m-%d"),
                "财报信号可用日": pd.Timestamp(g["信号可用日期"].iloc[0]).strftime("%Y-%m-%d"),
                "高壁垒标签": int(g["高壁垒标签"].iloc[0]),
                "TTM营收": rev,
                "TTM营业成本": cost,
                "TTM毛利率": gm_ttm,
                "TTM毛利率同比变化": yoy_change,
                "TTM营收同比增长率": rev_growth,
            })

        if not summary:
            continue
        cross = pd.DataFrame(summary)
        cross = cross.replace([np.inf, -np.inf], np.nan).dropna(subset=["TTM毛利率同比变化", "TTM营收同比增长率"])
        if cross.empty:
            continue
        # winsorize each metric in the cross-section before z-score
        winsorized = cross.copy()
        for col in ["TTM毛利率同比变化", "TTM营收同比增长率"]:
            q1 = winsorized[col].quantile(0.01)
            q99 = winsorized[col].quantile(0.99)
            winsorized[col] = winsorized[col].clip(lower=q1, upper=q99)
        for col in ["TTM毛利率同比变化", "TTM营收同比增长率"]:
            std = winsorized[col].std(ddof=0)
            if std == 0 or not np.isfinite(std):
                winsorized[col] = 0.0
            else:
                winsorized[col] = (winsorized[col] - winsorized[col].mean()) / std
        winsorized["财务升级Z-score"] = (
            0.5 * winsorized["TTM毛利率同比变化"]
            + 0.5 * winsorized["TTM营收同比增长率"]
        )
        winsorized["财务升级Z-score"] = (
            winsorized["财务升级Z-score"] * winsorized["高壁垒标签"]
        )
        high_barrier = winsorized["高壁垒标签"].eq(1)
        high_barrier_values = winsorized.loc[high_barrier, "财务升级Z-score"]
        if not high_barrier_values.empty:
            standard_deviation = high_barrier_values.std(ddof=0)
            if standard_deviation > 0 and np.isfinite(standard_deviation):
                winsorized.loc[high_barrier, "财务升级Z-score"] = (
                    high_barrier_values - high_barrier_values.mean()
                ) / standard_deviation
            else:
                winsorized.loc[high_barrier, "财务升级Z-score"] = 0.0
        winsorized.loc[~high_barrier, "财务升级Z-score"] = 0.0
        winsorized["最终因子值"] = winsorized["财务升级Z-score"]
        winsorized = winsorized.loc[:, ["月末日期", "股票代码", "高壁垒标签", "财务升级Z-score", "最终因子值"]]
        rows.append(winsorized)

    if not rows:
        raise ValueError("未生成任何月末因子截面")
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["月末日期", "股票代码"]).reset_index(drop=True)
    return result


def main():
    financials = pd.read_csv(FINANCIAL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    industry_history = load_industry_history()
    cross_section = month_end_cross_sections(financials, industry_history)
    OUTPUT_FILE.write_text("", encoding="utf-8")
    cross_section.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(cross_section.head(10).to_string(index=False))
    print(f"\n输出条数：{len(cross_section)}")
    print(f"输出文件：{OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
