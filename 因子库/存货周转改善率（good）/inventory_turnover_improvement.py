import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FINANCIAL_FILE = BASE_DIR / "processed_data" / "财报_清洗完成.csv"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
OUTPUT_RAW = SCRIPT_DIR / "存货周转改善率_raw.csv"
SAMPLE_START = pd.Period("2019Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2026Q2", freq="Q-DEC")

FINANCIAL_COLUMNS = {
    "股票代码",
    "报告期",
    "首次公告日期",
    "营业成本",
    "存货",
}
def _normalize_stock_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def build_turnover_improvement(financials, trading_days):
    missing_columns = FINANCIAL_COLUMNS.difference(financials.columns)
    if missing_columns:
        raise ValueError(f"财报数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if "股票代码" not in trading_days.columns or "日期" not in trading_days.columns:
        raise ValueError("交易日数据必须包含“股票代码”和“日期”字段")

    data = financials.loc[
        :, ["股票代码", "报告期", "首次公告日期", "营业成本", "存货"]
    ].copy()
    data["股票代码"] = _normalize_stock_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data["营业成本"] = pd.to_numeric(data["营业成本"], errors="coerce")
    data["存货"] = pd.to_numeric(data["存货"], errors="coerce")
    data["季度"] = data["报告期"].dt.to_period("Q-DEC")
    data = data.loc[data["季度"] <= SAMPLE_END].copy()

    if data.duplicated(["股票代码", "季度"]).any():
        raise ValueError("财报数据存在重复的股票代码和报告季度")
    if data.empty:
        raise ValueError("样本区间内没有财报记录")

    data = data.sort_values(["季度", "股票代码"])
    full_quarters = pd.period_range(
        data["季度"].min(), data["季度"].max(), freq="Q-DEC"
    )
    costs = data.pivot(index="季度", columns="股票代码", values="营业成本")
    inventories = data.pivot(index="季度", columns="股票代码", values="存货")
    costs = costs.reindex(full_quarters)
    inventories = inventories.reindex(full_quarters)

    ttm_cost = costs.rolling(window=4, min_periods=4).sum()
    average_inventory = inventories.rolling(window=4, min_periods=4).mean()
    turnover = ttm_cost.div(average_inventory.where(average_inventory > 0))
    baseline_turnover = turnover.shift(4)
    improvement = turnover.sub(baseline_turnover).div(baseline_turnover.abs())

    measure_rows = []
    for quarter in full_quarters:
        if quarter < SAMPLE_START or quarter > SAMPLE_END:
            continue
        quarter_end = quarter.end_time.normalize()
        for code in costs.columns:
            current_turnover = turnover.at[quarter, code]
            base_turnover = baseline_turnover.at[quarter, code]
            current_ttm_cost = ttm_cost.at[quarter, code]
            current_average_inventory = average_inventory.at[quarter, code]
            base_quarter = quarter - 4

            if not (
                np.isfinite(current_turnover)
                and current_turnover > 0
                and np.isfinite(base_turnover)
                and base_turnover > 0
            ):
                continue

            current_report = data.loc[
                (data["股票代码"] == code) & (data["季度"] == quarter)
            ]
            if current_report.empty:
                continue
            report = current_report.iloc[0]
            if pd.isna(report["首次公告日期"]):
                continue

            measure_rows.append(
                {
                    "股票代码": code,
                    "报告期": quarter_end,
                    "首次公告日期": report["首次公告日期"],
                    "基期报告期": base_quarter.end_time.normalize(),
                    "TTM营业成本": current_ttm_cost,
                    "平均存货余额": current_average_inventory,
                    "当期存货周转率": current_turnover,
                    "基期存货周转率": base_turnover,
                    "存货周转改善率（TTM同比）": improvement.at[quarter, code],
                }
            )

    result = pd.DataFrame(measure_rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "股票代码",
                "报告期",
                "首次公告日期",
                "可用日期",
                "基期报告期",
                "TTM营业成本",
                "平均存货余额",
                "当期存货周转率",
                "基期存货周转率",
                "存货周转改善率（TTM同比）",
            ]
        )

    calendar = trading_days.loc[:, ["日期"]].copy()
    calendar["日期"] = pd.to_datetime(calendar["日期"], errors="raise")
    calendar = calendar.dropna().drop_duplicates().sort_values("日期")
    if calendar.empty:
        raise ValueError("交易日数据没有有效日期")

    result["首次公告日期"] = pd.to_datetime(result["首次公告日期"])
    result = pd.merge_asof(
        result.sort_values("首次公告日期"),
        calendar.rename(columns={"日期": "可用日期"}),
        left_on="首次公告日期",
        right_on="可用日期",
        direction="forward",
        allow_exact_matches=False,
    )
    result = result.dropna(subset=["可用日期"])
    result["报告期"] = result["报告期"].dt.strftime("%Y-%m-%d")
    result["首次公告日期"] = result["首次公告日期"].dt.strftime("%Y-%m-%d")
    result["可用日期"] = result["可用日期"].dt.strftime("%Y-%m-%d")
    result["基期报告期"] = result["基期报告期"].dt.strftime("%Y-%m-%d")
    result = result[
        [
            "股票代码",
            "报告期",
            "首次公告日期",
            "可用日期",
            "基期报告期",
            "TTM营业成本",
            "平均存货余额",
            "当期存货周转率",
            "基期存货周转率",
            "存货周转改善率（TTM同比）",
        ]
    ]
    result = result.sort_values(["报告期", "股票代码"]).reset_index(drop=True)
    return result


def main():
    for input_file in (FINANCIAL_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    financials = pd.read_csv(
        FINANCIAL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    quotes = pd.read_csv(
        QUOTE_FILE, usecols=["股票代码", "日期"], dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    result = build_turnover_improvement(financials, quotes)
    if result.empty:
        log.error("样本区间内没有可计算的存货周转改善率")
        return 1

    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    log.info("因子名称：存货周转改善率（TTM同比）")
    log.info("输出文件：%s", OUTPUT_RAW)
    log.info(
        "有效记录：%s；报告期范围：%s ~ %s；可用日期范围：%s ~ %s",
        len(result),
        result["报告期"].min(),
        result["报告期"].max(),
        result["可用日期"].min(),
        result["可用日期"].max(),
    )
    log.info(
        "改善率描述统计：\n%s",
        result["存货周转改善率（TTM同比）"].describe().to_string(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
