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
OUTPUT_RAW = SCRIPT_DIR / "研发费用率_raw.csv"
SAMPLE_START = pd.Period("2019Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2024Q4", freq="Q-DEC")
FACTOR_COLUMN = "研发费用率（TTM）"

FINANCIAL_COLUMNS = {
    "股票代码",
    "报告期",
    "首次公告日期",
    "研发费用",
    "营业总收入",
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


def build_rd_expense_ratio(financials, trading_days):
    missing_columns = FINANCIAL_COLUMNS.difference(financials.columns)
    if missing_columns:
        raise ValueError(f"财报数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if not {"股票代码", "日期"}.issubset(trading_days.columns):
        raise ValueError("交易日数据必须包含“股票代码”和“日期”字段")

    data = financials.loc[
        :, ["股票代码", "报告期", "首次公告日期", "研发费用", "营业总收入"]
    ].copy()
    data["股票代码"] = _normalize_stock_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data["研发费用"] = pd.to_numeric(data["研发费用"], errors="coerce")
    data["营业总收入"] = pd.to_numeric(data["营业总收入"], errors="coerce")
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
    research_expenses = data.pivot(
        index="季度", columns="股票代码", values="研发费用"
    ).reindex(full_quarters)
    operating_revenues = data.pivot(
        index="季度", columns="股票代码", values="营业总收入"
    ).reindex(full_quarters)
    announcements = data.pivot(
        index="季度", columns="股票代码", values="首次公告日期"
    ).reindex(full_quarters)

    ttm_research_expenses = research_expenses.rolling(
        window=4, min_periods=4
    ).sum()
    ttm_operating_revenues = operating_revenues.rolling(
        window=4, min_periods=4
    ).sum()
    ratio = ttm_research_expenses.div(
        ttm_operating_revenues.where(ttm_operating_revenues > 0)
    )

    calendar = trading_days.loc[:, ["日期"]].copy()
    calendar["日期"] = pd.to_datetime(calendar["日期"], errors="raise").dt.normalize()
    calendar = calendar.dropna().drop_duplicates().sort_values("日期")
    if calendar.empty:
        raise ValueError("交易日数据没有有效日期")

    measure_rows = []
    for quarter in full_quarters:
        if quarter < SAMPLE_START or quarter > SAMPLE_END:
            continue
        quarter_index = full_quarters.get_loc(quarter)
        if quarter_index < 3:
            continue
        component_quarters = full_quarters[quarter_index - 3 : quarter_index + 1]
        for code in research_expenses.columns:
            expense = ttm_research_expenses.at[quarter, code]
            revenue = ttm_operating_revenues.at[quarter, code]
            factor_value = ratio.at[quarter, code]
            component_announcements = announcements.loc[component_quarters, code]
            if (
                not np.isfinite(expense)
                or expense < 0
                or not np.isfinite(revenue)
                or revenue <= 0
                or not np.isfinite(factor_value)
                or component_announcements.isna().any()
            ):
                continue

            latest_announcement = component_announcements.max()
            measure_rows.append(
                {
                    "股票代码": code,
                    "报告期": quarter.end_time.normalize(),
                    "TTM最新所需财报公告日": latest_announcement,
                    "TTM研发费用": expense,
                    "TTM营业总收入": revenue,
                    FACTOR_COLUMN: factor_value,
                }
            )

    result = pd.DataFrame(measure_rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "股票代码",
                "报告期",
                "TTM最新所需财报公告日",
                "可用日期",
                "TTM研发费用",
                "TTM营业总收入",
                FACTOR_COLUMN,
            ]
        )

    result = pd.merge_asof(
        result.sort_values("TTM最新所需财报公告日"),
        calendar.rename(columns={"日期": "可用日期"}),
        left_on="TTM最新所需财报公告日",
        right_on="可用日期",
        direction="forward",
        allow_exact_matches=False,
    )
    result = result.dropna(subset=["可用日期"])
    for column in ("报告期", "TTM最新所需财报公告日", "可用日期"):
        result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result.sort_values(["报告期", "股票代码"]).reset_index(drop=True)


def main():
    for input_file in (FINANCIAL_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    financials = pd.read_csv(
        FINANCIAL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    quotes = pd.read_csv(
        QUOTE_FILE,
        usecols=["股票代码", "日期"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )
    result = build_rd_expense_ratio(financials, quotes)
    if result.empty:
        log.error("样本区间内没有可计算的研发费用率（TTM）")
        return 1

    result.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    log.info("因子名称：%s", FACTOR_COLUMN)
    log.info("计算方式：TTM研发费用 / TTM营业总收入")
    log.info("输出文件：%s", OUTPUT_RAW)
    log.info(
        "有效记录：%s；报告期范围：%s ~ %s；可用日期范围：%s ~ %s",
        len(result),
        result["报告期"].min(),
        result["报告期"].max(),
        result["可用日期"].min(),
        result["可用日期"].max(),
    )
    log.info("%s 描述统计：\n%s", FACTOR_COLUMN, result[FACTOR_COLUMN].describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
