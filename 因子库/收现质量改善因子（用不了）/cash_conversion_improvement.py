# -*- coding: utf-8 -*-
"""收现质量改善因子（TTM 同比改善）构建脚本
==========================================
现金流质量类因子：衡量企业"销售回款质量（现金含量）"相对去年同期的改善程度。

因子定义：
    收现质量        = 经营活动产生的现金流量净额TTM / 营业总收入TTM
    收现质量改善因子 = 当期收现质量 − 去年同期收现质量
    因子为正 → 回款质量同比改善（方向：值越大越好）

数据口径（与 07_财报爬取.py / etl_clean_quarterly_report.py 一致）：
    1. processed_data/财报_清洗完成.csv 中"营业总收入""经营活动产生的现金流量净额"
       已经是**单季度值**（爬取阶段按同年内差分还原），此处直接使用；
    2. 单季度值在**连续季度网格**上滚动求和得到 TTM（缺季度 → NaN，不做插值）；
    3. 基期 = 季度网格 shift(4)，即去年同期 TTM（网格连续，等价于自然年同季度）；
    4. 可用日期 = 因子涉及的 8 个季度（当期 TTM 4 季 + 基期 TTM 4 季）中**最晚的首次公告日期**
       之后的第一个交易日（allow_exact_matches=False），保证不使用尚未披露的数据；
    5. 防未来函数：报告期 ≤ 2024Q3，可用日期 ≤ 2024-11-01。

输出：
    收现质量改善因子_raw.csv   —— 报告期级因子值（IC / 单调性 / 回归检验的输入）

运行环境：aksahre_demo/.venv（pandas / numpy）
    python cash_conversion_improvement.py
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
FINANCIAL_FILE = BASE_DIR / "processed_data" / "财报_清洗完成.csv"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
OUTPUT_RAW = SCRIPT_DIR / "收现质量改善因子_raw.csv"

FACTOR_COLUMN = "收现质量改善因子"
SAMPLE_START = pd.Period("2019Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2024Q3", freq="Q-DEC")
MAX_AVAILABLE_DATE = pd.Timestamp("2024-11-01")

FINANCIAL_COLUMNS = {
    "股票代码",
    "报告期",
    "首次公告日期",
    "营业总收入",
    "经营活动产生的现金流量净额",
}
OUTPUT_COLUMNS = [
    "股票代码",
    "报告期",
    "首次公告日期",
    "最晚披露日期",
    "可用日期",
    "基期报告期",
    "TTM营业总收入",
    "TTM经营活动现金流量净额",
    "当期收现质量",
    "基期收现质量",
    FACTOR_COLUMN,
]


def _normalize_stock_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def build_cash_conversion_improvement(financials, trading_days):
    """由单季度财报构建收现质量改善因子。

    参数
    ----
    financials : DataFrame
        processed_data/财报_清洗完成.csv 的内容，流量字段为单季度值。
    trading_days : DataFrame
        行情数据（至少包含"股票代码"和"日期"），用于确定"公告日后第一个交易日"。

    返回
    ----
    DataFrame : 报告期级因子值，列见 OUTPUT_COLUMNS。
    """
    missing_columns = FINANCIAL_COLUMNS.difference(financials.columns)
    if missing_columns:
        raise ValueError(f"财报数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if "股票代码" not in trading_days.columns or "日期" not in trading_days.columns:
        raise ValueError("交易日数据必须包含“股票代码”和“日期”字段")

    data = financials.loc[
        :, ["股票代码", "报告期", "首次公告日期", "营业总收入", "经营活动产生的现金流量净额"]
    ].copy()
    data["股票代码"] = _normalize_stock_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data["营业总收入"] = pd.to_numeric(data["营业总收入"], errors="coerce")
    data["经营活动产生的现金流量净额"] = pd.to_numeric(
        data["经营活动产生的现金流量净额"], errors="coerce"
    )
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
    revenues = data.pivot(index="季度", columns="股票代码", values="营业总收入")
    cashflows = data.pivot(
        index="季度", columns="股票代码", values="经营活动产生的现金流量净额"
    )
    notices = data.pivot(index="季度", columns="股票代码", values="首次公告日期")
    revenues = revenues.reindex(full_quarters)
    cashflows = cashflows.reindex(full_quarters)
    notices = notices.reindex(full_quarters)

    ttm_revenue = revenues.rolling(window=4, min_periods=4).sum()
    ttm_cashflow = cashflows.rolling(window=4, min_periods=4).sum()
    conversion = ttm_cashflow.div(ttm_revenue.where(ttm_revenue > 0))
    baseline_conversion = conversion.shift(4)
    improvement = conversion.sub(baseline_conversion)

    notice_values = (
        notices.to_numpy(dtype="datetime64[ns]").astype("int64").astype("float64")
    )
    notice_values = np.where(
        notice_values == np.iinfo("int64").min, np.nan, notice_values
    )
    notices_numeric = pd.DataFrame(
        notice_values, index=notices.index, columns=notices.columns
    )
    # 因子涉及的 8 个季度（当期 TTM 4 季 + 基期 TTM 4 季）中最晚的首次公告日期
    latest_notice = notices_numeric.rolling(window=8, min_periods=8).max()

    measure_rows = []
    for quarter in full_quarters:
        if quarter < SAMPLE_START or quarter > SAMPLE_END:
            continue
        quarter_end = quarter.end_time.normalize()
        base_quarter = quarter - 4
        for code in revenues.columns:
            current_conversion = conversion.at[quarter, code]
            base_conversion = baseline_conversion.at[quarter, code]
            notice = latest_notice.at[quarter, code]
            current_report = data.loc[
                (data["股票代码"] == code) & (data["季度"] == quarter)
            ]

            if current_report.empty or not np.isfinite(notice):
                continue
            if not (
                np.isfinite(current_conversion) and np.isfinite(base_conversion)
            ):
                continue

            measure_rows.append(
                {
                    "股票代码": code,
                    "报告期": quarter_end,
                    "首次公告日期": current_report.iloc[0]["首次公告日期"],
                    "最晚披露日期": pd.to_datetime(notice, unit="ns"),
                    "基期报告期": base_quarter.end_time.normalize(),
                    "TTM营业总收入": ttm_revenue.at[quarter, code],
                    "TTM经营活动现金流量净额": ttm_cashflow.at[quarter, code],
                    "当期收现质量": current_conversion,
                    "基期收现质量": base_conversion,
                    FACTOR_COLUMN: improvement.at[quarter, code],
                }
            )

    if not measure_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.DataFrame(measure_rows)
    for column in ("报告期", "首次公告日期", "最晚披露日期", "基期报告期"):
        result[column] = pd.to_datetime(result[column], errors="raise").astype(
            "datetime64[ns]"
        )

    calendar = trading_days.loc[:, ["日期"]].copy()
    calendar["日期"] = pd.to_datetime(calendar["日期"], errors="raise").astype(
        "datetime64[ns]"
    )
    calendar = calendar.dropna().drop_duplicates().sort_values("日期")
    if calendar.empty:
        raise ValueError("交易日数据没有有效日期")

    result = pd.merge_asof(
        result.sort_values("最晚披露日期"),
        calendar.rename(columns={"日期": "可用日期"}),
        left_on="最晚披露日期",
        right_on="可用日期",
        direction="forward",
        allow_exact_matches=False,
    )
    result = result.loc[result["可用日期"] <= MAX_AVAILABLE_DATE]
    result = result.dropna(subset=["可用日期"])
    if result.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result["报告期"] = result["报告期"].dt.strftime("%Y-%m-%d")
    result["首次公告日期"] = result["首次公告日期"].dt.strftime("%Y-%m-%d")
    result["最晚披露日期"] = result["最晚披露日期"].dt.strftime("%Y-%m-%d")
    result["可用日期"] = result["可用日期"].dt.strftime("%Y-%m-%d")
    result["基期报告期"] = result["基期报告期"].dt.strftime("%Y-%m-%d")
    result = result[OUTPUT_COLUMNS]
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
        QUOTE_FILE,
        usecols=["股票代码", "日期"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )
    result = build_cash_conversion_improvement(financials, quotes)
    if result.empty:
        log.error("样本区间内没有可计算的收现质量改善因子")
        return 1

    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    log.info("因子名称：收现质量改善因子（经营活动现金流净额TTM / 营业总收入TTM 的 TTM 同比改善）")
    log.info("输出文件：%s", OUTPUT_RAW)
    log.info(
        "有效记录：%s；股票数：%s；报告期范围：%s ~ %s；可用日期范围：%s ~ %s",
        len(result),
        result["股票代码"].nunique(),
        result["报告期"].min(),
        result["报告期"].max(),
        result["可用日期"].min(),
        result["可用日期"].max(),
    )
    log.info("因子描述统计：\n%s", result[FACTOR_COLUMN].describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
