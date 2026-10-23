import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_前复权.csv"
OUTPUT_RAW = SCRIPT_DIR / "动量因子_raw.csv"
OUTPUT_CLEANED = SCRIPT_DIR / "动量因子_cleaned.csv"
REQUIRED_COLUMNS = {"股票代码", "日期", "收盘"}
FACTOR_COLUMNS = {"股票代码", "截面日期", "动量因子"}


def calculate_momentum(quotes):
    missing_columns = REQUIRED_COLUMNS.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = (
        data["股票代码"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )
    data["日期"] = pd.to_datetime(data["日期"], errors="raise")
    data["收盘"] = pd.to_numeric(data["收盘"], errors="raise")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0]
    data["月份"] = data["日期"].dt.to_period("M")

    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    if data.empty:
        raise ValueError("行情数据中没有有效的前复权收盘价")

    monthly = (
        data.sort_values(["股票代码", "日期"])
        .groupby(["月份", "股票代码"], as_index=False)
        .tail(1)
    )
    prices = monthly.pivot(index="月份", columns="股票代码", values="收盘")

    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)

    # At month t, use the t-1 month-end close relative to the t-12 month-end close.
    prior_month_close = prices.shift(1)
    twelve_months_prior_close = prices.shift(12)
    momentum = prior_month_close.div(twelve_months_prior_close).sub(1)
    momentum.index.name = "月份"
    momentum.columns.name = "股票代码"

    result = momentum.stack().rename("动量因子").reset_index()
    month_end_dates = data.groupby("月份")["日期"].max()
    result["截面日期"] = result["月份"].map(month_end_dates).dt.strftime("%Y-%m-%d")
    result["月份"] = result["月份"].astype(str)
    result = result[["股票代码", "截面日期", "动量因子"]].sort_values(
        ["截面日期", "股票代码"]
    )
    return result.reset_index(drop=True)


def clean_factor(factor):
    missing_columns = FACTOR_COLUMNS.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    cleaned = factor.loc[:, ["股票代码", "截面日期", "动量因子"]].copy()
    cleaned["股票代码"] = cleaned["股票代码"].astype(str).str.strip().str.zfill(6)
    cleaned["截面日期"] = pd.to_datetime(cleaned["截面日期"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    cleaned["动量因子"] = pd.to_numeric(cleaned["动量因子"], errors="coerce")
    finite = np.isfinite(cleaned["动量因子"])
    missing_count = int((~finite).sum())
    cleaned = cleaned.loc[finite].copy()

    if cleaned.duplicated(["股票代码", "截面日期"]).any():
        cleaned = cleaned.sort_values(
            ["股票代码", "截面日期", "动量因子"],
            ascending=[True, True, False],
        )
        cleaned = cleaned.drop_duplicates(["股票代码", "截面日期"], keep="first")

    def winsorize_mad(values):
        median = values.median()
        mad = (values - median).abs().median()
        if mad == 0 or not np.isfinite(mad):
            return values
        lower = median - 5 * 1.4826 * mad
        upper = median + 5 * 1.4826 * mad
        return values.clip(lower=lower, upper=upper)

    cleaned["动量因子_winsor"] = cleaned.groupby("截面日期")["动量因子"].transform(
        winsorize_mad
    )
    cross_section_sizes = cleaned.groupby("截面日期")["动量因子"].transform("count")
    cleaned = cleaned.loc[cross_section_sizes > 1].copy()

    def standardize(values):
        standard_deviation = values.std(ddof=0)
        if standard_deviation == 0 or not np.isfinite(standard_deviation):
            return pd.Series(0.0, index=values.index)
        return (values - values.mean()) / standard_deviation

    cleaned["动量因子_cleaned"] = cleaned.groupby("截面日期")[
        "动量因子_winsor"
    ].transform(standardize)
    cleaned = cleaned.sort_values(["截面日期", "股票代码"]).reset_index(drop=True)

    log.info(
        "缺失或非有限值剔除: %s; 清洗后记录: %s; 截面数: %s",
        missing_count,
        len(cleaned),
        cleaned["截面日期"].nunique(),
    )
    return cleaned


def main():
    if not QUOTE_FILE.is_file():
        log.error("前复权行情文件不存在: %s", QUOTE_FILE)
        return 1

    log.info("读取前复权行情: %s", QUOTE_FILE)
    quotes = pd.read_csv(QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    result = calculate_momentum(quotes)
    if result.empty:
        log.error("没有生成有效动量因子，请检查行情覆盖期")
        return 1

    cleaned = clean_factor(result)
    if cleaned.empty:
        log.error("清洗后没有有效动量因子记录")
        return 1

    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    cleaned.to_csv(OUTPUT_CLEANED, index=False, encoding="utf-8-sig")
    log.info("原始因子已输出: %s", OUTPUT_RAW)
    log.info("清洗因子已输出: %s", OUTPUT_CLEANED)
    log.info(
        "有效记录: %s; 截面范围: %s 至 %s; 首个截面日期: %s",
        len(cleaned),
        cleaned["截面日期"].min(),
        cleaned["截面日期"].max(),
        cleaned["截面日期"].min(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
