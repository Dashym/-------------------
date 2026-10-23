import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rd_expense_ratio import FACTOR_COLUMN


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
FACTOR_FILE = SCRIPT_DIR / "研发费用率_raw.csv"
QUOTE_FILE = (
    SCRIPT_DIR.parent.parent
    / "aksahre_demo"
    / "A股日线_20190101-20260331_不复权.csv"
)
OUTPUT_FILE = SCRIPT_DIR / "研发费用率_cleaned.csv"
MIN_CROSS_SECTION_SIZE = 2
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826
WINSORIZED_COLUMN = "研发费用率_winsor"
STANDARDIZED_COLUMN = "研发费用率_zscore"

FACTOR_COLUMNS = {"股票代码", "报告期", "可用日期", FACTOR_COLUMN}


def _normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def clean_rd_factor(factor, quotes, min_cross_section=MIN_CROSS_SECTION_SIZE):
    missing_columns = FACTOR_COLUMNS.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if not {"股票代码", "日期"}.issubset(quotes.columns):
        raise ValueError("行情数据必须包含“股票代码”和“日期”字段")
    if min_cross_section < 2:
        raise ValueError("标准化截面至少需要 2 只股票")

    data = factor.copy()
    data["股票代码"] = _normalize_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["可用日期"] = pd.to_datetime(data["可用日期"], errors="raise").dt.normalize()
    data[FACTOR_COLUMN] = pd.to_numeric(data[FACTOR_COLUMN], errors="coerce")

    duplicate_mask = data.duplicated(["股票代码", "可用日期"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    data = data.sort_values(
        ["股票代码", "可用日期", "报告期"],
        ascending=[True, True, False],
    ).drop_duplicates(["股票代码", "可用日期"], keep="first")
    if data.duplicated(["股票代码", "可用日期"]).any():
        raise ValueError("因子去重后仍存在股票代码和可用日期重复记录")

    missing_factor_rows = int((~np.isfinite(data[FACTOR_COLUMN])).sum())
    data = data.loc[np.isfinite(data[FACTOR_COLUMN])].copy()

    calendar = quotes.loc[:, ["股票代码", "日期"]].copy()
    calendar["股票代码"] = _normalize_codes(calendar["股票代码"])
    calendar["日期"] = pd.to_datetime(calendar["日期"], errors="raise").dt.normalize()
    calendar = calendar.dropna(subset=["股票代码", "日期"])
    if calendar.empty:
        raise ValueError("行情数据没有有效交易日期")
    trading_dates = set(calendar["日期"].unique())
    invalid_dates = sorted(set(data["可用日期"]) - trading_dates)
    if invalid_dates:
        raise ValueError(
            f"因子可用日期不是行情交易日，示例：{invalid_dates[0].strftime('%Y-%m-%d')}"
        )

    known = {}
    retained = []
    clipped_count = 0
    skipped_small_cross_sections = 0
    for available_date, updates in data.groupby("可用日期", sort=True):
        updated_codes = set()
        for _, update in updates.iterrows():
            code = update["股票代码"]
            known[code] = update.drop(labels="可用日期").to_dict()
            updated_codes.add(code)

        snapshot = pd.DataFrame.from_dict(known, orient="index")
        snapshot.index.name = "股票代码"
        snapshot = snapshot.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[FACTOR_COLUMN]
        )
        if len(snapshot) < min_cross_section:
            skipped_small_cross_sections += 1
            continue

        values = snapshot[FACTOR_COLUMN].astype(float)
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            lower = median - MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad
            upper = median + MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad
            winsorized = values.clip(lower=lower, upper=upper)
            clipped_count += int((values != winsorized).sum())
        else:
            winsorized = values
        snapshot[WINSORIZED_COLUMN] = winsorized

        standard_deviation = winsorized.std(ddof=0)
        if not np.isfinite(standard_deviation):
            raise ValueError(
                f"{available_date:%Y-%m-%d} 截面标准差无法计算"
            )
        if standard_deviation == 0:
            snapshot[STANDARDIZED_COLUMN] = 0.0
        else:
            snapshot[STANDARDIZED_COLUMN] = (
                winsorized - winsorized.mean()
            ) / standard_deviation

        available_updates = updated_codes.intersection(snapshot.index)
        if available_updates:
            new_rows = snapshot.loc[sorted(available_updates)].copy()
            new_rows["股票代码"] = new_rows.index
            new_rows["可用日期"] = available_date
            retained.append(new_rows.reset_index(drop=True))

    if not retained:
        raise ValueError("没有符合最小样本数要求的有效截面可用于清洗")
    cleaned = pd.concat(retained, ignore_index=True)
    cleaned["报告期"] = pd.to_datetime(cleaned["报告期"]).dt.strftime("%Y-%m-%d")
    cleaned["可用日期"] = pd.to_datetime(cleaned["可用日期"]).dt.strftime("%Y-%m-%d")
    ordered_columns = [
        column
        for column in (
            "股票代码",
            "报告期",
            "TTM最新所需财报公告日",
            "可用日期",
            "TTM研发费用",
            "TTM营业总收入",
            FACTOR_COLUMN,
            WINSORIZED_COLUMN,
            STANDARDIZED_COLUMN,
        )
        if column in cleaned.columns
    ]
    cleaned = cleaned[ordered_columns].sort_values(
        ["可用日期", "股票代码"]
    ).reset_index(drop=True)
    cleaned.attrs["cleaning_stats"] = {
        "duplicate_rows": duplicate_rows,
        "missing_factor_rows": missing_factor_rows,
        "clipped_values": clipped_count,
        "skipped_small_cross_sections": skipped_small_cross_sections,
    }
    return cleaned


def main():
    for input_file in (FACTOR_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(
        FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    quotes = pd.read_csv(
        QUOTE_FILE,
        usecols=["股票代码", "日期"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )
    cleaned = clean_rd_factor(factor, quotes)
    stats = cleaned.attrs["cleaning_stats"]
    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("因子名称：%s", FACTOR_COLUMN)
    log.info("清洗方式：缺失值剔除、MAD(%g倍) 去极值、截面 Z-score", MAD_MULTIPLIER)
    log.info(
        "重复记录涉及 %s 行；缺失/无效值剔除 %s 行；去极值截尾 %s 个值；"
        "因截面不足跳过 %s 个日期",
        stats["duplicate_rows"],
        stats["missing_factor_rows"],
        stats["clipped_values"],
        stats["skipped_small_cross_sections"],
    )
    log.info(
        "输出 %s 条；日期范围 %s ~ %s；文件：%s",
        len(cleaned),
        cleaned["可用日期"].min(),
        cleaned["可用日期"].max(),
        OUTPUT_FILE,
    )
    log.info(
        "标准化因子描述统计：\n%s",
        cleaned[STANDARDIZED_COLUMN].describe().to_string(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
