import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "存货周转改善率_raw.csv"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
OUTPUT_FILE = SCRIPT_DIR / "存货周转改善率_cleaned.csv"
FACTOR_COLUMN = "存货周转改善率（TTM同比）"
MIN_CROSS_SECTION_SIZE = 3

FACTOR_COLUMNS = {"股票代码", "报告期", "可用日期", FACTOR_COLUMN}
QUOTE_COLUMNS = {"股票代码", "日期", "收盘", "成交量", "换手率"}


def _normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def _market_capitalization(quotes):
    missing_columns = QUOTE_COLUMNS.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    market = quotes.loc[:, ["股票代码", "日期", "收盘", "成交量", "换手率"]].copy()
    market["股票代码"] = _normalize_codes(market["股票代码"])
    market["日期"] = pd.to_datetime(market["日期"], errors="raise").dt.normalize()
    for column in ("收盘", "成交量", "换手率"):
        market[column] = pd.to_numeric(market[column], errors="coerce")

    if market.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")

    valid_turnover = market.loc[market["换手率"] > 0, "换手率"]
    if valid_turnover.empty:
        raise ValueError("行情中没有有效的正换手率，无法估算流通市值")
    turnover_is_percent = valid_turnover.max() > 1.5
    turnover_scale = 10000 if turnover_is_percent else 100

    market["流通股本估算"] = (
        market["成交量"] / market["换手率"] * turnover_scale
    )
    market["流通市值"] = market["收盘"] * market["流通股本估算"]
    market["ln流通市值"] = np.log(
        market["流通市值"].where(market["流通市值"] > 0)
    )
    market.loc[
        ~np.isfinite(market["ln流通市值"]), "ln流通市值"
    ] = np.nan
    return market[
        ["股票代码", "日期", "流通股本估算", "流通市值", "ln流通市值"]
    ], turnover_is_percent


def clean_inventory_factor(factor, quotes, min_cross_section=MIN_CROSS_SECTION_SIZE):
    missing_columns = FACTOR_COLUMNS.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if min_cross_section < 3:
        raise ValueError("市值中性化至少需要 3 只股票的截面")

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

    market, turnover_is_percent = _market_capitalization(quotes)
    missing_factor_rows = int((~np.isfinite(data[FACTOR_COLUMN])).sum())
    market_by_date = {
        date: frame.set_index("股票代码")
        for date, frame in market.groupby("日期", sort=False)
    }
    known = {}
    retained = []
    clipped_count = 0
    missing_market_rows = 0

    for available_date, updates in data.groupby("可用日期", sort=True):
        updated_codes = set()
        for _, update in updates.iterrows():
            code = update["股票代码"]
            if not np.isfinite(update[FACTOR_COLUMN]):
                known.pop(code, None)
                continue
            known[code] = update.drop(labels="可用日期").to_dict()
            updated_codes.add(code)

        current_market = market_by_date.get(available_date)
        if current_market is None or not known:
            missing_market_rows += len(updated_codes)
            continue

        snapshot = pd.DataFrame.from_dict(known, orient="index")
        snapshot.index.name = "股票代码"
        current_market = current_market.reindex(snapshot.index)
        snapshot = snapshot.join(current_market, how="inner")
        valid_snapshot = (
            np.isfinite(snapshot[FACTOR_COLUMN])
            & np.isfinite(snapshot["ln流通市值"])
        )
        missing_market_rows += sum(
            code in updated_codes and not valid
            for code, valid in valid_snapshot.items()
        )
        snapshot = snapshot.loc[valid_snapshot].copy()
        if len(snapshot) < min_cross_section:
            continue

        values = snapshot[FACTOR_COLUMN]
        median = values.median()
        mad = (values - median).abs().median()
        if mad != 0 and np.isfinite(mad):
            lower = median - 5 * 1.4826 * mad
            upper = median + 5 * 1.4826 * mad
            clipped_values = values.clip(lower=lower, upper=upper)
            clipped_count += int((values != clipped_values).sum())
        else:
            clipped_values = values
        snapshot["存货周转改善率_winsor"] = clipped_values

        design = np.column_stack(
            [np.ones(len(snapshot)), snapshot["ln流通市值"].to_numpy(dtype=float)]
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, clipped_values.to_numpy(dtype=float), rcond=None
        )
        residuals = clipped_values.to_numpy(dtype=float) - design @ coefficients
        snapshot["存货周转改善率_neutral"] = residuals
        residual_std = residuals.std(ddof=0)
        if residual_std == 0 or not np.isfinite(residual_std):
            snapshot["存货周转改善率_cleaned"] = 0.0
        else:
            snapshot["存货周转改善率_cleaned"] = (
                residuals - residuals.mean()
            ) / residual_std

        available_updates = updated_codes.intersection(snapshot.index)
        if available_updates:
            new_rows = snapshot.loc[sorted(available_updates)].copy()
            new_rows["可用日期"] = available_date
            new_rows["股票代码"] = new_rows.index
            retained.append(new_rows.reset_index(drop=True))

    if not retained:
        raise ValueError("没有符合最小样本数要求且有行情数据的截面可用于清洗")
    cleaned = pd.concat(retained, ignore_index=True)

    cleaned["报告期"] = cleaned["报告期"].dt.strftime("%Y-%m-%d")
    cleaned["可用日期"] = cleaned["可用日期"].dt.strftime("%Y-%m-%d")
    cleaned = cleaned[
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
            FACTOR_COLUMN,
            "流通股本估算",
            "流通市值",
            "ln流通市值",
            "存货周转改善率_winsor",
            "存货周转改善率_neutral",
            "存货周转改善率_cleaned",
        ]
    ]
    cleaned = cleaned.sort_values(["可用日期", "股票代码"]).reset_index(drop=True)

    log.info(
        "换手率单位：%s；同可用日期重复记录：%s 行（保留最新报告期）；缺失因子：%s 行；缺失/无效市值剔除：%s 行；MAD截尾：%s 次；清洗保留：%s 行/%s 个公告日截面",
        "百分比" if turnover_is_percent else "小数",
        duplicate_rows,
        missing_factor_rows,
        missing_market_rows,
        clipped_count,
        len(cleaned),
        cleaned["可用日期"].nunique(),
    )
    return cleaned


def main():
    for input_file in (FACTOR_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    quotes = pd.read_csv(
        QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    cleaned = clean_inventory_factor(factor, quotes)
    if cleaned.empty:
        log.error("清洗后没有有效因子记录")
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("已输出清洗因子：%s", OUTPUT_FILE)
    log.info(
        "标准化因子统计：均值 %.6f，标准差 %.6f；日期范围 %s ~ %s",
        cleaned["存货周转改善率_cleaned"].mean(),
        cleaned["存货周转改善率_cleaned"].std(),
        cleaned["可用日期"].min(),
        cleaned["可用日期"].max(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
