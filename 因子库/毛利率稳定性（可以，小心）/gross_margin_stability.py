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
RAW_FILE = SCRIPT_DIR / "毛利率_raw.csv"
CLEANED_FILE = SCRIPT_DIR / "毛利率_cleaned.csv"
FACTOR_FILE = SCRIPT_DIR / "毛利率稳定性_raw.csv"
SAMPLE_START = pd.Period("2019Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2026Q2", freq="Q-DEC")
WINDOW_QUARTERS = 12
DRIFT_MIN_PAIRS = 5
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99
WEIGHTS = {"z_sigma": 0.4, "z_cv": 0.3, "z_drift": 0.3}
FACTOR_COLUMN = "毛利率稳定性"

FINANCIAL_COLUMNS = {
    "股票代码",
    "报告期",
    "首次公告日期",
    "营业总收入",
    "营业成本",
}


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def build_raw_gross_margin(financials):
    missing_columns = FINANCIAL_COLUMNS.difference(financials.columns)
    if missing_columns:
        raise ValueError(f"财报数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = financials.loc[
        :, ["股票代码", "报告期", "首次公告日期", "营业总收入", "营业成本"]
    ].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data["营业总收入"] = pd.to_numeric(data["营业总收入"], errors="coerce")
    data["营业成本"] = pd.to_numeric(data["营业成本"], errors="coerce")
    data["季度"] = data["报告期"].dt.to_period("Q-DEC")
    data = data.loc[
        data["季度"].between(SAMPLE_START, SAMPLE_END)
    ].copy()
    if data.duplicated(["股票代码", "季度"]).any():
        raise ValueError("财报数据存在重复的股票代码和报告季度")

    revenue_is_valid = np.isfinite(data["营业总收入"]) & (data["营业总收入"] != 0)
    data["毛利率_raw"] = np.nan
    data.loc[revenue_is_valid, "毛利率_raw"] = (
        data.loc[revenue_is_valid, "营业总收入"]
        - data.loc[revenue_is_valid, "营业成本"]
    ) / data.loc[revenue_is_valid, "营业总收入"]
    data.loc[~np.isfinite(data["毛利率_raw"]), "毛利率_raw"] = np.nan
    data = data.loc[data["首次公告日期"].notna()].copy()
    data["季度"] = data["季度"].astype(str)
    data["报告期"] = data["报告期"].dt.strftime("%Y-%m-%d")
    return data[
        [
            "股票代码",
            "报告期",
            "季度",
            "首次公告日期",
            "营业总收入",
            "营业成本",
            "毛利率_raw",
        ]
    ].sort_values(["股票代码", "季度"]).reset_index(drop=True)


def clean_gross_margin_series(raw):
    required = {"股票代码", "季度", "首次公告日期", "毛利率_raw"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"毛利率数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    data = raw.copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["季度"] = pd.PeriodIndex(data["季度"], freq="Q-DEC")
    data["首次公告日期"] = pd.to_datetime(
        data["首次公告日期"], errors="coerce"
    ).dt.normalize()
    data["毛利率_raw"] = pd.to_numeric(data["毛利率_raw"], errors="coerce")
    data.loc[~np.isfinite(data["毛利率_raw"]), "毛利率_raw"] = np.nan
    if data.duplicated(["股票代码", "季度"]).any():
        raise ValueError("毛利率数据存在重复的股票代码和报告季度")

    periods = pd.period_range(SAMPLE_START, SAMPLE_END, freq="Q-DEC")
    filled_rows = []
    for code, stock_data in data.groupby("股票代码", sort=True):
        stock_data = stock_data.set_index("季度").reindex(periods)
        stock_data["股票代码"] = code
        stock_data.index.name = "季度"
        original = stock_data["毛利率_raw"]
        filled = original.interpolate(method="linear", limit_area="inside")
        announcements = stock_data["首次公告日期"]
        value_available_dates = announcements.copy()
        valid_positions = np.flatnonzero(original.notna().to_numpy())
        original_values = original.to_numpy(dtype=float)
        available_values = announcements.to_numpy()
        for left_position, right_position in zip(
            valid_positions[:-1], valid_positions[1:]
        ):
            if right_position - left_position <= 1:
                continue
            left_date = available_values[left_position]
            right_date = available_values[right_position]
            if pd.isna(left_date) or pd.isna(right_date):
                filled.iloc[left_position + 1 : right_position] = np.nan
                continue
            for position in range(left_position + 1, right_position):
                value_available_dates.iloc[position] = max(left_date, right_date)

        stock_data["毛利率_插值"] = filled
        stock_data["毛利率值可用公告日"] = value_available_dates
        stock_data["毛利率_插补标记"] = (
            original.isna() & stock_data["毛利率_插值"].notna()
        )
        filled_rows.append(stock_data.reset_index())

    result = pd.concat(filled_rows, ignore_index=True)
    result["毛利率_winsor"] = result["毛利率_插值"]
    for _, indices in result.groupby("季度", sort=True).groups.items():
        values = result.loc[indices, "毛利率_插值"].dropna()
        if values.empty:
            continue
        lower = values.quantile(WINSOR_LOWER_QUANTILE)
        upper = values.quantile(WINSOR_UPPER_QUANTILE)
        result.loc[indices, "毛利率_winsor"] = result.loc[
            indices, "毛利率_插值"
        ].clip(lower=lower, upper=upper)

    result["季度"] = result["季度"].astype(str)
    result["报告期"] = pd.PeriodIndex(result["季度"], freq="Q-DEC").end_time.strftime(
        "%Y-%m-%d"
    )
    for column in ("首次公告日期", "毛利率值可用公告日"):
        result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    result["毛利率_插补标记"] = result["毛利率_插补标记"].astype(bool)
    return result[
        [
            "股票代码",
            "报告期",
            "季度",
            "首次公告日期",
            "毛利率_raw",
            "毛利率_插值",
            "毛利率_winsor",
            "毛利率_插补标记",
            "毛利率值可用公告日",
        ]
    ].sort_values(["季度", "股票代码"]).reset_index(drop=True)


def _next_trading_day(announcement, trading_dates):
    position = trading_dates.searchsorted(pd.Timestamp(announcement), side="right")
    if position >= len(trading_dates):
        return pd.NaT
    return trading_dates[position]


def build_stability_factor(cleaned, trading_days):
    required = {
        "股票代码",
        "季度",
        "首次公告日期",
        "毛利率_raw",
        "毛利率_winsor",
    }
    missing_columns = required.difference(cleaned.columns)
    if missing_columns:
        raise ValueError(f"清洗毛利率数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if "日期" not in trading_days.columns:
        raise ValueError("交易日数据必须包含“日期”字段")

    data = cleaned.copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["季度"] = pd.PeriodIndex(data["季度"], freq="Q-DEC")
    data["首次公告日期"] = pd.to_datetime(
        data["首次公告日期"], errors="coerce"
    ).dt.normalize()
    data["毛利率_raw"] = pd.to_numeric(data["毛利率_raw"], errors="coerce")
    data["毛利率_winsor"] = pd.to_numeric(data["毛利率_winsor"], errors="coerce")
    if data.duplicated(["股票代码", "季度"]).any():
        raise ValueError("清洗毛利率数据存在重复的股票代码和报告季度")

    calendar = pd.DatetimeIndex(
        pd.to_datetime(trading_days["日期"], errors="raise")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if calendar.empty:
        raise ValueError("交易日数据没有有效日期")
    raw_panel = data.pivot(index="季度", columns="股票代码", values="毛利率_raw")
    announcement_panel = data.pivot(
        index="季度", columns="股票代码", values="首次公告日期"
    )
    all_periods = pd.period_range(SAMPLE_START, SAMPLE_END, freq="Q-DEC")
    raw_panel = raw_panel.reindex(all_periods)
    announcement_panel = announcement_panel.reindex(all_periods)

    factor_rows = []
    first_anchor = SAMPLE_START + WINDOW_QUARTERS - 1
    for anchor in all_periods:
        if anchor < first_anchor:
            continue
        window = pd.period_range(
            anchor - (WINDOW_QUARTERS - 1), anchor, freq="Q-DEC"
        )
        # Interpolation is limited to observations no later than the anchor quarter.
        raw_window = raw_panel.reindex(window)
        interpolated = raw_window.apply(
            lambda values: values.interpolate(method="linear", limit_area="inside"),
            axis=0,
        )
        candidate_codes = [
            code
            for code in raw_panel.columns
            if pd.notna(raw_panel.at[anchor, code])
            and interpolated[code].notna().all()
        ]
        if len(candidate_codes) < 2:
            continue
        gm_window = interpolated.loc[:, candidate_codes].copy()
        # Recompute the percentile winsorization using only the information available
        # within this as-of window, for every quarter used in the stability metrics.
        for period in window:
            values = gm_window.loc[period].dropna()
            if values.empty:
                continue
            lower = values.quantile(WINSOR_LOWER_QUANTILE)
            upper = values.quantile(WINSOR_UPPER_QUANTILE)
            gm_window.loc[period] = gm_window.loc[period].clip(
                lower=lower, upper=upper
            )

        latest_announcement = (
            announcement_panel.reindex(window)
            .loc[:, candidate_codes]
            .max()
            .max()
        )
        if pd.isna(latest_announcement):
            continue
        available_date = _next_trading_day(latest_announcement, calendar)
        if pd.isna(available_date):
            continue

        anchor_metrics = []
        for code in candidate_codes:
            series = gm_window[code]
            values = series.to_numpy(dtype=float)
            mean_gm = float(values.mean())
            sigma = float(values.std(ddof=1))
            if not np.isfinite(mean_gm) or mean_gm == 0:
                continue
            cv = sigma / mean_gm
            yoy_differences = np.abs(values[4:] - values[:-4])
            yoy_differences = yoy_differences[np.isfinite(yoy_differences)]
            if len(yoy_differences) < DRIFT_MIN_PAIRS:
                continue
            drift = float(yoy_differences.mean())
            if not np.isfinite(sigma) or not np.isfinite(cv) or not np.isfinite(drift):
                continue
            anchor_metrics.append(
                {
                    "股票代码": code,
                    "报告期": anchor.end_time.normalize(),
                    "季度": str(anchor),
                    "可用日期": available_date,
                    "毛利率标准差_sigma": sigma,
                    "毛利率变异系数_CV": cv,
                    "毛利率同比变化绝对值均值_Drift": drift,
                }
            )

        if len(anchor_metrics) < 2:
            continue
        cross_section = pd.DataFrame(anchor_metrics)
        component_columns = [
            ("毛利率标准差_sigma", "z_sigma"),
            ("毛利率变异系数_CV", "z_cv"),
            ("毛利率同比变化绝对值均值_Drift", "z_drift"),
        ]
        for source, target in component_columns:
            values = cross_section[source]
            standard_deviation = values.std(ddof=0)
            if not np.isfinite(standard_deviation) or standard_deviation == 0:
                cross_section[target] = 0.0
            else:
                cross_section[target] = (
                    values - values.mean()
                ) / standard_deviation
        cross_section[FACTOR_COLUMN] = -(
            WEIGHTS["z_sigma"] * cross_section["z_sigma"]
            + WEIGHTS["z_cv"] * cross_section["z_cv"]
            + WEIGHTS["z_drift"] * cross_section["z_drift"]
        )
        factor_rows.extend(cross_section.to_dict(orient="records"))

    factors = pd.DataFrame(factor_rows)
    if factors.empty:
        raise ValueError("没有满足 12 季度窗口及同行标准化要求的稳定性因子截面")
    factors["报告期"] = pd.to_datetime(factors["报告期"]).dt.strftime("%Y-%m-%d")
    factors["可用日期"] = pd.to_datetime(factors["可用日期"]).dt.strftime("%Y-%m-%d")
    return factors[
        [
            "股票代码",
            "报告期",
            "季度",
            "可用日期",
            "毛利率标准差_sigma",
            "毛利率变异系数_CV",
            "毛利率同比变化绝对值均值_Drift",
            "z_sigma",
            "z_cv",
            "z_drift",
            FACTOR_COLUMN,
        ]
    ].sort_values(["可用日期", "股票代码"]).reset_index(drop=True)


def main():
    for path in (FINANCIAL_FILE, QUOTE_FILE):
        if not path.is_file():
            log.error("输入文件不存在: %s", path)
            return 1
    financials = pd.read_csv(
        FINANCIAL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    trading_days = pd.read_csv(
        QUOTE_FILE,
        usecols=["股票代码", "日期"],
        dtype={"股票代码": str},
        encoding="utf-8-sig",
    )
    raw = build_raw_gross_margin(financials)
    cleaned = clean_gross_margin_series(raw)
    factors = build_stability_factor(cleaned, trading_days)
    raw.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    cleaned.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
    factors.to_csv(FACTOR_FILE, index=False, encoding="utf-8-sig")
    log.info("原始毛利率序列：%s 条，输出 %s", len(raw), RAW_FILE)
    log.info("插值并按季度 1%%/99%% 缩尾序列：%s 条，输出 %s", len(cleaned), CLEANED_FILE)
    log.info(
        "稳定性因子：%s 条，季度 %s ~ %s，可用日期 %s ~ %s，输出 %s",
        len(factors),
        factors["季度"].min(),
        factors["季度"].max(),
        factors["可用日期"].min(),
        factors["可用日期"].max(),
        FACTOR_FILE,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
