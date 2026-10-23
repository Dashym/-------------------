import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "毛利率稳定性_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
FACTOR_COLUMN = "毛利率稳定性"
MIN_CROSS_SECTION_SIZE = 10
FORWARD_HORIZON_MONTHS = 12
HAC_MAX_LAGS = 11
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def monthly_quote_data(quotes):
    required = {"股票代码", "日期", "收盘"}
    missing_columns = required.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    data = quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    data["月份"] = data["日期"].dt.to_period("M")
    return (
        data.sort_values(["股票代码", "日期"])
        .groupby(["股票代码", "月份"], as_index=False)
        .tail(1)
        .copy()
    )


def prepare_monthly_cross_sections(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
    required = {"股票代码", "可用日期", FACTOR_COLUMN}
    missing_columns = required.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")
    if min_cross_section < 2:
        raise ValueError("最小截面股票数必须至少为 2")

    signals = factor.loc[:, ["股票代码", "可用日期", FACTOR_COLUMN]].copy()
    signals["股票代码"] = normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(
        signals["可用日期"], errors="raise"
    ).dt.normalize()
    signals[FACTOR_COLUMN] = pd.to_numeric(signals[FACTOR_COLUMN], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", FACTOR_COLUMN]
    )
    signals = signals.sort_values(["股票代码", "可用日期"])
    signals = signals.drop_duplicates(["股票代码", "可用日期"], keep="last")

    market = monthly_quote_data(unadjusted_quotes)
    adjusted = monthly_quote_data(adjusted_quotes)
    if adjusted.empty or market.empty:
        raise ValueError("行情数据没有有效月末价格")
    prices = adjusted.pivot(index="月份", columns="股票代码", values="收盘")
    months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(months)
    returns = prices.shift(-FORWARD_HORIZON_MONTHS).div(prices).sub(1)
    returns.index.name = "月份"
    returns = (
        returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="未来12个月收益")
        .dropna(subset=["未来12个月收益"])
    )

    sections = []
    for month in sorted(market["月份"].unique()):
        current_market = market.loc[market["月份"] == month]
        month_end = current_market["日期"].max()
        as_of = signals.loc[signals["可用日期"] <= month_end]
        if as_of.empty:
            continue
        as_of = (
            as_of.sort_values(["股票代码", "可用日期"])
            .drop_duplicates("股票代码", keep="last")
        )
        section = as_of.merge(
            current_market[["股票代码", "月份"]],
            on="股票代码",
            how="inner",
            validate="one_to_one",
        ).merge(
            returns.loc[returns["月份"] == month],
            on=["股票代码", "月份"],
            how="inner",
            validate="one_to_one",
        )
        section = (
            section.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[FACTOR_COLUMN, "未来12个月收益"])
            .sort_values("股票代码")
        )
        if len(section) < min_cross_section:
            continue

        values = section[FACTOR_COLUMN].astype(float)
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            values = values.clip(
                lower=median - MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad,
                upper=median + MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad,
            )
        standard_deviation = values.std(ddof=0)
        if standard_deviation == 0 or not np.isfinite(standard_deviation):
            continue
        section["因子值_zscore"] = (values - values.mean()) / standard_deviation
        section["截面月份"] = str(month)
        section["截面日期"] = month_end.strftime("%Y-%m-%d")
        section["预测月份"] = str(month + FORWARD_HORIZON_MONTHS)
        sections.append(section)

    if not sections:
        return pd.DataFrame(
            columns=[
                "股票代码",
                "月份",
                "截面月份",
                "截面日期",
                "预测月份",
                "因子值_zscore",
                "未来12个月收益",
            ]
        )
    return pd.concat(sections, ignore_index=True)


def _beta_continued_fraction(a, b, x):
    maximum_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) < epsilon:
            return result
    raise ArithmeticError("Student-t p 值计算未收敛")


def _regularized_incomplete_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1 - front * _beta_continued_fraction(b, a, 1 - x) / b


def two_sided_t_pvalue(t_statistic, degrees_of_freedom):
    if degrees_of_freedom < 1 or not np.isfinite(t_statistic):
        return float("nan")
    x = degrees_of_freedom / (degrees_of_freedom + float(t_statistic) ** 2)
    return _regularized_incomplete_beta(x, degrees_of_freedom / 2, 0.5)


def newey_west_mean_standard_error(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("HAC 系数序列必须是一维有限数值")
    if max_lags < 0:
        raise ValueError("HAC 滞后期数不能为负数")
    count = len(values)
    if count <= max_lags:
        raise ValueError(
            f"HAC 滞后期数 {max_lags} 必须小于系数观测数 {count}"
        )
    residuals = values - values.mean()
    long_run_variance = float(np.dot(residuals, residuals) / count)
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)
        covariance = float(
            np.dot(residuals[lag:], residuals[:-lag]) / count
        )
        long_run_variance += 2 * weight * covariance
    variance_of_mean = long_run_variance / count * count / (count - 1)
    if variance_of_mean < 0 and np.isclose(variance_of_mean, 0):
        variance_of_mean = 0.0
    if variance_of_mean < 0:
        raise ArithmeticError("Newey-West HAC 方差估计为负")
    return float(np.sqrt(variance_of_mean))


def coefficient_summary(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = newey_west_mean_standard_error(values, max_lags)
    t_stat = (
        mean / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else float("nan")
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "p_value": two_sided_t_pvalue(t_stat, len(values) - 1),
    }


def significance(p_value):
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def report_dates(factor, unadjusted, adjusted):
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")
    return (
        (
            factor_dates.min().strftime("%Y-%m-%d"),
            factor_dates.max().strftime("%Y-%m-%d"),
        ),
        (
            market_dates.min().strftime("%Y-%m-%d"),
            market_dates.max().strftime("%Y-%m-%d"),
        ),
        (
            adjusted_dates.min().strftime("%Y-%m-%d"),
            adjusted_dates.max().strftime("%Y-%m-%d"),
        ),
    )


def load_inputs():
    for path in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")
    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    return factor, unadjusted, adjusted
