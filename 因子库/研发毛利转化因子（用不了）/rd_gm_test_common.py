# -*- coding: utf-8 -*-
"""研发-毛利转化因子（方向哑变量）三大检验的公共模块。

季度截面口径（兜底策略）：
    截面日 = 季度末最后一个交易日；
    因子值 = 截至该日已公告的最新方向哑变量（0/1）与基础连续因子；
    收益   = 未来 1 年（4 个季度）累计后复权收益 = t+4 季度末后复权收盘 / t 季度末后复权收盘 - 1；
    预处理 = 收益率按申万三级行业中位数去均值，因子值与收益率做 1%/99% 双缩尾。
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "研发毛利转化因子_raw.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
FACTOR_COLUMN = "最终因子值"
DUMMY_COLUMN = "方向哑变量"
INDUSTRY_COLUMN = "三级行业名称"
MIN_CROSS_SECTION_SIZE = 10
FORWARD_HORIZON_QUARTERS = 4   # 未来 1 年 = 4 个季度
HAC_MAX_LAGS = 3                # Newey-West 滞后阶数
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
RETURN_COLUMN = "未来1年收益"


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def quarterly_quote_data(quotes):
    """按（股票代码，季度）取最后一个交易日的收盘价。"""
    required = {"股票代码", "日期", "收盘"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing))}")
    data = quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="raise").dt.normalize()
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("行情数据存在重复的股票代码和日期记录")
    data["季度"] = data["日期"].dt.to_period("Q-DEC")
    return (
        data.sort_values(["股票代码", "日期"])
        .groupby(["股票代码", "季度"], as_index=False)
        .tail(1)
        .copy()
    )


def prepare_quarterly_cross_sections(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
    required = {"股票代码", "可用日期", FACTOR_COLUMN, DUMMY_COLUMN, INDUSTRY_COLUMN}
    missing = required.difference(factor.columns)
    if missing:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing))}")
    if min_cross_section < 2:
        raise ValueError("最小截面股票数必须至少为 2")

    signals = factor.loc[
        :, ["股票代码", "可用日期", DUMMY_COLUMN, FACTOR_COLUMN, INDUSTRY_COLUMN]
    ].copy()
    signals["股票代码"] = normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(signals["可用日期"], errors="raise").dt.normalize()
    signals[DUMMY_COLUMN] = pd.to_numeric(signals[DUMMY_COLUMN], errors="coerce")
    signals[FACTOR_COLUMN] = pd.to_numeric(signals[FACTOR_COLUMN], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", DUMMY_COLUMN, FACTOR_COLUMN]
    )
    signals = signals.sort_values(["股票代码", "可用日期"]).drop_duplicates(
        ["股票代码", "可用日期"], keep="last"
    )

    market = quarterly_quote_data(unadjusted_quotes)
    adjusted = quarterly_quote_data(adjusted_quotes)
    if adjusted.empty or market.empty:
        raise ValueError("行情数据没有有效的季度末价格")

    prices = adjusted.pivot(index="季度", columns="股票代码", values="收盘")
    all_quarters = pd.period_range(prices.index.min(), prices.index.max(), freq="Q-DEC")
    prices = prices.reindex(all_quarters)
    forward_returns = prices.shift(-FORWARD_HORIZON_QUARTERS).div(prices).sub(1)
    forward_returns.index.name = "季度"
    forward_returns = (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="季度", var_name="股票代码", value_name=RETURN_COLUMN)
        .dropna(subset=[RETURN_COLUMN])
    )

    results = []
    for quarter in sorted(market["季度"].unique()):
        market_quarter = market.loc[market["季度"] == quarter]
        quarter_end = market_quarter["日期"].max()
        as_of = signals.loc[signals["可用日期"] <= quarter_end]
        if as_of.empty:
            continue
        as_of = (
            as_of.sort_values(["股票代码", "可用日期"])
            .drop_duplicates("股票代码", keep="last")
        )
        cross_section = as_of.merge(
            market_quarter[["股票代码", "季度"]],
            on="股票代码",
            how="inner",
        ).merge(
            forward_returns.loc[forward_returns["季度"] == quarter],
            on=["股票代码", "季度"],
            how="inner",
        )
        cross_section = (
            cross_section.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[DUMMY_COLUMN, FACTOR_COLUMN, RETURN_COLUMN])
            .sort_values("股票代码")
        )
        if len(cross_section) < min_cross_section:
            continue

        # 收益率按申万三级行业中位数去均值
        cross_section[RETURN_COLUMN] = (
            cross_section.groupby(INDUSTRY_COLUMN)[RETURN_COLUMN]
            .transform(lambda x: x - x.median())
        )
        # 因子值（连续）与收益率 1%/99% 双缩尾
        factor_values = cross_section[FACTOR_COLUMN].astype(float)
        f_lo, f_hi = factor_values.quantile(WINSOR_LOWER), factor_values.quantile(WINSOR_UPPER)
        cross_section[FACTOR_COLUMN] = factor_values.clip(lower=f_lo, upper=f_hi)
        return_values = cross_section[RETURN_COLUMN].astype(float)
        r_lo, r_hi = return_values.quantile(WINSOR_LOWER), return_values.quantile(WINSOR_UPPER)
        cross_section[RETURN_COLUMN] = return_values.clip(lower=r_lo, upper=r_hi)

        cross_section["截面季度"] = str(quarter)
        cross_section["截面日期"] = quarter_end.strftime("%Y-%m-%d")
        cross_section["预测季度"] = str(quarter + FORWARD_HORIZON_QUARTERS)
        results.append(cross_section)

    if not results:
        return pd.DataFrame(
            columns=[
                "股票代码", "季度", "截面季度", "截面日期", "预测季度",
                DUMMY_COLUMN, FACTOR_COLUMN, INDUSTRY_COLUMN, RETURN_COLUMN,
            ]
        )
    return pd.concat(results, ignore_index=True)


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
        numerator = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
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
    observations = len(values)
    if observations <= max_lags:
        raise ValueError(f"HAC 滞后期数 {max_lags} 必须小于系数观测数 {observations}")
    residuals = values - values.mean()
    long_run_variance = float(np.dot(residuals, residuals) / observations)
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)
        autocovariance = float(np.dot(residuals[lag:], residuals[:-lag]) / observations)
        long_run_variance += 2 * weight * autocovariance
    variance_of_mean = long_run_variance / observations
    variance_of_mean *= observations / (observations - 1)
    if variance_of_mean < 0 and np.isclose(variance_of_mean, 0):
        variance_of_mean = 0.0
    if variance_of_mean < 0:
        raise ArithmeticError("Newey-West HAC 方差估计为负")
    return float(np.sqrt(variance_of_mean))


def coefficient_summary(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = newey_west_mean_standard_error(values, max_lags=max_lags)
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


def report_dates(factor, unadjusted_quotes, adjusted_quotes):
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted_quotes["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted_quotes["日期"], errors="raise")
    return (
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        (market_dates.min().strftime("%Y-%m-%d"), market_dates.max().strftime("%Y-%m-%d")),
        (adjusted_dates.min().strftime("%Y-%m-%d"), adjusted_dates.max().strftime("%Y-%m-%d")),
    )


def load_inputs():
    for input_file in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not input_file.is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    return factor, unadjusted, adjusted
