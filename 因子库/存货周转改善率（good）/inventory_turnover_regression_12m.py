import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_turnover_ic import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_COLUMN,
    FACTOR_FILE,
    MIN_CROSS_SECTION_SIZE,
    UNADJUSTED_QUOTE_FILE,
    _monthly_market_data,
    _normalize_codes,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "存货周转改善率_回归结果_未来12个月.txt"
FORWARD_HORIZON_MONTHS = 12
HAC_MAX_LAGS = 11


def calculate_monthly_regressions(
    factor,
    unadjusted_quotes,
    adjusted_quotes,
    min_cross_section=MIN_CROSS_SECTION_SIZE,
):
    required = {"股票代码", "可用日期", FACTOR_COLUMN}
    missing_columns = required.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    signals = factor.loc[:, ["股票代码", "可用日期", FACTOR_COLUMN]].copy()
    signals["股票代码"] = _normalize_codes(signals["股票代码"])
    signals["可用日期"] = pd.to_datetime(
        signals["可用日期"], errors="raise"
    ).dt.normalize()
    signals[FACTOR_COLUMN] = pd.to_numeric(signals[FACTOR_COLUMN], errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", "可用日期", FACTOR_COLUMN]
    )
    signals = signals.sort_values(["股票代码", "可用日期"])
    signals = signals.drop_duplicates(["股票代码", "可用日期"], keep="last")

    market = _monthly_market_data(unadjusted_quotes)
    adjusted = _monthly_market_data(adjusted_quotes, adjusted=True)
    prices = adjusted.pivot(index="月份", columns="股票代码", values="收盘")
    all_months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(all_months)
    forward_returns = prices.shift(-12).div(prices).sub(1)
    forward_returns.index.name = "月份"
    forward_returns = (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="未来12个月收益")
        .dropna(subset=["未来12个月收益"])
    )

    results = []
    for month in sorted(market["月份"].unique()):
        month_end = market.loc[market["月份"] == month, "日期"].max()
        as_of = signals.loc[signals["可用日期"] <= month_end]
        if as_of.empty:
            continue
        as_of = (
            as_of.sort_values(["股票代码", "可用日期"])
            .drop_duplicates("股票代码", keep="last")
        )

        cross_section = as_of.merge(
            market.loc[
                market["月份"] == month,
                ["股票代码", "月份", "ln流通市值"],
            ],
            on="股票代码",
            how="inner",
            validate="one_to_one",
        ).merge(
            forward_returns.loc[forward_returns["月份"] == month],
            on=["股票代码", "月份"],
            how="inner",
            validate="one_to_one",
        )
        cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[FACTOR_COLUMN, "ln流通市值", "未来12个月收益"]
        )
        if len(cross_section) < min_cross_section:
            continue

        values = cross_section[FACTOR_COLUMN]
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            values = values.clip(
                lower=median - 5 * 1.4826 * mad,
                upper=median + 5 * 1.4826 * mad,
            )

        size = cross_section["ln流通市值"].to_numpy(dtype=float)
        neutral_design = np.column_stack([np.ones(len(cross_section)), size])
        neutral_coefficients, _, _, _ = np.linalg.lstsq(
            neutral_design, values.to_numpy(dtype=float), rcond=None
        )
        residuals = values.to_numpy(dtype=float) - neutral_design @ neutral_coefficients
        residual_std = residuals.std(ddof=0)
        if residual_std == 0 or not np.isfinite(residual_std):
            continue
        x = (residuals - residuals.mean()) / residual_std
        y = cross_section["未来12个月收益"].to_numpy(dtype=float)
        regression_design = np.column_stack([np.ones(len(x)), x])
        if np.linalg.matrix_rank(regression_design) < 2:
            continue

        coefficients, _, _, _ = np.linalg.lstsq(regression_design, y, rcond=None)
        prediction = regression_design @ coefficients
        residual_sum_squares = float(np.sum((y - prediction) ** 2))
        total_sum_squares = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            1 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else 0.0
        )
        results.append(
            {
                "月份": str(month),
                "截面日期": month_end.strftime("%Y-%m-%d"),
                "预测月份": str(month + FORWARD_HORIZON_MONTHS),
                "股票数": len(cross_section),
                "alpha": coefficients[0],
                "beta": coefficients[1],
                "R-squared": r_squared,
            }
        )

    return pd.DataFrame(
        results,
        columns=["月份", "截面日期", "预测月份", "股票数", "alpha", "beta", "R-squared"],
    )


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


def _newey_west_mean_standard_error(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("HAC 系数序列必须是一维有限数值")
    if max_lags < 0:
        raise ValueError("HAC 滞后期数不能为负数")
    observations = len(values)
    if observations <= max_lags:
        raise ValueError(
            f"HAC 滞后期数 {max_lags} 必须小于系数观测数 {observations}"
        )

    residuals = values - values.mean()
    long_run_variance = float(np.dot(residuals, residuals) / observations)
    for lag in range(1, max_lags + 1):
        bartlett_weight = 1 - lag / (max_lags + 1)
        autocovariance = float(
            np.dot(residuals[lag:], residuals[:-lag]) / observations
        )
        long_run_variance += 2 * bartlett_weight * autocovariance

    variance_of_mean = long_run_variance / observations
    variance_of_mean *= observations / (observations - 1)
    if variance_of_mean < 0 and np.isclose(variance_of_mean, 0):
        variance_of_mean = 0.0
    if variance_of_mean < 0:
        raise ArithmeticError("Newey-West HAC 方差估计为负")
    return float(np.sqrt(variance_of_mean))


def _coefficient_summary(values, max_lags=HAC_MAX_LAGS):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = _newey_west_mean_standard_error(values, max_lags=max_lags)
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


def _significance(p_value):
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def render_report(regressions, factor_rows, factor_dates, market_dates, adjusted_dates):
    if len(regressions) < 2:
        raise ValueError("有效回归截面不足 2 期，无法进行 Fama-MacBeth 系数检验")

    alpha = _coefficient_summary(regressions["alpha"])
    beta = _coefficient_summary(regressions["beta"])
    mean_r_squared = regressions["R-squared"].mean()
    direction = "正" if beta["mean"] > 0 else "负" if beta["mean"] < 0 else "零"
    if beta["p_value"] < 0.01:
        conclusion = "因子回归系数在 1% 水平显著。"
    elif beta["p_value"] < 0.05:
        conclusion = "因子回归系数在 5% 水平显著。"
    elif beta["p_value"] < 0.1:
        conclusion = "因子回归系数在 10% 水平显著，统计证据较弱。"
    else:
        conclusion = "因子回归系数未达到 10% 显著性水平。"

    lines = [
        "=" * 72,
        "存货周转改善率（TTM同比）Fama-MacBeth 回归报告",
        "=" * 72,
        "",
        "回归设定：",
        "  因变量 y：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1",
        "  自变量 X：月末时点已公告的最新存货周转改善率，MAD(5倍)截尾、对数流通市值中性化并截面 Z-score",
        "  每月截面 OLS：y(i,t+12) = alpha(t) + beta(t) * X(i,t) + error(i,t)",
        "  汇总：Fama-MacBeth，报告逐月系数的时间序列均值",
        f"  第二阶段标准误：Newey-West HAC，Bartlett 核，最大滞后期数={HAC_MAX_LAGS}，含 n/(n-1) 有限样本修正",
        "  t 统计量 = 系数均值 / Newey-West HAC 标准误；p 值采用双侧 Student-t(df=截面数-1)",
        "  说明：HAC 用于处理12个月重叠收益窗口导致的月度系数序列自相关",
        f"  每期最少有效股票数：{MIN_CROSS_SECTION_SIZE}",
        "  beta 单位：因子增加 1 个截面标准差对应的未来 12 个月收益变化（收益率小数）",
        "",
        "数据概况：",
        f"  因子记录：{factor_rows:,} 条，可用日期 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  不复权行情（流通市值）：{market_dates[0]} ~ {market_dates[1]}",
        f"  后复权行情（收益）：{adjusted_dates[0]} ~ {adjusted_dates[1]}",
        f"  有效回归截面：{len(regressions)} 期，月份 {regressions['月份'].iloc[0]} ~ {regressions['月份'].iloc[-1]}",
        f"  总观测数：{int(regressions['股票数'].sum()):,}",
        f"  平均有效股票数/截面：{regressions['股票数'].mean():.1f}，范围 {regressions['股票数'].min()} ~ {regressions['股票数'].max()}",
        f"  平均 R-squared：{mean_r_squared:.4f}",
        "",
        f"{'变量':<18}{'均值系数':>14}{'标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 70,
        f"{'const':<18}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}{alpha['t_stat']:>12.4f}{alpha['p_value']:>12.4f} {_significance(alpha['p_value'])}",
        f"{'存货周转改善率 beta':<18}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}{beta['t_stat']:>12.4f}{beta['p_value']:>12.4f} {_significance(beta['p_value'])}",
        "-" * 70,
        "显著性标记：*** p<0.01，** p<0.05，* p<0.1",
        "",
        f"beta 方向：{direction}；beta={beta['mean']:.6f}（约 {beta['mean'] * 100:.4f} 个百分点/因子标准差）",
        f"Fama-MacBeth t={beta['t_stat']:.4f}，p={beta['p_value']:.4f}",
        f"结论：{conclusion}",
        "",
        "逐月回归结果：",
        f"{'月份':<12}{'预测月份':<12}{'样本数':>8}{'alpha':>14}{'beta':>14}{'R-squared':>14}",
        "-" * 74,
    ]
    lines.extend(
        f"{row['月份']:<12}{row['预测月份']:<12}{row['股票数']:>8d}{row['alpha']:>14.6f}{row['beta']:>14.6f}{row['R-squared']:>14.4f}"
        for _, row in regressions.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    for input_file in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    regressions = calculate_monthly_regressions(factor, unadjusted, adjusted)
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")
    report = render_report(
        regressions,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        (market_dates.min().strftime("%Y-%m-%d"), market_dates.max().strftime("%Y-%m-%d")),
        (
            adjusted_dates.min().strftime("%Y-%m-%d"),
            adjusted_dates.max().strftime("%Y-%m-%d"),
        ),
    )
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("未来 12 个月回归报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
