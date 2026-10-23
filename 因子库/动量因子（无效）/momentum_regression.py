import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_FILE = SCRIPT_DIR / "动量因子_cleaned.csv"
QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_后复权.csv"
OUTPUT_FILE = SCRIPT_DIR / "动量因子_回归结果.txt"
FACTOR_VALUE_COLUMN = "动量因子_cleaned"
MIN_STOCKS_PER_CROSS_SECTION = 10


def prepare_forward_returns(quotes):
    required = {"股票代码", "日期", "收盘"}
    missing_columns = required.difference(quotes.columns)
    if missing_columns:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = quotes.loc[:, ["股票代码", "日期", "收盘"]].copy()
    data["股票代码"] = data["股票代码"].astype(str).str.strip().str.zfill(6)
    data["日期"] = pd.to_datetime(data["日期"], errors="raise")
    data["收盘"] = pd.to_numeric(data["收盘"], errors="raise")
    data = data.dropna(subset=["股票代码", "日期", "收盘"])
    data = data.loc[data["收盘"] > 0].copy()
    data["月份"] = data["日期"].dt.to_period("M")

    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("后复权行情存在重复的股票代码和日期记录")
    if data.empty:
        raise ValueError("后复权行情中没有有效收盘价")

    month_end = (
        data.sort_values(["股票代码", "日期"])
        .groupby(["月份", "股票代码"], as_index=False)
        .tail(1)
    )
    prices = month_end.pivot(index="月份", columns="股票代码", values="收盘")
    months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")
    prices = prices.reindex(months)
    forward_returns = prices.shift(-1).div(prices).sub(1)
    forward_returns.index.name = "月份"

    return (
        forward_returns.rename_axis(columns="股票代码")
        .reset_index()
        .melt(id_vars="月份", var_name="股票代码", value_name="下期收益")
        .dropna(subset=["下期收益"])
    )


def run_fama_macbeth_regression(factor, quotes, min_stocks=MIN_STOCKS_PER_CROSS_SECTION):
    required = {"股票代码", "截面日期", FACTOR_VALUE_COLUMN}
    missing_columns = required.difference(factor.columns)
    if missing_columns:
        raise ValueError(f"因子数据缺少必要字段: {', '.join(sorted(missing_columns))}")

    data = factor.loc[:, ["股票代码", "截面日期", FACTOR_VALUE_COLUMN]].copy()
    data["股票代码"] = data["股票代码"].astype(str).str.strip().str.zfill(6)
    data["截面日期"] = pd.to_datetime(data["截面日期"], errors="raise")
    data["月份"] = data["截面日期"].dt.to_period("M")
    data[FACTOR_VALUE_COLUMN] = pd.to_numeric(data[FACTOR_VALUE_COLUMN], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["股票代码", FACTOR_VALUE_COLUMN]
    )

    if data.duplicated(["月份", "股票代码"]).any():
        raise ValueError("因子数据存在重复的月份和股票代码")

    returns = prepare_forward_returns(quotes)
    merged = data.merge(returns, on=["月份", "股票代码"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[FACTOR_VALUE_COLUMN, "下期收益"]
    )

    rows = []
    for month, cross_section in merged.groupby("月份", sort=True):
        if len(cross_section) < min_stocks:
            continue

        x = cross_section[FACTOR_VALUE_COLUMN].to_numpy(dtype=float)
        y = cross_section["下期收益"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        if np.linalg.matrix_rank(design) < 2:
            continue

        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coefficients
        residual_sum_squares = float(np.sum((y - prediction) ** 2))
        total_sum_squares = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            1 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else 0.0
        )
        rows.append(
            {
                "月份": str(month),
                "截面日期": cross_section["截面日期"].max().strftime("%Y-%m-%d"),
                "股票数": len(cross_section),
                "alpha": coefficients[0],
                "beta": coefficients[1],
                "R-squared": r_squared,
            }
        )

    return pd.DataFrame(
        rows, columns=["月份", "截面日期", "股票数", "alpha", "beta", "R-squared"]
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
    x = degrees_of_freedom / (
        degrees_of_freedom + float(t_statistic) ** 2
    )
    return _regularized_incomplete_beta(x, degrees_of_freedom / 2, 0.5)


def _summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    )
    t_statistic = (
        mean / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else float("nan")
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": two_sided_t_pvalue(t_statistic, len(values) - 1),
    }


def _significance(p_value):
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def render_report(regressions, factor_count, factor_dates, quote_count, quote_dates):
    if len(regressions) < 2:
        raise ValueError("有效回归截面不足 2 期，无法汇总 Fama-MacBeth 标准误")

    alpha = _summary(regressions["alpha"])
    beta = _summary(regressions["beta"])
    average_r_squared = regressions["R-squared"].mean()
    average_stocks = regressions["股票数"].mean()
    if beta["p_value"] < 0.01:
        conclusion = "因子回归系数在 1% 水平显著。"
    elif beta["p_value"] < 0.05:
        conclusion = "因子回归系数在 5% 水平显著。"
    elif beta["p_value"] < 0.1:
        conclusion = "因子回归系数在 10% 水平显著，证据较弱。"
    else:
        conclusion = "因子回归系数未达到 10% 显著性水平。"
    direction = "正" if beta["mean"] > 0 else "负" if beta["mean"] < 0 else "零"

    lines = [
        "=" * 72,
        "动量因子 Fama-MacBeth 回归检验报告",
        "=" * 72,
        "",
        "回归设定：",
        "  因变量 y：下一自然月后复权月末收盘收益 = 下月月末收盘 / 本月月末收盘 - 1",
        "  自变量 X：动量因子_cleaned（基于前复权数据计算、截面去极值及 Z-score）",
        "  每期截面回归：y(i,t+1) = alpha(t) + beta(t) * X(i,t) + error(i,t)",
        "  汇总方法：Fama-MacBeth，逐月 OLS 系数取时间序列均值",
        "  标准误：逐月系数的样本标准差 / sqrt(有效期数)，未作 Newey-West 自相关调整",
        f"  每期最少有效股票数：{MIN_STOCKS_PER_CROSS_SECTION}",
        "  beta 含义：动量因子每增加 1 个截面标准差，对应的下期收益变化（收益率小数）",
        "",
        "数据概况：",
        f"  清洗因子：{factor_count:,} 条，日期范围 {factor_dates[0]} ~ {factor_dates[1]}",
        f"  后复权行情：{quote_count:,} 条，日期范围 {quote_dates[0]} ~ {quote_dates[1]}",
        f"  有效回归截面：{len(regressions)} 期，范围 {regressions['月份'].iloc[0]} ~ {regressions['月份'].iloc[-1]}",
        f"  平均有效股票数/截面：{average_stocks:.1f}，范围 {regressions['股票数'].min()} ~ {regressions['股票数'].max()}",
        f"  平均 R-squared：{average_r_squared:.4f}",
        "",
        f"{'变量':<16}{'均值系数':>14}{'标准误':>14}{'t统计量':>12}{'p值':>12}",
        "-" * 68,
        f"{'const':<16}{alpha['mean']:>14.6f}{alpha['standard_error']:>14.6f}{alpha['t_statistic']:>12.4f}{alpha['p_value']:>12.4f} {_significance(alpha['p_value'])}",
        f"{'动量因子 beta':<16}{beta['mean']:>14.6f}{beta['standard_error']:>14.6f}{beta['t_statistic']:>12.4f}{beta['p_value']:>12.4f} {_significance(beta['p_value'])}",
        "-" * 68,
        "显著性标记：*** p<0.01，** p<0.05，* p<0.1",
        "",
        f"动量因子 beta：{beta['mean']:.6f}（约 {beta['mean'] * 100:.4f} 个百分点/因子标准差）",
        f"beta 方向：{direction}；Fama-MacBeth t={beta['t_statistic']:.4f}，p={beta['p_value']:.4f}",
        f"结论：{conclusion}",
        "",
        "逐月回归系数：",
        f"{'月份':<12}{'样本数':>8}{'alpha':>14}{'beta':>14}{'R-squared':>14}",
        "-" * 62,
    ]
    lines.extend(
        f"{row['月份']:<12}{row['股票数']:>8d}{row['alpha']:>14.6f}{row['beta']:>14.6f}{row['R-squared']:>14.4f}"
        for _, row in regressions.iterrows()
    )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    for input_file in (FACTOR_FILE, QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    quotes = pd.read_csv(QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    regressions = run_fama_macbeth_regression(factor, quotes)
    factor_dates = pd.to_datetime(factor["截面日期"], errors="raise")
    quote_dates = pd.to_datetime(quotes["日期"], errors="raise")
    report = render_report(
        regressions,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        len(quotes),
        (quote_dates.min().strftime("%Y-%m-%d"), quote_dates.max().strftime("%Y-%m-%d")),
    )

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("回归检验文本报告已输出: %s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
