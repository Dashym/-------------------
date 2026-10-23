# -*- coding: utf-8 -*-
"""研发-毛利转化因子回测（2024Q4 截面前 20% 高分股持有至 2025 年全年）。

回测设置（兜底策略）：
    1. 取 2024Q4 报告期截面；
    2. 候选池 = 方向哑变量 = 1（研发强度_t-8 同比上升 且 毛利率_t 同比上升）的股票；
    3. 在候选池内按基础连续因子值降序，选出前 20% 高分股；
    4. 买入日 = 选中股票可用日期的最大值（保证所有财报均已公告，防未来函数）；
    5. 等权持有至 2025 年最后一个交易日；
    6. 对比基准：候选池等权、哑变量=0 等权、全样本等权。
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
FACTOR_FILE = SCRIPT_DIR / "研发毛利转化因子_raw.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR
    / "clean_temp"
    / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)
OUTPUT_FILE = SCRIPT_DIR / "研发毛利转化因子_回测结果.txt"

DUMMY_COLUMN = "方向哑变量"
FACTOR_COLUMN = "最终因子值"
TOP_FRACTION = 0.20
BACKTEST_QUARTER = "2024Q4"
HOLD_END_DATE = pd.Timestamp("2025-12-31")


def normalize_codes(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def load_adjusted_prices():
    quotes = pd.read_csv(ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    quotes["股票代码"] = normalize_codes(quotes["股票代码"])
    quotes["日期"] = pd.to_datetime(quotes["日期"], errors="raise").dt.normalize()
    quotes["收盘"] = pd.to_numeric(quotes["收盘"], errors="coerce")
    quotes = quotes.dropna(subset=["股票代码", "日期", "收盘"])
    quotes = quotes.loc[quotes["收盘"] > 0].copy()
    quotes = quotes.sort_values(["股票代码", "日期"]).drop_duplicates(
        ["股票代码", "日期"], keep="last"
    )
    return quotes


def period_return(quotes, codes, buy_date, sell_date):
    """等权持有期收益：买入日收盘 -> 卖出日收盘。"""
    rows = []
    for code in codes:
        hist = quotes.loc[quotes["股票代码"] == code].sort_values("日期")
        if hist.empty:
            continue
        buy = hist.loc[hist["日期"] >= buy_date]
        if buy.empty:
            continue
        buy_price = buy.iloc[0]["收盘"]
        sell = hist.loc[hist["日期"] <= sell_date]
        if sell.empty:
            continue
        sell_price = sell.iloc[-1]["收盘"]
        rows.append({"股票代码": code, "买入价": buy_price, "卖出价": sell_price,
                     "持有收益": sell_price / buy_price - 1})
    return pd.DataFrame(rows)


def summarize(detail):
    if detail.empty:
        return {"股票数": 0, "平均收益": float("nan"), "中位数收益": float("nan")}
    return {
        "股票数": len(detail),
        "平均收益": detail["持有收益"].mean(),
        "中位数收益": detail["持有收益"].median(),
    }


def render_report(strategy, benchmarks, detail):
    lines = [
        "=" * 72,
        "研发-毛利转化因子回测报告（2024Q4 -> 2025 全年）",
        "=" * 72,
        "",
        "回测设置：",
        f"  截面：{BACKTEST_QUARTER} 报告期",
        "  候选池：方向哑变量 = 1（研发强度_t-8 同比上升 且 毛利率_t 同比上升）",
        f"  选股：候选池内按基础连续因子降序取前 {TOP_FRACTION:.0%} 高分股，等权持有",
        f"  买入日：{detail['买入日']}（选中股票可用日期最大值，防未来函数）",
        f"  卖出日：{detail['卖出日']}（2025 年最后一个交易日）",
        "  持有期：约 1 年",
        "",
        "选股概况：",
        f"  2024Q4 截面股票数：{detail['截面股票数']}",
        f"  哑变量=1 候选池：{detail['候选池数']} 只",
        f"  前 20% 高分股：{detail['策略股票数']} 只",
        "",
        "持有期收益对比（等权）：",
        f"{'组合':<26}{'股票数':>8}{'平均收益':>14}{'中位数收益':>14}",
        "-" * 62,
        f"{'策略组合（前20%高分）':<26}{strategy['股票数']:>8}"
        f"{strategy['平均收益']:>14.4%}{strategy['中位数收益']:>14.4%}",
    ]
    for name, bench in benchmarks.items():
        lines.append(
            f"{name:<26}{bench['股票数']:>8}{bench['平均收益']:>14.4%}{bench['中位数收益']:>14.4%}"
        )
    lines.extend(
        [
            "-" * 62,
            f"策略相对候选池超额：{detail['超额_候选池']:.4%}",
            f"策略相对哑变量=0 超额：{detail['超额_哑变量0']:.4%}",
            f"策略相对全样本超额：{detail['超额_全样本']:.4%}",
            "",
            "策略组合持仓明细（按持有收益降序）：",
            f"{'股票代码':<10}{'买入价':>12}{'卖出价':>12}{'持有收益':>12}",
            "-" * 46,
        ]
    )
    for _, row in detail["持仓明细"].iterrows():
        lines.append(
            f"{row['股票代码']:<10}{row['买入价']:>12.3f}{row['卖出价']:>12.3f}"
            f"{row['持有收益']:>12.4%}"
        )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def main():
    if not FACTOR_FILE.is_file() or not ADJUSTED_QUOTE_FILE.is_file():
        log.error("输入文件不存在")
        return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    factor["股票代码"] = normalize_codes(factor["股票代码"])
    section = factor.loc[factor["截面季度"] == BACKTEST_QUARTER].copy()
    if section.empty:
        log.error("因子文件没有 %s 截面", BACKTEST_QUARTER)
        return 1

    quotes = load_adjusted_prices()
    sell_date = quotes.loc[quotes["日期"] <= HOLD_END_DATE, "日期"].max()

    candidates = section.loc[section[DUMMY_COLUMN] == 1].copy()
    non_candidates = section.loc[section[DUMMY_COLUMN] == 0].copy()
    candidates = candidates.sort_values(FACTOR_COLUMN, ascending=False)
    top_n = max(1, int(np.ceil(TOP_FRACTION * len(candidates))))
    strategy_codes = candidates.head(top_n)["股票代码"].tolist()

    buy_date = pd.to_datetime(
        section.loc[section["股票代码"].isin(strategy_codes), "可用日期"]
    ).max()

    strategy_detail = period_return(quotes, strategy_codes, buy_date, sell_date)
    candidate_detail = period_return(
        quotes, candidates["股票代码"].tolist(), buy_date, sell_date
    )
    non_candidate_detail = period_return(
        quotes, non_candidates["股票代码"].tolist(), buy_date, sell_date
    )
    all_detail = period_return(quotes, section["股票代码"].tolist(), buy_date, sell_date)

    strategy = summarize(strategy_detail)
    benchmarks = {
        "候选池（哑变量=1）": summarize(candidate_detail),
        "哑变量=0 对照": summarize(non_candidate_detail),
        "全样本等权": summarize(all_detail),
    }
    detail = {
        "截面股票数": len(section),
        "候选池数": len(candidates),
        "策略股票数": len(strategy_codes),
        "买入日": buy_date.strftime("%Y-%m-%d"),
        "卖出日": sell_date.strftime("%Y-%m-%d"),
        "超额_候选池": strategy["平均收益"] - benchmarks["候选池（哑变量=1）"]["平均收益"],
        "超额_哑变量0": strategy["平均收益"] - benchmarks["哑变量=0 对照"]["平均收益"],
        "超额_全样本": strategy["平均收益"] - benchmarks["全样本等权"]["平均收益"],
        "持仓明细": strategy_detail.sort_values("持有收益", ascending=False),
    }

    report = render_report(strategy, benchmarks, detail)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("回测报告已输出：%s", OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

