# -*- coding: utf-8 -*-
"""
销售收现 & 营业收入 爬取（东财「按报告期」接口）
================================================
背景：仓库现有财报数据集（raw_data/processed_data）只包含利润表/资产负债表的
      部分字段，**没有**「销售商品、提供劳务收到的现金」，因此销售收现比改善因子
      无法从现有数据计算。本脚本通过 akshare 东方财富接口补齐该字段。

数据口径：
  - 利润表        ak.stock_profit_sheet_by_report_em   → TOTAL_OPERATE_INCOME（营业总收入，年初至期末累计值）
  - 现金流量表    ak.stock_cash_flow_sheet_by_report_em → SALES_SERVICES（销售商品、提供劳务收到的现金，累计值）
  - 首次公告日期  优先取接口 NOTICE_DATE，缺失时退化为 UPDATE_DATE
  - 两张表都是**累计值**，还原为单季度值的差分工作交由
    sales_receipt_ratio_improvement.py 完成（与 07_财报爬取.py 口径一致）

输出（缓存，可断点续跑）：
  aksahre_demo/raw_data/汽车零部件_销售收现与营业收入_原始.csv

运行环境：aksahre_demo/.venv（akshare 1.19.1）
    python sales_receipt_cash_flow_crawler.py
"""

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
POOL_FILE = BASE_DIR / "汽车零部件_申万二级成分股.csv"
OUTPUT_FILE = BASE_DIR / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"

# 接口限流：每次请求之间的最小间隔（秒）
REQUEST_INTERVAL = 0.5
# 报告期过滤：TTM 同比需要当年 + 去年同期共 8 个季度
# 保留 2017Q1 起，可支撑 2019Q1 起的因子值（截面自 2019-04 起）
REPORT_START = "2017-01-01"

# 测试模式：只爬前 N 只，验证接口与字段
TEST_MODE = False
N_TEST_STOCKS = 3

OUTPUT_COLUMNS = [
    "股票代码",
    "报告期",
    "首次公告日期",
    "营业总收入_累计",
    "销售收现_累计",
]


def progress(message):
    """按仓库脚本惯例：即时打印并 flush，便于观察长任务进度。"""
    print(message, flush=True)


def market_prefix(code):
    """6 开头 → SH，其余（0/3/4/8）→ SZ/BJ，东财接口统一可用。"""
    return "SH" if str(code).zfill(6).startswith("6") else "SZ"


def fetch_table(symbol, fetcher, field, notice_candidates):
    """抓取单张财报，返回 (报告期, 首次公告日期, 字段值) 的 DataFrame。"""
    frame = fetcher(symbol=symbol)
    if field not in frame.columns:
        raise RuntimeError(f"接口返回缺少字段 {field}，实际字段示例：{list(frame.columns)[:15]}")
    notice_column = next((c for c in notice_candidates if c in frame.columns), None)
    if notice_column is None:
        raise RuntimeError(f"接口返回缺少公告日期字段，候选 {notice_candidates}")
    data = frame.loc[:, ["REPORT_DATE", notice_column, field]].copy()
    data.columns = ["报告期", "首次公告日期", field]
    data["报告期"] = pd.to_datetime(data["报告期"], errors="coerce")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    data = data.dropna(subset=["报告期"])
    data = data.loc[data["报告期"] >= pd.Timestamp(REPORT_START)]
    return data.drop_duplicates("报告期", keep="first")


def fetch_one(code):
    """抓取单只股票的营业总收入与销售收现（累计值）。"""
    symbol = f"{market_prefix(code)}{str(code).zfill(6)}"
    profit = fetch_table(
        symbol,
        ak.stock_profit_sheet_by_report_em,
        "TOTAL_OPERATE_INCOME",
        ["NOTICE_DATE", "UPDATE_DATE"],
    )
    cash = fetch_table(
        symbol,
        ak.stock_cash_flow_sheet_by_report_em,
        "SALES_SERVICES",
        ["NOTICE_DATE", "UPDATE_DATE"],
    )
    merged = profit.merge(cash, on="报告期", how="outer", suffixes=("_利润表", "_现金流量表"))
    merged["首次公告日期"] = pd.concat(
        [merged["首次公告日期_利润表"], merged["首次公告日期_现金流量表"]], axis=1
    ).max(axis=1)
    merged["股票代码"] = str(code).zfill(6)
    merged = merged.rename(
        columns={"TOTAL_OPERATE_INCOME": "营业总收入_累计", "SALES_SERVICES": "销售收现_累计"}
    )
    return merged.loc[:, OUTPUT_COLUMNS]


def load_pool():
    pool = pd.read_csv(POOL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    codes = (
        pool["股票代码"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return codes


def load_cache():
    if not OUTPUT_FILE.is_file():
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    cached = pd.read_csv(OUTPUT_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    cached["股票代码"] = cached["股票代码"].astype(str).str.zfill(6)
    return cached, set(cached["股票代码"].unique())


def save(frames):
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined["股票代码"] = combined["股票代码"].astype(str).str.zfill(6)
    combined = combined.sort_values(["股票代码", "报告期"], kind="mergesort")
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main():
    codes = load_pool()
    if TEST_MODE:
        codes = codes[:N_TEST_STOCKS]
        progress(f"[测试模式] 仅爬取前 {N_TEST_STOCKS} 只：{codes}")

    cached, done = load_cache()
    pending = [code for code in codes if code not in done]
    progress(f"股票池 {len(codes)} 只，缓存已有 {len(done)} 只，本次待爬 {len(pending)} 只")
    progress(f"输出文件：{OUTPUT_FILE}")

    frames = [] if cached.empty else [cached]
    failed = []
    for index, code in enumerate(pending, start=1):
        try:
            frames.append(fetch_one(code))
        except Exception as exc:  # noqa: BLE001 - 单只失败不应中断整体爬取
            failed.append((code, repr(exc)))
            progress(f"  [警告] {code} 抓取失败：{exc}")
        if index % 10 == 0 or index == len(pending):
            save(frames)
            frames = [pd.read_csv(OUTPUT_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")]
            progress(f"爬取进度：{index}/{len(pending)}，已保存 {OUTPUT_FILE.name}")
        time.sleep(REQUEST_INTERVAL)

    save(frames)
    final = pd.read_csv(OUTPUT_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    progress("=" * 72)
    progress(
        f"完成：{final['股票代码'].nunique()} 只股票，{len(final)} 条记录，"
        f"报告期 {final['报告期'].min()} ~ {final['报告期'].max()}"
    )
    progress(f"销售收现缺失 {int(final['销售收现_累计'].isna().sum())} 条，"
             f"营业总收入缺失 {int(final['营业总收入_累计'].isna().sum())} 条")
    if failed:
        progress(f"失败 {len(failed)} 只：{failed}")
    progress(f"已保存 → {OUTPUT_FILE}")
    progress("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
