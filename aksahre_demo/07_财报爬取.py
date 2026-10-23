# -*- coding: utf-8 -*-
"""
07_财报爬取.py（Builder 模式实现）
==================================
批量爬取 208 只汽车零部件股票的季度财报核心数据。
三张表均使用 akshare 东方财富 "按报告期" 接口；
利润表、现金流量表返回的是年初至期末的**累计值**，
需在爬取后对 Q2 / Q3 / Q4 做差分还原为**单季度值**；
资产负债表为时点快照数据，直接取值即可。

输出：raw_data / A股汽车零部件_季度财报_原始_2019-2026.csv

Builder 链式调用示例：
    FinancialCrawler(stock_codes) \
        .prepare() \
        .fetch_all() \
        .diff_to_quarterly() \
        .sort_and_save()
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd


# ========================================================================== #
# 路径 & 常量配置
# ========================================================================== #
BASE_DIR = Path(__file__).resolve().parent          # 当前脚本所在目录
RAW_DIR = BASE_DIR / "raw_data"                     # 原始数据目录
OUTPUT_FILE = RAW_DIR / "A股汽车零部件_季度财报_原始_2019-2026.csv"

# 报告期起止（仅保留 2019Q1 及之后的数据）
YEAR_START = 2019
YEAR_END = 2026

# 接口限流控制：每次请求之间的最小间隔（秒）
REQUEST_INTERVAL = 0.8

# 爬取接口对应的市场前缀规则：6 开头 → SH，其他 → SZ / BJ（北交所统一归到 SZ 也能请求到，实际测试用 SZ 即可覆盖 0/3/4 开头）
def _market_prefix(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("6"):
        return "SH"
    return "SZ"


# ========================================================================== #
# 字段映射：akshare 原始列名  →  最终输出列名
# ========================================================================== #

# 利润表（累计值，需差分取单季度）
PROFIT_FIELDS = {
    "TOTAL_OPERATE_INCOME":         "营业总收入",
    "TOTAL_OPERATE_COST":           "营业成本",
    "RESEARCH_EXPENSE":            "研发费用",
    "PARENT_NETPROFIT":             "归属于母公司股东的净利润",
    "DEDUCT_PARENT_NETPROFIT":      "扣除非经常性损益后的净利润",
}

# 资产负债表（时点数据，直接取值）
BALANCE_FIELDS = {
    "INVENTORY":                    "存货",
    "ACCOUNTS_RECE":                "应收账款",
    "TOTAL_ASSETS":                 "总资产",
    "TOTAL_PARENT_EQUITY":          "归属于母公司股东权益合计",
    "GOODWILL":                     "商誉",
    "FIXED_ASSET":                  "固定资产",
}

# 现金流量表（累计值，需差分取单季度）
CASHFLOW_FIELDS = {
    "NETCASH_OPERATE":              "经营活动产生的现金流量净额",
}

# 同时出现在多张表中的通用字段
KEY_FIELDS = ["REPORT_DATE", "NOTICE_DATE", "SECURITY_CODE"]


# ========================================================================== #
# Builder 类
# ========================================================================== #
class FinancialCrawler:
    """
    财报爬取 Builder。
    设计目标：每一步职责单一，可通过链式调用拼出完整流程；
             允许在任意阶段插入自定义逻辑（比如跳过已爬过的股票）。
    """

    # ------------------------------------------------------------------ #
    # 内部状态
    # ------------------------------------------------------------------ #
    _cumulative_mask: pd.Series            # 标记哪些输出列是"累计值"，需要差分
    _drop_mask: pd.Series                  # 标记哪些输出列是"时点值"，直接保留

    # ------------------------------------------------------------------ #
    # Step 0 · 构造器：接收股票代码列表
    # ------------------------------------------------------------------ #
    def __init__(self, stock_codes: List[str]) -> None:
        # 统一补前导零
        self._stock_codes: List[str] = [str(c).zfill(6) for c in stock_codes]

        # 运行中状态
        self._frames: List[pd.DataFrame] = []    # 累积每只股票合并后的 DataFrame
        self._success_codes: List[str] = []       # 爬取成功的股票代码
        self._fail_codes: List[str] = []          # 爬取失败的股票代码

        # 用于最终标记哪些字段需要差分还原为单季度
        self._cumulative_fields = list(PROFIT_FIELDS.values()) + list(CASHFLOW_FIELDS.values())
        self._point_in_time_fields = list(BALANCE_FIELDS.values())

    # ------------------------------------------------------------------ #
    # Step 1 · 检查输出文件 & 目录（禁止覆盖已有文件）
    # ------------------------------------------------------------------ #
    def prepare(self) -> "FinancialCrawler":
        if OUTPUT_FILE.exists():
            raise FileExistsError(
                f"输出文件已存在，为防止覆盖请先手动删除或改名：{OUTPUT_FILE}"
            )
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print(f"准备爬取 {len(self._stock_codes)} 只股票的季度财报")
        print(f"输出路径：{OUTPUT_FILE}")
        print("=" * 60)
        return self

    # ------------------------------------------------------------------ #
    # Step 2 · 爬取单只股票的三张表 → 合并成一张
    # ------------------------------------------------------------------ #
    def _fetch_one(self, code: str) -> Optional[pd.DataFrame]:
        """爬取一只股票的三张表并按报告期 inner join。"""
        symbol = f"{_market_prefix(code)}{code}"
        try:
            # 利润表
            profit = ak.stock_profit_sheet_by_report_em(symbol=symbol)
            profit = profit[KEY_FIELDS + list(PROFIT_FIELDS.keys())].copy()

            # 资产负债表
            balance = ak.stock_balance_sheet_by_report_em(symbol=symbol)
            balance = balance[KEY_FIELDS + list(BALANCE_FIELDS.keys())].copy()

            # 现金流量表
            cash = ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
            cash = cash[KEY_FIELDS + list(CASHFLOW_FIELDS.keys())].copy()

        except Exception as exc:
            print(f"  [失败] {code}: {exc}")
            return None

        # 按 REPORT_DATE 三表 inner join（取三者都有记录的报告期）
        merged = profit.merge(balance, on="REPORT_DATE", how="inner", suffixes=("_p", "_b"))
        merged = merged.merge(cash, on="REPORT_DATE", how="inner", suffixes=("", "_c"))

        # 字段重命名为中文输出列名
        rename_dict = {}
        rename_dict.update({k: v for k, v in PROFIT_FIELDS.items() if k in merged.columns})
        rename_dict.update({k: v for k, v in BALANCE_FIELDS.items() if k in merged.columns})
        rename_dict.update({k: v for k, v in CASHFLOW_FIELDS.items() if k in merged.columns})
        merged = merged.rename(columns=rename_dict)

        # 规范化股票代码（从 SECURITY_CODE 里提取 6 位数字）
        merged["股票代码"] = (
            merged["SECURITY_CODE"]
            .astype("string")
            .str.extract(r"(\d{6})", expand=False)
            .str.zfill(6)
        )

        # 报告期 & 公告日期转 datetime
        merged["报告期"] = pd.to_datetime(merged["REPORT_DATE"], errors="coerce")
        merged["首次公告日期"] = pd.to_datetime(merged["NOTICE_DATE"], errors="coerce")

        # 按报告期升序（后面差分需要按时间顺序）
        merged = merged.sort_values("报告期").reset_index(drop=True)

        # 仅保留目标年份范围（YEAR_START - YEAR_END）
        mask_year = (
            merged["报告期"].dt.year >= YEAR_START
        ) & (
            merged["报告期"].dt.year <= YEAR_END
        )
        merged = merged.loc[mask_year].copy()

        # 清理中间列
        drop_cols = ["REPORT_DATE", "NOTICE_DATE", "SECURITY_CODE"]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

        # 把数值列强制转为 float（非法字符 → NaN）
        numeric_cols = self._cumulative_fields + self._point_in_time_fields
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # 调整列顺序为：股票代码 → 报告期 → 首次公告日期 → 利润表 → 资产负债表 → 现金流量表
        front = ["股票代码", "报告期", "首次公告日期"]
        order = (
            front
            + list(PROFIT_FIELDS.values())
            + list(BALANCE_FIELDS.values())
            + list(CASHFLOW_FIELDS.values())
        )
        order = [c for c in order if c in merged.columns]
        merged = merged[order]

        return merged

    # ------------------------------------------------------------------ #
    # Step 3 · 循环爬取全部股票
    # ------------------------------------------------------------------ #
    def fetch_all(self) -> "FinancialCrawler":
        total = len(self._stock_codes)
        for idx, code in enumerate(self._stock_codes, start=1):
            print(f"[{idx}/{total}] 正在爬取 {code} ...", end=" ", flush=True)
            df = self._fetch_one(code)

            if df is not None and not df.empty:
                self._frames.append(df)
                self._success_codes.append(code)
                print(f"成功（{len(df)} 条报告期）")
            else:
                self._fail_codes.append(code)
                print("无有效数据或失败")

            # 请求间隔（最后一只可以跳过）
            if idx < total:
                time.sleep(REQUEST_INTERVAL)

        # 汇总统计
        print("-" * 60)
        print(f"爬取完成：成功 {len(self._success_codes)} 只 / 失败 {len(self._fail_codes)} 只")
        if self._fail_codes:
            print(f"失败股票代码列表：{', '.join(self._fail_codes)}")
        return self

    # ------------------------------------------------------------------ #
    # Step 4 · 累计值 → 单季度值（Q2/Q3/Q4 做差分，Q1 直接保留）
    # ------------------------------------------------------------------ #
    def diff_to_quarterly(self) -> "FinancialCrawler":
        if not self._frames:
            print("[警告] 没有任何数据，跳过分步差分")
            return self

        print("-" * 60)
        print("开始将累计值还原为单季度数据（Q2/Q3/Q4 差分，Q1 直接保留）")

        # 对每只股票、每个累计值字段做"同年内"差分
        # 逻辑：
        #   同年 Q2 = Q2累计 - Q1累计
        #   同年 Q3 = Q3累计 - Q2累计
        #   同年 Q4 = Q4累计 - Q3累计
        #   跨年或 Q1：直接用当期累计值（因为 Q1 的累计值就是 Q1 本身）
        for col in self._cumulative_fields:
            for i, df in enumerate(self._frames):
                if col not in df.columns:
                    continue

                # 取上一条的累计值；同一股票的第一行会得到 NaN
                prev = df[col].shift(1)

                # 判断"上一条和当前是否在同一年"
                cur_year = df["报告期"].dt.year
                prev_year = df["报告期"].dt.year.shift(1)
                same_year = cur_year == prev_year      # True=同年；False=跨年或首行

                # 构造差分减数：同一年用上一期累计值，跨年/首行当作 0
                prev_effective = prev.where(same_year, 0).fillna(0)

                self._frames[i][col] = df[col] - prev_effective

        print("累计值 → 单季度值转换完成")
        return self

    # ------------------------------------------------------------------ #
    # Step 5 · 合并、排序、保存
    # ------------------------------------------------------------------ #
    def sort_and_save(self) -> None:
        if not self._frames:
            raise RuntimeError("没有任何数据可以保存")

        # 合并所有股票
        df_all = pd.concat(self._frames, ignore_index=True)

        # 强制恢复股票代码为 6 位字符串（concat 后可能被 pandas 推断为 int64，
        # 导致写 CSV 时前导零丢失，比如 "000030" 变成 30）
        df_all["股票代码"] = df_all["股票代码"].astype("string").str.zfill(6)

        # 按股票代码升序、报告期升序排列
        df_all = df_all.sort_values(
            by=["股票代码", "报告期"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        # 保存到 raw_data
        df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print("=" * 60)
        print(f"汇总：总行数 {len(df_all)}，股票数量 {df_all['股票代码'].nunique()}")
        print(f"最早报告期：{df_all['报告期'].min()}")
        print(f"最晚报告期：{df_all['报告期'].max()}")
        print(f"已保存 → {OUTPUT_FILE}")
        print("=" * 60)


# ========================================================================== #
# 入口函数
# ========================================================================== #

# ===== 测试模式开关 =====
# True  = 只爬前 N 只做冒烟测试，快速验证接口 & 差分逻辑
# False = 全量爬取 208 只（正式生产）
TEST_MODE = False
N_TEST_STOCKS = 5
# ======================


def main() -> None:
    # 方式一：直接在代码里写股票代码列表（与股票池.py 底部的 stock_codes 保持一致）
    from 股票池 import stock_codes

    # 也可以从 CSV 文件读取：
    # pool_csv = BASE_DIR / "汽车零部件_申万二级成分股.csv"
    # stock_codes = pd.read_csv(pool_csv, dtype={"股票代码": "string"})["股票代码"].tolist()

    # 根据开关裁剪股票池 & 调整输出文件名
    if TEST_MODE:
        stock_codes = stock_codes[:N_TEST_STOCKS]
        # 全局替换输出文件名，加 _TEST 后缀避免污染正式结果
        global OUTPUT_FILE
        OUTPUT_FILE = RAW_DIR / "A股汽车零部件_季度财报_原始_2019-2026_TEST.csv"
        print(f"[测试模式] 仅爬取前 {N_TEST_STOCKS} 只：{stock_codes}")

    FinancialCrawler(stock_codes).prepare().fetch_all().diff_to_quarterly().sort_and_save()


if __name__ == "__main__":
    main()