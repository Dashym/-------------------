# -*- coding: utf-8 -*-
"""
06_格式统一.py（Builder 模式实现）
==================================
独立的"行情数据格式标准化"步骤，可单独运行、可被复用。
读取 clean_temp 文件夹内三份"剔除异常 K 线后"的 CSV，
对日期、股票代码、排序等字段做格式统一，并输出统计校验信息。

处理规则：
    1. date   → 强制转换为 pandas datetime 类型
    2. symbol → 强制转为 6 位字符串，不足自动补前导 0
    3. 按 symbol、date 升序排序，重置索引
    4. 打印校验信息：股票数量 / 总行数 / 最早日期 / 最晚日期

输出：另存为 "原名_格式统一.csv"，绝不覆盖输入文件。

设计：采用 Builder 模式，支持链式调用，每一步职责单一，方便插入/裁剪。
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 路径 & 文件名配置
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent          # 当前脚本所在目录
CLEAN_DIR = BASE_DIR / "clean_temp"                 # 清洗中间文件目录

# 输入文件：上一步（03_剔除异常K线.py）生成的三份 CSV
INPUT_FILES = [
    "A股日线_20190101-20260331_不复权_去重排序_剔除异常.csv",
    "A股日线_20190101-20260331_前复权_去重排序_剔除异常.csv",
    "A股日线_20190101-20260331_后复权_去重排序_剔除异常.csv",
]

# 中文列名 → 本脚本内部统一使用的英文标准名（便于跨文件复用）
COLUMN_ALIAS = {
    "股票代码": "symbol",
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

# 反向映射，保存时把列名恢复为中文，保证与上下游脚本列名风格一致
COLUMN_ALIAS_REVERSE = {v: k for k, v in COLUMN_ALIAS.items()}

# 必需的核心列（缺失则直接报错）
REQUIRED_COLUMNS_ZH = {"股票代码", "日期"}


# --------------------------------------------------------------------------- #
# Builder 类
# --------------------------------------------------------------------------- #
class FormatStandardizer:
    """
    行情数据格式统一 Builder。

    使用示例（链式调用）：
        (
            FormatStandardizer()
            .load(input_path)
            .rename_columns()
            .validate_columns()
            .cast_date()
            .cast_symbol()
            .sort_and_reset()
            .print_summary()
            .save(output_path)
        )

    每一步都返回 self，用户可以按需省略某一步，或调整步骤顺序。
    """

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None    # 持有当前 DataFrame
        self._input_path: Optional[Path] = None     # 输入文件路径

    # ------------------------------------------------------------------ #
    # Step 1 · 读取 CSV
    # ------------------------------------------------------------------ #
    def load(self, path: Path) -> "FormatStandardizer":
        """读取 CSV 文件，股票代码列按字符串读取以保留前导零。"""
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")

        self._input_path = path
        self._df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"股票代码": "string", "symbol": "string"},
        )
        print(f"[加载] {path.name}  原始行数 = {len(self._df)}")
        return self

    # ------------------------------------------------------------------ #
    # Step 2 · 列名标准化
    # ------------------------------------------------------------------ #
    def rename_columns(self) -> "FormatStandardizer":
        """把中文列名统一映射为英文标准名（内部处理用）。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        rename_dict = {
            cn: en for cn, en in COLUMN_ALIAS.items() if cn in self._df.columns
        }
        self._df = self._df.rename(columns=rename_dict)
        return self

    # ------------------------------------------------------------------ #
    # Step 3 · 必需列校验
    # ------------------------------------------------------------------ #
    def validate_columns(self) -> "FormatStandardizer":
        """校验必需的 symbol / date 列是否存在，缺失则抛异常。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        missing_zh = REQUIRED_COLUMNS_ZH.difference(COLUMN_ALIAS_REVERSE.get(col, col)
                                                     for col in self._df.columns)
        # 同时兼容已经是英文名的情况
        required_en = {"symbol", "date"}
        missing_en = required_en.difference(self._df.columns)

        if missing_en:
            raise ValueError(
                f"{self._input_path.name} 缺少必需列：{missing_en}，"
                f"当前列有：{list(self._df.columns)}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Step 4 · date 字段 → datetime
    # ------------------------------------------------------------------ #
    def cast_date(self) -> "FormatStandardizer":
        """把 date 列强制转换为 pandas datetime 类型。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        self._df["date"] = pd.to_datetime(self._df["date"], errors="coerce")

        nan_count = int(self._df["date"].isna().sum())
        if nan_count:
            print(f"[日期] 发现 {nan_count} 条无法解析的日期（已转为 NaT，保留在数据中）")
        return self

    # ------------------------------------------------------------------ #
    # Step 5 · symbol 股票代码 → 6 位字符串
    # ------------------------------------------------------------------ #
    def cast_symbol(self) -> "FormatStandardizer":
        """把 symbol 列统一转为字符串，清理 .0 后缀，zfill(6) 补足前导零。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        series = self._df["symbol"].astype("string").str.strip()
        series = series.str.replace(r"\.0+$", "", regex=True)   # 清理 float 转 str 带来的 .0
        self._df["symbol"] = series.str.zfill(6)               # 补前导零到 6 位
        return self

    # ------------------------------------------------------------------ #
    # Step 6 · 排序 + 重置索引
    # ------------------------------------------------------------------ #
    def sort_and_reset(self) -> "FormatStandardizer":
        """按 symbol 升序、date 升序排序，然后重置索引。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        self._df = self._df.sort_values(
            by=["symbol", "date"],
            ascending=[True, True],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        return self

    # ------------------------------------------------------------------ #
    # Step 7 · 统计校验（打印，不改变数据）
    # ------------------------------------------------------------------ #
    def print_summary(self) -> "FormatStandardizer":
        """打印当前数据的校验统计信息。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        print("-" * 60)
        print(f"[校验] {self._input_path.name}")
        print(f"  股票数量    : {self._df['symbol'].nunique()}")
        print(f"  总行数      : {len(self._df)}")
        print(f"  最早日期    : {self._df['date'].min()}")
        print(f"  最晚日期    : {self._df['date'].max()}")
        print(f"  date 类型   : {self._df['date'].dtype}")
        print(f"  symbol 类型 : {self._df['symbol'].dtype}")
        return self

    # ------------------------------------------------------------------ #
    # Step 8 · 保存（恢复中文列名）
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """保存到 CSV；保存前把列名恢复为中文，保证与上下游一致。"""
        if self._df is None:
            raise RuntimeError("请先调用 .load(path) 读取数据")

        # 反向把英文列名换回中文，缺失的列（可能未在映射中）保持原样
        rename_back = {
            en: zh for en, zh in COLUMN_ALIAS_REVERSE.items()
            if en in self._df.columns
        }
        df_to_save = self._df.rename(columns=rename_back)

        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[保存] → {path.name}")

    # ------------------------------------------------------------------ #
    # 快捷方法：一键跑完整链路
    # ------------------------------------------------------------------ #
    def run_full_pipeline(self, input_path: Path, output_path: Path) -> None:
        """一次性走完 load → rename → validate → cast_date → cast_symbol
        → sort → summary → save 全链路。"""
        (
            self.load(input_path)
                .rename_columns()
                .validate_columns()
                .cast_date()
                .cast_symbol()
                .sort_and_reset()
                .print_summary()
                .save(output_path)
        )


# --------------------------------------------------------------------------- #
# 入口函数
# --------------------------------------------------------------------------- #
def main() -> None:
    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(f"找不到 clean_temp 目录：{CLEAN_DIR}")

    print("=" * 60)
    print("开始统一行情数据格式")
    print(f"工作目录：{CLEAN_DIR}")
    print("=" * 60)

    for filename in INPUT_FILES:
        input_path = CLEAN_DIR / filename
        output_path = CLEAN_DIR / f"{Path(filename).stem}_格式统一.csv"

        if not input_path.is_file():
            print(f"[跳过] 找不到文件：{filename}")
            continue

        # 使用 Builder 模式
        FormatStandardizer().run_full_pipeline(input_path, output_path)

    print("=" * 60)
    print("行情数据格式统一完成")
    print("=" * 60)


if __name__ == "__main__":
    main()