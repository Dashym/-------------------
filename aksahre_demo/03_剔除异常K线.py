# -*- coding: utf-8 -*-
"""
03_剔除异常K线.py
读取 clean_temp 文件夹里三份去重排序后的 CSV 行情文件，
剔除以下异常 K 线：
    1. 最高价 high 小于最低价 low
    2. 开盘、收盘、最高、最低中任意价格小于 0
    3. 成交量 volume 小于 0
    4. 停牌 K 线：open == high == low == close 四价相等，
       且成交量为空（NaN）或 ≤ 0
       —— 一字涨停/一字跌停（四价相等但成交量 > 0）保留不动
清洗后的结果以 "_剔除异常" 为后缀另存到 clean_temp 文件夹，
绝不覆盖原始输入文件。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# 路径配置
BASE_DIR = Path(__file__).resolve().parent          # 当前脚本所在目录
CLEAN_DIR = BASE_DIR / "clean_temp"                 # 清洗中间文件目录

# 待处理的三份"去重排序"CSV 文件（由 去重排序.py 生成）
INPUT_FILES = [
    "A股日线_20190101-20260331_不复权_去重排序.csv",
    "A股日线_20190101-20260331_前复权_去重排序.csv",
    "A股日线_20190101-20260331_后复权_去重排序.csv",
]

# 中文列名 → 英文标准名 映射（本脚本内部统一使用英文列名做判断）
COLUMN_MAP = {
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}

# 需要参与"价格 < 0"判断的四个价格字段
PRICE_COLS = ["open", "close", "high", "low"]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 的中文列名统一映射为英文标准名，便于后续规则判断。"""
    rename_dict = {}
    for cn_name, en_name in COLUMN_MAP.items():
        if cn_name in frame.columns:
            rename_dict[cn_name] = en_name
    return frame.rename(columns=rename_dict)


def validate_required_columns(frame: pd.DataFrame, filename: str) -> None:
    """校验当前文件是否包含清洗所需的全部关键字段，缺失则抛异常。"""
    missing = [col for col in COLUMN_MAP.values() if col not in frame.columns]
    if missing:
        raise ValueError(
            f"{filename} 缺少必需列：{missing}，当前列有：{list(frame.columns)}"
        )


def remove_abnormal_kline(input_path: Path) -> None:
    """对单个 CSV 文件执行异常 K 线剔除，并保存新文件。"""
    filename = input_path.name

    # 1. 读取 CSV（股票代码保留为字符串以免丢失前导零）
    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype={"股票代码": "string"})
    original_rows = len(df)

    # 2. 列名标准化 + 必需字段校验
    df = normalize_columns(df)
    validate_required_columns(df, filename)

    # 3. 价格、成交量统一转数值（非法字符串会变成 NaN，后续也会被剔除）
    for col in PRICE_COLS + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. 构造四条异常规则的布尔掩码
    mask_high_lt_low = df["high"] < df["low"]                      # 规则1：最高 < 最低
    mask_price_neg = (df[PRICE_COLS] < 0).any(axis=1)              # 规则2：任意价格 < 0
    mask_volume_neg = df["volume"] < 0                             # 规则3：成交量 < 0（兜底脏数据）

    # 规则4：停牌 K 线 —— 两个条件必须同时成立才剔除：
    #   条件A：open == high == low == close  （四价相等）
    #   条件B：成交量为空(NaN) 或 ≤ 0        （无成交）
    # 这样一字涨停/跌停（四价相等 但 成交量 > 0）不会被误伤
    mask_ohlc_flat = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )
    mask_volume_still = df["volume"].isna() | (df["volume"] <= 0)
    mask_suspend = mask_ohlc_flat & mask_volume_still
    print(f"  [DEBUG] mask_suspend = {mask_suspend.sum()} 行"
          f"  (其中 四价等={mask_ohlc_flat.sum()}, "
          f"量≤0或NaN={mask_volume_still.sum()}, "
          f"一字板保护={mask_ohlc_flat.sum() - mask_suspend.sum()})")

    # 同时把价格或成交量为 NaN 的脏行也剔除掉
    mask_nan = df[PRICE_COLS + ["volume"]].isna().any(axis=1)

    # 合并：只要命中任意一条异常规则就剔除
    mask_abnormal = (
        mask_high_lt_low
        | mask_price_neg
        | mask_volume_neg
        | mask_suspend
        | mask_nan
    )

    # 5. 剔除非正常 K 线
    df_clean = df.loc[~mask_abnormal].copy()
    removed_rows = original_rows - len(df_clean)

    # 6. 恢复中文列名，保证输出文件与上游脚本列名风格一致
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df_clean = df_clean.rename(columns=reverse_map)

    # 7. 另存为 "原名_剔除异常.csv"，绝不覆盖输入文件
    output_path = CLEAN_DIR / f"{input_path.stem}_剔除异常.csv"
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 8. 打印统计信息
    print("-" * 60)
    print(f"文件：{filename}")
    print(f"  原始行数：{original_rows}")
    print(f"  剔除行数：{removed_rows}  "
          f"(最高<最低={mask_high_lt_low.sum()}, "
          f"价格<0={mask_price_neg.sum()}, "
          f"成交量<0={mask_volume_neg.sum()}, "
          f"停牌(四价等+量<=0)={mask_suspend.sum()}, "
          f"含NaN={mask_nan.sum()})")
    print(f"  剩余行数：{len(df_clean)}")
    print(f"  输出文件：{output_path.name}")


def main() -> None:
    """遍历 clean_temp 下的三份去重排序 CSV，逐一执行异常 K 线剔除。"""
    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(f"找不到 clean_temp 目录：{CLEAN_DIR}")

    print("=" * 60)
    print("开始剔除异常 K 线")
    print(f"工作目录：{CLEAN_DIR}")
    print("=" * 60)

    for filename in INPUT_FILES:
        input_path = CLEAN_DIR / filename
        if not input_path.is_file():
            print(f"[跳过] 找不到文件：{filename}")
            continue
        remove_abnormal_kline(input_path)

    print("=" * 60)
    print("异常 K 线剔除完成")
    print("=" * 60)


if __name__ == "__main__":
    main()