# -*- coding: utf-8 -*-
"""
A股汽车零部件季度财报基础清洗脚本
输入：raw_data/A股汽车零部件_季度财报_原始_2019-2026.csv
输出：processed_data/财报_清洗完成.csv
"""

import os
import pandas as pd

# ==================== 步骤1：导入库并读取原始CSV ====================
RAW_DIR = "raw_data"
RAW_FILE = os.path.join(RAW_DIR, "A股汽车零部件_季度财报_原始_2019-2026.csv")
OUTPUT_DIR = "processed_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "财报_清洗完成.csv")

# 创建输出目录（已存在则不报错）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取原始财报CSV（只读，不修改源文件）
# dtype={'股票代码': str} 防止pandas自动把股票代码解析为整数导致前导0丢失
try:
    df = pd.read_csv(RAW_FILE, encoding="utf-8", dtype={"股票代码": str})
except UnicodeDecodeError:
    df = pd.read_csv(RAW_FILE, encoding="gbk", dtype={"股票代码": str})
except Exception as e:
    print(f"读取原始文件异常：{e}")
    raise

# 股票代码标准化：去空格 + 左补零到6位（防止源数据有残缺）
df["股票代码"] = df["股票代码"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)

# 步骤2：基础统计，打印清洗前总行数
total_before = len(df)
print(f"【清洗前总行数】: {total_before}")
print(f"【股票代码数量】: {df['股票代码'].nunique()}")
print(f"【股票代码样例】: {df['股票代码'].unique()[:5].tolist()}")

# ==================== 步骤3：清洗规则（按顺序执行，不可颠倒） ====================

# 3.1 删除 report_date 为空、股票代码 stock_id 为空的无效空行
# 对应中文列名：报告期、股票代码
before_31 = len(df)
df = df.dropna(subset=["报告期", "股票代码"])
# 同时清理空字符串（有些空值可能被存为""）
df = df[~df["报告期"].astype(str).str.strip().eq("")]
df = df[~df["股票代码"].astype(str).str.strip().eq("")]
print(f"步骤3.1 删除报告期或股票代码为空的无效行：{before_31 - len(df)} 行")

# 3.2 去除完全重复的整行数据，只保留第一条重复记录
before_32 = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"步骤3.2 删除完全重复整行：{before_32 - len(df)} 行")

# 3.3 保留 report_date 合法日期（YYYY-MM-DD格式），剔除日期格式错乱、无法解析的行
before_33 = len(df)
# 尝试将报告期解析为日期，解析失败的转为NaT，然后过滤掉
df["报告期_parsed"] = pd.to_datetime(df["报告期"], errors="coerce", format="mixed")
invalid_date_mask = df["报告期_parsed"].isna()
invalid_date_count = invalid_date_mask.sum()
if invalid_date_count > 0:
    print(f"步骤3.3 发现 {invalid_date_count} 行报告期日期格式非法，已剔除")
    df = df[~invalid_date_mask]
# 同步清洗首次公告日期（能解析最好，解析不了也别炸，留着参与后续排序）
df["首次公告日期_parsed"] = pd.to_datetime(df["首次公告日期"], errors="coerce", format="mixed")
print(f"步骤3.3 日期合法性校验后剩余：{len(df)} 行")

# 3.4 同股票 + 同报告期 可能有多条记录（原始公告、更正公告）
# 去重规则：同一股票、同一报告期，只保留首次公告日期最早那一条（原始财报，丢弃更正/补充公告）
before_34 = len(df)
# 按股票代码 + 报告期 分组，按首次公告日期升序排列，每组取第一条
# 公告日期缺失的排在最后（sort_values 的 na_position="last"）
df = (
    df.sort_values(
        by=["股票代码", "报告期_parsed", "首次公告日期_parsed"],
        ascending=[True, True, True],
        na_position="last",
    )
    .groupby(["股票代码", "报告期_parsed"], as_index=False)
    .first()
)
print(f"步骤3.4 同股票+同报告期保留首次公告：{before_34 - len(df)} 行 (更正/补充公告已剔除)")

# 清理临时解析列
df = df.drop(columns=["报告期_parsed", "首次公告日期_parsed"])

# ==================== 步骤4：清洗完成后统计 ====================
total_after = len(df)
deleted = total_before - total_after
print(f"\n【清洗后总行数】: {total_after}")
print(f"【合计删除行数】: {deleted}")

# ==================== 步骤5：保存清洗后文件 ====================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ==================== 步骤6：打印输出校验 ====================
print("\n【清洗后前10行预览】(仅展示 股票代码 / 报告期 / 首次公告日期)")
# 兼容列名：如果有中文列名就用中文列名，否则用英文别名
cols_show = ["股票代码", "报告期", "首次公告日期"]
# 转为字符串展示避免长科学计数
for c in cols_show:
    if c in df.columns:
        df[c] = df[c].astype(str)
print(df[cols_show].head(10).to_string(index=False))

# 最终提示
print(f"\n财报清洗任务执行完成，输出文件路径为 {OUTPUT_FILE}")