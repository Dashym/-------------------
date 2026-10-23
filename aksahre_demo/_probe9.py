# -*- coding: utf-8 -*-
"""探测申万行业历史的可用性：能否构造动态成分股。"""
import os
import sys

import pandas as pd

ROOT = r"C:\Users\Heyuanhang\Desktop\因子库"
BASE = r"C:\Users\Heyuanhang\Desktop\aksahre_demo"
OUT = []

pool = pd.read_csv(os.path.join(BASE, "汽车零部件_申万二级成分股.csv"), dtype=str, encoding="utf-8-sig")
codes = set(pool["股票代码"].str.extract(r"(\d{6})", expand=False).dropna())

history = pd.read_csv(os.path.join(ROOT, "申万_行业分类_全量历史.csv"), dtype=str, encoding="utf-8-sig")
OUT.append("全量历史 cols=%s rows=%d stocks=%d" % (list(history.columns), len(history), history["股票代码"].nunique()))
OUT.append("行业代码 取值数=%d，示例=%s" % (history["行业代码"].nunique(), sorted(history["行业代码"].unique())[:20]))

sub = history.loc[history["股票代码"].isin(codes)]
OUT.append("")
OUT.append("池内股票的行业历史：%d 条，覆盖 %d 只（池 %d 只）" % (len(sub), sub["股票代码"].nunique(), len(codes)))
OUT.append("池内股票行业代码取值：%s" % sorted(sub["行业代码"].unique()))
OUT.append("每只股票记录数分布：%s" % sub.groupby("股票代码").size().value_counts().to_dict())

sample = sub.loc[sub["股票代码"] == "000030"].sort_values("计入日期")
OUT.append("")
OUT.append("000030 行业历史：\n%s" % sample.to_string(index=False))

# 三级分类里的汽车零部件代码集合
three = pd.read_csv(os.path.join(ROOT, "申万_汽车零部件_三级分类.csv"), dtype=str, encoding="utf-8-sig")
OUT.append("")
OUT.append("三级分类 cols=%s rows=%d" % (list(three.columns), len(three)))
if "三级行业代码" in three.columns:
    OUT.append("三级行业代码取值：%s" % sorted(three["三级行业代码"].dropna().unique()))
OUT.append("快照日期取值：%s" % sorted(three["快照日期"].dropna().unique())[:5])
OUT.append("纳入日 > 2024-11-01 的股票数：%d"
           % (pd.to_datetime(three["纳入日"], errors="coerce") > pd.Timestamp("2024-11-01")).sum())

# 与池的匹配
three_codes = set(three["股票代码"].str.extract(r"(\d{6})", expand=False).dropna())
OUT.append("三级分类 ∩ 池：%d；池中不在三级分类：%s" % (len(three_codes & codes), sorted(codes - three_codes)[:20]))
OUT.append("三级分类中不在池（已剔除/已离开行业）：%d" % len(three_codes - codes))

with open(os.path.join(BASE, "_probe9.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
print("probe9 done", flush=True)
sys.stdout.flush()
