# -*- coding: utf-8 -*-
"""探测本地行情/财报文件的字段与规模（为构建真实因子做准备）。"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = r"C:\Users\Heyuanhang\Desktop\aksahre_demo"
OUT = []
p = lambda s: OUT.append(s)  # noqa: E731

fin = pd.read_csv(
    os.path.join(BASE, "processed_data", "财报_清洗完成.csv"),
    dtype={"股票代码": str},
    encoding="utf-8-sig",
)
p("财报 rows=%d stocks=%d" % (len(fin), fin["股票代码"].nunique()))
p("报告期 range: %s ~ %s" % (fin["报告期"].min(), fin["报告期"].max()))
p("首次公告日期 null=%d" % fin["首次公告日期"].isna().sum())
p("公告日 < 报告期 的行数: %d" % (pd.to_datetime(fin["首次公告日期"]) < pd.to_datetime(fin["报告期"])).sum())
p("公告日 range: %s ~ %s" % (fin["首次公告日期"].min(), fin["首次公告日期"].max()))
p("stocks: %s" % ", ".join(sorted(fin["股票代码"].unique())[:15]))
p("每只股票报告期数分布: %s" % fin.groupby("股票代码")["报告期"].count().value_counts().to_dict())

adj = pd.read_csv(
    os.path.join(BASE, "clean_temp", "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"),
    dtype={"股票代码": str},
    encoding="utf-8-sig",
)
p("")
p("后复权(格式统一) rows=%d cols=%s" % (len(adj), list(adj.columns)))
p("  日期 range: %s ~ %s" % (adj["日期"].min(), adj["日期"].max()))
p("  stocks=%d" % adj["股票代码"].nunique())

raw = pd.read_csv(
    os.path.join(BASE, "clean_temp", "A股日线_20190101-20260331_不复权_去重排序_剔除异常_格式统一.csv"),
    dtype={"股票代码": str},
    encoding="utf-8-sig",
)
p("")
p("不复权(格式统一) rows=%d cols=%s" % (len(raw), list(raw.columns)))
p("  换手率 dtype=%s min=%s max=%s" % (raw["换手率"].dtype, raw["换手率"].min(), raw["换手率"].max()))
p("  成交量 min=%s max=%s" % (raw["成交量"].min(), raw["成交量"].max()))

# 月末交易日日历（2019-01 ~ 2024-11）
adj["日期"] = pd.to_datetime(adj["日期"])
cal = pd.Series(sorted(adj["日期"].unique()))
cal = pd.DataFrame({"日期": cal})
cal["月份"] = cal["日期"].dt.to_period("M")
month_end = cal.groupby("月份")["日期"].max()
p("")
p("交易日历 %s ~ %s, 交易日数=%d" % (cal["日期"].min().date(), cal["日期"].max().date(), len(cal)))
p("月末交易日 2019-01~2024-11 (共%d个):" % len(month_end))
p("  " + ", ".join(str(d.date()) for d in month_end.loc[:pd.Period("2024-11")]))

with open(os.path.join(BASE, "_probe7.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
print("probe7 done", flush=True)
sys.stdout.flush()
