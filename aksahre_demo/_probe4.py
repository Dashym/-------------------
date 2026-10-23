# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd

BASE = r"C:\Users\Heyuanhang\Desktop\aksahre_demo"
out = []
for name in [
    "汽车零部件_申万二级成分股.csv",
    "汽车零部件_剔除ST_上市日期截至2024-12-31.csv",
    "汽车零部件_剔除ST_上市日期2025-01-01及以后.csv",
    "规模因子.csv",
    "ic_result.txt",
    "factor_quality_check.py",
]:
    fp = os.path.join(BASE, name)
    if not os.path.isfile(fp):
        out.append("MISSING %s" % name)
        continue
    if name.endswith(".csv"):
        df = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
        out.append("%-55s rows=%d cols=%s" % (name, len(df), list(df.columns)))
    else:
        out.append("%-55s (%d bytes)" % (name, os.path.getsize(fp)))

# 汽车零部件三级分类 2024 年底快照里的二级成员数
fp = r"C:\Users\Heyuanhang\Desktop\因子库\申万_汽车零部件_三级分类_2024年底快照.csv"
df = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
out.append("")
out.append("2024年快照 二级行业取值: %s" % df["二级行业"].unique().tolist())
out.append("2024年快照 股票数=%d 三级行业分布=\n%s" % (len(df), df["三级行业名称"].value_counts().to_string()))

with open(os.path.join(BASE, "_probe4.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("probe4 done")
sys.stdout.flush()
