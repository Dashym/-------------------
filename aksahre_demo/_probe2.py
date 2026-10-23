# -*- coding: utf-8 -*-
import os
import sys

BASE = r"C:\Users\Heyuanhang\Desktop\aksahre_demo"
ROOT = r"C:\Users\Heyuanhang\Desktop\因子库"
out = []

# 1) 依赖可用性
for mod in ["scipy", "statsmodels", "akshare", "numpy", "pandas"]:
    try:
        m = __import__(mod)
        out.append("MODULE %-12s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as exc:  # noqa: BLE001
        out.append("MODULE %-12s MISSING (%s)" % (mod, exc))

# 2) 股票池 / 申万历史文件
import pandas as pd

pool = pd.read_csv(os.path.join(BASE, "汽车零部件_申万二级成分股.csv"), dtype={"股票代码": str}, encoding="utf-8-sig")
out.append("")
out.append("POOL 汽车零部件_申万二级成分股.csv rows=%d cols=%s" % (len(pool), list(pool.columns)))
out.append("POOL codes sample: %s" % ", ".join(pool["股票代码"].head(40).tolist()))

for name in ["申万_汽车零部件_三级分类.csv", "申万_汽车零部件_全量历史变动.csv", "申万_行业分类_全量历史.csv", "申万_汽车零部件_三级分类_2024年底快照.csv"]:
    fp = os.path.join(ROOT, name)
    df = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
    out.append("")
    out.append("HIST %s rows=%d cols=%s" % (name, len(df), list(df.columns)))
    out.append("     head: %s" % df.head(3).to_dict("records"))
    for col in df.columns:
        if "日" in col or "date" in col.lower():
            out.append("     %s range: %s ~ %s" % (col, df[col].min(), df[col].max()))

# 3) 行情文件
q = pd.read_csv(os.path.join(BASE, "A股日线_20190101-20260331_后复权.csv"), dtype={"股票代码": str}, encoding="utf-8-sig", usecols=["股票代码", "日期", "收盘"])
out.append("")
out.append("QUOTES 后复权 rows=%d stocks=%d dates=%s ~ %s" % (len(q), q["股票代码"].nunique(), q["日期"].min(), q["日期"].max()))

with open(os.path.join(BASE, "_probe2.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("probe2 done")
sys.stdout.flush()
