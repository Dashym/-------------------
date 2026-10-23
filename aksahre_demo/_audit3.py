# -*- coding: utf-8 -*-
"""逐步复现 load_crawl_cache，定位到底哪一步把 190 只缩成 10 只。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
from sales_receipt_ratio_improvement import CUMULATIVE_COLUMNS, normalize_codes  # noqa: E402

CACHE = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\raw_data\汽车零部件_销售收现与营业收入_原始.csv")
out = []


def snap(tag, frame):
    codes = frame["股票代码"].astype(str).nunique() if "股票代码" in frame else -1
    out.append("%-46s rows=%5d 唯一代码=%4d" % (tag, len(frame), codes))


data = pd.read_csv(CACHE, dtype={"股票代码": str}, encoding="utf-8-sig")
snap("1. read_csv 原始", data)
out.append("   股票代码样例（前3/后3）：%s ... %s" % (
    data["股票代码"].head(3).tolist(), data["股票代码"].tail(3).tolist()))
out.append("   报告期样例（前3/后3）：%s ... %s" % (
    data["报告期"].head(3).tolist(), data["报告期"].tail(3).tolist()))

data["股票代码"] = normalize_codes(data["股票代码"])
snap("2. normalize_codes 之后", data)

data["报告期"] = pd.to_datetime(data["报告期"], errors="coerce").dt.normalize()
out.append("   报告期 NaT 数量 = %d" % data["报告期"].isna().sum())
snap("3. 报告期 to_datetime 之后", data)

data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce").dt.normalize()
out.append("   首次公告日期 NaT 数量 = %d" % data["首次公告日期"].isna().sum())
snap("4. 公告日 to_datetime 之后", data)

for column in CUMULATIVE_COLUMNS:
    data[column] = pd.to_numeric(data[column], errors="coerce")
    out.append("   %s 数值化后 NaN = %d" % (column, data[column].isna().sum()))
snap("5. 累计值 to_numeric 之后", data)

data = data.dropna(subset=["股票代码", "报告期"])
snap("6. dropna(代码,报告期) 之后", data)

data = data.drop_duplicates(["股票代码", "报告期"], keep="last")
snap("7. drop_duplicates 之后", data)

out.append("   幸存代码（前 15 个）：%s" % sorted(data["股票代码"].unique())[:15])
out.append("   幸存报告期范围：%s ~ %s" % (data["报告期"].min(), data["报告期"].max()))

Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_audit3_out.txt").write_text(
    "\n".join(out), encoding="utf-8"
)
print("done", flush=True)
