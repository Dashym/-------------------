# -*- coding: utf-8 -*-
"""核对真实数据前置条件，结果写入 _data_check_out.txt。"""
from pathlib import Path

import pandas as pd

BASE = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
CACHE = BASE / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
POOL = BASE / "汽车零部件_申万二级成分股.csv"
QUOTES = BASE / "clean_temp" / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"

lines = []
for path in (CACHE, POOL, QUOTES):
    lines.append("%s 存在=%s 大小=%.1f MB" % (path.name, path.is_file(), path.stat().st_size / 1e6 if path.is_file() else 0))

cache = pd.read_csv(CACHE, dtype={"股票代码": str}, encoding="utf-8-sig")
lines.append("")
lines.append("爬虫缓存：%d 行，%d 只股票，列=%s" % (len(cache), cache["股票代码"].nunique(), list(cache.columns)))
lines.append("报告期范围：%s ~ %s" % (cache["报告期"].min(), cache["报告期"].max()))
lines.append("公告日期空值：%d" % cache["首次公告日期"].isna().sum())

pool = pd.read_csv(POOL, dtype=str, encoding="utf-8-sig")
lines.append("")
lines.append("股票池：%d 行，列=%s" % (len(pool), list(pool.columns)))
lines.append("股票池前 3 行：%s" % pool.head(3).to_dict("records"))

quotes = pd.read_csv(QUOTES, usecols=["日期"], encoding="utf-8-sig")
dates = pd.to_datetime(quotes["日期"], errors="coerce")
lines.append("")
lines.append("后复权行情：%d 行，日期 %s ~ %s" % (len(quotes), dates.min(), dates.max()))
November = sorted({value for value in dates.dt.strftime("%Y-%m-%d") if value.startswith("2024-11")})
lines.append("2024-11 交易日（前 5）：%s" % November[:5])
lines.append("2024-10-31 是否交易日：%s" % (dates.dt.strftime("%Y-%m-%d") == "2024-10-31").any())
lines.append("2024-11-01 是否交易日：%s" % (dates.dt.strftime("%Y-%m-%d") == "2024-11-01").any())

Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_data_check_out.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("done", flush=True)
