# -*- coding: utf-8 -*-
"""扫描 sales_receipt_ratio_* 脚本的措辞残留与函数定义，并查看爬虫进度。"""
from pathlib import Path

DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
PHRASES = ["公告日之后第一个交易日", "财报可用日期", "收现质量", "经营活动现金流", "cash_conversion"]
OUT = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_inspect_out.txt")

lines = []
for name in sorted(p.name for p in DIR.glob("sales_receipt_ratio_*.py")) + sorted(
    p.name for p in DIR.glob("test_sales_receipt_ratio_*.py")
):
    text = (DIR / name).read_text(encoding="utf-8")
    lines.append("===== %s (%d 行) =====" % (name, len(text.splitlines())))
    for phrase in PHRASES:
        hits = [i for i, row in enumerate(text.splitlines(), 1) if phrase in row]
        if hits:
            lines.append("  残留 %s @ 行 %s" % (phrase, hits))
    for i, row in enumerate(text.splitlines(), 1):
        if row.startswith("def ") or row.startswith("    def "):
            lines.append("  %4d %s" % (i, row.strip()))
    lines.append("")

csv = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\raw_data\汽车零部件_销售收现与营业收入_原始.csv")
import csv as csv_module

with csv.open(encoding="utf-8-sig") as handle:
    rows = list(csv_module.DictReader(handle))
stocks = {row["股票代码"] for row in rows}
lines.append("爬虫缓存：%d 行，%d 只股票" % (len(rows), len(stocks)))
log = DIR / "_crawler_run2.log"
if log.is_file():
    tail = log.read_text(encoding="gbk", errors="replace").strip().splitlines()
    lines.append("日志尾部：" + (tail[-1] if tail else "(空)"))

OUT.write_text("\n".join(lines), encoding="utf-8")
print("ok", flush=True)
