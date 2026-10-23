# -*- coding: utf-8 -*-
"""诊断：爬虫日志尾部、流水线状态、爬虫脚本写入方式，汇总到 _diag_out.txt。"""
import csv
import re
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
CRAWLER = DEMO / "sales_receipt_cash_flow_crawler.py"

lines = []


def add(title, body):
    lines.append("######## %s" % title)
    lines.extend(body)
    lines.append("")


add("流水线状态（全）", (DEMO / "_pipeline_status.txt").read_text(encoding="utf-8").splitlines())

for name in ("_crawler_run2.log", "_crawler_run2.err"):
    path = DEMO / name
    if path.is_file():
        text = path.read_text(encoding="gbk", errors="replace").strip().splitlines()
        add("%s（最后 12 行 / 共 %d 行）" % (name, len(text)), text[-12:])
    else:
        add("%s" % name, ["（不存在）"])

if CACHE.is_file():
    with CACHE.open(encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    with_format = sum(1 for row in raw if "00:00:00" in (row["报告期"] or ""))
    add(
        "缓存格式统计",
        [
            "总行数=%d，带时间戳的报告期=%d，纯日期=%d"
            % (len(raw), with_format, len(raw) - with_format),
            "股票数（去重）=%d" % len({row["股票代码"] for row in raw}),
        ],
    )

if CRAWLER.is_file():
    text = CRAWLER.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [
        "%d | %s" % (number, line.rstrip())
        for number, line in enumerate(text, start=1)
        if re.search(r"to_csv|open\(|mode=|MODE|resume|append|__main__|sleep|for ", line)
    ]
    add("爬虫脚本写入/流程相关行（共 %d 行）" % len(hits), hits[:60])

(DEMO / "_diag_out.txt").write_text("\n".join(lines), encoding="utf-8")
print("done", flush=True)
