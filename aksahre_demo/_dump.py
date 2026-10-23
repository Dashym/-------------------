# -*- coding: utf-8 -*-
"""把 因子库 里的模板脚本按行号 dump 成工作区内的文本，便于直接阅读。"""
from pathlib import Path

RANGES = [
    (Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子\sales_receipt_ratio_improvement.py"), 108, 184),
]
out = []
for path, start, end in RANGES:
    lines = path.read_text(encoding="utf-8").splitlines()
    last = min(end, len(lines))
    out.append("################ %s (%d-%d of %d) ################" % (path.name, start, last, len(lines)))
    for number in range(start, last + 1):
        out.append("%4d| %s" % (number, lines[number - 1]))
    out.append("")
Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_dump_out.txt").write_text("\n".join(out), encoding="utf-8")
print("dump ok", flush=True)
