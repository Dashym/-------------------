# -*- coding: utf-8 -*-
"""扫描迁移后残留的旧因子措辞，结果写入 _wordcheck_out.txt。"""
from pathlib import Path

FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
OUTPUT = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_wordcheck_out.txt")

TOKENS = ["收现质量", "经营活动现金流", "cash_conversion", "CashConversion", "现金转换"]
lines = []
for path in sorted(FACTOR_DIR.glob("*.py")):
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        hits = [token for token in TOKENS if token in line]
        if hits:
            lines.append("%s:%d | %s | %s" % (path.name, number, ",".join(hits), line.strip()))
lines.append("命中行数：%d" % len(lines))
lines.append("扫描文件数：%d" % len(list(FACTOR_DIR.glob("*.py"))))
OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print("done", flush=True)
