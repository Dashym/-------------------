# -*- coding: utf-8 -*-
"""打印 4 个脚本的关键路径/常量，写入 _const_out.txt（校验迁移后指向正确的因子文件）。"""
import importlib
import os
import sys
from pathlib import Path

FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
OUTPUT = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_const_out.txt")

os.chdir(FACTOR_DIR)
sys.path.insert(0, str(FACTOR_DIR))

NAMES = [
    "sales_receipt_ratio_improvement",
    "sales_receipt_ratio_ic",
    "sales_receipt_ratio_monotonicity",
    "sales_receipt_ratio_regression",
]

lines = []
for name in NAMES:
    module = importlib.import_module(name)
    lines.append("### %s" % name)
    for attribute in sorted(vars(module)):
        if attribute.isupper():
            value = getattr(module, attribute)
            printable = (
                isinstance(value, (str, int))
                or value is None
                or isinstance(value, os.PathLike)
                or type(value).__name__ == "Timestamp"
            )
            if printable:
                lines.append("  %s = %r" % (attribute, value))
                if isinstance(value, os.PathLike):
                    lines.append("      存在 = %s" % Path(value).exists())
lines.append("")
lines.append("因子文件存在：%s" % (FACTOR_DIR / "销售收现比改善因子_raw.csv").is_file())
lines.append("明细文件存在：%s" % (FACTOR_DIR / "销售收现比改善因子_明细.csv").is_file())
OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print("done", flush=True)
