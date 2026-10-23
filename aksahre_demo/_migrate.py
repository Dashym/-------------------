# -*- coding: utf-8 -*-
"""把「收现质量改善因子」的三大检验脚本与单测（已跑通的模板）迁移为「销售收现比改善因子」版本。

只做机械替换：模块名 cash_conversion_* → sales_receipt_ratio_*，
因子名 收现质量改善因子/收现质量 → 销售收现比改善因子/销售收现比。
"""
from pathlib import Path

LIB = Path(r"C:\Users\Heyuanhang\Desktop\因子库")
SRC = LIB / "收现质量改善因子"
DST = LIB / "销售收现比改善因子"

FILES = [
    "cash_conversion_ic.py",
    "cash_conversion_monotonicity.py",
    "cash_conversion_regression.py",
    "test_cash_conversion_ic.py",
    "test_cash_conversion_monotonicity.py",
    "test_cash_conversion_regression.py",
]

REPLACEMENTS = [
    ("cash_conversion_monotonicity", "sales_receipt_ratio_monotonicity"),
    ("cash_conversion_regression", "sales_receipt_ratio_regression"),
    ("cash_conversion_ic", "sales_receipt_ratio_ic"),
    ("cash_conversion", "sales_receipt_ratio"),
    ("收现质量改善因子", "销售收现比改善因子"),
    ("收现质量", "销售收现比"),
]

report = []
for name in FILES:
    text = (SRC / name).read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    target = DST / name.replace("cash_conversion", "sales_receipt_ratio")
    target.write_text(text, encoding="utf-8")
    hits = [old for old, _ in REPLACEMENTS if old in text]
    report.append("%s -> %s | 残留=%s" % (name, target.name, hits or "无"))

print("\n".join(report), flush=True)
