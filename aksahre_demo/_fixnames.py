# -*- coding: utf-8 -*-
"""清理迁移后残留的英文标识 CashConversion → SalesReceiptRatio。"""
from pathlib import Path

DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
changed = []
for path in DIR.glob("*sales_receipt_ratio*.py"):
    text = path.read_text(encoding="utf-8")
    if "CashConversion" in text:
        path.write_text(text.replace("CashConversion", "SalesReceiptRatio"), encoding="utf-8")
        changed.append(path.name)
print("已修正：", changed or "无", flush=True)
