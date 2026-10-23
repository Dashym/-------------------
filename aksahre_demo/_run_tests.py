# -*- coding: utf-8 -*-
"""运行销售收现比改善因子的全部单测，结果写入 _unit_all.txt（UTF-8）。"""
import io
import os
import sys
import unittest
from pathlib import Path

FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
OUTPUT = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_unit_all.txt")

os.chdir(FACTOR_DIR)
sys.path.insert(0, str(FACTOR_DIR))

suite = unittest.TestLoader().discover(
    str(FACTOR_DIR), pattern="test_sales_receipt_ratio_*.py", top_level_dir=str(FACTOR_DIR)
)
buffer = io.StringIO()
result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
OUTPUT.write_text(buffer.getvalue(), encoding="utf-8")
print(
    "tests=%d failures=%d errors=%d" % (result.testsRun, len(result.failures), len(result.errors)),
    flush=True,
)
