# -*- coding: utf-8 -*-
import os
import sys

BASE = r"C:\Users\Heyuanhang\Desktop\aksahre_demo"
ROOT = r"C:\Users\Heyuanhang\Desktop\因子库"
out = []

files = [
    os.path.join(BASE, "raw_data", "A股汽车零部件_季度财报_原始_2019-2026.csv"),
    os.path.join(BASE, "raw_data", "A股汽车零部件_季度财报_原始_2019-2026_TEST.csv"),
    os.path.join(BASE, "A股日线_20190101-20260331_后复权.csv"),
    os.path.join(BASE, "汽车零部件_申万二级成分股.csv"),
    os.path.join(BASE, "processed_data", "财报_清洗完成.csv"),
]
for fp in files:
    out.append("=" * 100)
    out.append("FILE: %s  (%d bytes)" % (fp, os.path.getsize(fp) if os.path.isfile(fp) else -1))
    if os.path.isfile(fp):
        with open(fp, "r", encoding="utf-8-sig", errors="replace") as fh:
            for i in range(4):
                line = fh.readline()
                if not line:
                    break
                out.append("L%d: %s" % (i, line.rstrip("\n")[:1500]))

out.append("=" * 100)
out.append("=== ic_env dir ===")
env = os.path.join(BASE, "ic_env")
for dirpath, dirnames, filenames in os.walk(env):
    rel = os.path.relpath(dirpath, env)
    for f in filenames[:40]:
        out.append(os.path.join(rel, f))

with open(os.path.join(BASE, "_probe1.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("probe1 done")
sys.stdout.flush()
