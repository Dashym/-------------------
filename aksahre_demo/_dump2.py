# -*- coding: utf-8 -*-
"""通用片段导出：python _dump2.py <文件绝对路径> <起> <止> [输出文件]"""
import sys
from pathlib import Path

source = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
target = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_dump2_out.txt")

rows = source.read_text(encoding="utf-8").splitlines()
chunk = ["%4d| %s" % (i, row) for i, row in enumerate(rows[start - 1 : end], start)]
header = "######## %s (%d-%d of %d) ########" % (source.name, start, end, len(rows))
target.write_text("\n".join([header] + chunk), encoding="utf-8")
print("ok", flush=True)
