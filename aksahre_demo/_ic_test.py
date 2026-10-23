import csv, numpy as np
from collections import defaultdict
print('start', flush=True)
fc=set()
with open(r'c:\Users\Heyuanhang\Desktop\因子库\scale_factor_cleaned_v2.csv','r',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): fc.add(r['股票代码'])
print('factor codes:', len(fc), flush=True)

q=defaultdict(list); n=0
with open(r'c:\Users\Heyuanhang\Desktop\aksahre_demo\clean_temp\A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv','r',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        c=r['股票代码']
        if c not in fc: continue
        q[c].append((r['日期'],float(r['收盘']))); n+=1
print('read:', n, flush=True)