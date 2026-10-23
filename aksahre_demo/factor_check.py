"""
因子质量体检脚本 - 纯 numpy 实现，不依赖 pandas/scipy
"""
import csv
import numpy as np
from collections import defaultdict

CSV_PATH = r'c:\Users\Heyuanhang\Desktop\因子库\scale_factor_cleaned_v2.csv'

def calc_skew(vals):
    """手动计算偏度"""
    v = np.asarray(vals)
    v = v - v.mean()
    m2 = (v ** 2).mean()
    m3 = (v ** 3).mean()
    if m2 < 1e-15:
        return 0.0
    return m3 / (m2 ** 1.5)

def calc_kurt(vals):
    """手动计算超额峰度"""
    v = np.asarray(vals)
    v = v - v.mean()
    m2 = (v ** 2).mean()
    m4 = (v ** 4).mean()
    if m2 < 1e-15:
        return 0.0
    return m4 / (m2 ** 2) - 3.0

# ---- 读取 CSV ----
rows = []
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append((
            r['可用日期'],
            r['股票代码'],
            float(r['ln流通市值']),
            float(r['ln流通市值_cleaned']),
        ))

by_date = defaultdict(list)
for d, c, raw, cln in rows:
    by_date[d].append((c, raw, cln))

by_stock = defaultdict(list)
for d, c, raw, cln in rows:
    by_stock[c].append((d, raw, cln))

dates = sorted(by_date.keys())
stocks = sorted(by_stock.keys())
print(f'数据量: {len(rows):,}行  {len(dates)}截面  {len(stocks)}只股票')
print()

# ============================================================
# 1. 缺失率
# ============================================================
missing_rates = []
for d in dates:
    vals = [r[2] for r in by_date[d]]
    mr = sum(1 for v in vals if np.isnan(v)) / len(vals)
    missing_rates.append(mr)
missing_rates = np.array(missing_rates)
print('=' * 52)
print('1. 缺失率')
print('=' * 52)
print(f'  全样本平均缺失率: {missing_rates.mean():.6%}')
print(f'  每期缺失率最大值: {missing_rates.max():.6%}')
print(f'  每期缺失率标准差: {missing_rates.std():.6%}')
print()

# ============================================================
# 2. 截面分布
# ============================================================
means, stds, skews, kurts = [], [], [], []
for d in dates:
    vals = np.array([r[2] for r in by_date[d]])
    vals = vals[~np.isnan(vals)]
    if len(vals) < 8:
        continue
    means.append(vals.mean())
    stds.append(vals.std(ddof=0))
    skews.append(calc_skew(vals))
    kurts.append(calc_kurt(vals))

print('=' * 52)
print('2. 截面分布 (各期汇总)')
print('=' * 52)
for label, arr in [('均值', means), ('标准差', stds), ('偏度', skews), ('峰度', kurts)]:
    a = np.array(arr)
    print(f'  {label:>6s}: 全样本均值={a.mean():.6f}  范围=[{a.min():.6f}, {a.max():.6f}]')
print(f'  有效截面数: {len(means)}')
print()

# ============================================================
# 3. 因子自相关
# ============================================================
lags = [1, 5, 10, 20]
ac_all = {lag: [] for lag in lags}
stock_n_obs = []  # 每只股票的有效观测数

for stock in stocks:
    items = sorted(by_stock[stock], key=lambda x: x[0])
    series = np.array([r[2] for r in items])
    series = series[~np.isnan(series)]
    stock_n_obs.append(len(series))
    # 对每个 lag 单独判断: 至少需要 lag+5 个点
    for lag in lags:
        if len(series) <= lag + 4:
            continue
        s1 = series[:-lag]
        s2 = series[lag:]
        s1m = s1 - s1.mean()
        s2m = s2 - s2.mean()
        denom = np.sqrt((s1m ** 2).sum() * (s2m ** 2).sum())
        if denom == 0:
            continue
        ac = (s1m * s2m).sum() / denom
        if not np.isnan(ac):
            ac_all[lag].append(ac)

stock_n_obs = np.array(stock_n_obs)
print('=' * 52)
print('3. 因子自相关 (全样本均值)')
print('=' * 52)
print(f'  每只股票观测数: mean={stock_n_obs.mean():.1f}  '
      f'median={np.median(stock_n_obs):.0f}  '
      f'范围=[{stock_n_obs.min():.0f}, {stock_n_obs.max():.0f}]')
for lag in lags:
    arr = ac_all[lag]
    m = np.mean(arr) if arr else np.nan
    print(f'  lag{lag:>2}: 均值={m:.6f}  有效股票数={len(arr)}')

lag1_mean = np.mean(ac_all[1]) if ac_all[1] else np.nan
if not np.isnan(lag1_mean) and 0 < lag1_mean < 1:
    hl = np.log(0.5) / np.log(lag1_mean)
    print(f'  半衰期估算: {hl:.2f} 天')
else:
    print('  半衰期估算: N/A')
print()

# ============================================================
# 4. 分组区分度 (原始ln流通市值 max组/min组)
# ============================================================
ratios = []
for d in dates:
    items = [(r[1], r[2]) for r in by_date[d]]
    cleaned_vals = np.array([x[1] for x in items])
    raw_vals = np.array([x[0] for x in items])
    valid = ~np.isnan(cleaned_vals) & ~np.isnan(raw_vals)
    cleaned_vals = cleaned_vals[valid]
    raw_vals = raw_vals[valid]
    n = len(cleaned_vals)
    if n < 10:
        continue

    try:
        boundaries = np.percentile(cleaned_vals, np.linspace(0, 100, 11))
        boundaries = np.unique(boundaries)
        if len(boundaries) < 5:
            continue
        labels = np.digitize(cleaned_vals, boundaries[1:-1], right=True)
    except (ValueError, IndexError):
        continue

    group_means = []
    for g in set(labels):
        mask = labels == g
        if mask.sum() > 0:
            group_means.append(raw_vals[mask].mean())
    if len(group_means) < 4:
        continue
    gm = np.array(group_means)
    mx, mn = gm.max(), gm.min()
    if mn <= 0:
        continue
    ratios.append(float(mx / mn))

print('=' * 52)
print('4. 分组区分度 (原始ln流通市值, max组/min组)')
print('=' * 52)
print(f'  全样本平均比值: {np.mean(ratios):.4f}')
print(f'  比值标准差:     {np.std(ratios):.4f}')
print(f'  有效截面数:     {len(ratios)}')