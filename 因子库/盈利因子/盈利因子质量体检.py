# -*- coding: utf-8 -*-
"""
盈利因子(ROA) 因子质量前置体检
================================
对已清洗的 ROA 因子 (profit_factor_cleaned.csv) 进行 4 项质量检验：
  1. 缺失率检查
  2. 截面分布检查 (均值/标准差/偏度/峰度)
  3. 因子自相关与衰减 (lag 1/5/10/20 + 半衰期)
  4. 分组区分度 (按 cleaned 分组, 比较原始 ROA 高低组差异)

纯 numpy 实现，不依赖 pandas/scipy，与目录内其余检验脚本风格一致。

用法:
    python 盈利因子质量体检.py
输出:
    quality_result.txt
"""
import csv
import numpy as np
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FACTOR = os.path.join(HERE, 'profit_factor_cleaned.csv')
OUT = os.path.join(HERE, 'quality_result.txt')

FACTOR_COL = 'ROA_cleaned'
RAW_COL = 'ROA'
N_GROUPS = 5

lines = []


def p(s):
    lines.append(s)
    print(s, flush=True)


def calc_skew(vals):
    """手动计算偏度 (无偏近似)"""
    v = np.asarray(vals, dtype=float)
    v = v - v.mean()
    m2 = (v ** 2).mean()
    m3 = (v ** 3).mean()
    if m2 < 1e-15:
        return 0.0
    return m3 / (m2 ** 1.5)


def calc_kurt(vals):
    """手动计算超额峰度"""
    v = np.asarray(vals, dtype=float)
    v = v - v.mean()
    m2 = (v ** 2).mean()
    m4 = (v ** 4).mean()
    if m2 < 1e-15:
        return 0.0
    return m4 / (m2 ** 2) - 3.0


p('=' * 60)
p('盈利因子(ROA) 因子质量前置体检')
p('=' * 60)
p('')

# ---- 读取因子 (双层索引: 可用日期, 股票代码) ----
by_date = defaultdict(list)
by_stock = defaultdict(list)
with open(FACTOR, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        d = r['可用日期']
        c = r['股票代码']
        raw = float(r[RAW_COL])
        cln = float(r[FACTOR_COL])
        by_date[d].append((c, raw, cln))
        by_stock[c].append((d, raw, cln))

dates = sorted(by_date.keys())
stocks = sorted(by_stock.keys())
p('因子文件: {}'.format(os.path.basename(FACTOR)))
p('数据量: {:,} 行, {} 截面, {} 只股票'.format(
    sum(len(v) for v in by_date.values()), len(dates), len(stocks)))
p('')

# ===================================================================
# 检验1: 缺失率检查
# ===================================================================
missing_rates = []
for d in dates:
    vals = [r[2] for r in by_date[d]]
    mr = sum(1 for v in vals if np.isnan(v)) / len(vals)
    missing_rates.append(mr)
missing_rates = np.array(missing_rates)

p('=' * 60)
p('检验1: 缺失率')
p('=' * 60)
p('  全样本平均缺失率: {:.4%}'.format(missing_rates.mean()))
p('  每期缺失率最大值: {:.4%}'.format(missing_rates.max()))
p('  每期缺失率标准差: {:.4%}'.format(missing_rates.std()))
if missing_rates.mean() > 0.15:
    p('  判断: [WARN] 平均缺失率超过 15%')
else:
    p('  判断: [PASS] 平均缺失率在可接受范围')
p('')

# ===================================================================
# 检验2: 截面分布检查
# ===================================================================
means, stds, skews, kurts = [], [], [], []
for d in dates:
    vals = np.array([r[2] for r in by_date[d]], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 8:
        continue
    means.append(vals.mean())
    stds.append(vals.std(ddof=0))
    skews.append(calc_skew(vals))
    kurts.append(calc_kurt(vals))

means = np.array(means)
stds = np.array(stds)
skews = np.array(skews)
kurts = np.array(kurts)

p('=' * 60)
p('检验2: 截面分布 (ROA_cleaned)')
p('=' * 60)
p('  均值: 全样本均值={:.4f}, 范围=[{:.4f}, {:.4f}]'.format(
    means.mean(), means.min(), means.max()))
p('  标准差: 全样本均值={:.4f}, 范围=[{:.4f}, {:.4f}]'.format(
    stds.mean(), stds.min(), stds.max()))
p('  偏度: 全样本均值={:.4f}, 范围=[{:.4f}, {:.4f}]'.format(
    skews.mean(), skews.min(), skews.max()))
p('  峰度(超额): 全样本均值={:.4f}, 范围=[{:.4f}, {:.4f}]'.format(
    kurts.mean(), kurts.min(), kurts.max()))
p('  有效截面数: {}'.format(len(means)))
if abs(skews.mean()) > 2:
    p('  判断: [WARN] 平均偏度绝对值超过 2, 分布高度偏斜')
else:
    p('  判断: [PASS] 平均偏度在可接受范围')
p('')

# ===================================================================
# 检验3: 因子自相关与衰减
# ===================================================================
lags = [1, 5, 10, 20]
ac_all = {lag: [] for lag in lags}
stock_n_obs = []

for stock in stocks:
    items = sorted(by_stock[stock], key=lambda x: x[0])
    series = np.array([r[2] for r in items], dtype=float)
    series = series[~np.isnan(series)]
    stock_n_obs.append(len(series))
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
p('=' * 60)
p('检验3: 因子自相关与衰减')
p('=' * 60)
p('  每只股票观测数: mean={:.1f}, median={:.0f}, 范围=[{:.0f}, {:.0f}]'.format(
    stock_n_obs.mean(), np.median(stock_n_obs), stock_n_obs.min(), stock_n_obs.max()))
for lag in lags:
    arr = ac_all[lag]
    m = np.mean(arr) if arr else np.nan
    p('  lag{:>2}: 均值={:.4f}, 有效股票数={}'.format(lag, m, len(arr)))

lag1_mean = np.mean(ac_all[1]) if ac_all[1] else np.nan
if not np.isnan(lag1_mean) and 0 < lag1_mean < 1:
    hl = np.log(0.5) / np.log(lag1_mean)
    p('  半衰期估算: {:.2f} 天'.format(hl))
else:
    p('  半衰期估算: N/A')

if np.isnan(lag1_mean):
    p('  判断: [FAIL] 无法计算自相关')
elif lag1_mean < 0.6:
    p('  判断: [WARN] lag1 自相关低于 0.6, 因子变化较快')
else:
    p('  判断: [PASS] lag1 自相关稳定')
p('')

# ===================================================================
# 检验4: 分组区分度 (按 cleaned 分组, 比较原始 ROA 高低组差异)
# ===================================================================
spreads = []
n_valid = 0
for d in dates:
    items = [(r[0], r[1], r[2]) for r in by_date[d]]  # (code, raw, cleaned)
    cleaned = np.array([x[2] for x in items], dtype=float)
    raw = np.array([x[1] for x in items], dtype=float)
    valid = ~np.isnan(cleaned) & ~np.isnan(raw)
    cleaned = cleaned[valid]
    raw = raw[valid]
    n = len(cleaned)
    if n < 10:
        continue
    order = np.argsort(cleaned)
    raw_s = raw[order]
    # 分 5 组, 取最高组(G5)与最低组(G1)的原始 ROA 均值
    g1 = raw_s[:int(n * 1 / N_GROUPS)]
    g5 = raw_s[int(n * (N_GROUPS - 1) / N_GROUPS):]
    if len(g1) == 0 or len(g5) == 0:
        continue
    spreads.append(float(g5.mean() - g1.mean()))
    n_valid += 1

spreads = np.array(spreads)
p('=' * 60)
p('检验4: 分组区分度 (原始 ROA 高低组差异)')
p('=' * 60)
p('  高组(G5) - 低组(G1) 原始ROA均值差: {:.4f}'.format(spreads.mean()))
if len(spreads) > 1:
    p('  差值的标准差: {:.4f}'.format(spreads.std(ddof=1)))
p('  有效截面数: {}'.format(n_valid))
p('  差值>0 占比: {:.2%}'.format((spreads > 0).mean()))
if spreads.mean() < 0.01:
    p('  判断: [WARN] 高低组原始 ROA 差异过小 (<1%), 因子区分度弱')
else:
    p('  判断: [PASS] 高低组原始 ROA 有明显差异')
p('')

p('=' * 60)
p('盈利因子(ROA) 质量体检完成')
p('=' * 60)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

