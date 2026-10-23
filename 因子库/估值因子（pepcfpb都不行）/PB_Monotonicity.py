# 虚拟环境: python -m venv factor_env && factor_env\Scripts\activate && pip install numpy
import csv, numpy as np, sys, math
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / 'aksahre_demo'

FACTOR = SCRIPT_DIR / 'PB_cleaned.csv'
QUOTE  = BASE_DIR / 'clean_temp' / 'A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv'
OUT    = SCRIPT_DIR / 'PB_单调性结果.txt'
N_GROUPS = 5

lines = []
def p(s):
    lines.append(s)
    print(s, flush=True)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def t_pvalue(t_stat, df):
    if df < 1:
        return float('nan')
    return 2.0 * (1.0 - norm_cdf(abs(t_stat)))

p('=' * 60)
p('PB 价值因子 单调性检验 (分组收益)')
p('=' * 60)
p('')

p('[1/4] 读取因子...')
fb = defaultdict(dict); fc = set()
with open(FACTOR, 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        v = float(r['lnPB_cleaned'])
        if np.isnan(v): continue
        c = r['股票代码']
        fb[r['可用日期']][c] = v; fc.add(c)
p('  因子: {:,}条, {}截面, {}只股票'.format(sum(len(v) for v in fb.values()), len(fb), len(fc)))

p('[2/4] 读取行情并计算T+1收益率...')
q = defaultdict(list); n = 0
with open(QUOTE, 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        c = r['股票代码']
        if c not in fc: continue
        q[c].append((r['日期'], float(r['收盘']))); n += 1
p('  行情: {:,}条'.format(n))

fr = defaultdict(dict)
for c, items in q.items():
    items.sort(key=lambda x: x[0])
    for i in range(len(items)-1):
        d0, c0 = items[i]; d1, c1 = items[i+1]
        if c0 <= 0: continue
        ret = (c1 / c0) - 1
        if ret < -0.105 or ret > 0.105: continue
        fr[d0][c] = ret
del q
p('  有效收益: {:,}条 (已过滤涨跌停)'.format(sum(len(v) for v in fr.values())))

p('[3/4] 逐期分组计算收益...')
fd = sorted(fb)
group_returns = {g: [] for g in range(N_GROUPS)}
spread_returns = []
valid_dates = 0

for d in fd:
    fdd = fb[d]
    rd  = fr.get(d, {})
    cm  = sorted(set(fdd) & set(rd))
    n   = len(cm)
    if n < N_GROUPS * 3:
        continue
    valid_dates += 1
    fv = np.array([fdd[c] for c in cm])
    rv = np.array([rd[c] for c in cm])
    order = np.argsort(fv)
    rv_sorted = rv[order]

    for g in range(N_GROUPS):
        start = int(n * g / N_GROUPS)
        end   = int(n * (g+1) / N_GROUPS)
        if end > start:
            group_returns[g].append(rv_sorted[start:end].mean())

    ret_q1 = rv_sorted[0:int(n/N_GROUPS)].mean()
    ret_q5 = rv_sorted[int(n*(N_GROUPS-1)/N_GROUPS):].mean()
    spread_returns.append(ret_q1 - ret_q5)

p('  有效截面: {}'.format(valid_dates))

if valid_dates < 2:
    p('[ERROR] 有效截面不足')
    sys.exit(1)

p('[4/4] 汇总单调性结果...')
gmean = {}; gstd = {}
for g in range(N_GROUPS):
    arr = np.array(group_returns[g])
    gmean[g] = arr.mean()
    gstd[g] = arr.std(ddof=1)

spread_arr = np.array(spread_returns)
spread_mean = spread_arr.mean()
spread_std  = spread_arr.std(ddof=1)
spread_t    = spread_mean / (spread_std / np.sqrt(valid_dates)) if spread_std > 0 else float('nan')
spread_p    = t_pvalue(spread_t, valid_dates - 1)

Q1_arr = np.array(group_returns[0])
Q5_arr = np.array(group_returns[N_GROUPS-1])
Q1_mean = Q1_arr.mean(); Q5_mean = Q5_arr.mean()
Q1_std  = Q1_arr.std(ddof=1); Q5_std = Q5_arr.std(ddof=1)

annual_q1 = (1 + Q1_mean)**250 - 1
annual_q5 = (1 + Q5_mean)**250 - 1
annual_spread = (1 + Q1_mean)**250 - (1 + Q5_mean)**250

sorted_means = sorted(gmean.items(), key=lambda x: x[1])
monotonic_up   = all(gmean[i] <= gmean[i+1] for i in range(N_GROUPS-1))
monotonic_down = all(gmean[i] >= gmean[i+1] for i in range(N_GROUPS-1))

p('')
p('=' * 60)
p('PB 价值因子 单调性检验报告')
p('=' * 60)
p('')
p('分组数: {} 组 (按PB从低到高)'.format(N_GROUPS))
p('有效截面数: {} 期'.format(valid_dates))
p('日期范围: {} ~ {}'.format(fd[0], fd[-1]))
p('')

p('-' * 75)
p('{:20s} {:>12s} {:>12s} {:>12s} {:>12s}'.format('分组', '日均收益', '标准差', '年化收益', '夏普比率'))
p('-' * 75)

for g in range(N_GROUPS):
    label_map = {0: '最低PB', 1: '次低PB', 2: '中等PB', 3: '次高PB', 4: '最高PB'}
    label = 'G{} ({})'.format(g+1, label_map.get(g, ''))
    annual = (1 + gmean[g])**250 - 1
    sharpe = (gmean[g] / gstd[g]) * np.sqrt(250) if gstd[g] > 0 else float('nan')
    p('{:20s} {:>+11.4%} {:>11.4%} {:>+11.2%} {:>12.4f}'.format(label, gmean[g], gstd[g], annual, sharpe))

p('-' * 75)
p('{:20s} {:>+11.4%} {:>11.4%} {:>+11.2%}'.format('多空(G1-G5)', spread_mean, spread_std, annual_spread))
p('')

p('-' * 75)
p('  多空组合显著性检验')
p('-' * 75)
p('  t 统计量: {:.4f}'.format(spread_t))
p('  p 值: {:.4f}'.format(spread_p))

if spread_p < 0.01:
    sig = '*** (1%水平显著, 多空收益显著不为零)'
elif spread_p < 0.05:
    sig = '** (5%水平显著)'
elif spread_p < 0.1:
    sig = '* (10%水平显著)'
else:
    sig = '(不显著, p>0.1)'
p('  显著性: {}'.format(sig))
p('  多空胜率: {:.1%} ({}/{})'.format((spread_arr > 0).mean(), (spread_arr > 0).sum(), valid_dates))
p('')

if monotonic_down:
    jie = '严格单调递减: PB越低 → 收益越高 ✓ (符合价值因子理论)'
elif monotonic_up:
    jie = '严格单调递增: PB越高 → 收益越高 ✗ (与价值因子理论相反)'
else:
    jie = '非严格单调'

p('单调性判断: {}'.format(jie))

dn_count = 0; up_count = 0
for i in range(N_GROUPS-1):
    if gmean[i] > gmean[i+1]: dn_count += 1
    elif gmean[i] < gmean[i+1]: up_count += 1
p('收益递减步数(G1>G2>...>G5): {}/{}'.format(dn_count, N_GROUPS-1))
p('收益递增步数(G1<G2<...<G5): {}/{}'.format(up_count, N_GROUPS-1))
p('')

p('各组分位收益趋势:')
for g in range(N_GROUPS-1):
    diff = gmean[g] - gmean[g+1]
    label_map = {0: '最低', 1: '次低', 2: '中等', 3: '次高', 4: '最高'}
    p('  G{}→G{}: {:+.2%}'.format(g+1, g+2, diff))
p('')
p('=' * 60)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))