import csv, numpy as np, sys, os
from collections import defaultdict

HERE   = os.path.dirname(os.path.abspath(__file__))
FACTOR = os.path.join(HERE, 'profit_factor_cleaned.csv')
QUOTE  = r'c:\Users\Heyuanhang\Desktop\aksahre_demo\clean_temp\A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv'
OUT    = os.path.join(HERE, 'monotonicity_result.txt')
PNG    = os.path.join(HERE, 'monotonicity.png')
N_GROUPS = 5

lines = []
def p(s):
    lines.append(s)
    print(s, flush=True)

p('='*60)
p('盈利因子(ROA) 单调性检验 (分组收益)')
p('='*60)
p('')

p('[1/4] 读取因子...')
fb = defaultdict(dict); fc = set()
with open(FACTOR, 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        v = float(r['ROA_cleaned'])
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
        d0,c0=items[i]; d1,c1=items[i+1]
        if c0<=0: continue
        ret=(c1/c0)-1
        if ret<-0.105 or ret>0.105: continue
        fr[d0][c]=ret
del q
p('  有效收益: {:,}条 (已过滤涨跌停)'.format(sum(len(v) for v in fr.values())))

p('[3/4] 逐期分组计算收益...')
fd = sorted(fb)
group_returns = {g: [] for g in range(N_GROUPS)}
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
    fv_sorted = fv[order]
    rv_sorted = rv[order]
    for g in range(N_GROUPS):
        start = int(n * g / N_GROUPS)
        end   = int(n * (g+1) / N_GROUPS)
        if end > start:
            group_returns[g].append(rv_sorted[start:end].mean())

p('  有效截面: {}'.format(valid_dates))

if valid_dates < 2:
    p('[ERROR] 有效截面不足')
    sys.exit(1)

p('[4/4] 汇总单调性结果...')
gmean = {}
for g in range(N_GROUPS):
    arr = np.array(group_returns[g])
    gmean[g] = arr.mean()

top_bottom_spread = gmean[N_GROUPS-1] - gmean[0]

monotonic_up   = all(gmean[i] <= gmean[i+1] for i in range(N_GROUPS-1))
monotonic_down = all(gmean[i] >= gmean[i+1] for i in range(N_GROUPS-1))

p('')
p('='*60)
p('盈利因子(ROA) 单调性检验报告')
p('='*60)
p('')
p('分组数: {} 组'.format(N_GROUPS))
p('有效截面数: {} 期'.format(valid_dates))
p('')
p('-'*60)
p('{:20s} {:>12s} {:>12s}'.format('分组', '平均T+1收益', '年化'))
p('-'*60)
for g in range(N_GROUPS):
    label_map = {0: '最低', 1: '次低', 2: '中等', 3: '次高', 4: '最高'}
    label = 'G{} ({})'.format(g+1, label_map.get(g, ''))
    annual = (1+gmean[g])**250 - 1
    p('{:20s} {:>+11.4%} {:>+11.2%}'.format(label, gmean[g], annual))
p('-'*60)
spread_annual = (1+gmean[N_GROUPS-1])**250 - (1+gmean[0])**250
p('{:20s} {:>+11.4%} {:>+11.2%}'.format('多空(G{}-G1)'.format(N_GROUPS), top_bottom_spread, spread_annual))
p('')

if monotonic_up:
    jie = '严格单调递增: 因子值越大, 收益越高'
elif monotonic_down:
    jie = '严格单调递减: 因子值越大, 收益越低'
else:
    jie = '非严格单调, 需进一步分析'

p('单调性判断: {}'.format(jie))

up_count = 0; dn_count = 0
for i in range(N_GROUPS-1):
    if gmean[i+1] > gmean[i]: up_count += 1
    elif gmean[i+1] < gmean[i]: dn_count += 1
p('递增步数: {}/{}'.format(up_count, N_GROUPS-1))
p('递减步数: {}/{}'.format(dn_count, N_GROUPS-1))

pw = 0
for i in range(N_GROUPS-1):
    if gmean[i+1] > gmean[i]: pw += 1
    elif gmean[i+1] < gmean[i]: pw -= 1
if pw > 0: p('趋势: 正向 (因子值越大, 收益越高)')
elif pw < 0: p('趋势: 负向 (因子值越大, 收益越低)')
else: p('趋势: 无明显方向')

p('')
p('='*60)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

p('')
p('结果已写入: {}'.format(OUT))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    group_labels = ['G1\n(低)', 'G2', 'G3', 'G4', 'G5\n(高)']
    vals = [gmean[g] for g in range(N_GROUPS)]
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in vals]

    ax1.bar(group_labels, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax1.set_title('Group Mean Returns', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean T+1 Return')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax1.text(i, v + (0.0003 if v>=0 else -0.0008),
                 '{:+.2%}'.format(v), ha='center', fontsize=10, fontweight='bold', color=c)

    cumulative = np.cumsum(vals)
    ax2.plot(range(1, N_GROUPS+1), cumulative, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax2.set_title('Cumulative Return by Group', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Group (low -> high)')
    ax2.set_ylabel('Cumulative Return')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    for i, v in enumerate(cumulative):
        ax2.annotate('{:+.2%}'.format(v), (i+1, v),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', fontsize=9, fontweight='bold')

    spread_text = 'Spread: {:+.2%}'.format(top_bottom_spread)
    fig.suptitle('ROA Factor Monotonicity Test ({})'.format(spread_text),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PNG, dpi=150, bbox_inches='tight')
    plt.close()
    p('图表已保存: {}'.format(PNG))

except ImportError:
    p('[WARN] matplotlib 未安装, 跳过图表生成')