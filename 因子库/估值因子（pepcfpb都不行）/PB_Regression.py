# 虚拟环境: python -m venv factor_env && factor_env\Scripts\activate && pip install numpy
import csv, numpy as np, sys, math
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / 'aksahre_demo'

FACTOR = SCRIPT_DIR / 'PB_cleaned.csv'
QUOTE  = BASE_DIR / 'clean_temp' / 'A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv'
OUT    = SCRIPT_DIR / 'PB_回归结果.txt'

lines = []
def p(s):
    lines.append(s)
    print(s, flush=True)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def t_pvalue(t_stat, df):
    if df < 1:
        return float('nan')
    p = 2.0 * (1.0 - norm_cdf(abs(t_stat)))
    return p

p('PB 价值因子 Fama-MacBeth 回归检验')
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

p('[3/4] 逐期 OLS 回归...')
fd = sorted(fb)
betas = []; alphas = []; r2s = []; nobs_list = []
for di, d in enumerate(fd):
    fdd = fb[d]; rd = fr.get(d, {})
    cm = sorted(set(fdd) & set(rd))
    n = len(cm)
    if n < 10:
        continue
    y = np.array([rd[c] for c in cm])
    x = np.array([fdd[c] for c in cm])
    X = np.column_stack([np.ones(n), x])
    try:
        coeff, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        continue
    alpha = coeff[0]
    beta  = coeff[1]
    alphas.append(alpha)
    betas.append(beta)
    nobs_list.append(n)
    y_pred = X @ coeff
    ss_res = ((y - y_pred)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2s.append(r2)
    if (di+1) % 50 == 0:
        p('  进度: {}/{} 截面'.format(di+1, len(fd)))
p('  有效截面: {}'.format(len(betas)))

if len(betas) < 2:
    p('[ERROR] 有效截面不足')
    sys.exit(1)

p('[4/4] 汇总 Fama-MacBeth 结果...')
N = len(betas)
total_obs = sum(nobs_list)

alpha_arr = np.array(alphas)
beta_arr  = np.array(betas)
r2_arr    = np.array(r2s)
nobs_arr  = np.array(nobs_list)

alpha_mean = alpha_arr.mean()
beta_mean  = beta_arr.mean()
r2_mean    = r2_arr.mean()

alpha_se = alpha_arr.std(ddof=1) / np.sqrt(N)
beta_se  = beta_arr.std(ddof=1) / np.sqrt(N)

alpha_t = alpha_mean / alpha_se if alpha_se > 0 else float('nan')
beta_t  = beta_mean / beta_se if beta_se > 0 else float('nan')

alpha_p = t_pvalue(alpha_t, N-1)
beta_p  = t_pvalue(beta_t, N-1)

def star(pv):
    if pv < 0.01: return '***'
    elif pv < 0.05: return '**'
    elif pv < 0.1: return '*'
    else: return ''

alpha_s = star(alpha_p)
beta_s  = star(beta_p)

p('')
p('=' * 60)
p('PB 价值因子 Fama-MacBeth 回归检验报告')
p('=' * 60)
p('')
p('样本区间: {} 至 {}'.format(fd[0], fd[-1]))
p('总截面数: {} 期'.format(N))
p('总观测数: {}'.format(total_obs))
p('平均 R-squared: {:.4f}'.format(r2_mean))
p('-' * 70)
p('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('变量', '系数', '标准误', 't统计量', 'p值'))
p('-' * 70)
p('{:20s} {:>10.6f} {:>10.6f} {:>10.4f} {:>10.4f} {}'.format('const', alpha_mean, alpha_se, alpha_t, alpha_p, alpha_s))
p('{:20s} {:>10.6f} {:>10.6f} {:>10.4f} {:>10.4f} {}'.format('lnPB_cleaned', beta_mean, beta_se, beta_t, beta_p, beta_s))
p('-' * 70)
p('显著性标记: *** p<0.01, ** p<0.05, * p<0.1')
p('')
p('')
p('【PB因子解读】')
p('  系数: {:.6f}'.format(beta_mean))
p('  t统计量: {:.4f}'.format(beta_t))
p('  p值: {:.4f}'.format(beta_p))
if beta_p < 0.01:
    jie = '系数在1%水平上显著，PB因子对收益率有显著的解释能力。'
elif beta_p < 0.05:
    jie = '系数在5%水平上显著，PB因子对收益率有一定的解释能力。'
elif beta_p < 0.1:
    jie = '系数在10%水平上显著，PB因子解释能力较弱。'
else:
    jie = '系数不显著（p>0.1），在汽车零部件行业内，PB因子对截面收益率缺乏显著的解释力。'
p('  -> {}'.format(jie))
p('')
p('=' * 60)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))