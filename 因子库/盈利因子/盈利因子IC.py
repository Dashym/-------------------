import csv, numpy as np, sys, os
from collections import defaultdict

HERE   = os.path.dirname(os.path.abspath(__file__))
FACTOR = os.path.join(HERE, 'profit_factor_cleaned.csv')
QUOTE  = r'c:\Users\Heyuanhang\Desktop\aksahre_demo\clean_temp\A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv'
OUT    = os.path.join(HERE, 'ic_result.txt')

lines = []
def p(s):
    lines.append(s)
    print(s, flush=True)

p('='*60)
p('盈利因子(ROA) IC 检验')
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
p('')

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
p('')

p('[3/4] 逐期计算 Rank IC / Pearson IC...')
ic=[]; fd=sorted(fb)
for di,d in enumerate(fd):
    fdd=fb[d]; rd=fr.get(d,{})
    cm=sorted(set(fdd)&set(rd)); n=len(cm)
    if n<10: continue
    fv=np.array([fdd[c] for c in cm])
    rv=np.array([rd[c] for c in cm])
    frr=np.argsort(np.argsort(fv)).astype(float)+1
    rrr=np.argsort(np.argsort(rv)).astype(float)+1
    fm=frr.mean(); rmm=rrr.mean()
    fd_=frr-fm; rd_=rrr-rmm
    dnr=np.sqrt((fd_**2).sum()*(rd_**2).sum())
    ri=(fd_*rd_).sum()/dnr if dnr>0 else np.nan
    fm2=fv.mean(); rm2=rv.mean()
    fd2=fv-fm2; rd2=rv-rm2
    dnp=np.sqrt((fd2**2).sum()*(rd2**2).sum())
    pi=(fd2*rd2).sum()/dnp if dnp>0 else np.nan
    ic.append((d,n,ri,pi))
    if (di+1)%50==0: p('  进度: {}/{} 截面'.format(di+1, len(fd)))
p('  有效截面: {}'.format(len(ic)))
p('')

if not ic:
    p('[ERROR] 无有效IC数据')
    sys.exit(1)

da=np.array([r[0] for r in ic]); na=np.array([r[1] for r in ic])
ra=np.array([r[2] for r in ic]); pa=np.array([r[3] for r in ic])
vr=~np.isnan(ra); rc=ra[vr]; vp=~np.isnan(pa); pc=pa[vp]
rm=rc.mean(); rs=rc.std(ddof=1); pm=pc.mean(); ps=pc.std(ddof=1)
rir=rm/rs if rs>0 else float('nan')
pir=pm/ps if ps>0 else float('nan')
tr=rm/(rs/np.sqrt(vr.sum())) if vr.sum()>1 else float('nan')
tp=pm/(ps/np.sqrt(vp.sum())) if vp.sum()>1 else float('nan')
cum=np.nancumsum(ra)

p('='*60)
p('IC 检验结果')
p('='*60)
p('')
p('  截面数: {}'.format(len(ic)))
p('  平均股票数/截面: {:.1f} [范围 {:.0f}~{:.0f}]'.format(na.mean(), na.min(), na.max()))
p('  日期范围: {} ~ {}'.format(da[0], da[-1]))
p('')
p('  {:16s} {:>10s} {:>12s}'.format('', 'Rank IC', 'Pearson IC'))
p('  ' + '-'*40)
p('  {:16s} {:>10.4f} {:>12.4f}'.format('IC均值', rm, pm))
p('  {:16s} {:>10.4f} {:>12.4f}'.format('IC标准差', rs, ps))
p('  {:16s} {:>10.4f} {:>12.4f}'.format('ICIR', rir, pir))
p('  {:16s} {:>9.2%} {:>11.2%}'.format('IC>0占比', (rc>0).mean(), (pc>0).mean()))
p('  {:16s} {:>10.4f} {:>12s}'.format('abs(IC)均值', np.abs(rc).mean(), '-'))
p('  {:16s} {:>10.4f} {:>12.4f}'.format('t-stat', tr, tp))
p('')
p('  累计Rank IC: {:.4f} -> {:.4f}'.format(cum[0], cum[-1]))
p('    (范围: [{:.4f}, {:.4f}])'.format(np.nanmin(cum), np.nanmax(cum)))
p('')

ai=abs(rm)
if ai>=0.05 and (np.isnan(rir) or abs(rir)>0.5):
    v='强势有效因子, 建议进入分层回测'
elif ai>=0.03 and (np.isnan(rir) or abs(rir)>0.3):
    v='有效因子, 可考虑进入分层回测'
elif ai>=0.02:
    v='弱有效因子, 建议结合其他因子或优化后重试'
else:
    v='因子无效, 建议检查数据或放弃'
p('  综合判断: {}'.format(v))
p('    (|Rank IC|={:.4f}, ICIR={:.2f})'.format(ai, rir))
p('')
p('='*60)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))