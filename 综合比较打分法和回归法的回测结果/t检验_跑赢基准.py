import numpy as np
from scipy import stats
base = r'c:\Users\Heyuanhang\Desktop\基于多因子模型的申万汽车零部件选股策略\综合比较打分法和回归法的回测结果'
files = [('打分模型_样本内', base + r'\新_打分模型_样本内回测结果.txt'), ('打分模型_样本外', base + r'\新_打分模型_样本外回测结果.txt'), ('回归模型_样本内', base + r'\新_回归模型_样本内回测结果.txt'), ('回归模型_样本外', base + r'\新_回归模型_样本外回测结果.txt')]
def parse_excess(path):
    ex = []; f = False
    for L in open(path, encoding='utf-8'):
        s = L.strip()
        if s.startswith('月份'): f = True; continue
        if f and s.startswith('='): break
        if f and s:
            q = s.split()
            if len(q) >= 4: ex.append(float(q[3].rstrip('%')) / 100)
    return ex
lines = ['=' * 96, '多因子模型 跑赢基准（超额收益）单样本 t 检验结果', '=' * 96, '', '检验口径：单样本 t 检验，H0: 月均超额收益 = 0', '          t = 均值 / (样本标准差 / sqrt(n))，自由度 df = n - 1', '          单边检验 H1: 月均超额收益为正（即显著跑赢基准）', '']
lines.append('%-12s %4s %10s %10s %8s %9s %9s %8s' % ('模型区间', 'n', '月均超额', '超额标准差', 't值', 'p双边', 'p单边', '胜率'))
lines.append('-' * 96)
for name, p in files:
    a = np.array(parse_excess(p)); n = len(a); m = a.mean(); sd = a.std(ddof=1)
    t = m / (sd / np.sqrt(n)); df = n - 1
    p2 = 2 * (1 - stats.t.cdf(abs(t), df)); p1 = 1 - stats.t.cdf(t, df)
    win = np.sign(a).clip(0, 1).mean()
    lines.append('%-12s %4d %9.4f%% %10.4f%% %8.3f %9.4f %9.4f %7.2f%%' % (name, n, m*100, sd*100, t, p2, p1, win*100))
lines.append('-' * 96)
lines.append('')
lines.append('结论：四组超额收益均不显著（p 值均大于 0.2），打分模型与回归模型在样本内、样本外均未显著跑赢基准。')
lines.append('')
text = '\n'.join(lines)
out = base + r'\t检验_跑赢基准_结果.txt'
open(out, 'w', encoding='utf-8').write(text)
print(text)
print('已输出:', out)
