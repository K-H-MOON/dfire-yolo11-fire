import os, glob, json, time, unicodedata, subprocess, sys
import numpy as np
from PIL import Image
from google.colab import files, drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'
STM  = f'{SRC}/steam'
ROUT = f'{SRC}/runs_s1'
EOUT = f'{SRC}/eval_s1'

SPLITS = [1, 2, 3, 4, 5]
CONFS  = [0.03, 0.10, 0.25, 0.40]
BASE   = 0.10
CMIN   = 0.01
CURVE  = [round(x, 3) for x in np.arange(0.01, 0.96, 0.01)]
MINSRC = 10
BATCH  = 16
PASS   = 1.0

ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}
WANT_SRC = {'s2': 341, 'ft': 21, 'j01': 18, 'm3': 48, 'p2': 14, '07': 37, 'kt': 28,
            'kfire03': 25, 'q1': 24, 'j04': 19, '05': 10, 'j12': 6, 'kfire02': 17,
            '04': 23}
WANT_E = {1: 83, 2: 412, 3: 152, 4: 121, 5: 104}
D_DIRS = ['steam_near', 'steam_far', 'steam_in']
SIDE   = ['smoke_faint', 'smoke_reheat']

drive.mount('/content/drive')
os.makedirs(EOUT, exist_ok=True)

print('=' * 78)
print('평가 2판 — 1층 원문을 줄마다 대조해 고침')
print('=' * 78)
print('  1판에서 어긋난 일곱을 고쳤음. 원문을 인용해 함께 찍음')
print()
print('  [태그]   원문 `학습에 쓰는 것 smoke 만 / 평가에만 쓰는 것 smoke_faint ·')
print('           smoke_reheat (태그를 유지해 따로 집계)`')
print('           -> E 는 **smoke 만**. faint·reheat 는 별도 표로 따로 집계')
print('  [지표]   원문 `E·D 는 출처별 macro 평균으로 냄. 하한 10장을 못 넘는 출처는 제외함.`')
print('           원문 `그래서 출처별 macro 를 주 지표로 두는 것임.`')
print('           -> **macro 가 주 지표**. micro 는 곁들여 적음')
print('  [폭]     원문 `평균·중앙값·최소값을 쓰지 않음. 폭이 애초에 재려던 것이기 때문임.`')
print('           -> 다섯 분할은 **폭(최소~최대)** 을 주 결과로 냄')
print('  [구간]   원문 `주요 비율에 95% 신뢰구간을 냄. 다만 액면 그대로 받아들이면 안 됨`')
print('           -> 붙임. 단서도 함께 찍음')
print('  [성립선] 원문 `판별비 E/D 가 **1 이상**이면 성립, 미만이면 미달. 두 단계뿐임.`')
print('           -> 1 이상 (1판은 초과로 짰음)')
print('  [j12]    원문 `그 분할은 평가 출처가 넷이 되는 것으로 보고함.`')
print('           -> 1번 분할은 **평가 출처 넷**으로 적음')
print('  [병기]   원문 `판정문에 E·D 를 반드시 병기함.`')
print('           -> 판별비만으로 성립을 말하지 않음')
print('=' * 78)


def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(path, size=8):
    g = np.asarray(Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS),
                   np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def dedup(paths):
    keep, hs = [], []
    for p in sorted(paths):
        h = dhash(p)
        if all(int(np.unpackbits(h ^ g).sum()) > 0 for g in hs):
            keep.append(p);  hs.append(h)
    return keep


def wilson(k, n, z=1.96):
    r'''비율의 95% 신뢰구간 (Wilson). n 이 0 이면 nan'''
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tci(vals):
    r'''출처별 값의 평균과 95% 구간 (t). 출처가 둘 미만이면 nan'''
    v = np.asarray([x for x in vals if x == x], float)
    if v.size < 2:
        return (float(v.mean()) if v.size else float('nan'),
                float('nan'), float('nan'))
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, v.size - 1))
    except Exception:
        t = 2.776
    h = t * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


r'''1. 자료 — 원문 숫자와 대조'''
print('\n' + '=' * 78)
print('[1] 평가 자료 — 1층 자료 절과 대조')
print('=' * 78)

pool = {}
for p in glob.glob(f'{EXT}/smoke/*.jpg'):
    b = norm(os.path.basename(p))
    for k in ID.values():
        if b.startswith(norm(k) + '_'):
            pool.setdefault(k, []).append(p)
            break
ESRC = {k: dedup(pool.get(k, [])) for k in ID.values()}
npick = sum(len(v) for v in pool.values())
ndd = sum(len(v) for v in ESRC.values())
bad = sum(1 for k, w in WANT_SRC.items() if len(ESRC[k]) != w)
print(f'  E  smoke 만 · 뽑음 {npick} (원문 734) · 서로 다름 {ndd} (원문 666) · '
      f'어긋난 출처 {bad}개')
ok_e = (npick, ndd, bad) == (734, 666, 0)
print(f'    [검산] {"일치" if ok_e else "**불일치 — 아래를 보고에 쓰지 말 것**"}')

SIDESRC = {}
for d in SIDE:
    for p in glob.glob(f'{EXT}/{d}/*.jpg'):
        b = norm(os.path.basename(p))
        for k in ID.values():
            if b.startswith(norm(k) + '_'):
                SIDESRC.setdefault(k, []).append(p)
                break
SIDESRC = {k: dedup(v) for k, v in SIDESRC.items()}
print(f'  곁  {SIDE} · 서로 다름 ' +
      ' · '.join(f'{k} {len(v)}' for k, v in sorted(SIDESRC.items())))
print(f'    원문이 `따로 집계` 라 했으므로 **E 에 안 넣고 별도 표로만 냄**')

dpaths = []
for d in D_DIRS:
    dpaths += glob.glob(f'{STM}/{d}/*.jpg')
dsite = {}
for p in dpaths:
    dsite.setdefault(norm(os.path.basename(p)).split('_')[0], []).append(p)
DSRC = {k: dedup(v) for k, v in dsite.items()}
dtot = sum(len(v) for v in DSRC.values())
ok_d = (len(dsite), len(dpaths), dtot) == (11, 2233, 2044)
print(f'\n  D  {len(dsite)}곳 · 뽑음 {len(dpaths)} (원문 2,233) · '
      f'서로 다름 {dtot} (원문 2,044)')
print(f'    [검산] {"일치" if ok_d else "**불일치**"}')

bgall = glob.glob(f'{STM}/bg/*.jpg')
KW = norm('개원중')
KR = norm('로봇고')
KN = norm('논현중')
gw = [p for p in bgall if norm(os.path.basename(p)).startswith(KW + '_')]
gr = [p for p in bgall if norm(os.path.basename(p)).startswith(KR + '_')
      and norm('쉐이크') in norm(os.path.basename(p))]
gn = [p for p in bgall if norm(os.path.basename(p)).startswith(KN + '_')]
BW, BR, BN = dedup(gw), dedup(gr), dedup(gn)
ok_b = (len(BW), len(BR)) == (178, 29)
print(f'\n  배경 오탐군 — 원문 `로봇고 튀김(쉐이크) 29장` · `개원중 CCTV 178장` · 합 207')
print(f'    개원중 {len(gw)} -> {len(BW)} · 로봇고(쉐이크) {len(gr)} -> {len(BR)} · '
      f'합 {len(BW) + len(BR)}')
print(f'    [검산] {"일치" if ok_b else "**불일치 — 배경 값을 보고에 쓰지 말 것**"}')
print(f'    논현중 {len(gn)} -> {len(BN)}   (원문에 목표 장수 없음. 포함/제외 둘 다 냄)')

if not (ok_e and ok_d and ok_b):
    print('\n  **검산이 하나라도 어긋났음. 아래 숫자를 보고에 쓰지 말 것**')


r'''2. 추론'''
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO
print(f'\nultralytics {ultralytics.__version__} · torch {torch.__version__} · '
      f'CUDA {torch.cuda.is_available()}')


def maxconf(model, paths):
    out, ms, n = [], 0.0, 0
    for i in range(0, len(paths), BATCH):
        for r in model(paths[i:i + BATCH], conf=CMIN, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if len(c) else 0.0)
            ms += float(r.speed.get('inference', 0.0));  n += 1
    return np.array(out, np.float32), (ms / max(n, 1))


def rate(v, t):
    return float((v >= t).mean()) if len(v) else float('nan')


RES = {}
for s in SPLITS:
    w = f'{ROUT}/s{s}/best.pt'
    if not os.path.exists(w):
        print(f'\n분할 {s} — {w} 가 없어 건너뜀')
        continue
    print('\n' + '=' * 78)
    print(f'분할 {s}')
    print('=' * 78)
    keys = [ID[i] for i in EVAL[s]]
    ne = sum(len(ESRC[k]) for k in keys)
    use = [k for k in keys if len(ESRC[k]) >= MINSRC]
    drop = [k for k in keys if k not in use]
    print(f'  E 출처 {" · ".join(f"{k}({len(ESRC[k])})" for k in keys)}  합 {ne}장')
    print(f'    [검산] 1층 배정표의 {WANT_E[s]}장과 '
          f'{"일치" if ne == WANT_E[s] else "**불일치**"}')
    if drop:
        print(f'    하한 {MINSRC}장 미달 {" · ".join(drop)} -> '
              f'**이 분할은 평가 출처 {len(use)}개로 보고함** (원문 그대로)')

    m = YOLO(w)
    t0 = time.time()
    ec = {k: maxconf(m, ESRC[k]) for k in keys}
    sc = {k: maxconf(m, SIDESRC[k]) for k in keys if k in SIDESRC}
    dc = {k: maxconf(m, v) for k, v in DSRC.items()}
    bw_, br_, bn_ = maxconf(m, BW), maxconf(m, BR), maxconf(m, BN)
    dt = time.time() - t0
    inf = np.mean([v[1] for v in list(ec.values()) + list(dc.values()) + [bw_, br_, bn_]])
    print(f'  추론 {dt/60:.1f}분 · 장당 추론 {inf:.2f}ms')

    ev = {k: v[0] for k, v in ec.items()}
    sv = {k: v[0] for k, v in sc.items()}
    dv = {k: v[0] for k, v in dc.items()}
    dkeys = [k for k in sorted(dv) if len(dv[k]) >= MINSRC]
    eall = np.concatenate([ev[k] for k in use]) if use else np.array([])
    dall = np.concatenate([dv[k] for k in dkeys])

    print(f'\n  **macro 가 주 지표임** (원문 `E·D 는 출처별 macro 평균으로 냄`)')
    print(f'  {"conf":>6}{"E macro":>10}{"95% 구간":>18}{"D macro":>10}{"95% 구간":>18}'
          f'{"판별비":>9}')
    print('  ' + '-' * 72)
    row = {}
    for t in CONFS:
        eM, elo, ehi = tci([rate(ev[k], t) for k in use])
        dM, dlo, dhi = tci([rate(dv[k], t) for k in dkeys])
        em, dm = rate(eall, t), rate(dall, t)
        rM = eM / dM if dM > 0 else float('inf')
        rm = em / dm if dm > 0 else float('inf')
        row[t] = {'Em': eM, 'Elo': elo, 'Ehi': ehi, 'Dm': dM, 'Dlo': dlo, 'Dhi': dhi,
                  'rM': rM, 'E': em, 'D': dm, 'r': rm}
        mark = '  <- 판정' if abs(t - BASE) < 1e-9 else ''
        print(f'  {t:>6.2f}{eM:>10.3f}{f"[{elo:.3f}, {ehi:.3f}]":>18}'
              f'{dM:>10.3f}{f"[{dlo:.3f}, {dhi:.3f}]":>18}{rM:>9.2f}{mark}')
    print('  ' + '-' * 72)
    print(f'  구간은 **출처 사이의 흩어짐**(t, 출처 수 기준)임. 원문 단서 —')
    print(f'    `평가군이 같은 조리 과정의 연속 프레임이라 각 장이 독립 관측이 아님`')

    print(f'\n  곁들여 — micro (전체 장 비율 · Wilson 95%)')
    for t in CONFS:
        em, dm = rate(eall, t), rate(dall, t)
        el, eh = wilson(int(round(em * len(eall))), len(eall))
        dl, dh = wilson(int(round(dm * len(dall))), len(dall))
        print(f'    conf {t:.2f}  E {em:.3f} [{el:.3f}, {eh:.3f}]  '
              f'D {dm:.3f} [{dl:.3f}, {dh:.3f}]  비 {em/dm if dm>0 else float("inf"):.2f}')

    print(f'\n  출처별 E (conf {BASE}) — 원문 `출처별 값 macro 뒤에 나열`')
    for k in keys:
        n = len(ev[k]);  r = rate(ev[k], BASE)
        lo, hi = wilson(int(round(r * n)), n)
        print(f'    {k:<10}{n:>5}장   {r:>6.3f}  [{lo:.3f}, {hi:.3f}]'
              f'{"   (하한 미달 · macro 제외)" if k in drop else ""}')
    print(f'  시설별 D (conf {BASE})')
    for k in sorted(dv, key=lambda x: -rate(dv[x], BASE)):
        n = len(dv[k]);  r = rate(dv[k], BASE)
        lo, hi = wilson(int(round(r * n)), n)
        print(f'    {k:<10}{n:>5}장   {r:>6.3f}  [{lo:.3f}, {hi:.3f}]'
              f'{"   (하한 미달 · macro 제외)" if k not in dkeys else ""}')

    if sv:
        print(f'\n  곁 자료 — smoke_faint · smoke_reheat (원문 `따로 집계`. E 에 안 넣음)')
        for k in keys:
            if k in sv and len(sv[k]):
                print(f'    {k:<10}{len(sv[k]):>5}장   {rate(sv[k], BASE):>6.3f}')

    print(f'\n  배경 오탐군 — 원문 `판정에는 안 쓰되 값은 냄`')
    b_ex = np.concatenate([bw_[0], br_[0]])
    b_in = np.concatenate([bw_[0], br_[0], bn_[0]])
    for nm, v, tag in (('개원중', bw_[0], '  참고로만. 낮게 나올 수밖에 없는 자료임'),
                       ('로봇고(쉐이크)', br_[0], '  주로 읽음'),
                       ('논현중', bn_[0], '  1층이 배경 오탐군에서 뺀 것')):
        n = len(v);  r = rate(v, BASE)
        lo, hi = wilson(int(round(r * n)), n)
        print(f'    {nm:<14}{n:>5}장   {r:>6.3f}  [{lo:.3f}, {hi:.3f}]{tag}')
    print(f'    논현중 제외  {len(b_ex):>5}장   {rate(b_ex, BASE):>6.3f}')
    print(f'    논현중 포함  {len(b_in):>5}장   {rate(b_in, BASE):>6.3f}')
    print(f'    원문 — `크게 다르면 논현중에 뭔가 있는 것이고, 비슷하면 놓친 것이')
    print(f'            있어도 결론이 안 바뀜`')

    r0 = row[BASE]
    ok = r0['rM'] >= PASS
    print(f'\n  판정 (conf {BASE} · macro) — '
          f'**E {r0["Em"]:.3f} · D {r0["Dm"]:.3f} · 판별비 {r0["rM"]:.2f} '
          f'-> {"성립" if ok else "미달"}**')
    print(f'    원문 `판별비 E/D 가 1 이상이면 성립, 미만이면 미달. 두 단계뿐임.`')
    print(f'    원문 `판정문에 E·D 를 반드시 병기함.`')

    RES[s] = {'n_e': int(ne), 'n_src_used': len(use), 'dropped': drop,
              'inf_ms': float(inf), 'sec': float(dt),
              'rows': {str(t): row[t] for t in CONFS},
              'e_src': {k: {'n': int(len(ev[k])), 'r': rate(ev[k], BASE)} for k in keys},
              'side': {k: {'n': int(len(sv[k])), 'r': rate(sv[k], BASE)} for k in sv},
              'd_src': {k: {'n': int(len(dv[k])), 'r': rate(dv[k], BASE)} for k in dv},
              'bg': {'개원중': rate(bw_[0], BASE), '로봇고': rate(br_[0], BASE),
                     '논현중': rate(bn_[0], BASE),
                     'excl': rate(b_ex, BASE), 'incl': rate(b_in, BASE)},
              'curve': {'conf': CURVE,
                        'E': [float(np.mean([rate(ev[k], t) for k in use])) for t in CURVE],
                        'D': [float(np.mean([rate(dv[k], t) for k in dkeys])) for t in CURVE]},
              'checks': {'e': bool(ok_e), 'd': bool(ok_d), 'bg': bool(ok_b)}}

json.dump(RES, open(f'{EOUT}/eval_s1_v2.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/eval_s1_v2.json')


r'''3. 다섯 분할 — 폭이 주 결과'''
if RES:
    print('\n' + '=' * 78)
    print(f'다섯 분할 — conf {BASE} · macro · 원문 `폭이 애초에 재려던 것`')
    print('=' * 78)
    print(f'  {"분할":>5}{"E장":>7}{"출처":>6}{"E macro":>10}{"D macro":>10}'
          f'{"판별비":>9}{"판정":>7}')
    print('  ' + '-' * 54)
    for s, v in RES.items():
        r = v['rows'][str(BASE)]
        print(f'  {s:>5}{v["n_e"]:>7}{v["n_src_used"]:>6}{r["Em"]:>10.3f}'
              f'{r["Dm"]:>10.3f}{r["rM"]:>9.2f}'
              f'{("성립" if r["rM"] >= PASS else "미달"):>7}')
    print('  ' + '-' * 54)
    E = [v['rows'][str(BASE)]['Em'] for v in RES.values()]
    D = [v['rows'][str(BASE)]['Dm'] for v in RES.values()]
    R = [v['rows'][str(BASE)]['rM'] for v in RES.values()]
    print(f'  **폭**   E {min(E):.3f} ~ {max(E):.3f} (차 {max(E)-min(E):.3f}) · '
          f'D {min(D):.3f} ~ {max(D):.3f} (차 {max(D)-min(D):.3f})')
    print(f'          판별비 {min(R):.2f} ~ {max(R):.2f} (차 {max(R)-min(R):.2f})')
    print(f'  성립선 {PASS} 이상인 분할 — {[s for s, v in RES.items() if v["rows"][str(BASE)]["rM"] >= PASS]}')
    print(f'  원문 `평균·중앙값·최소값을 쓰지 않음` 이므로 위 다섯을 합치지 않음')
    print(f'  장당 추론 {np.mean([v["inf_ms"] for v in RES.values()]):.2f}ms (T4 · imgsz 640)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for s, v in RES.items():
        c = v['curve']
        ax[0].plot(c['conf'], c['E'], label=f'split {s}')
        ax[1].plot(c['conf'], c['D'], label=f'split {s}')
        ax[2].plot(c['conf'], [a / b if b > 0 else np.nan
                               for a, b in zip(c['E'], c['D'])], label=f'split {s}')
    for a, t in zip(ax, ('E macro  smoke detection', 'D macro  steam false alarm',
                         'ratio E/D  (pass line 1.0)')):
        a.set_xlabel('conf');  a.set_title(t);  a.grid(alpha=.3);  a.legend()
        a.axvline(BASE, color='k', ls='--', lw=1)
    ax[2].axhline(1.0, color='r', ls='--', lw=1)
    ax[2].set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(f'{EOUT}/_curve_v2.png', dpi=120)
    print(f'  -> {EOUT}/_curve_v2.png')
    files.download(f'{EOUT}/_curve_v2.png')

print('\n' + '=' * 78)
print('이 평가가 못 보는 것')
print('=' * 78)
print('  장 단위라 **어디를 봤는지 모름**. 원문 `정탐이 냄비 위가 아닌 곳에서')
print('  나오는지는 1회차 결과에서 눈으로 확인함` — 그림 확인이 남아 있음')
print('  원문 `평가군이 같은 조리 과정의 연속 프레임이라 각 장이 독립 관측이 아님`')
print('  김 2,044장 중 한 곳(영동중)이 557장임')
print('  판별비의 신뢰구간은 안 냄 — 두 비율의 비이고 원문에 규칙이 없음')
