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

ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}
E_USE  = ['smoke', 'smoke_faint', 'smoke_reheat']
E_SKIP = ['smoke_frag', 'smoke_mixed']
WANT_E = {1: 83, 2: 412, 3: 152, 4: 121, 5: 104}
WANT_SRC = {'ft': 21, 'j01': 18, 'p2': 14, 'q1': 24, 'j12': 6, 's2': 341, '05': 10,
            'kfire02': 17, '04': 23, 'm3': 48, '07': 37, 'kfire03': 25, 'kt': 28,
            'j04': 19}
D_DIRS = ['steam_near', 'steam_far', 'steam_in']
BG_IN  = ['개원중', '로봇고']
BG_OUT = ['논현중']

drive.mount('/content/drive')
os.makedirs(EOUT, exist_ok=True)

print('=' * 78)
print('평가 — 1층 보고 규칙 여섯 줄을 전부 채움')
print('=' * 78)
print(f'  채점    장 단위. 한 장에 임계값 이상인 상자가 하나라도 있으면 정탐')
print(f'  임계값   {CONFS} · 판정은 **{BASE}** 한 점')
print(f'  E       분할마다 평가 배정 출처 · {E_USE} · {E_SKIP} 제외 · 출처별 서로 다름')
print(f'  D       김 11곳 {D_DIRS} · 서로 다름')
print(f'  배경    개원중 · 로봇고 (판정에 안 씀) · 논현중 포함/제외 둘 다')
print(f'  macro   출처당 서로 다른 것 {MINSRC}장 이상만 (1층 하한)')
print('=' * 78)
print('  **판별비를 micro 와 macro 둘 다 냄** — 1층이 어느 쪽인지 안 정했음.')
print('  지금 하나를 고르면 그 선택이 결과를 보고 내린 것이 되므로 둘 다 적음')
print('=' * 78)


def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(path, size=8):
    g = np.asarray(Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS),
                   np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def dedup(paths):
    r'''파일 이름 차례로 훑으며 해밍 > 0 인 것만 남김 — 본 합성과 같은 규칙'''
    keep, hs = [], []
    for p in sorted(paths):
        h = dhash(p)
        if all(int(np.unpackbits(h ^ g).sum()) > 0 for g in hs):
            keep.append(p);  hs.append(h)
    return keep


r'''1. 자료를 모음'''
print('\n' + '=' * 78)
print('[1] 평가 자료')
print('=' * 78)

pool = {}
for d in E_USE:
    for p in glob.glob(f'{EXT}/{d}/*.jpg'):
        b = norm(os.path.basename(p))
        for k in ID.values():
            if b.startswith(norm(k) + '_'):
                pool.setdefault(k, []).append(p)
                break
ESRC = {}
print(f'  {"출처":<10}{"뽑음":>7}{"서로 다름":>10}{"1층 표":>9}{"판정":>8}')
print('  ' + '-' * 46)
bad = 0
for k in ID.values():
    ps = pool.get(k, [])
    dd = dedup(ps)
    ESRC[k] = dd
    w = WANT_SRC.get(k)
    ok = (w is None) or (len(dd) == w)
    bad += (not ok)
    print(f'  {k:<10}{len(ps):>7}{len(dd):>10}'
          f'{("-" if w is None else w):>9}{("-" if w is None else ("일치" if ok else "**불일치**")):>8}')
print('  ' + '-' * 46)
tot = sum(len(v) for v in ESRC.values())
print(f'  {"합":<10}{sum(len(v) for v in pool.values()):>7}{tot:>10}{666:>9}'
      f'{("일치" if tot == 666 else "**불일치**"):>8}')
if bad or tot != 666:
    print('  **1층 자료 절과 어긋남. 아래 숫자를 보고에 쓰지 말 것**')

dpaths = []
for d in D_DIRS:
    dpaths += glob.glob(f'{STM}/{d}/*.jpg')
dsite = {}
for p in dpaths:
    dsite.setdefault(norm(os.path.basename(p)).split('_')[0], []).append(p)
DSRC = {k: dedup(v) for k, v in dsite.items()}
dtot = sum(len(v) for v in DSRC.values())
print(f'\n  D 김  {len(dsite)}곳 · 뽑음 {len(dpaths)}장 · 서로 다름 {dtot}장')
print(f'    [검산] 1층의 11곳 · 2,233 · 2,044 와 '
      f'{"일치" if (len(dsite), len(dpaths), dtot) == (11, 2233, 2044) else "**불일치**"}')

bgall = glob.glob(f'{STM}/bg/*.jpg')
bgsrc = {}
for p in bgall:
    bgsrc.setdefault(norm(os.path.basename(p)).split('_')[0], []).append(p)
BSRC = {k: dedup(v) for k, v in bgsrc.items()}
print(f'\n  배경  ' + ' · '.join(f'{k} {len(bgsrc[k])}->{len(BSRC[k])}' for k in sorted(BSRC)))
b207 = sum(len(BSRC.get(k, [])) for k in BG_IN)
print(f'    [검산] 개원중 178 · 로봇고 29 · 합 207 과 '
      f'{"일치" if (len(BSRC.get("개원중", [])), len(BSRC.get("로봇고", [])), b207) == (178, 29, 207) else "**불일치**"}')
print(f'    숭곡중 {len(BSRC.get("숭곡중", []))}장은 1층이 배경 오탐군을 2출처로 못 박아 **안 씀**')


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
    r'''장마다 가장 신뢰도 높은 상자의 값. 상자가 없으면 0. 추론 시간도 모음'''
    out, ms, n = [], 0.0, 0
    for i in range(0, len(paths), BATCH):
        rs = model(paths[i:i + BATCH], conf=CMIN, verbose=False)
        for r in rs:
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
    print(f'  E 출처 {" · ".join(f"{k}({len(ESRC[k])})" for k in keys)}  합 {ne}장')
    print(f'    [검산] 1층 표의 {WANT_E[s]}장과 '
          f'{"일치" if ne == WANT_E[s] else "**불일치**"}')

    m = YOLO(w)
    t0 = time.time()
    ec = {k: maxconf(m, ESRC[k]) for k in keys}
    dc = {k: maxconf(m, v) for k, v in DSRC.items()}
    bc = {k: maxconf(m, v) for k, v in BSRC.items() if k in BG_IN + BG_OUT}
    dt = time.time() - t0
    inf = np.mean([v[1] for v in list(ec.values()) + list(dc.values()) + list(bc.values())])
    print(f'  추론 {dt/60:.1f}분 · 장당 추론 {inf:.2f}ms (전처리·후처리 제외)')

    ev = {k: v[0] for k, v in ec.items()}
    dv = {k: v[0] for k, v in dc.items()}
    bv = {k: v[0] for k, v in bc.items()}
    eall = np.concatenate([ev[k] for k in keys]) if keys else np.array([])
    dall = np.concatenate([dv[k] for k in sorted(dv)])
    emacro_keys = [k for k in keys if len(ev[k]) >= MINSRC]
    dmacro_keys = [k for k in sorted(dv) if len(dv[k]) >= MINSRC]
    drop = [k for k in keys if k not in emacro_keys]
    if drop:
        print(f'  macro 에서 뺀 출처 (서로 다름 {MINSRC}장 미만) — {" · ".join(drop)}')

    print(f'\n  {"conf":>6}{"E micro":>10}{"E macro":>10}{"D micro":>10}{"D macro":>10}'
          f'{"비 micro":>11}{"비 macro":>11}')
    print('  ' + '-' * 68)
    row = {}
    for t in CONFS:
        em, dm = rate(eall, t), rate(dall, t)
        eM = float(np.mean([rate(ev[k], t) for k in emacro_keys])) if emacro_keys else float('nan')
        dM = float(np.mean([rate(dv[k], t) for k in dmacro_keys])) if dmacro_keys else float('nan')
        rm = em / dm if dm > 0 else float('inf')
        rM = eM / dM if dM > 0 else float('inf')
        row[t] = {'E': em, 'Em': eM, 'D': dm, 'Dm': dM, 'r': rm, 'rm': rM}
        mark = '  <- 판정' if abs(t - BASE) < 1e-9 else ''
        print(f'  {t:>6.2f}{em:>10.3f}{eM:>10.3f}{dm:>10.3f}{dM:>10.3f}'
              f'{rm:>11.2f}{rM:>11.2f}{mark}')
    print('  ' + '-' * 68)
    print(f'  성립선은 1 임. 화재 저장소는 이 라인을 0.54 -> 0.75 로 못 넘고 종결했음')

    print(f'\n  출처별 E (conf {BASE})')
    for k in keys:
        print(f'    {k:<10}{len(ev[k]):>5}장   {rate(ev[k], BASE):>6.3f}'
              f'{"   (macro 제외)" if k in drop else ""}')
    print(f'  시설별 D (conf {BASE})')
    for k in sorted(dv, key=lambda x: -rate(dv[x], BASE)):
        print(f'    {k:<10}{len(dv[k]):>5}장   {rate(dv[k], BASE):>6.3f}'
              f'{"   (macro 제외)" if k not in dmacro_keys else ""}')

    print(f'\n  배경 오탐군 — 판정에 안 씀')
    bin_ = np.concatenate([bv[k] for k in BG_IN if k in bv])
    bwith = np.concatenate([bv[k] for k in BG_IN + BG_OUT if k in bv])
    for k in BG_IN + BG_OUT:
        if k in bv:
            tag = '' if k in BG_IN else '   (1층이 배경 오탐군에서 뺀 것)'
            print(f'    {k:<10}{len(bv[k]):>5}장   {rate(bv[k], BASE):>6.3f}{tag}')
    print(f'    논현중 제외  {len(bin_):>5}장   {rate(bin_, BASE):>6.3f}')
    print(f'    논현중 포함  {len(bwith):>5}장   {rate(bwith, BASE):>6.3f}')
    print(f'    **개원중은 낮게 나올 수밖에 없는 자료임. 낮게 나와도 아무 말도 하지 않음**')

    RES[s] = {'n_e': int(ne), 'inf_ms': float(inf), 'sec': float(dt),
              'rows': {str(t): row[t] for t in CONFS},
              'e_src': {k: {'n': int(len(ev[k])), 'r': rate(ev[k], BASE)} for k in keys},
              'd_src': {k: {'n': int(len(dv[k])), 'r': rate(dv[k], BASE)} for k in dv},
              'bg': {k: {'n': int(len(bv[k])), 'r': rate(bv[k], BASE)} for k in bv},
              'bg_excl': rate(bin_, BASE), 'bg_incl': rate(bwith, BASE),
              'curve': {'conf': CURVE,
                        'E': [rate(eall, t) for t in CURVE],
                        'D': [rate(dall, t) for t in CURVE]},
              'macro_dropped': drop}

json.dump(RES, open(f'{EOUT}/eval_s1.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/eval_s1.json')


r'''3. 모아 보기'''
if RES:
    print('\n' + '=' * 78)
    print(f'다섯 분할 — conf {BASE} · 합치지 않고 각각')
    print('=' * 78)
    print(f'  {"분할":>5}{"E장":>7}{"E micro":>10}{"D micro":>10}{"비 micro":>11}'
          f'{"E macro":>10}{"D macro":>10}{"비 macro":>11}')
    print('  ' + '-' * 74)
    for s, v in RES.items():
        r = v['rows'][str(BASE)]
        print(f'  {s:>5}{v["n_e"]:>7}{r["E"]:>10.3f}{r["D"]:>10.3f}{r["r"]:>11.2f}'
              f'{r["Em"]:>10.3f}{r["Dm"]:>10.3f}{r["rm"]:>11.2f}')
    print('  ' + '-' * 74)
    print(f'  성립선 1 · 다섯 중 몇이 넘었는가 — micro '
          f'{sum(1 for v in RES.values() if v["rows"][str(BASE)]["r"] > 1)}/{len(RES)} · '
          f'macro {sum(1 for v in RES.values() if v["rows"][str(BASE)]["rm"] > 1)}/{len(RES)}')
    print(f'  장당 추론 {np.mean([v["inf_ms"] for v in RES.values()]):.2f}ms (T4 · imgsz 640)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s, v in RES.items():
        ax[0].plot(v['curve']['conf'], v['curve']['E'], label=f'split {s}')
        ax[1].plot(v['curve']['conf'], v['curve']['D'], label=f'split {s}')
    for a, t in zip(ax, ('E  smoke detection (higher better)',
                         'D  steam false alarm (lower better)')):
        a.set_xlabel('conf');  a.set_title(t);  a.grid(alpha=.3);  a.legend()
        a.axvline(BASE, color='k', ls='--', lw=1)
    fig.tight_layout()
    fig.savefig(f'{EOUT}/_curve.png', dpi=120)
    print(f'  -> {EOUT}/_curve.png')
    files.download(f'{EOUT}/_curve.png')

print('\n' + '=' * 78)
print('이 평가가 못 보는 것')
print('=' * 78)
print('  장 단위라 **어디를 봤는지 모름** — 뒤쪽 국솥에 반응해도 정탐이 됨.')
print('  1층이 `정탐이 냄비 위가 아닌 곳에서 나오는지 눈으로 확인함` 이라 적었으므로')
print('  그림 확인이 남아 있음')
print('  평가군이 같은 조리 과정의 연속 프레임이라 각 장이 독립 관측이 아님')
print('  김 2,044장 중 한 곳(영동중)이 557장이라 micro 가 그 한 곳에 끌림')
