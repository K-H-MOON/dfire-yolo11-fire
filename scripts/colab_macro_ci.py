import os, glob, json, unicodedata, subprocess, sys
import numpy as np
from PIL import Image
from google.colab import drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'
EOUT = f'{SRC}/eval_s1'
ROUT = f'{SRC}/runs_s1'
SPLITS = [1, 2, 3, 4, 5]
CONF, CMIN, MINSRC, BATCH = 0.10, 0.01, 10, 16
ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}

drive.mount('/content/drive')

print('=' * 78)
print('95% 신뢰구간 되살리기 — 1층 `주요 비율에 95% 신뢰구간을 냄`')
print('=' * 78)
print('  기록 92·93 이 이 규칙을 안 지켰음. 평가 1판 결함 4 를 그대로 반복한 것임')
print('  D 는 geom_sensitivity.json 에 출처별 값이 남아 있어 다시 안 재도 됨')
print('  E 만 다시 잼. 구간은 평가 2판과 같은 방법(출처별 값에 t 구간)으로 냄')
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


def tci(vals):
    r'''출처별 값의 평균과 95% 구간 (t). 평가 2판과 같은 함수'''
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


try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
LB = LetterBox((640, 640), auto=False, stride=32)
print(f'\nultralytics {ultralytics.__version__} · CUDA {torch.cuda.is_available()}')

G = json.load(open(f'{EOUT}/geom_sensitivity.json'))
pool = {}
for p in glob.glob(f'{EXT}/smoke/*.jpg'):
    b = norm(os.path.basename(p))
    for k in ID.values():
        if b.startswith(norm(k) + '_'):
            pool.setdefault(k, []).append(p)
            break
ESRC = {k: dedup(pool.get(k, [])) for k in ID.values()}
print(f'[검산] E 서로 다름 {sum(len(v) for v in ESRC.values())} (666)')


def rate_a(model, paths):
    out = []
    for i in range(0, len(paths), BATCH):
        for r in model(paths[i:i + BATCH], conf=CMIN, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if len(c) else 0.0)
    return float(np.mean(np.array(out) >= CONF))


def rate_c(model, paths):
    out = []
    for i in range(0, len(paths), BATCH):
        arrs = [LB(image=cv2.imread(p)) for p in paths[i:i + BATCH]]
        for r in model(arrs, conf=CMIN, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if len(c) else 0.0)
    return float(np.mean(np.array(out) >= CONF))


OUT = {}
for s in SPLITS:
    w = f'{ROUT}/s{s}/best.pt'
    if not os.path.exists(w) or str(s) not in G:
        print(f'\n분할 {s} — 자료가 없어 건너뜀');  continue
    m = YOLO(w)
    ek = [ID[i] for i in EVAL[s] if len(ESRC[ID[i]]) >= MINSRC]
    ea = {k: rate_a(m, ESRC[k]) for k in ek}
    ec = {k: rate_c(m, ESRC[k]) for k in ek}
    row = {}
    for tag, ev in [('가', ea), ('다', ec)]:
        dv = G[str(s)][tag][2]
        E, elo, ehi = tci(list(ev.values()))
        D, dlo, dhi = tci(list(dv.values()))
        row[tag] = dict(E=E, elo=elo, ehi=ehi, D=D, dlo=dlo, dhi=dhi,
                        e_src=ev, d_src=dv)
    OUT[s] = row
    print(f'\n분할 {s}  E 출처 {len(ek)}개 · D 출처 {len(G[str(s)]["가"][2])}개')
    print(f'  {"방식":5}{"E macro":>9}{"95% 구간":>20}{"D macro":>9}{"95% 구간":>20}{"판별비":>8}')
    for tag in ['가', '다']:
        r = row[tag]
        print(f'  {tag:5}{r["E"]:9.3f}   [{r["elo"]:6.3f}, {r["ehi"]:6.3f}]'
              f'{r["D"]:9.3f}   [{r["dlo"]:6.3f}, {r["dhi"]:6.3f}]{r["E"]/r["D"]:8.2f}')
    a = row['가']
    print(f'  [검산] 가 의 E·D 가 기록된 본 결과와 같은가 — '
          f'E {a["E"]:.3f} · D {a["D"]:.3f}')
    print(f'  E 출처별  ' + ' · '.join(f'{k} {v:.3f}' for k, v in a['e_src'].items()))

json.dump({str(k): {t: {kk: vv for kk, vv in r.items()} for t, r in v.items()}
           for k, v in OUT.items()},
          open(f'{EOUT}/macro_ci.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/macro_ci.json  (출처별 값까지 저장. 다음부터 다시 안 재도 됨)')

print('\n' + '=' * 78)
print('읽을 때 붙일 단서 — 1층 원문')
print('=' * 78)
print('  `주요 비율에 95% 신뢰구간을 냄. 다만 액면 그대로 받아들이면 안 됨 —')
print('   평가군이 같은 조리 과정의 연속 프레임이라 각 장이 독립 관측이 아님`')
print('  출처가 넷에서 다섯뿐이라 t 구간이 매우 넓게 나올 것임. 그것도 결과임')
print('  1층은 판별비의 구간을 어떻게 낼지는 안 정했음. E 와 D 만 냄')
