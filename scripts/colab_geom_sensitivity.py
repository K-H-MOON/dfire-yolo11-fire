import os, glob, json, unicodedata, subprocess, sys, time
import numpy as np
from PIL import Image
from google.colab import drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'
STM  = f'{SRC}/steam'
ROUT = f'{SRC}/runs_s1'
EOUT = f'{SRC}/eval_s1'

SPLITS = [1, 2, 3, 4, 5]
CONF, CMIN, MINSRC, BATCH = 0.10, 0.01, 10, 16
D_DIRS = ['steam_near', 'steam_far', 'steam_in']
ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}
REC = {1: (0.650, 0.600), 2: (0.807, 0.898), 3: (0.634, 0.411),
       4: (0.901, 0.925), 5: (0.398, 0.412)}

drive.mount('/content/drive')

print('=' * 78)
print('민감도 시험 + 숭곡중 32장 되돌리기 — 한 번에')
print('=' * 78)
print('  세 방식으로 같은 자료를 잼')
print('    가 재현    원래 코드와 같음. 출처별 dedup 정렬 뒤 16장씩 묶어 추론')
print('              숭곡중만 한 사이트 안에 크기가 둘이라 묶음 경계 32장이 정사각이 됨')
print('    나 고침    (출처, 크기)별로 묶음. 크기가 섞이는 일이 없음 -> 전부 직사각')
print('    다 정사각  전부 640x640 으로 채워 넣음')
print()
print('  가 는 본 결과의 재현이므로 기록된 값과 맞아야 함 (검산)')
print('  나 - 가 = **숭곡중 32장 사고의 크기**')
print('  다 - 나 = **입력 기하 자유도의 크기**')
print()
print('  못 박아 둔 것 — 본 결과는 가 로 하고, 다 가 어떻게 나오든 바꾸지 않음')
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


try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
print(f'\nultralytics {ultralytics.__version__} · CUDA {torch.cuda.is_available()}')
LB = LetterBox((640, 640), auto=False, stride=32)

pool = {}
for p in glob.glob(f'{EXT}/smoke/*.jpg'):
    b = norm(os.path.basename(p))
    for k in ID.values():
        if b.startswith(norm(k) + '_'):
            pool.setdefault(k, []).append(p)
            break
ESRC = {k: dedup(pool.get(k, [])) for k in ID.values()}
dsite = {}
for d in D_DIRS:
    for p in glob.glob(f'{STM}/{d}/*.jpg'):
        dsite.setdefault(norm(os.path.basename(p)).split('_')[0], []).append(p)
DSRC = {k: dedup(v) for k, v in dsite.items()}
print(f'[검산] E 서로 다름 {sum(len(v) for v in ESRC.values())} (666) · '
      f'D {sum(len(v) for v in DSRC.values())} (2044)')


def mx_chunks(model, chunks):
    out = []
    for ch in chunks:
        for r in model(ch, conf=CMIN, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if len(c) else 0.0)
    return out


def run_a(model, paths):
    ch = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    return mx_chunks(model, ch)


def run_b(model, paths):
    g = {}
    for p in paths:
        g.setdefault(Image.open(p).size, []).append(p)
    ch = []
    for k in sorted(g):
        v = g[k]
        ch += [v[i:i + BATCH] for i in range(0, len(v), BATCH)]
    return mx_chunks(model, ch)


def run_c(model, paths):
    out = []
    for i in range(0, len(paths), BATCH):
        arrs = [LB(image=cv2.imread(p)) for p in paths[i:i + BATCH]]
        for r in model(arrs, conf=CMIN, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if len(c) else 0.0)
    return out


def rate(v):
    return float(np.mean(np.array(v) >= CONF)) if len(v) else float('nan')


RES = {}
for s in SPLITS:
    w = f'{ROUT}/s{s}/best.pt'
    if not os.path.exists(w):
        print(f'\n분할 {s} — 가중치가 없어 건너뜀');  continue
    m = YOLO(w)
    t0 = time.time()
    ekeys = [ID[i] for i in EVAL[s] if len(ESRC[ID[i]]) >= MINSRC]
    dkeys = [k for k in sorted(DSRC) if len(DSRC[k]) >= MINSRC]
    row = {}
    for tag, fn in [('가', run_a), ('나', run_b), ('다', run_c)]:
        e = [rate(fn(m, ESRC[k])) for k in ekeys]
        d = {k: rate(fn(m, DSRC[k])) for k in dkeys}
        row[tag] = (float(np.mean(e)), float(np.mean(list(d.values()))), d)
    RES[s] = row
    print(f'\n분할 {s}  평가 출처 {len(ekeys)}개 · D 출처 {len(dkeys)}개 · '
          f'{(time.time()-t0)/60:.1f}분')
    print(f'  {"방식":6}{"E macro":>10}{"D macro":>10}{"판별비":>9}{"판정":>7}')
    for tag in ['가', '나', '다']:
        E, D, _ = row[tag]
        print(f'  {tag:6}{E:10.3f}{D:10.3f}{E/D:9.2f}{("성립" if E/D >= 1 else "미달"):>7}')
    rE, rD = REC[s]
    E, D, _ = row['가']
    ok = abs(E - rE) <= 0.001 and abs(D - rD) <= 0.001
    print(f'  [검산] 가 와 기록된 본 결과 ({rE:.3f} · {rD:.3f}) — '
          f'{"일치" if ok else "**불일치. 아래를 믿지 말 것**"}')
    da, db = row['가'][2], row['나'][2]
    diff = [k for k in da if abs(da[k] - db[k]) > 1e-9]
    print(f'  [검산] 가 와 나 가 다른 D 출처 — '
          f'{" · ".join(f"{k} {da[k]:.4f} -> {db[k]:.4f}" for k in diff) if diff else "없음"}')
    print(f'         숭곡중만 나와야 맞음')

json.dump({str(k): {t: [v[0], v[1], v[2]] for t, v in r.items()} for k, r in RES.items()},
          open(f'{EOUT}/geom_sensitivity.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/geom_sensitivity.json')

if RES:
    print('\n' + '=' * 78)
    print('모아 보기')
    print('=' * 78)
    print(f'  {"분할":5}{"가 판별비":>10}{"나 판별비":>10}{"다 판별비":>10}   판정 가 / 나 / 다')
    for s in sorted(RES):
        r = {t: RES[s][t][0] / RES[s][t][1] for t in ['가', '나', '다']}
        j = ' / '.join('성립' if r[t] >= 1 else '미달' for t in ['가', '나', '다'])
        print(f'  {s:<5}{r["가"]:10.2f}{r["나"]:10.2f}{r["다"]:10.2f}   {j}')
    flip = [s for s in RES if (RES[s]['가'][0] / RES[s]['가'][1] >= 1)
            != (RES[s]['다'][0] / RES[s]['다'][1] >= 1)]
    print(f'\n  가 와 다 사이에서 판정이 뒤집힌 분할 — '
          f'{" · ".join(map(str, flip)) if flip else "없음"}')
    print('  뒤집힌 분할이 있으면 그것이 곧 `1층이 안 정한 자유도가 결론을 바꾼다` 는 뜻임')
