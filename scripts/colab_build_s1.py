import os, glob, json, io, shutil, unicodedata, time
import numpy as np
from PIL import Image
from google.colab import drive

SRC   = '/content/drive/MyDrive/smoke_frames'
MAT   = f'{SRC}/matte'
TRIAL = f'{SRC}/synth_trial'
BGDIR = f'{SRC}/steam/bg'
BGKEY = '개원중'
WORK  = '/content/ds_s1'
DOUT  = f'{SRC}/ds_s1'

KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SID   = {'m3': 4, 'kfire03': 9, 'q1': 10, 'j04': 11, '07': 6, 'p2': 5}

SEED   = 1
THR    = 0.06
IMGSZ, STRIDE = 640, 8
UMIN   = 0.30
COVHI  = 0.60
NPER   = 476
GRID   = 3
NPAIR  = 24
RETRY  = 20
JPEGQ  = 95
SPLITS = [1, 2, 3, 4, 5]
WANT_PIECE = {1: 43, 2: 67, 3: 21, 4: 60, 5: 15}

drive.mount('/content/drive')

print('=' * 78)
print('본 합성 — 2층 절차 8단계를 그대로 돌림. 2층은 닫힌 문서이므로 여기서 안 바꿈')
print('=' * 78)
print(f'  합성식   배경 x (1 - 알파) + 255 x 알파      틴트 안 씀')
print(f'  배경     개원중 해밍 > 0 서로 다름 178장')
print(f'  라벨     질량99 상자 · YOLO txt · 클래스 0 = smoke')
print(f'  소재     피복률 상한 {COVHI:.0%} -> 크기 규칙 -> 67장')
print(f'  크기     가로폭 {UMIN} \\~ 배율 1.0 · 로그 균등      개수 한 장에 하나')
print(f'  장수     분할마다 양성 {NPER}장 · 학습 음성 0장      시드 {SEED}')
print(f'  제외     합성한 장마다 상자 안 평균 변화 < 크기별 잡음 바닥이면 버림')
print('=' * 78)


r'''공통 — size_rule2 와 같은 식을 씀'''
def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(g, size=8):
    x = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size + 1, size),
                                                              Image.LANCZOS), np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


def box_q(a, q=0.005):
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def rgbf(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float32)


def lumaf(p):
    return np.asarray(Image.open(p).convert('L'), np.float32)


r'''1. 배경 — 2층이 못 박은 해밍 > 0 서로 다름'''
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(lumaf(p))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = np.asarray(Image.open(bgs[0]).convert('RGB')).shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ))
PWMIN = int(round(W * UMIN))
print(f'\n배경 {len(bgs_all)}장 -> 서로 다름 {len(bgs)}장 · {W}x{H}')
print(f'  [검산] 2층의 178장과 {"일치" if len(bgs) == 178 else "**불일치 — 멈춤**"}')
assert len(bgs) == 178, '배경 장수가 2층과 다름'


r'''2. 잡음 바닥 — size_rule2 와 한 글자도 다르지 않게'''
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
FA, FV = [], []
diffs = []
for i in pair_idx:
    f1, f2 = rgbf(bgs_all[i]), rgbf(bgs_all[i + 1])
    if f1.shape == f2.shape:
        diffs.append(np.abs(f2 - f1).mean(2))
for ar in AREAS:
    bw = int(min(W, max(MINSIDE, round(np.sqrt(ar * 16 / 9)))))
    bh = int(min(H, max(MINSIDE, round(ar / bw))))
    v = []
    for d in diffs:
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(H - bh, 0) * gy / max(GRID - 1, 1))
                xx = int(max(W - bw, 0) * gx / max(GRID - 1, 1))
                v.append(float(d[yy:yy + bh, xx:xx + bw].mean()))
    FA.append(float(np.log(bw * bh)));  FV.append(float(np.median(v)))
del diffs
ok_floor = abs(min(FV) - 2.85) < 0.15 and abs(max(FV) - 5.41) < 0.15
print(f'\n잡음 바닥 {min(FV):.2f} \\~ {max(FV):.2f} 계조')
print(f'  [검산] 2층의 2.85\\~5.41 과 {"일치" if ok_floor else "**불일치 — 멈춤**"}')
assert ok_floor, '잡음 바닥이 2층과 다름'


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


r'''저장 품질을 정함 — 2층이 안 정한 칸. 안 정하면 파일을 못 씀'''
qbg = [bgs[i] for i in np.linspace(0, len(bgs) - 1, 8).round().astype(int)]
dq = []
for p in qbg:
    a = rgbf(p)
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(b, 'JPEG', quality=JPEGQ)
    b.seek(0)
    dq.append(float(np.abs(np.asarray(Image.open(b).convert('RGB'), np.float32) - a).mean()))
dqm = float(np.median(dq))
print(f'\n저장 품질 q{JPEGQ} 로 배경만 되감았을 때 생기는 변화 {dqm:.4f} 계조')
print(f'  가장 낮은 잡음 바닥 {min(FV):.3f} 의 **{dqm / min(FV):.2%}**')
print(f'  2층은 저장 형식을 안 정했음. 안 정하면 진행이 불가능하므로 여기서 정하고')
print(f'  기록 91 에 적음. 바닥보다 훨씬 작은 것을 재서 고른 값임')


r'''3. 소재 — 절차 1단계(피복률) -> 2단계(크기 범위)'''
sr = {(x['key'], x['file']): x for x in json.load(open(f'{TRIAL}/size_rule2.json'))['pieces']}
cov = {(r['key'], r['file']): r['cov_frame']
       for r in json.load(open(f'{TRIAL}/coverage.json'))}
all175 = sorted(sr)
step1 = [k for k in all175 if cov.get(k, 0.0) <= COVHI]
usable = [k for k in step1 if any(d['ok'] for d in sr[k]['draw'])]
print(f'\n소재 {len(all175)}장 -> 피복률 상한 {len(step1)}장 -> 크기 규칙 {len(usable)}장')
print(f'  [검산] 2층의 175 -> 174 -> 67 과 '
      f'{"일치" if (len(all175), len(step1), len(usable)) == (175, 174, 67) else "**불일치 — 멈춤**"}')
assert (len(all175), len(step1), len(usable)) == (175, 174, 67), '소재 수가 2층과 다름'

by_src = {k: sum(1 for t in usable if t[0] == k) for k in KEYS}
print('  출처별  ' + ' · '.join(f'{k} {by_src[k]}' for k in KEYS))

r'''사전 등록된 분할 배정 — split_s1.json 의 train 목록을 그대로 옮겨 박았음.
   파일에 의존하지 않게 하려는 것이고, 아래 분할별 소재 수 검산이 이 값을 되짚음.
   번호 대조는 assign_split.py 와 같음 — 4 m3 · 5 p2 · 6 07 · 9 kfire03 · 10 q1 · 11 j04'''
TRAIN = {1: {1, 4, 6, 7, 8, 9, 11, 12, 14, 15},
         2: {3, 4, 5, 6, 7, 8, 9, 10, 11, 13},
         3: {1, 2, 5, 7, 8, 11, 12, 13, 14, 15},
         4: {1, 3, 6, 8, 9, 10, 11, 13, 14, 15},
         5: {1, 2, 3, 4, 7, 8, 9, 13, 14, 15}}
_sp = f'{TRIAL}/split_s1.json'
if os.path.exists(_sp):
    _f = {i + 1: set(s['train']) for i, s in enumerate(json.load(open(_sp))['splits'])}
    print(f'  [검산] 박아 둔 분할 배정이 split_s1.json 과 '
          f'{"일치" if _f == TRAIN else "**불일치 — 멈춤**"}')
    assert _f == TRAIN, '분할 배정이 사전 등록 파일과 다름'
else:
    print('  [주의] split_s1.json 이 없어 분할 배정을 대조하지 못했음')


ALPHA = {}


def alpha8(key, file):
    if (key, file) not in ALPHA:
        a = np.asarray(Image.open(f'{MAT}/{key}/{file}'))[..., 3].copy()
        a[a < int(round(THR * 255))] = 0
        ALPHA[(key, file)] = a
    return ALPHA[(key, file)]


def size_range(a8):
    ph0, pw0 = a8.shape
    smax = min(1.0, W / pw0, H / ph0)
    smin = PWMIN / pw0
    return (smax, smax) if smin > smax else (smin, smax)


r'''4\~8. 분할마다 476장'''
os.makedirs(DOUT, exist_ok=True)
summary = {}
for sp_no in SPLITS:
    t0 = time.time()
    pieces = [k for k in usable if SID[k[0]] in TRAIN[sp_no]]
    n_p = len(pieces)
    okc = n_p == WANT_PIECE[sp_no]
    print('\n' + '=' * 78)
    print(f'분할 {sp_no}  소재 {n_p}장  '
          f'[검산] 2층의 {WANT_PIECE[sp_no]}장과 {"일치" if okc else "**불일치**"}')
    print('  출처  ' + ' · '.join(f'{k} {sum(1 for t in pieces if t[0] == k)}'
                                  for k in KEYS if any(t[0] == k for t in pieces)))
    assert okc, f'분할 {sp_no} 소재 수가 2층과 다름'

    root = f'{WORK}/split{sp_no}'
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(f'{root}/images/train');  os.makedirs(f'{root}/labels/train')

    porder = list(np.random.default_rng(SEED).permutation(n_p))
    border = list(np.random.default_rng(SEED).permutation(len(bgs)))
    rng = np.random.default_rng(SEED)

    made, t, tries, skipped = 0, 0, 0, 0
    man = []
    bg_cache = (None, None)
    while made < NPER and t < NPER * 40:
        pk = pieces[porder[t % n_p]]
        bp = bgs[border[t % len(bgs)]]
        t += 1
        a8 = alpha8(*pk)
        lo, hi = size_range(a8)
        if bg_cache[0] != bp:
            bg_cache = (bp, np.asarray(Image.open(bp).convert('RGB')))
        bg = bg_cache[1]

        hit = None
        for _ in range(RETRY):
            tries += 1
            s = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo < hi else hi
            pw = max(int(round(a8.shape[1] * s)), 4)
            ph = max(int(round(a8.shape[0] * s)), 4)
            if pw > W or ph > H:
                continue
            al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS),
                            np.float32) / 255.0
            al[al < THR] = 0
            b = box_q(al)
            if b is None:
                continue
            sub = al[b[1]:b[3], b[0]:b[2]]
            bh, bw = sub.shape
            if min(bw, bh) < MINSIDE:
                continue
            yy = int(rng.integers(0, max(H - ph, 0) + 1))
            xx = int(rng.integers(0, max(W - pw, 0) + 1))
            y0, x0 = yy + b[1], xx + b[0]
            patch = bg[y0:y0 + bh, x0:x0 + bw].astype(np.float32).mean(2)
            chg = float(((255.0 - patch) * sub).mean())   # 합성물의 |변화| 와 같은 값. size_rule2 와 같은 식
            fl = floor_at(bw * bh)
            if chg >= fl:
                hit = (s, al, pw, ph, yy, xx, y0, x0, bw, bh, chg, fl)
                break
        if hit is None:
            skipped += 1
            continue

        s, al, pw, ph, yy, xx, y0, x0, bw, bh, chg, fl = hit
        img = bg.copy()
        reg = img[yy:yy + ph, xx:xx + pw].astype(np.float32)
        img[yy:yy + ph, xx:xx + pw] = np.clip(
            reg * (1 - al[..., None]) + 255.0 * al[..., None], 0, 255).astype(np.uint8)

        name = f's{sp_no}_{made:04d}'
        Image.fromarray(img).save(f'{root}/images/train/{name}.jpg', quality=JPEGQ)
        cx, cy = (x0 + bw / 2) / W, (y0 + bh / 2) / H
        with open(f'{root}/labels/train/{name}.txt', 'w') as f:
            f.write(f'0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}\n')
        man.append({'name': name, 'key': pk[0], 'file': pk[1],
                    'bg': os.path.basename(bp), 's': s,
                    'pos': [xx, yy], 'box': [x0, y0, x0 + bw, y0 + bh],
                    'chg': chg, 'floor': fl})
        made += 1

    open(f'{root}/data.yaml', 'w').write(
        f'path: {os.path.abspath(root)}\ntrain: images/train\nval: images/train\n'
        f'nc: 1\nnames: [\'smoke\']\n')
    json.dump({'split': sp_no, 'n_piece': n_p, 'made': made, 'skipped': skipped,
               'draws': tries, 'seed': SEED, 'jpegq': JPEGQ, 'rows': man},
              open(f'{root}/manifest.json', 'w'), ensure_ascii=False, default=float)

    pair = {(r['file'], r['bg']) for r in man}
    used_p = len({r['file'] for r in man});  used_b = len({r['bg'] for r in man})
    cnt = {}
    for r in man:
        cnt[r['file']] = cnt.get(r['file'], 0) + 1
    print(f'  만든 장수 {made} · 버린 소재 자리 {skipped} · 뽑기 {tries}회 '
          f'({time.time() - t0:.0f}초)')
    if made != NPER:
        print(f'  **{NPER}장을 못 채웠음 — 이 분할은 못 씀**')
    oob = [r for r in man if r['box'][0] < 0 or r['box'][1] < 0
           or r['box'][2] > W or r['box'][3] > H]
    print(f'  [검산] 라벨 상자가 화면 밖으로 나간 장 '
          f'{"없음" if not oob else f"**{len(oob)}장 — 못 씀**"}')
    print(f'  [검산] 같은 (소재, 배경) 짝이 두 번 나온 적 '
          f'{"없음" if len(pair) == made else f"**{made - len(pair)}회 — 2층 4단계와 어긋남**"}')
    print(f'  쓴 소재 {used_p}/{n_p}종 · 쓴 배경 {used_b}/{len(bgs)}종 · '
          f'소재당 재사용 {made / max(used_p, 1):.2f}회 (최소 {min(cnt.values())} · 최대 {max(cnt.values())})')
    ch = np.array([r['chg'] for r in man]);  flr = np.array([r['floor'] for r in man])
    ss = np.array([r['s'] for r in man])
    print(f'  변화 중앙 {np.median(ch):.2f} · 바닥 중앙 {np.median(flr):.2f} · '
          f'바닥 밑인 장 {int((ch < flr).sum())}장')
    print(f'  배율 최소 {ss.min():.3f} · 중앙 {np.median(ss):.3f} · 최대 {ss.max():.3f}')
    summary[sp_no] = {'made': made, 'skipped': skipped, 'used_p': used_p,
                      'reuse': made / max(used_p, 1)}

    z = shutil.make_archive(f'{DOUT}/split{sp_no}', 'zip', root)
    print(f'  -> {z}  ({os.path.getsize(z) / 1e6:.0f} MB)')

print('\n' + '=' * 78)
print('요약')
print('=' * 78)
print(f'  {"분할":>5}{"만든 장":>9}{"버린 자리":>11}{"쓴 소재":>9}{"소재당 재사용":>14}')
for k, v in summary.items():
    print(f'  {k:>5}{v["made"]:>9}{v["skipped"]:>11}{v["used_p"]:>9}{v["reuse"]:>14.2f}')
print(f'\n  2층이 맞춘 값 — 소재당 재사용 476/67 = {476 / 67:.2f} '
      f'(화재 저장소 1400/197 = {1400 / 197:.2f})')
print(f'  분할마다 소재 수가 다르므로 **재사용 횟수는 분할마다 다름.** 2층 한계에 적혀 있음')
print(f'  2층 4단계는 67 과 178 이 서로소인 것을 근거로 들었으나, 소재 수가 다른 분할도')
print(f'  lcm(소재수, 178) 이 {NPER} 보다 크므로 같은 짝이 안 나옴. 위 분할별 검산이 실제로 확인함')
print(f'\n  압축본은 {DOUT} 에 있음. 작업본은 {WORK} 에 있고 런타임이 끊기면 사라짐')
