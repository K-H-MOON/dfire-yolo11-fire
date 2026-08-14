# ===== 배경 적응 틴트를 잼 — 연기색을 255 로 고정한 것이 맞는가 =====
#
# 화재 저장소 `web3d/PREREGISTER_SMOKE.md` 원문.
#
#   배경이 밝으면 검댕 연기 쪽으로 기울인다. **흰 연기는 흰 급식실에서 보이지 않아
#   라벨만 남고 신호가 없는 학습 예제가 되기 때문이다.**
#
# 우리는 연기색을 **255 고정**으로 얹고 배경은 **밝은 스테인리스 급식실**임.
# 175장 중 107장이 제외된 것이 그 증상일 수 있음.
#
# 그리고 우리 사전 등록 118\~122줄이 오려내기에서 세 출처를 버린 사유가
# `**밝은 배경이라 연기가 배경보다 안 밝음**`(j12 은박 · 05 밝은 타일 · ft 콘크리트) 임 —
# **같은 물리의 반대편**임. 오려낼 때는 그 이유로 버려 놓고 얹을 때는 흰색으로 얹고 있음.
#
# ---------------------------------------------------------------------------
# 재는 방법 — **같은 뽑기에서 색만 바꿈**
#
# 크기·자리·소재를 `colab_size_rule2.py` 와 **같은 시드로 그대로 되살리고**,
# 연기색만 갈아 끼워 변화를 다시 잼. 그래야 색의 효과만 남음.
#
#   흰색 255      지금 쓰는 것
#   고정 180 · 128 · 80    회색\~검댕
#   배경대비 D    연기색을 배경보다 **항상 D 만큼** 떨어뜨림. 변화가 배경과 무관해지나,
#                **배경이 어두운 곳에서는 오히려 대비를 줄임**(흰색이면 255−bg = 150 인데
#                D=60 이면 60). 견주기용으로만 둠
#   밝으면검댕 C  배경이 어두우면 흰색 255 · 밝으면 검댕 C. **화재 저장소 문장이
#                `배경이 밝으면 검댕 쪽으로 기운다` 이므로 어두운 곳에서는 흰색을
#                유지하는 것이 그 뜻에 맞음.** 예행에서 이걸 빠뜨린 것을 잡았음
#
# ---------------------------------------------------------------------------
# **미리 못 박는 읽는 기준** — 숫자가 나오기 전에 정함
#
#   m3 는 **눈으로 확인한 `연기 없음` 대조군**임(`_m3.jpg` 여덟 장 · `_m3_pass.jpg` 다섯 장
#   어디에도 연기가 없었음). 그러므로
#
#       **m3 가 통과하면 그 틴트는 참 소재가 아니라 잡음까지 살린 것으로 봄**
#
#   고르는 기준 — **m3 통과 0장을 지키는 틴트 중 나머지 출처의 통과 소재가 가장 많은 것**
#   그 틴트로도 눈에 연기처럼 보이는지는 **시트로 따로 봄.** 숫자로 못 정함
#
# ---------------------------------------------------------------------------
# 이 시험이 **못 재는 것**
#   틴트가 학습에 좋은가 — 학습을 돌려야 앎
#   **틴트는 학습 자료를 실제와 다르게 만듦.** 평가 대상(실제 연기·김)은 배경 대비가
#   보장돼 있지 않음. 도메인 차이를 만드는 조작이고, 화재 저장소도 이 조작의 효과를
#   따로 재지는 않았음(단일 변수가 아니었음)
#   큰 상자에서 평균이 희석되는 문제는 그대로임
#
# 계산을 줄이려고 상자가 15만 화소를 넘으면 **성기게 골라 평균**을 냄(오차 1% 미만).
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 15~25분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
BGDIR = f'{SRC}/steam/bg'
MAT   = f'{SRC}/matte'
OUT   = f'{SRC}/synth_trial'
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SEED  = 1                      # size_rule2 와 **같아야** 뽑기가 되살아남
THR   = 0.06
IMGSZ, STRIDE = 640, 8
UMIN  = 0.30
K     = 8
NBG, GRID = 4, 3
NPAIR = 24
MAXPX = 150_000                # 상자가 이보다 크면 성기게 고름

TINTS = [('흰색 255 (지금)', 'fix', 255), ('고정 180', 'fix', 180),
         ('검댕 80', 'fix', 80), ('배경대비 60', 'adapt', 60),
         ('밝으면검댕 80', 'flip', 80), ('밝으면검댕 40', 'flip', 40)]
MID = 127.5                    # 밝고 어두움을 가르는 값 (8비트 한가운데)

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')

print('=' * 74)
print('미리 못 박은 읽는 기준 — 숫자가 나오기 전에 정함')
print('=' * 74)
print('  m3 는 눈으로 확인한 **연기 없음 대조군**임')
print('  → **m3 가 통과하면 그 틴트는 잡음까지 살린 것으로 봄**')
print('  고름  m3 통과 0장을 지키는 틴트 중 나머지 출처 통과가 가장 많은 것')
print('  눈    그 틴트가 연기처럼 보이는지는 시트로 따로 봄. 숫자로 못 정함')
print('=' * 74)


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


# ---- 배경 ----
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(np.asarray(Image.open(p).convert('L'), np.float32))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = np.asarray(Image.open(bgs[0]).convert('RGB')).shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ));  PWMIN = int(round(W * UMIN))
idx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
BGM = [np.asarray(Image.open(bgs[i]).convert('RGB'), np.float32).mean(2) for i in idx]
print(f'\n배경 서로 다름 {len(bgs)}장 · 판정에 {NBG}장 · {W}x{H}')

# ---- 크기별 잡음 바닥 (틴트와 무관 — 배경끼리의 차이임) ----
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
diffs = []
for i in pair_idx:
    f1 = np.asarray(Image.open(bgs_all[i]).convert('RGB'), np.float32)
    f2 = np.asarray(Image.open(bgs_all[i + 1]).convert('RGB'), np.float32)
    if f1.shape == f2.shape:
        diffs.append(np.abs(f2 - f1).mean(2))
FA, FV = [], []
for ar in AREAS:
    bw = int(min(W, max(MINSIDE, round(np.sqrt(ar * 16 / 9)))))
    bh = int(min(H, max(MINSIDE, round(ar / bw))))
    v = [float(d[int(max(H-bh,0)*gy/max(GRID-1,1)):int(max(H-bh,0)*gy/max(GRID-1,1))+bh,
                 int(max(W-bw,0)*gx/max(GRID-1,1)):int(max(W-bw,0)*gx/max(GRID-1,1))+bw].mean())
         for d in diffs for gy in range(GRID) for gx in range(GRID)]
    FA.append(float(np.log(bw * bh)));  FV.append(float(np.median(v)))
print(f'잡음 바닥 {min(FV):.2f}\\~{max(FV):.2f} 계조 (상자 크기마다) — **틴트와 무관**')


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


# ---- 소재 · 뽑기 (size_rule2 와 같은 시드·같은 차례) ----
mats = [(k, p) for k in KEYS for p in sorted(glob.glob(f'{MAT}/{k}/*.png'))]
pieces = []
for k, p in mats:
    a8 = np.asarray(Image.open(p))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    if a8.max() == 0:
        continue
    ph0, pw0 = a8.shape
    pieces.append({'key': k, 'file': os.path.basename(p), 'path': p, 'a8': a8,
                   'pw0': pw0, 'ph0': ph0,
                   'smin': PWMIN / pw0, 'smax': min(1.0, W / pw0, H / ph0)})
for x in pieces:
    lo, hi = (x['smax'], x['smax']) if x['smin'] > x['smax'] else (x['smin'], x['smax'])
    x['ss'] = sorted([lo] if lo >= hi else
                     list(np.exp(rng.uniform(np.log(lo), np.log(hi), K))))
print(f'소재 {len(pieces)}장 · 소재마다 크기 {K}개 (시드 {SEED} — size_rule2 와 같음)')

# ---------------------------------------------------------------------------
# 틴트별로 판정
# ---------------------------------------------------------------------------
print(f'\n[재는 중] 소재 {len(pieces)} x 크기 {K} x 자리 {NBG*GRID*GRID} x 틴트 {len(TINTS)}')
res = {t[0]: {'ok': set(), 'chg': []} for t in TINTS}
for x in pieces:
    for s in x['ss']:
        pw = max(int(round(x['pw0'] * s)), 4);  ph = max(int(round(x['ph0'] * s)), 4)
        if pw > W or ph > H:
            continue
        al = np.asarray(Image.fromarray(x['a8']).resize((pw, ph), Image.LANCZOS),
                        np.float32) / 255.0
        al[al < THR] = 0
        b = box_q(al)
        if b is None or min(b[2] - b[0], b[3] - b[1]) < MINSIDE:
            continue
        sub = al[b[1]:b[3], b[0]:b[2]]
        step = max(1, int(np.sqrt(sub.size / MAXPX)))
        sub_s = sub[::step, ::step]
        fl = floor_at((b[2] - b[0]) * (b[3] - b[1]))
        acc = {t[0]: [] for t in TINTS}
        for bm in BGM:
            for gy in range(GRID):
                for gx in range(GRID):
                    yy = int(max(H - ph, 0) * gy / max(GRID - 1, 1))
                    xx = int(max(W - pw, 0) * gx / max(GRID - 1, 1))
                    pt = bm[yy + b[1]:yy + b[3], xx + b[0]:xx + b[2]][::step, ::step]
                    for name, kind, val in TINTS:
                        if kind == 'fix':
                            acc[name].append(float((np.abs(val - pt) * sub_s).mean()))
                        elif kind == 'adapt':       # 배경보다 항상 val 만큼 떨어뜨림
                            acc[name].append(float(val * sub_s.mean()))
                        else:                       # 어두우면 흰색 · 밝으면 검댕 val
                            d = np.where(pt < MID, 255.0 - pt, pt - val)
                            acc[name].append(float((d * sub_s).mean()))
        for name, _, _ in TINTS:
            m = float(np.median(acc[name]))
            res[name]['chg'].append((x['key'], m, fl))
            if m >= fl:
                res[name]['ok'].add((x['key'], x['file']))

# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
tot = {k: sum(1 for x in pieces if x['key'] == k) for k in KEYS}
print(f'\n{"틴트":<16}' + ''.join(f'{k:>9}' for k in KEYS) + f'{"합":>7}{"변화 중앙":>11}')
print('-' * (16 + 9 * len(KEYS) + 18))
print(f'{"(소재 수)":<16}' + ''.join(f'{tot[k]:>9}' for k in KEYS)
      + f'{len(pieces):>7}')
print('-' * (16 + 9 * len(KEYS) + 18))
best = None
for name, _, _ in TINTS:
    ok = res[name]['ok']
    row = [sum(1 for kk, _ in ok if kk == k) for k in KEYS]
    md = float(np.median([c for _, c, _ in res[name]['chg']]))
    mark = ''
    if row[0] > 0:
        mark = '   ← **m3 통과. 잡음까지 살림**'
    else:
        if best is None or sum(row) > best[1]:
            best = (name, sum(row))
    print(f'{name:<16}' + ''.join(f'{v:>9}' for v in row)
          + f'{sum(row):>7}{md:>11.2f}{mark}')
print('-' * (16 + 9 * len(KEYS) + 18))
if best:
    print(f'\n미리 못 박은 기준 적용 -> **{best[0]}**  (m3 0장 · 나머지 {best[1]}장)')
    print(f'  지금(흰색 255)은 {sum(1 for kk, _ in res[TINTS[0][0]]["ok"] if kk != "m3")}장')
else:
    print('\n**m3 통과 0장을 지키는 틴트가 없음 — 틴트를 안 쓰는 것이 맞음**')

json.dump({name: sorted(f'{k}/{f}' for k, f in res[name]['ok']) for name, _, _ in TINTS},
          open(f'{OUT}/tint.json', 'w'), ensure_ascii=False)
print(f'-> {OUT}/tint.json')

# ---------------------------------------------------------------------------
# 시트 — 같은 소재·같은 배경·같은 크기에서 색만 바꿔 나란히
# ---------------------------------------------------------------------------
def korean_font(size):
    c = (glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
         + glob.glob('/usr/share/fonts/**/*Nanum*.ttf', recursive=True)
         + glob.glob('/usr/share/fonts/**/NotoSansCJK*', recursive=True))
    if not c:
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        c = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
    for f in c:
        try:
            ft = ImageFont.truetype(f, size)
            t = Image.new('L', (size * 4, size * 2), 0)
            ImageDraw.Draw(t).text((2, 2), '연기', fill=255, font=ft)
            if np.asarray(t).max() > 0:
                return ft
        except Exception:
            pass
    return ImageFont.load_default()


F = korean_font(22)
BG0 = np.asarray(Image.open(bgs[int(idx[0])]).convert('RGB'), np.float32)
BG0M = BG0.mean(2)

# 볼 것 — 출처마다 하나 + **m3 하나(대조군)**
show = []
for k in ['q1', '07', 'p2', 'm3']:
    c = [x for x in pieces if x['key'] == k]
    if c:
        show.append(c[len(c) // 2])

PANEL, rows = 380, []
for x in show:
    s = x['ss'][len(x['ss']) // 2]
    pw = max(int(round(x['pw0'] * s)), 4);  ph = max(int(round(x['ph0'] * s)), 4)
    im = Image.open(x['path']).convert('RGBA').resize((pw, ph), Image.LANCZOS)
    q = np.asarray(im, np.float32);  al = q[..., 3] / 255.0;  al[al < THR] = 0
    b = box_q(al)
    if b is None:
        continue
    xx = (W - pw) // 2;  yy = int((H - ph) * 0.25)
    panels = []
    for name, kind, val in TINTS:
        o = BG0.copy()
        reg = o[yy:yy+ph, xx:xx+pw]
        bl = BG0M[yy:yy+ph, xx:xx+pw][..., None] * np.ones((1, 1, 3), np.float32)
        if kind == 'fix':
            col = np.full_like(reg, float(val))
        elif kind == 'adapt':
            col = np.where(bl >= val, bl - val, bl + val)
        else:
            col = np.where(bl < MID, 255.0, float(val))
        o[yy:yy+ph, xx:xx+pw] = reg * (1 - al[..., None]) + col * al[..., None]
        cx0, cy0 = max(b[0]+xx-40, 0), max(b[1]+yy-40, 0)
        cx1, cy1 = min(b[2]+xx+40, W), min(b[3]+yy+40, H)
        panels.append((name, o[cy0:cy1, cx0:cx1]))
    rows.append((f'{x["key"]} {x["file"]} · 배율 {s:.3f}'
                 + ('   ← **대조군: 연기 없음**' if x['key'] == 'm3' else ''), panels))

Ht = 0
for _, ps in rows:
    h, w = ps[0][1].shape[:2]
    Ht += round(PANEL * h / w) + 56
sh = Image.new('RGB', (PANEL * len(TINTS), Ht + 8), (16, 16, 16))
dr = ImageDraw.Draw(sh);  y = 0
for lab, ps in rows:
    h, w = ps[0][1].shape[:2];  ch = round(PANEL * h / w)
    dr.text((6, y + 4), lab, fill=(255, 220, 0), font=F)
    for j, (nm, im) in enumerate(ps):
        dr.text((j * PANEL + 6, y + 30), nm, fill=(180, 220, 255), font=F)
        sh.paste(Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
                 .resize((PANEL, ch), Image.LANCZOS), (j * PANEL, y + 56))
    y += ch + 56
path = f'{OUT}/_tint.jpg'
sh.save(path, quality=92)
print(f'\n-> {path}   ({len(rows)}줄 x 틴트 {len(TINTS)}칸 · 같은 소재·배경·크기에서 색만 다름)')
files.download(path)
print('\n**맨 아랫줄이 m3 대조군임.** 거기서 무언가 보이기 시작하면 그 틴트는 잡음을 살린 것임.')
