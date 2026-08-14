# ===== 크기 규칙을 정하고 제외를 다시 판정함 =====
#
# 3회차의 제외 판정(175 → 70장)은 **가로폭을 0.30 으로 고정해 잰 것**이었고,
# 크기를 바꾸면 07 은 0.48배 · p2 는 0.63배로 흔들렸음(가장 큰 어긋남 51.6%).
# 원인은 **축소가 잡음을 지우는 것** — 07 은 알파 덩어리가 4149개라 줄이면 사라짐.
#
# 그래서 판정하는 자리를 옮김.
#
#   버린 방법   `원본 알파에서 판정` — 매트에 대해서는 옳으나 **쓰임에 대해서는 틀림.**
#               07 을 가로 0.3 으로 놓으면 모델이 보는 것은 축소된 알파인데,
#               원본에서 판정하면 실제로 쓸 때 멀쩡한 소재가 통째로 잘림
#   쓰는 방법   **합성한 장마다 판정.** 크기를 뽑아 얹고, 그 장에서 변화를 재고,
#               문턱 아래면 그 장을 안 씀. 재는 알파가 곧 모델이 보는 알파임
#
# `제외된 소재` 는 고정 목록이 아니라 **어떤 크기로도 못 통과한 소재**가 됨.
#
# ---------------------------------------------------------------------------
# 크기 규칙 — 임의로 정한 것은 분포 모양 하나뿐임
#
#   하한   학습 설정이 **imgsz 640** 으로 고정돼 있음(사전 등록). 1920 배경을 640 으로
#          줄이면 3배 축소. YOLO 의 가장 촘촘한 격자가 stride 8 이므로 640 화면에서
#          8화소 미만은 격자 한 칸에도 못 미침 → **라벨 상자 짧은 변 ≥ 24화소(1920 기준)**
#   상한   **배율 1.0.** 축소는 사전 등록이 이미 허용하지만(imgsz 640) **확대는 근거가 없고**
#          알파 경계를 뭉갬. 화면 밖으로 나가는 것도 함께 막음
#   분포   그 사이 **로그 균등** ← 이것만 제가 정한 것이고 근거가 없음.
#          작은 쪽과 큰 쪽에 비슷한 수의 장이 가게 하려는 것뿐임
#   위치   무작위 (논거로 정함 — 모델이 자리를 단서로 못 쓰게 함)
#   개수   한 장에 하나 (가로나 세로가 화면 절반을 넘는 소재가 70/70장 이라 겹침이 불가피)
#
# 소재마다 하한이 다르고 그 아래로는 안 놓으므로,
# **작게 놓아 안 보이는데 상자만 있는 거짓 라벨이 구조적으로 안 생김.**
#
# ---------------------------------------------------------------------------
# 미리 적는 예측 — 결과와 코드가 대조함
#
#   07 · p2   작은 쪽에서 통과하고 **큰 쪽이 잘릴 것**
#   m3        **어떤 크기로도 못 통과할 것.** 통과하면 지표가 잡음 위에 상자를 씌운 것이므로
#             통과한 m3 를 반드시 그림으로 내어 눈으로 봄
#   나머지 셋  범위가 거의 안 잘릴 것 (고정/원본 비가 1.00\~1.04 였음)
#
# 이 시험이 **못 재는 것**
#   학습이 되는가 · 김이 화면에서 차지하는 크기(11곳 중 2곳만 판을 잡을 수 있음)
#   확대를 막았으므로 큰 연기는 원본이 큰 출처에서만 나옴 — 출처와 크기가 얽힘
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 15~25분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
BGDIR = f'{SRC}/steam/bg'
MAT   = f'{SRC}/matte'
OUT   = f'{SRC}/synth_trial'
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SEED  = 1
THR   = 0.06
IMGSZ = 640          # 사전 등록 학습 설정
STRIDE = 8           # YOLO 의 가장 촘촘한 격자
K     = 12           # 소재마다 뽑아 볼 크기 수
NPAIR = 24           # 잡음 바닥을 잴 프레임 쌍
GRID  = 3

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

print('=' * 76)
print('미리 못 박은 규칙 — 아래 숫자가 나오기 전에 정한 것임')
print('=' * 76)
print('  판정      **합성한 장마다** 변화를 재고 문턱 아래면 그 장을 안 씀')
print('  문턱      잡음 바닥 — 개원중 프레임 쌍의 **같은 크기 상자** 안 평균 차의 중앙값')
print('  하한      라벨 상자 짧은 변 ≥ 24화소(1920 기준) — imgsz 640 · stride 8 이 정함')
print('  상한      배율 1.0 — 확대는 근거가 없음')
print('  분포      로그 균등  ← 이것만 임의임')
print('  위치      무작위 · 개수 한 장에 하나')
print('예측  07·p2 는 큰 쪽이 잘림 · m3 는 전멸 · 나머지 셋은 거의 안 잘림')
print('=' * 76)


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


# ---------------------------------------------------------------------------
# 1. 배경
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(np.asarray(Image.open(p).convert('L'), np.float32))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
bg0 = np.asarray(Image.open(bgs[0]).convert('RGB'), np.float32)
H, W = bg0.shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ))        # 1920 기준 24화소
print(f'\n배경 {len(bgs_all)}장 -> 서로 다름 {len(bgs)}장 · {W}x{H}')
print(f'라벨 상자 짧은 변 하한 {MINSIDE}화소  (= stride {STRIDE} x {W}/{IMGSZ})')

NBG = 8
idx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
bgimgs = [np.asarray(Image.open(bgs[i]).convert('RGB'), np.float32) for i in idx]

# ---------------------------------------------------------------------------
# 2. 잡음 바닥 — **상자 크기마다** 잼. 크기에 따라 달라지는지도 봄
# ---------------------------------------------------------------------------
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
diffs = []
for i in pair_idx:
    f1 = np.asarray(Image.open(bgs_all[i]).convert('RGB'), np.float32)
    f2 = np.asarray(Image.open(bgs_all[i + 1]).convert('RGB'), np.float32)
    if f1.shape == f2.shape:
        diffs.append(np.abs(f2 - f1).mean(2))

print(f'\n[잡음 바닥]  프레임 쌍 {len(diffs)}개 × 자리 {GRID*GRID}곳 · 상자 크기마다')
print(f'{"상자":>13}{"바닥(계조)":>12}{"표본":>7}')
print('-' * 34)
FA, FV = [], []
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
    print(f'{f"{bw}x{bh}":>13}{FV[-1]:>12.3f}{len(v):>7}')
print('-' * 34)
spread = (max(FV) - min(FV)) / max(np.median(FV), 1e-9)
print(f'  바닥의 최대-최소 폭이 중앙값의 {spread:.1%}   '
      f'{"크기에 거의 안 달림 — 한 값으로 써도 됨" if spread < 0.15 else "**크기에 달림 — 상자 크기마다 다른 바닥을 씀**"}')
print(f'  3회차가 쓴 값 4.446 (480x312 에서 잰 것)')


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


# ---------------------------------------------------------------------------
# 3. 소재 — 원본 알파를 한 번만 읽음
# ---------------------------------------------------------------------------
mats = [(k, p) for k in KEYS for p in sorted(glob.glob(f'{MAT}/{k}/*.png'))]
pieces = []
for k, p in mats:
    a8 = np.asarray(Image.open(p))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    b = box_q(a8.astype(np.float32) / 255.0)
    if b is None:
        continue
    short0 = min(b[2] - b[0], b[3] - b[1])
    ph0, pw0 = a8.shape
    smax = min(1.0, W / pw0, H / ph0)
    smin = MINSIDE / max(short0, 1)
    pieces.append({'key': k, 'file': os.path.basename(p), 'path': p, 'a8': a8,
                   'pw0': pw0, 'ph0': ph0, 'short0': short0,
                   'smin': smin, 'smax': smax})
print(f'\n소재 {len(pieces)}장 (전수)')

bad = [x for x in pieces if x['smin'] > x['smax']]
print(f'  [검산] 하한이 상한보다 큰 소재 {len(bad)}장   '
      f'{"통과" if not bad else "**이 소재는 어떤 크기로도 못 놓음**"}')
for x in bad[:5]:
    print(f'    {x["key"]:<9}{x["file"]:<22}하한 {x["smin"]:.3f} > 상한 {x["smax"]:.3f}')

# ---------------------------------------------------------------------------
# 4. 소재마다 K 개 크기를 뽑아 얹고 장마다 판정
# ---------------------------------------------------------------------------
print(f'\n[판정]  소재마다 크기 {K}개를 로그 균등으로 뽑아 무작위 자리에 얹고 잼')
short_bad = 0
for x in pieces:
    x['draw'] = []
    if x['smin'] > x['smax']:
        continue
    ss = np.exp(rng.uniform(np.log(x['smin']), np.log(x['smax']), K))
    for s in ss:
        pw = max(int(round(x['pw0'] * s)), 4)
        ph = max(int(round(x['ph0'] * s)), 4)
        if pw > W or ph > H:
            continue
        al = np.asarray(Image.fromarray(x['a8']).resize((pw, ph), Image.LANCZOS),
                        np.float32) / 255.0
        al[al < THR] = 0
        b = box_q(al)
        if b is None:
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        if min(bw, bh) < MINSIDE:                 # 하한 규칙
            short_bad += 1
            continue
        sub = al[b[1]:b[3], b[0]:b[2]]
        bi = bgimgs[int(rng.integers(0, len(bgimgs)))]
        yy = int(rng.integers(0, max(H - ph, 1)));  xx = int(rng.integers(0, max(W - pw, 1)))
        patch = bi[yy + b[1]:yy + b[3], xx + b[0]:xx + b[2]]
        chg = float(((255.0 - patch) * sub[..., None]).mean())
        fl = floor_at(bw * bh)
        x['draw'].append({'s': float(s), 'wr': bw / W, 'hr': bh / H,
                          'chg': chg, 'floor': fl, 'ok': chg >= fl,
                          'pos': [xx, yy], 'box': [int(v) for v in b]})

print(f'  하한 미달로 버린 뽑기 {short_bad}개  (라벨 상자 짧은 변 < {MINSIDE}화소)')
inr = all(x['smin'] - 1e-9 <= d['s'] <= x['smax'] + 1e-9 for x in pieces for d in x['draw'])
print(f'  [검산] 뽑은 배율이 범위 안인가 — {"통과" if inr else "**실패**"}')
sb = min([min(d['wr'] * W, d['hr'] * H) for x in pieces for d in x['draw']] or [999])
print(f'  [검산] 통과 뽑기의 상자 짧은 변 최솟값 {sb:.0f}화소 ≥ {MINSIDE}   '
      f'{"통과" if sb >= MINSIDE else "**실패**"}')

# ---------------------------------------------------------------------------
# 5. 출처별 결과
# ---------------------------------------------------------------------------
print(f'\n{"출처":<9}{"장":>5}{"쓸수있음":>9}{"통과율":>8}{"통과 배율 최소~최대":>22}'
      f'{"통과 가로비 최소~최대":>24}')
print('-' * 78)
usable = {}
for k in KEYS:
    ps = [x for x in pieces if x['key'] == k]
    ok = [x for x in ps if any(d['ok'] for d in x['draw'])]
    usable[k] = len(ok)
    dr = [d for x in ps for d in x['draw']]
    pd = [d for d in dr if d['ok']]
    rate = len(pd) / max(len(dr), 1)
    if pd:
        s1 = f'{min(d["s"] for d in pd):.3f}~{max(d["s"] for d in pd):.3f}'
        w1 = f'{min(d["wr"] for d in pd):.3f}~{max(d["wr"] for d in pd):.3f}'
    else:
        s1 = w1 = '—'
    print(f'{k:<9}{len(ps):>5}{len(ok):>9}{rate:>8.1%}{s1:>22}{w1:>24}')
print('-' * 78)
print(f'{"합":<9}{len(pieces):>5}{sum(usable.values()):>9}')

never = [x for x in pieces if x['draw'] and not any(d['ok'] for d in x['draw'])]
print(f'\n어떤 크기로도 못 통과한 소재 {len(never)}장')
for k in KEYS:
    n = sum(1 for x in never if x['key'] == k)
    if n:
        print(f'  {k:<9}{n}장')

# ---------------------------------------------------------------------------
# 6. 예측 대조
# ---------------------------------------------------------------------------
print('\n[예측 대조]  결과를 보기 전에 적어 둔 것')
def cut_hi(k):
    """큰 쪽이 잘렸는가 — 통과한 배율의 최댓값이 상한에 못 미치는가"""
    ps = [x for x in pieces if x['key'] == k]
    pd = [d for x in ps for d in x['draw'] if d['ok']]
    if not pd:
        return None
    smx = max(x['smax'] for x in ps)
    return max(d['s'] for d in pd) / smx


for k, want in (('07', '큰 쪽 잘림'), ('p2', '큰 쪽 잘림'),
                ('m3', '전멸'), ('kfire03', '거의 안 잘림'),
                ('q1', '거의 안 잘림'), ('j04', '거의 안 잘림')):
    r = cut_hi(k)
    if want == '전멸':
        got = '전멸' if usable[k] == 0 else f'**{usable[k]}장 통과**'
    elif r is None:
        got = '전멸'
    else:
        got = f'통과 최대배율 / 상한 = {r:.2f}'
    mark = ''
    if want == '전멸' and usable[k] > 0:
        mark = '   ← **예측과 어긋남. 통과한 것을 눈으로 볼 것**'
    print(f'  {k:<9}예측 {want:<12}얻음 {got}{mark}')

SPLIT = {1: ['m3', 'kfire03', 'j04', '07'],
         2: ['m3', 'kfire03', 'q1', 'j04', '07', 'p2'],
         3: ['p2', 'j04'], 4: ['kfire03', 'q1', 'j04', '07'],
         5: ['m3', 'kfire03']}
print('\n분할마다 쓸 수 있는 소재 (상한 32장 적용)')
for s, ks in SPLIT.items():
    print(f'  {s}번  {" · ".join(ks):<44}{sum(min(usable[k], 32) for k in ks):>4}장')

json.dump({'minside': MINSIDE, 'floor_area': FA, 'floor_val': FV, 'seed': SEED,
           'pieces': [{'key': x['key'], 'file': x['file'], 'smin': x['smin'],
                       'smax': x['smax'], 'draw': x['draw']} for x in pieces]},
          open(f'{OUT}/size_rule.json', 'w'), ensure_ascii=False)
print(f'\n-> {OUT}/size_rule.json')

# ---------------------------------------------------------------------------
# 7. 시트 — 통과 예시(크기를 흩어) + **통과한 m3 는 반드시**
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


F = korean_font(24)


def render(x, d):
    im = Image.open(x['path']).convert('RGBA')
    pw = max(int(round(x['pw0'] * d['s'])), 4);  ph = max(int(round(x['ph0'] * d['s'])), 4)
    q = np.asarray(im.resize((pw, ph), Image.LANCZOS), np.float32)
    al = q[..., 3] / 255.0;  al[al < THR] = 0
    xx, yy = d['pos']
    o = bgimgs[0].copy()
    o[yy:yy + ph, xx:xx + pw] = o[yy:yy + ph, xx:xx + pw] * (1 - al[..., None]) + 255.0 * al[..., None]
    b = d['box']
    return o, (b[0] + xx, b[1] + yy, b[2] + xx, b[3] + yy)


def sheet(items, path, cw=1100):
    Ht = sum(round(cw * im.shape[0] / im.shape[1]) + 40 for _, im, _ in items) + 8
    sh = Image.new('RGB', (cw, Ht), (16, 16, 16));  d = ImageDraw.Draw(sh);  y = 0
    for lab, im, bx in items:
        h0, w0 = im.shape[:2];  ch = round(cw * h0 / w0)
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS)
        dd = ImageDraw.Draw(pic);  s = cw / w0
        if bx:
            dd.rectangle([bx[0]*s, bx[1]*s, bx[2]*s, bx[3]*s], outline=(120, 255, 120), width=3)
        d.text((6, y + 8), lab, fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 40));  y += ch + 40
    sh.save(path, quality=90);  return path


rows = []
for k in KEYS:
    cand = [(x, d) for x in pieces if x['key'] == k for d in x['draw'] if d['ok']]
    if not cand:
        continue
    cand.sort(key=lambda t: t[1]['s'])
    for j in (0, len(cand) // 2, len(cand) - 1):        # 작은 · 가운데 · 큰
        x, d = cand[j]
        o, bx = render(x, d)
        rows.append((f'{k} {x["file"]}  배율 {d["s"]:.3f} · 가로비 {d["wr"]:.3f} · '
                     f'변화 {d["chg"]:.2f} ≥ 바닥 {d["floor"]:.2f}', o, bx))
p1 = sheet(rows, f'{OUT}/_size_pass.jpg')
print(f'-> {p1}   (출처마다 작은·가운데·큰 셋)')
files.download(p1)

m3ok = [(x, d) for x in pieces if x['key'] == 'm3' for d in x['draw'] if d['ok']]
if m3ok:
    m3ok.sort(key=lambda t: -t[1]['chg'])
    rows2 = []
    for x, d in m3ok[:5]:
        o, bx = render(x, d)
        rows2.append((f'**예측과 어긋남** m3 {x["file"]}  배율 {d["s"]:.3f} · '
                      f'변화 {d["chg"]:.2f} ≥ 바닥 {d["floor"]:.2f}', o, bx))
    p2 = sheet(rows2, f'{OUT}/_m3_pass.jpg')
    print(f'-> {p2}   **m3 가 통과했음. 연기인지 잡음인지 눈으로 볼 것**')
    files.download(p2)
else:
    print('m3 는 어떤 크기로도 통과하지 못했음 — 예측대로임')

print('\n**이 시험도 학습 자료를 만들지 않음.** 2층을 닫고 나서 본 합성을 돌림.')
