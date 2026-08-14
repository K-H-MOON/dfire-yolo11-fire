# ===== 크기 규칙 — 눈으로 대조한 조건 아래로는 안 내려감 =====
#
# 앞 스크립트(`colab_size_rule.py`)에서 **m3 51장이 통과했고, 그림을 보니 상자 안에
# 아무것도 없었음.** 배율 0.073\~0.174 로 아주 작게 놓였을 때만 통과했음.
#
#   축소가 흩어진 잡음을 뭉쳐 알파를 부풀리고
#   동시에 작은 상자의 잡음 바닥이 3.0 으로 낮아져
#   **양쪽에서 통과 쪽으로 밀림**
#
# ---------------------------------------------------------------------------
# 제가 제안했던 `R ≤ 1.2` 는 **쓰기 전에 반증됐음**
#
# R = 배율 s 의 상자 안 알파 평균 ÷ 원본의 상자 안 알파 평균.
# `colab_size_check.py` 가 이미 재 놓은 값 — kfire03 1.00 · q1 1.00 · j04 0.96 ·
# **07 2.06 · p2 1.60**.
#
# 07 은 U=0.30 에서 **눈으로 흰 연기가 확인된 소재**임(`_lowpass.jpg` 의 07_00057.0).
# 07·p2 는 큰 쪽에서 잡음 바닥에 걸리므로(원본 변화 3.5·2.1 < 큰 상자 바닥 5.4),
# 작은 쪽을 R 이 막으면 **두 출처가 통째로 사라짐.**
# R 은 `잡음이 뭉친 것` 과 `얇은 가장자리가 문턱 아래로 떨어져 상자가 좁아진 것` 을
# 못 가름. 두 가지가 같은 방향으로 값을 올림. **그래서 안 씀.**
#
# ---------------------------------------------------------------------------
# 이번 규칙 — 결과를 보기 전에 정한 것
#
#   판정      합성한 장마다 · **크기별** 잡음 바닥과 견줌            (필요조건)
#   크기 하한  **가로폭 0.30** (1920 기준 576화소).
#             3회차에서 **눈으로 대조한 유일한 조건**이 이것이고, 그 아래는 한 번도
#             확인한 적이 없음. m3 가 뚫린 경로가 바로 그 미확인 영역이었음
#   크기 상한  배율 1.0 — 확대는 근거가 없고 알파 경계를 뭉갬
#   분포      그 사이 로그 균등  ← 임의
#   위치      무작위 · 개수 한 장에 하나
#   제외      **어떤 크기로도 못 통과한 소재**
#
# 판정하는 자리도 고침 — 앞 스크립트는 뽑기마다 **자리 하나**에서 판정해
# 07 이 배율 0.99 까지 통과한 것이 자리 운인지 소재 차이인지 못 갈랐음.
# 이번에는 **배경 4장 × 자리 9곳의 중앙값**으로 판정함.
#   한계 — 실제 합성은 자리 하나에 놓이므로 **밝은 자리에 놓인 장은 문턱에 가까울 수 있음.**
#
# ---------------------------------------------------------------------------
# 미리 적는 예측 — 코드가 결과와 대조함
#
#   m3         **전멸** (하한이 576화소라 작게 못 감)
#   07 · p2    **큰 쪽이 잘림**
#   kfire03·q1·j04   거의 안 잘림
#   3회차 대조   가로폭 0.30 에서 잰 변화가 **3회차 값과 15% 안으로 맞을 것**
#               (구현이 눈으로 확인된 그 조건을 되살리는지 보는 검산)
#
# 이 시험이 **못 재는 것**
#   학습이 되는가 · 김이 화면에서 차지하는 크기(11곳 중 2곳만 판을 잡을 수 있음)
#   `0.30 이 옳은 하한인가` — 눈으로 대조했을 뿐 재서 정한 값이 아님
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
SEED  = 1
THR   = 0.06
IMGSZ, STRIDE = 640, 8
UMIN  = 0.30          # 눈으로 대조한 조건 = 가로폭 0.30
K     = 8             # 소재마다 뽑아 볼 크기 수
NBG   = 4             # 판정에 쓸 배경 장수
GRID  = 3             # 자리 3x3
NPAIR = 24
TOL   = 0.15          # 3회차 대조 허용 폭

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

print('=' * 78)
print('미리 못 박은 규칙 — 아래 숫자가 나오기 전에 정한 것임')
print('=' * 78)
print('  판정      합성한 장마다 · **크기별** 잡음 바닥과 견줌 (필요조건)')
print(f'  크기 하한  **가로폭 {UMIN}** — 3회차에서 눈으로 대조한 유일한 조건.')
print('            그 아래는 확인한 적이 없고, m3 가 뚫린 곳이 바로 거기였음')
print('  크기 상한  배율 1.0 — 확대는 근거가 없음')
print('  분포      로그 균등  ← 임의')
print(f'  판정 자리  배경 {NBG}장 × 자리 {GRID}x{GRID} 의 **중앙값** (자리 운을 섞지 않으려고)')
print('  제외      어떤 크기로도 못 통과한 소재')
print('예측  m3 전멸 · 07·p2 큰 쪽 잘림 · 나머지 셋 거의 안 잘림')
print(f'      가로폭 {UMIN} 에서 잰 변화가 3회차 값과 {TOL:.0%} 안으로 맞을 것')
print('=' * 78)


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
# 1. 배경 — 서로 다름(해밍 > 0) 에서 고르게 흩어 NBG 장
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(np.asarray(Image.open(p).convert('L'), np.float32))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = np.asarray(Image.open(bgs[0]).convert('RGB')).shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ))
PWMIN = int(round(W * UMIN))
print(f'\n배경 {len(bgs_all)}장 -> 서로 다름 {len(bgs)}장 · {W}x{H}')
print(f'크기 하한 = 가로 {PWMIN}화소 · 라벨 상자 짧은 변 하한 {MINSIDE}화소(보조 검사)')

idx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
# **밝기 평균 화면**만 씀 — 변화는 채널마다 같은 알파를 곱하므로
# `알파 x (255 - 채널평균)` 과 정확히 같음. 3회차와 같은 값이 나옴
BGM = [np.asarray(Image.open(bgs[i]).convert('RGB'), np.float32).mean(2) for i in idx]

# ---------------------------------------------------------------------------
# 2. 잡음 바닥 — 상자 크기마다
# ---------------------------------------------------------------------------
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
diffs = []
for i in pair_idx:
    f1 = np.asarray(Image.open(bgs_all[i]).convert('RGB'), np.float32)
    f2 = np.asarray(Image.open(bgs_all[i + 1]).convert('RGB'), np.float32)
    if f1.shape == f2.shape:
        diffs.append(np.abs(f2 - f1).mean(2))

print(f'\n[잡음 바닥]  프레임 쌍 {len(diffs)}개 × 자리 {GRID*GRID}곳')
print(f'{"상자":>13}{"바닥(계조)":>12}')
print('-' * 27)
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
    print(f'{f"{bw}x{bh}":>13}{FV[-1]:>12.3f}')
print('-' * 27)
print(f'  폭이 중앙값의 {(max(FV)-min(FV))/np.median(FV):.1%} — 크기마다 다른 바닥을 씀')


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


# ---------------------------------------------------------------------------
# 3. 소재 · 크기 범위
# ---------------------------------------------------------------------------
mats = [(k, p) for k in KEYS for p in sorted(glob.glob(f'{MAT}/{k}/*.png'))]
pieces = []
for k, p in mats:
    a8 = np.asarray(Image.open(p))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    if a8.max() == 0:
        continue
    ph0, pw0 = a8.shape
    smax = min(1.0, W / pw0, H / ph0)
    smin = PWMIN / pw0
    pieces.append({'key': k, 'file': os.path.basename(p), 'path': p, 'a8': a8,
                   'pw0': pw0, 'ph0': ph0, 'smin': smin, 'smax': smax,
                   'narrow': smin > smax})
print(f'\n소재 {len(pieces)}장 (전수)')
nar = [x for x in pieces if x['narrow']]
print(f'  [검산] 원본이 {PWMIN}화소보다 좁아 하한을 못 지키는 소재 {len(nar)}장'
      f'{"" if not nar else "   ← 이 소재는 배율 1.0 하나로만 놓되 **확인 안 된 영역**임"}')
for x in nar[:5]:
    print(f'    {x["key"]:<9}{x["file"]:<22}원본 가로 {x["pw0"]}화소')


def chg_at(a8, s):
    """배율 s 에서 (변화 중앙값, 상자, 상자넓이, 가로비). 못 재면 None."""
    pw = max(int(round(a8.shape[1] * s)), 4)
    ph = max(int(round(a8.shape[0] * s)), 4)
    if pw > W or ph > H:
        return None
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS),
                    np.float32) / 255.0
    al[al < THR] = 0
    b = box_q(al)
    if b is None:
        return None
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(bw, bh) < MINSIDE:
        return None
    sub = al[b[1]:b[3], b[0]:b[2]]
    v = []
    for bm in BGM:
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(H - ph, 0) * gy / max(GRID - 1, 1))
                xx = int(max(W - pw, 0) * gx / max(GRID - 1, 1))
                patch = bm[yy + b[1]:yy + b[3], xx + b[0]:xx + b[2]]
                v.append(float(((255.0 - patch) * sub).mean()))
    return float(np.median(v)), [int(t) for t in b], bw * bh, bw / W


# ---------------------------------------------------------------------------
# 4. 검산 — 가로폭 0.30 에서 잰 값이 3회차와 맞는가
# ---------------------------------------------------------------------------
prev = f'{OUT}/synth_trial3.json'
print(f'\n[검산] 가로폭 {UMIN} 에서 잰 변화가 3회차 값과 맞는가')
if os.path.exists(prev):
    old = {(r['key'], r['file']): r['chg'] for r in json.load(open(prev))['records']}
    rat = []
    for x in pieces[::7]:                       # 골고루 25장 안팎
        o = old.get((x['key'], x['file']))
        r = chg_at(x['a8'], PWMIN / x['pw0'])
        if o and o > 0.2 and r:
            rat.append(r[0] / o)
    if rat:
        md = float(np.median(rat))
        print(f'  표본 {len(rat)}장 · 새값/3회차값 중앙 {md:.3f} '
              f'(최소 {min(rat):.3f} · 최대 {max(rat):.3f})')
        print(f'  {"통과 — 구현이 3회차 조건을 되살림" if abs(md-1) <= TOL else "**실패 — 구현이 3회차와 다름. 아래 숫자를 쓰지 말 것**"}')
    else:
        print('  견줄 표본이 없어 검산 못 함')
else:
    print(f'  {prev} 가 없어 검산 못 함')

# ---------------------------------------------------------------------------
# 5. 소재마다 K 개 크기 · 장마다 판정
# ---------------------------------------------------------------------------
print(f'\n[판정]  소재마다 크기 {K}개를 로그 균등으로 · 배경 {NBG} × 자리 {GRID}x{GRID} 중앙값')
nshort = 0
for x in pieces:
    lo, hi = (x['smax'], x['smax']) if x['narrow'] else (x['smin'], x['smax'])
    ss = [lo] if lo >= hi else list(np.exp(rng.uniform(np.log(lo), np.log(hi), K)))
    x['draw'] = []
    for s in sorted(ss):
        r = chg_at(x['a8'], s)
        if r is None:
            nshort += 1;  continue
        chg, b, area, wr = r
        fl = floor_at(area)
        x['draw'].append({'s': float(s), 'wr': wr, 'chg': chg, 'floor': fl,
                          'ok': chg >= fl, 'box': b})
print(f'  상자 짧은 변 미달로 버린 뽑기 {nshort}개')
inr = all((x['smin'] - 1e-9 <= d['s'] <= x['smax'] + 1e-9) or x['narrow']
          for x in pieces for d in x['draw'])
print(f'  [검산] 뽑은 배율이 범위 안인가 — {"통과" if inr else "**실패**"}')

# ---------------------------------------------------------------------------
# 6. 결과
# ---------------------------------------------------------------------------
print(f'\n{"출처":<9}{"장":>5}{"쓸수있음":>9}{"통과율":>8}'
      f'{"통과 배율 최소~최대":>22}{"통과 가로비 최소~최대":>24}')
print('-' * 78)
usable = {}
for k in KEYS:
    ps = [x for x in pieces if x['key'] == k]
    ok = [x for x in ps if any(d['ok'] for d in x['draw'])]
    usable[k] = len(ok)
    dr = [d for x in ps for d in x['draw']];  pd = [d for d in dr if d['ok']]
    s1 = (f'{min(d["s"] for d in pd):.3f}~{max(d["s"] for d in pd):.3f}' if pd else '—')
    w1 = (f'{min(d["wr"] for d in pd):.3f}~{max(d["wr"] for d in pd):.3f}' if pd else '—')
    print(f'{k:<9}{len(ps):>5}{len(ok):>9}{len(pd)/max(len(dr),1):>8.1%}{s1:>22}{w1:>24}')
print('-' * 78)
print(f'{"합":<9}{len(pieces):>5}{sum(usable.values()):>9}')

never = [x for x in pieces if x['draw'] and not any(d['ok'] for d in x['draw'])]
print(f'\n어떤 크기로도 못 통과한 소재 {len(never)}장')
for k in KEYS:
    n = sum(1 for x in never if x['key'] == k)
    if n:
        print(f'  {k:<9}{n}장')

print('\n[예측 대조]  결과를 보기 전에 적어 둔 것')
for k, want in (('m3', '전멸'), ('07', '큰 쪽 잘림'), ('p2', '큰 쪽 잘림'),
                ('kfire03', '거의 안 잘림'), ('q1', '거의 안 잘림'),
                ('j04', '거의 안 잘림')):
    ps = [x for x in pieces if x['key'] == k]
    pd = [d for x in ps for d in x['draw'] if d['ok']]
    if want == '전멸':
        got = '전멸' if not pd else f'**{usable[k]}장 통과**'
        mark = '' if not pd else '   ← **어긋남. 통과한 것을 눈으로 볼 것**'
    else:
        smx = max(x['smax'] for x in ps)
        r = (max(d['s'] for d in pd) / smx) if pd else 0.0
        got = f'통과 최대배율 / 상한 = {r:.2f}' if pd else '전멸'
        mark = ''
    print(f'  {k:<9}예측 {want:<12}얻음 {got}{mark}')

SPLIT = {1: ['m3', 'kfire03', 'j04', '07'],
         2: ['m3', 'kfire03', 'q1', 'j04', '07', 'p2'],
         3: ['p2', 'j04'], 4: ['kfire03', 'q1', 'j04', '07'], 5: ['m3', 'kfire03']}
print('\n분할마다 쓸 수 있는 소재 (상한 32장 적용)')
for s, ks in SPLIT.items():
    print(f'  {s}번  {" · ".join(ks):<44}{sum(min(usable[k], 32) for k in ks):>4}장')

json.dump({'umin': UMIN, 'pwmin': PWMIN, 'seed': SEED,
           'floor_area': FA, 'floor_val': FV,
           'pieces': [{'key': x['key'], 'file': x['file'], 'smin': x['smin'],
                       'smax': x['smax'], 'narrow': bool(x['narrow']),
                       'draw': x['draw']} for x in pieces]},
          open(f'{OUT}/size_rule2.json', 'w'), ensure_ascii=False)
print(f'\n-> {OUT}/size_rule2.json')

# ---------------------------------------------------------------------------
# 7. 시트 — 출처마다 **통과 범위의 양끝**. 양끝이 다 눈에 맞아야 범위를 확정함
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
BGRGB = np.asarray(Image.open(bgs[idx[0]]).convert('RGB'), np.float32)


def render(x, d):
    im = Image.open(x['path']).convert('RGBA')
    pw = max(int(round(x['pw0'] * d['s'])), 4);  ph = max(int(round(x['ph0'] * d['s'])), 4)
    q = np.asarray(im.resize((pw, ph), Image.LANCZOS), np.float32)
    al = q[..., 3] / 255.0;  al[al < THR] = 0
    xx = (W - pw) // 2;  yy = int((H - ph) * 0.25)
    o = BGRGB.copy()
    o[yy:yy+ph, xx:xx+pw] = o[yy:yy+ph, xx:xx+pw] * (1 - al[..., None]) + 255.0 * al[..., None]
    b = d['box']
    return o, (b[0]+xx, b[1]+yy, b[2]+xx, b[3]+yy)


def sheet(items, path, cw=1100):
    Ht = sum(round(cw * im.shape[0] / im.shape[1]) + 40 for _, im, _ in items) + 8
    sh = Image.new('RGB', (cw, Ht), (16, 16, 16));  dr = ImageDraw.Draw(sh);  y = 0
    for lab, im, bx in items:
        h0, w0 = im.shape[:2];  ch = round(cw * h0 / w0)
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS)
        dd = ImageDraw.Draw(pic);  sc = cw / w0
        if bx:
            dd.rectangle([bx[0]*sc, bx[1]*sc, bx[2]*sc, bx[3]*sc], outline=(120, 255, 120), width=3)
        dr.text((6, y + 8), lab, fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 40));  y += ch + 40
    sh.save(path, quality=90);  return path


rows = []
for k in KEYS:
    cand = [(x, d) for x in pieces if x['key'] == k for d in x['draw'] if d['ok']]
    if not cand:
        continue
    cand.sort(key=lambda t: t[1]['s'])
    for tag, (x, d) in (('가장 작은 통과', cand[0]), ('가장 큰 통과', cand[-1])):
        o, bx = render(x, d)
        rows.append((f'{tag}  {k} {x["file"]}  배율 {d["s"]:.3f} · 가로비 {d["wr"]:.3f} · '
                     f'변화 {d["chg"]:.2f} ≥ 바닥 {d["floor"]:.2f}', o, bx))
if rows:
    p1 = sheet(rows, f'{OUT}/_range_ends.jpg')
    print(f'-> {p1}   (출처마다 통과 범위의 **양끝**)')
    files.download(p1)

m3ok = [(x, d) for x in pieces if x['key'] == 'm3' for d in x['draw'] if d['ok']]
if m3ok:
    m3ok.sort(key=lambda t: -t[1]['chg'])
    rows2 = []
    for x, d in m3ok[:5]:
        o, bx = render(x, d)
        rows2.append((f'**예측과 어긋남** m3 {x["file"]}  배율 {d["s"]:.3f} · '
                      f'변화 {d["chg"]:.2f} ≥ 바닥 {d["floor"]:.2f}', o, bx))
    p2 = sheet(rows2, f'{OUT}/_m3_pass2.jpg')
    print(f'-> {p2}   **m3 가 또 통과했음. 눈으로 볼 것**')
    files.download(p2)
else:
    print('m3 는 어떤 크기로도 통과하지 못했음 — 예측대로임')

print('\n**이 시험도 학습 자료를 만들지 않음.** 2층을 닫고 나서 본 합성을 돌림.')
