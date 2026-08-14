# ===== 라벨 상자를 잘라 원본 배율로 봄 — 눈대중을 없앰 =====
#
# `_range_ends.jpg` 는 화면 전체를 1100화소로 줄여 그렸음. 07 의 가장 작은 통과는
# **라벨 상자 가로가 223화소**라 그 시트에서는 60화소도 안 됨.
# 그 크기에서 `옅은 김이 있는가` 를 판정한 것은 눈대중이었음 — 규칙 9 위반.
#
# 여기서는 **상자만 잘라 원본 배율로** 세 칸에 나란히 놓음.
#
#   배경만 (그 자리)  |  합성  |  차이 (줄마다 다른 배수로 증폭 · 배수를 딱지에 적음)
#
# **차이 칸이 결정적임.** 눈으로 못 볼 만큼 옅어도 차이 그림에는 드러나고,
# 그 모양이 **연기 모양이면 참 라벨 · 흩어진 점이면 거짓 라벨**임.
# (m3 를 판정할 때 쓴 눈대중을 이걸로 바꿈)
#
# 앞 시트의 결함도 함께 고침
#   앞 시트는 판정을 **배경 4장 x 자리 9곳 중앙값**으로 해 놓고 그림은 **배경 한 장 ·
#   자리 한 곳**을 그렸음. 딱지의 변화 값이 그 그림의 값이 아니었음.
#   여기서는 36개 자리 중 **중앙값에 가장 가까운 자리**를 골라 그림 — 딱지와 그림이 맞음.
#
# 덤으로 하나 더 — **3회차 70장과 이번 68장 사이에 어느 소재가 뒤집혔는지** 셈.
#
# 이 시험이 **못 재는 것**
#   `사람이 보이면 학습이 된다` — 학습을 돌려야 앎
#   차이 칸의 증폭은 보이라고 곱한 것이지 학습이 보는 값이 아님.
#   **줄마다 배수가 달라 줄끼리 밝기를 견주면 안 됨** — 모양만 볼 것
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

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
THR   = 0.06
NBG, GRID = 4, 3          # size_rule2 와 **같은 값**이어야 자리가 되살아남
MARGIN = 0.20             # 상자 둘레로 이만큼 더 잘라 냄
PANEL  = 520              # 칸 하나의 최대 가로 화소
AMPMAX = 50               # 차이 증폭의 상한 (줄마다 다르게 씀)

drive.mount('/content/drive')


def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(g, size=8):
    x = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size + 1, size),
                                                              Image.LANCZOS), np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


# ---------------------------------------------------------------------------
# 1. 배경 — size_rule2 와 같은 방식으로 되살림
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(np.asarray(Image.open(p).convert('L'), np.float32))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = np.asarray(Image.open(bgs[0]).convert('RGB')).shape[:2]
idx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
BGRGB = [np.asarray(Image.open(bgs[i]).convert('RGB'), np.float32) for i in idx]
print(f'배경 서로 다름 {len(bgs)}장 · 판정에 쓴 {NBG}장을 되살림 (index {[int(i) for i in idx]})')

# ---------------------------------------------------------------------------
# 2. 3회차 70장과 이번 68장 대조 — 어느 소재가 뒤집혔는가
# ---------------------------------------------------------------------------
p3 = f'{OUT}/synth_trial3.json'
p4 = f'{OUT}/size_rule2.json'
J4 = json.load(open(p4))
new_ok = {(x['key'], x['file']) for x in J4['pieces'] if any(d['ok'] for d in x['draw'])}
print(f'\n[대조] 3회차(고정 0.30 · 단일 바닥) vs 이번(장마다 · 크기별 바닥)')
if os.path.exists(p3):
    J3 = json.load(open(p3))
    fl3 = J3['floor']
    old_ok = {(r['key'], r['file']) for r in J3['records'] if r['chg'] >= fl3}
    both = old_ok & new_ok
    only3 = old_ok - new_ok
    only4 = new_ok - old_ok
    print(f'  3회차 통과 {len(old_ok)}장 · 이번 통과 {len(new_ok)}장 · 양쪽 다 {len(both)}장')
    print(f'  3회차만 통과 {len(only3)}장 · 이번만 통과 {len(only4)}장')
    for tag, s in (('3회차만', only3), ('이번만', only4)):
        for k, f in sorted(s)[:12]:
            print(f'    {tag:<7}{k:<9}{f}')
    print(f'  **뒤집힌 소재가 {len(only3)+len(only4)}장뿐이면 답이 방법 선택에 안 흔들린 것임**')
else:
    print(f'  {p3} 가 없어 대조 못 함')

# ---------------------------------------------------------------------------
# 3. 범위 양끝 고르기 — 출처마다 가장 작은 통과 · 가장 큰 통과
# ---------------------------------------------------------------------------
picks = []
for k in KEYS:
    cand = [(x, d) for x in J4['pieces'] if x['key'] == k
            for d in x['draw'] if d['ok']]
    if not cand:
        print(f'{k}: 통과한 뽑기가 없음 — 건너뜀');  continue
    cand.sort(key=lambda t: t[1]['s'])
    picks.append(('가장 작은 통과', k, cand[0][0], cand[0][1]))
    picks.append(('가장 큰 통과',   k, cand[-1][0], cand[-1][1]))
print(f'\n고른 뽑기 {len(picks)}개 (출처 {len(picks)//2}곳 x 양끝)')


def load_alpha(k, f):
    a8 = np.asarray(Image.open(f'{MAT}/{k}/{f}'))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    return a8


def place_median(a8, s):
    """36개 자리에서 변화를 재고 **중앙값에 가장 가까운 자리**를 돌려줌."""
    pw = max(int(round(a8.shape[1] * s)), 4);  ph = max(int(round(a8.shape[0] * s)), 4)
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS),
                    np.float32) / 255.0
    al[al < THR] = 0
    ys, xs = np.nonzero(al)
    ry, rx = al.sum(1), al.sum(0)
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, 0.005)), int(np.searchsorted(c, 0.995)) + 1
    y0, y1 = span(ry);  x0, x1 = span(rx)
    b = (x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))
    sub = al[b[1]:b[3], b[0]:b[2]]
    cand = []
    for bi, bg in enumerate(BGRGB):
        bm = bg.mean(2)
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(H - ph, 0) * gy / max(GRID - 1, 1))
                xx = int(max(W - pw, 0) * gx / max(GRID - 1, 1))
                v = float(((255.0 - bm[yy + b[1]:yy + b[3], xx + b[0]:xx + b[2]]) * sub).mean())
                cand.append((v, bi, xx, yy))
    med = float(np.median([c[0] for c in cand]))
    v, bi, xx, yy = min(cand, key=lambda c: abs(c[0] - med))
    return al, b, bi, xx, yy, v, med


# ---------------------------------------------------------------------------
# 4. 잘라 세 칸으로
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
rows, worst = [], 0.0
print(f'\n{"":<15}{"출처":<9}{"배율":>7}{"상자(화소)":>13}{"저장된 변화":>12}'
      f'{"그림의 변화":>12}{"어긋남":>8}')
print('-' * 78)
for tag, k, x, d in picks:
    a8 = load_alpha(k, x['file'])
    al, b, bi, xx, yy, v, med = place_median(a8, d['s'])
    ph, pw = al.shape
    bg = BGRGB[bi]
    comp = bg.copy()
    comp[yy:yy+ph, xx:xx+pw] = (bg[yy:yy+ph, xx:xx+pw] * (1 - al[..., None])
                                + 255.0 * al[..., None])
    # 상자를 화면 좌표로 옮기고 둘레를 넓힘
    bx0, by0, bx1, by1 = b[0] + xx, b[1] + yy, b[2] + xx, b[3] + yy
    mw = int((bx1 - bx0) * MARGIN);  mh = int((by1 - by0) * MARGIN)
    cx0, cy0 = max(bx0 - mw, 0), max(by0 - mh, 0)
    cx1, cy1 = min(bx1 + mw, W), min(by1 + mh, H)
    A = bg[cy0:cy1, cx0:cx1]
    B = comp[cy0:cy1, cx0:cx1]
    raw = B - A
    # **줄마다 증폭을 달리 함** — 한 값으로 곱하면 짙은 줄은 하얗게 타서 모양이 안 보이고
    # 옅은 줄은 안 보임. 쓴 배수를 딱지에 적어 두므로 줄끼리 견줄 때 그것을 볼 것
    amp = float(np.clip(220.0 / max(raw.max(), 1e-6), 1.0, AMPMAX))
    Dif = np.clip(raw * amp, 0, 255)
    err = abs(v - d['chg']) / max(d['chg'], 1e-9)
    worst = max(worst, err)
    print(f'{tag:<15}{k:<9}{d["s"]:>7.3f}{f"{b[2]-b[0]}x{b[3]-b[1]}":>13}'
          f'{d["chg"]:>12.2f}{v:>12.2f}{err:>8.1%}')
    lab = (f'{tag}  {k} {x["file"]}  배율 {d["s"]:.3f} · 상자 {b[2]-b[0]}x{b[3]-b[1]}화소 · '
           f'변화 {v:.2f} ≥ 바닥 {d["floor"]:.2f}   |  배경 · 합성 · '
           f'차이 x{amp:.0f} (최대 {raw.max():.0f}계조)')
    rows.append((lab, A, B, Dif))
print('-' * 78)
print(f'  [검산] 저장된 변화와 그림의 변화가 어긋난 최대 폭 {worst:.1%}   '
      f'{"통과 — 딱지와 그림이 같은 값임" if worst < 0.10 else "**실패 — 고른 자리가 중앙값과 다름**"}')

Ht = 0
for _, A, _, _ in rows:
    h, w = A.shape[:2]
    Ht += round(PANEL * h / w) + 34
sh = Image.new('RGB', (PANEL * 3, Ht + 8), (16, 16, 16))
dr = ImageDraw.Draw(sh);  y = 0
for lab, A, B, Dif in rows:
    h, w = A.shape[:2];  ch = round(PANEL * h / w)
    for j, im in enumerate((A, B, Dif)):
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
        if pic.size[0] != PANEL:
            pic = pic.resize((PANEL, ch), Image.LANCZOS)
        sh.paste(pic, (j * PANEL, y + 34))
    dr.text((6, y + 6), lab, fill=(255, 220, 0), font=F)
    y += ch + 34
path = f'{OUT}/_crop_check.jpg'
sh.save(path, quality=92)
print(f'\n-> {path}   ({len(rows)}줄 · 칸마다 배경 · 합성 · 차이)')
files.download(path)

print('\n**차이 칸을 보고 판정할 것** — 연기 모양이면 참 라벨,')
print('흩어진 점이면 거짓 라벨임. 상자가 원본 배율이라 눈대중이 아님.')
print('(상자가 520화소보다 크면 그만큼 줄여 그렸고, 줄인 배율은 상자 화소 수로 알 수 있음)')
