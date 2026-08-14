# ===== m3 는 왜 66장 전부 제외됐는가 — 원인을 잼 =====
#
# 3회차에서 **m3 66장이 통째로 제외**됐음(변화 1.00\~3.97 · 문턱 4.446).
# 그런데 m3 의 **봉우리는 103\~166** 으로 다른 출처와 다르지 않음.
# `평균은 바닥 아래인데 봉우리는 높다` — 이게 무슨 뜻인지 갈라야 함.
#
#   갈래 ㄱ   m3 에 연기가 실제로 옅음        → 제외가 옳음. m3 는 원래 못 쓸 소재였음
#   갈래 ㄴ   m3 상자가 넓어 평균이 희석됨    → **지표가 못 재는 것**. 문턱을 다시 봐야 함
#
# **이 스크립트는 기준을 바꾸지 않음. 원인만 잼.** 새 기준을 세우는 것은
# 이 결과를 보고 따로 정함.
#
# 재는 것 넷
#   (1) 문턱을 잡음 바닥의 1사분\~3사분으로 흔들면 출처별로 몇 장이 살아나는가 (민감도)
#   (2) 출처별 질량99 **상자 넓이비** — m3 상자가 정말 넓은가
#   (3) 상자 안 **알파 평균** — 넓이를 뺀 순수한 짙기
#   (4) m3 66장을 전 구간에 고르게 뽑아 **눈으로** 봄
#
# 이 시험이 **못 재는 것**
#   m3 의 알파가 연기인지 압축 잡음인지 — 눈으로 볼 뿐 숫자로 못 가름
#   `사람이 보이면 학습이 된다` 는 보장 — 학습을 돌려야 알 수 있음
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 5~8분.

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
THR   = 0.06
U     = 0.30
NSHOW = 8                      # m3 에서 볼 장수 (전 구간에 고르게)

drive.mount('/content/drive')


def norm(s):
    return unicodedata.normalize('NFC', s)


def box_q(a, q=0.005):
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


# ---------------------------------------------------------------------------
# (1) 민감도 — 문턱을 흔들면 몇 장이 살아나는가
# ---------------------------------------------------------------------------
J = json.load(open(f'{OUT}/synth_trial3.json'))
rec = J['records']
FLOOR = J['floor']
print(f'3회차 문턱 {FLOOR:.3f} · 규칙 {J["rule"]} · 소재 {len(rec)}장')
print('\n[민감도] 문턱을 잡음 바닥 분포의 1사분\\~3사분으로 흔들었을 때 남는 장수')
print(f'{"문턱":>8}' + ''.join(f'{k:>9}' for k in KEYS) + f'{"합":>7}')
print('-' * (8 + 9 * len(KEYS) + 7))
for f in (3.133, FLOOR, 5.663):
    row = [sum(1 for r in rec if r['key'] == k and r['chg'] >= f) for k in KEYS]
    tag = ' *' if abs(f - FLOOR) < 1e-6 else '  '
    print(f'{f:>6.3f}{tag}' + ''.join(f'{v:>9}' for v in row) + f'{sum(row):>7}')
print('  * 이 줄이 3회차가 쓴 문턱임. 위아래는 잡음 바닥 표본의 1사분·3사분')
print('  **이 표는 기준을 바꾸자는 게 아니라 기준이 얼마나 흔들리는지 보이는 것임**')

# ---------------------------------------------------------------------------
# 소재 다시 읽기
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bg0 = np.asarray(Image.open(bgs_all[0]).convert('RGB'), np.float32)
H, W = bg0.shape[:2]


def load_piece(p):
    q = Image.open(p).convert('RGBA')
    pw = max(int(round(W * U)), 8)
    ph = max(int(round(pw * q.size[1] / q.size[0])), 8)
    if ph > H - 2:
        ph = H - 2
        pw = max(int(round(ph * q.size[0] / q.size[1])), 8)
    r = np.asarray(q.resize((pw, ph), Image.LANCZOS), np.float32)
    al = r[..., 3] / 255.0
    al[al < THR] = 0
    return r[..., :3], al


# ---------------------------------------------------------------------------
# (2)(3) 상자 넓이비와 상자 안 알파 평균 — 넓이 탓인지 짙기 탓인지
# ---------------------------------------------------------------------------
print(f'\n[상자와 짙기]  질량99 상자 · 가로폭 {U:g} 고정')
print(f'{"출처":<9}{"장":>5}{"상자넓이비":>11}{"상자안알파":>11}{"조각알파":>10}{"덩어리수":>9}')
print('-' * 56)
tab = {}
for k in KEYS:
    ps = sorted(glob.glob(f'{MAT}/{k}/*.png'))
    ar, ab, ap, nb = [], [], [], []
    for p in ps:
        _, al = load_piece(p)
        b = box_q(al)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        sub = al[y0:y1, x0:x1]
        ar.append(sub.size / (H * W))
        ab.append(float(sub.mean()))
        ap.append(float(al.mean()))
        nb.append(int(ndimage.label(al > 0)[1]))
    tab[k] = (np.median(ar), np.median(ab), np.median(ap), np.median(nb))
    print(f'{k:<9}{len(ps):>5}{np.median(ar):>11.4f}{np.median(ab):>11.4f}'
          f'{np.median(ap):>10.4f}{np.median(nb):>9.0f}')
print('-' * 56)
m = tab['m3']
o = [tab[k] for k in KEYS if k != 'm3']
print(f'm3 상자넓이비가 나머지 중앙값의 {m[0] / np.median([x[0] for x in o]):.2f}배')
print(f'm3 상자안알파가 나머지 중앙값의 {m[1] / np.median([x[1] for x in o]):.2f}배')
print(f'  넓이비가 크고 상자안알파가 작으면 **희석**, 넓이비가 보통인데 알파만 작으면 **정말 옅음**')

# ---------------------------------------------------------------------------
# (4) m3 를 눈으로 — 전 구간에서 고르게
# ---------------------------------------------------------------------------
def korean_font(size):
    cand = (glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
            + glob.glob('/usr/share/fonts/**/*Nanum*.ttf', recursive=True)
            + glob.glob('/usr/share/fonts/**/NotoSansCJK*', recursive=True))
    if not cand:
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        cand = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
    for c in cand:
        try:
            f = ImageFont.truetype(c, size)
            t = Image.new('L', (size * 4, size * 2), 0)
            ImageDraw.Draw(t).text((2, 2), '연기', fill=255, font=f)
            if np.asarray(t).max() > 0:
                return f
        except Exception:
            pass
    return ImageFont.load_default()


F = korean_font(24)
chg = {(r['key'], r['file']): r['chg'] for r in rec}
pk = {(r['key'], r['file']): r['peak'] for r in rec}


def rows_for(key, n):
    ps = sorted(glob.glob(f'{MAT}/{key}/*.png'))
    idx = np.linspace(0, len(ps) - 1, min(n, len(ps))).round().astype(int)
    out = []
    for j, i in enumerate(idx):
        p = ps[int(i)]
        rgb, al = load_piece(p)
        bg = np.asarray(Image.open(bgs_all[(j * 37) % len(bgs_all)]).convert('RGB'),
                        np.float32)
        ph, pw = al.shape
        x, y = (W - pw) // 2, int(H * 0.25)
        oo = bg.copy()
        oo[y:y+ph, x:x+pw] = (bg[y:y+ph, x:x+pw] * (1 - al[..., None])
                              + 255.0 * al[..., None])
        b = box_q(al)
        bb = None if b is None else (b[0]+x, b[1]+y, b[2]+x, b[3]+y)
        f = os.path.basename(p)
        c = chg.get((key, f));  q = pk.get((key, f))
        col = (255, 80, 80) if (c is not None and c < FLOOR) else (120, 255, 120)
        out.append((f'{"제외" if col[0] == 255 else "통과"} {key} {f}  '
                    f'변화 {c if c is None else round(c, 2)} · 봉우리 '
                    f'{q if q is None else round(q, 1)} · 문턱 {FLOOR:.2f}', oo,
                   [(bb, col)]))
    return out


def sheet(items, path, cw=1100):
    Ht = sum(round(cw * im.shape[0] / im.shape[1]) + 40 for _, im, _ in items) + 8
    sh = Image.new('RGB', (cw, Ht), (16, 16, 16))
    d = ImageDraw.Draw(sh);  y = 0
    for lab, im, boxes in items:
        h0, w0 = im.shape[:2]
        ch = round(cw * h0 / w0)
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS)
        dd = ImageDraw.Draw(pic);  s = cw / w0
        for b, col in boxes:
            if b:
                dd.rectangle([b[0]*s, b[1]*s, b[2]*s, b[3]*s], outline=col, width=3)
        d.text((6, y + 8), lab, fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 40))
        y += ch + 40
    sh.save(path, quality=90)
    return path


p1 = sheet(rows_for('m3', NSHOW), f'{OUT}/_m3.jpg')
print(f'\n-> {p1}   (m3 {NSHOW}장 · 전 구간에서 고르게)')
files.download(p1)

# 대조 — 통과한 출처에서 **변화가 가장 낮은** 두 장. m3 와 눈으로 견주려는 것
low = sorted([r for r in rec if r['chg'] >= FLOOR], key=lambda r: r['chg'])[:3]
rows = []
for j, r in enumerate(low):
    rgb, al = load_piece(f'{MAT}/{r["key"]}/{r["file"]}')
    bg = np.asarray(Image.open(bgs_all[(j * 53) % len(bgs_all)]).convert('RGB'), np.float32)
    ph, pw = al.shape
    x, y = (W - pw) // 2, int(H * 0.25)
    oo = bg.copy()
    oo[y:y+ph, x:x+pw] = bg[y:y+ph, x:x+pw] * (1 - al[..., None]) + 255.0 * al[..., None]
    b = box_q(al)
    rows.append((f'통과(가장 낮은 축) {r["key"]} {r["file"]}  변화 {r["chg"]:.2f} · '
                 f'봉우리 {r["peak"]:.1f}', oo,
                 [(None if b is None else (b[0]+x, b[1]+y, b[2]+x, b[3]+y), (120, 255, 120))]))
p2 = sheet(rows, f'{OUT}/_lowpass.jpg')
print(f'-> {p2}   (통과한 것 중 변화가 가장 낮은 셋 — m3 와 눈으로 견주려는 것)')
files.download(p2)

print('\n**이 스크립트는 기준을 바꾸지 않았음.** 원인을 재고 눈으로 볼 자료만 만들었음.')
