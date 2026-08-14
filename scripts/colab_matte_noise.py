# ===== 소재 알파의 잡음 재기 =====
#
# `matte` 폴더의 소재 175장을 전수로 잼. **사전 등록 값(문턱 0.06)은 안 바꿈 — 재기만 함.**
#
# 재는 것 둘
#   덩어리비   알파 질량 중 **가장 큰 연결 덩어리**가 차지하는 비율.
#              1 에 가까우면 연기 하나로 뭉쳐 있고, 낮으면 잡음이 흩어진 것
#   상자99     알파 질량의 **99% 를 담는 최소 상자**가 소재 조각에서 차지하는 넓이 비율.
#              1 에 가까우면 상자를 좁힐 수 없고, 작으면 상자만 좁히면 되는 문제
#
# **검산** — 시트에서 q1 은 깨끗하고 j04 는 의심스러웠음.
# 새 지표가 그 둘을 실제로 가르는지 코드가 확인함. **안 갈리면 지표가 실패한 것임**
# (컷 수를 1fps 에서 재려다 실패한 전례가 있음).
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 2~3분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files

SRC = '/content/drive/MyDrive/smoke_frames/matte'
KEYS = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
Q = 0.005                                  # 양끝 0.5% 씩 잘라 99% 상자


def main_ratio(a):
    """알파 질량 중 가장 큰 연결 덩어리의 몫."""
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return 0.0, 0
    mass = ndimage.sum(a, lab, range(1, n + 1))
    return float(mass.max() / mass.sum()), int(n)


def box99(a):
    """질량 99% 를 담는 상자의 넓이 비율. 행·열을 따로 잘라 냄."""
    ry, rx = a.sum(1), a.sum(0)
    def span(v):
        c = np.cumsum(v) / max(v.sum(), 1e-9)
        lo = int(np.searchsorted(c, Q))
        hi = int(np.searchsorted(c, 1 - Q))
        return max(hi - lo + 1, 1)
    return float(span(ry) * span(rx)) / a.size


rows, per = [], {}
print(f'{"출처":<9}{"장":>5}{"덩어리비":>10}{"상자99":>9}{"덩어리수":>9}')
print('-' * 44)
for k in KEYS:
    ps = sorted(glob.glob(f'{SRC}/{k}/*.png'))
    if not ps:
        print(f'{k:<9}[없음] {SRC}/{k}');  continue
    recs = []
    for p in ps:
        a = np.asarray(Image.open(p))[..., 3].astype(np.float32) / 255.0
        mr, nc = main_ratio(a)
        recs.append({'file': os.path.basename(p), 'main_ratio': round(mr, 4),
                     'box99': round(box99(a), 4), 'blobs': nc})
    per[k] = recs
    mr = np.median([r['main_ratio'] for r in recs])
    b9 = np.median([r['box99'] for r in recs])
    nb = np.median([r['blobs'] for r in recs])
    rows.append((k, len(recs), mr, b9, nb))
    print(f'{k:<9}{len(recs):>5}{mr:>10.3f}{b9:>9.3f}{nb:>9.0f}')

print('-' * 44)
print('*중앙값. 덩어리비는 1 에 가까울수록 좋고, 상자99 는 작을수록 좋음\n')

# ---- 검산 — q1(깨끗)과 j04(의심)가 갈리는가 ----
if 'q1' in per and 'j04' in per:
    q = np.median([r['main_ratio'] for r in per['q1']])
    j = np.median([r['main_ratio'] for r in per['j04']])
    qb = np.median([r['box99'] for r in per['q1']])
    jb = np.median([r['box99'] for r in per['j04']])
    print(f'[검산] 덩어리비  q1 {q:.3f} · j04 {j:.3f}   차 {q - j:+.3f}')
    print(f'      상자99    q1 {qb:.3f} · j04 {jb:.3f}   차 {jb - qb:+.3f}')
    if q - j > 0.15 or jb - qb > 0.15:
        print('      **지표가 둘을 가름.** 아래 목록을 근거로 써도 됨')
    else:
        print('      **지표가 둘을 못 가름 — 실패로 봄.** 이 숫자를 근거로 읽지 말 것')
else:
    print('[검산] q1 또는 j04 소재가 없어 검산을 못 함')

# ---- 나쁜 쪽 스무 장 ----
allr = [(k, r) for k, rs in per.items() for r in rs]
allr.sort(key=lambda t: (t[1]['main_ratio'], -t[1]['box99']))
print(f'\n덩어리비가 낮은 스무 장')
for k, r in allr[:20]:
    print(f'  {k:<9}{r["file"]:<22}덩어리비 {r["main_ratio"]:.3f} · '
          f'상자99 {r["box99"]:.3f} · 덩어리 {r["blobs"]}개')

json.dump(per, open(f'{SRC}/noise.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n-> {SRC}/noise.json')

# ---- 의심스러운 것만 원본 비율로 크게 ----
# **가로세로비를 소재마다 지켜 그림.** 앞 시트는 첫 행 비율로 전부 늘여
# j04(1280x418)가 세로로 2.3배 늘어나 보였음.
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
except Exception:
    F = ImageFont.load_default()

pick = allr[:6] + [(k, r) for k, r in allr if k == 'q1'][:2]     # 나쁜 여섯 + 대조 q1 둘
CW, imgs = 900, []
for k, r in pick:
    im = Image.open(f'{SRC}/{k}/{r["file"]}')
    rgb = Image.new('RGB', im.size, (0, 0, 0));  rgb.paste(im, mask=im.split()[3])
    al = im.split()[3].convert('RGB')
    imgs.append((f'{k} {r["file"]}  덩어리비 {r["main_ratio"]:.3f} · 상자99 {r["box99"]:.3f}',
                 rgb, al, im.size))

H = sum(round(CW * s[1] / s[0]) + 30 for *_, s in imgs) + 8
sh = Image.new('RGB', (2 * CW, H), (16, 16, 16))
d = ImageDraw.Draw(sh);  y = 0
for lab, rgb, al, (w, h) in imgs:
    ch = round(CW * h / w)
    d.text((6, y + 4), lab, fill=(255, 220, 0), font=F)
    sh.paste(rgb.resize((CW, ch), Image.LANCZOS), (0, y + 30))
    sh.paste(al.resize((CW, ch), Image.LANCZOS), (CW, y + 30))
    y += ch + 30
sh.save(f'{SRC}/_noise.jpg', quality=90)
print(f'-> {SRC}/_noise.jpg  ({len(imgs)}줄 · 가로세로비 유지)')
files.download(f'{SRC}/_noise.jpg')
