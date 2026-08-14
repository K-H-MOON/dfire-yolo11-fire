# ===== 선명도 재기 — 합성 배경 후보와 평가 대상의 도메인 차이 =====
#
# 합성 배경 후보는 **개원중 CCTV** 와 **로봇고 쉐이크** 둘뿐임
# (김이 없는 급식실 프레임이 그 둘밖에 없음).
#
# 개원중이 흐리다는 것은 지금까지 **제 눈대중**이었음. 재서 확인함.
# 학습 배경과 평가 대상(김 11곳)의 선명도가 크게 다르면 도메인 차이가 되고,
# 그것이 판별비를 낮추는 쪽으로 작용함.
#
# 재는 것
#   라플라시안 분산   흐릴수록 낮음. 초점·압축 흐림을 잡는 표준 지표
#   대비 정규화값     위 값 ÷ 화소 분산. **어두운 영상이 불리해지는 것을 덜어냄**
#   해상도           낮으면 그 자체로 흐림
#
# **검산** — 같은 프레임을 일부러 흐리게 만들어 값이 떨어지는지 확인함.
# 안 떨어지면 지표가 실패한 것이고 숫자를 근거로 쓰지 않음.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, unicodedata
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

STEAM = '/content/drive/MyDrive/smoke_frames/steam'
N = 30                                    # 그룹당 최대 표본

STEAMKEY = ['숭곡중', '영동중', '원촌중', '내곡중', '진선여고', '남일고',
            '인화여중', '금정초', '로봇고', '부산체고', '울산현대차']
BGKEY = ['개원중', '논현중', '로봇고_bg', '숭곡중_bg']
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def norm(s):
    return unicodedata.normalize('NFC', s)


def sharp(p):
    im = Image.open(p).convert('L')
    g = np.asarray(im, dtype=np.float32)
    lv = float(ndimage.convolve(g, LAP).var())
    return lv, lv / max(float(g.var()), 1e-6), im.size


def collect(folder, keys):
    """파일명 앞에 붙은 급식실 이름으로 묶음. 긴 이름부터 맞춰 `로봇고_bg` 를 먼저 잡음."""
    out = {k: [] for k in keys}
    for p in glob.glob(f'{folder}/*.jpg'):
        b = norm(os.path.basename(p))
        for k in sorted(keys, key=len, reverse=True):
            if b.startswith(norm(k)):
                out[k].append(p);  break
    return out


# ---- 검산 — 흐리게 만들면 값이 떨어지는가 ----
probe = None
for f in ('steam_near', 'steam_in', 'steam_far', 'bg'):
    ps = glob.glob(f'{STEAM}/{f}/*.jpg')
    if ps:
        probe = sorted(ps)[0];  break
if probe is None:
    raise SystemExit(f'{STEAM} 아래에 프레임이 없음')

im = Image.open(probe).convert('L')
base = float(ndimage.convolve(np.asarray(im, np.float32), LAP).var())
print('[검산] 같은 프레임을 흐리게 만들며 값이 떨어지는지')
print(f'  원본        {base:9.1f}')
prev, okay = base, True
for r in (0.5, 1.0, 2.0, 4.0):
    v = float(ndimage.convolve(
        np.asarray(im.filter(ImageFilter.GaussianBlur(r)), np.float32), LAP).var())
    print(f'  흐림 {r:<4}  {v:9.1f}')
    if v > prev:
        okay = False
    prev = v
print('  ' + ('지표가 흐림에 반응함 — 아래 값을 읽어도 됨' if okay
             else '**흐릴수록 오히려 오름 — 지표 실패. 근거로 쓰지 말 것**'))

# ---- 그룹별 측정 ----
steam = {k: [] for k in STEAMKEY}
for tag in ('steam_in', 'steam_near', 'steam_far'):
    for k, v in collect(f'{STEAM}/{tag}', STEAMKEY).items():
        steam[k] += v
bg = collect(f'{STEAM}/bg', BGKEY)

rows = []
print(f'\n{"그룹":<14}{"장":>5}{"라플라시안":>12}{"정규화":>10}{"해상도":>13}   쓰임')
print('-' * 66)
for name, group, use in ([(k, steam[k], '평가 D(김)') for k in STEAMKEY] +
                         [(k, bg[k], '배경 오탐군 / 합성 배경 후보') for k in BGKEY]):
    ps = sorted(group)[:N]
    if not ps:
        print(f'{name:<14}[없음]');  continue
    vals = [sharp(p) for p in ps]
    lv = float(np.median([v[0] for v in vals]))
    nv = float(np.median([v[1] for v in vals]))
    sz = vals[0][2]
    rows.append((name, len(ps), lv, nv, sz, use))
    print(f'{name:<14}{len(ps):>5}{lv:>12.1f}{nv:>10.3f}{f"{sz[0]}x{sz[1]}":>13}   {use}')

# ---- 견주기 ----
d = [r for r in rows if r[0] in STEAMKEY]
if d:
    md = float(np.median([r[2] for r in d]))
    mdn = float(np.median([r[3] for r in d]))
    print('\n' + '=' * 66)
    print(f'평가 대상(김 11곳) 중앙값   라플라시안 {md:.1f} · 정규화 {mdn:.3f}')
    for k in ('개원중', '로봇고_bg'):
        r = next((x for x in rows if x[0] == k), None)
        if r:
            print(f'  {k:<10}라플라시안 {r[2]:8.1f} ({r[2] / md:5.2f}배)'
                  f' · 정규화 {r[3]:.3f} ({r[3] / mdn:5.2f}배)')
    print('\n1 에 가까울수록 평가 대상과 닮은 것임. 크게 낮으면 흐린 배경으로 학습하고')
    print('선명한 자료로 평가하는 셈이라 전이가 어려워짐.')
