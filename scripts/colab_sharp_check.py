# ===== 선명도 지표를 장면 간에 견줄 수 있는가 =====
#
# 앞 검산은 `같은 프레임을 흐리게 하면 값이 떨어지는가` 만 봤음 — **한 장면 안에서만**
# 확인한 것임. 여기서는 **장면이 달라도 견줄 수 있는가** 를 봄.
#
# 방법 — 각 급식실 프레임에 **알려진 흐림**을 걸어 (장면 × 흐림) 조합을 만들고,
# 지표값으로 줄 세웠을 때 흐림 순서대로 서는지 봄.
#
#   같은 장면 쌍의 위반율    0 에 가까워야 정상 (앞 검산이 본 것)
#   **다른 장면 쌍의 위반율** ← 이번에 묻는 것. 0.5 에 가까우면 동전 던지기
#   겹침                   같은 흐림 단계 안의 값 범위가 단계 사이 간격보다 큰가
#
# ---------------------------------------------------------------------------
# 이 검증이 확인하지 **못하는** 것도 적어 둠
#
#   가우시안 흐림으로 시험함. **압축 흐림은 성격이 다름**(블로킹·모스키토 잡음)
#   개원중의 높은 값이 세부인지 잡음인지는 여전히 못 가름
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 2~3분.

import os, glob, unicodedata
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

STEAM = '/content/drive/MyDrive/smoke_frames/steam'
N = 5                                       # 그룹당 표본
BLUR = [0.0, 0.5, 1.0, 2.0]

STEAMKEY = ['숭곡중', '영동중', '원촌중', '내곡중', '진선여고', '남일고',
            '인화여중', '금정초', '로봇고', '부산체고', '울산현대차']
BGKEY = ['개원중', '논현중', '로봇고_bg', '숭곡중_bg']
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def norm(s):
    return unicodedata.normalize('NFC', s)


def vals(im, r):
    g = np.asarray(im if r == 0 else im.filter(ImageFilter.GaussianBlur(r)), np.float32)
    lv = float(ndimage.convolve(g, LAP).var())
    return lv, lv / max(float(g.var()), 1e-6)


def collect(folder, keys):
    out = {k: [] for k in keys}
    for p in glob.glob(f'{folder}/*.jpg'):
        b = norm(os.path.basename(p))
        for k in sorted(keys, key=len, reverse=True):
            if b.startswith(norm(k)):
                out[k].append(p);  break
    return out


files = {k: [] for k in STEAMKEY}
for tag in ('steam_in', 'steam_near', 'steam_far'):
    for k, v in collect(f'{STEAM}/{tag}', STEAMKEY).items():
        files[k] += v
files.update(collect(f'{STEAM}/bg', BGKEY))

# (장면, 흐림, 원값, 정규화)
pts = []
for k, ps in files.items():
    for p in sorted(ps)[:N]:
        im = Image.open(p).convert('L')
        for r in BLUR:
            a, b = vals(im, r)
            pts.append((k, r, a, b))
if not pts:
    raise SystemExit(f'{STEAM} 아래에 프레임이 없음')
print(f'표본 {len(pts)}개 — 장면 {len(files)}곳 × 흐림 {len(BLUR)}단계\n')

for name, idx in (('라플라시안 원값', 2), ('대비 정규화', 3)):
    # 덜 흐린 쪽(ri)이 더 높아야 정상. 뒤집히면 위반, 같으면 동점(구별 못 함)
    same_v = same_t = same_n = diff_v = diff_t = diff_n = 0
    for i in range(len(pts)):
        for j in range(len(pts)):
            ki, ri, vi = pts[i][0], pts[i][1], pts[i][idx]
            kj, rj, vj = pts[j][0], pts[j][1], pts[j][idx]
            if ri >= rj:
                continue
            if ki == kj:
                same_n += 1;  same_v += (vi < vj);  same_t += (vi == vj)
            else:
                diff_n += 1;  diff_v += (vi < vj);  diff_t += (vi == vj)
    sr = (same_v + same_t) / max(same_n, 1)
    dr = (diff_v + diff_t) / max(diff_n, 1)
    print(f'[{name}]')
    print(f'  같은 장면 쌍 못 가름   {sr:6.1%}  (뒤집힘 {same_v} · 동점 {same_t} / {same_n})')
    print(f'  다른 장면 쌍 못 가름   {dr:6.1%}  (뒤집힘 {diff_v} · 동점 {diff_t} / {diff_n})'
          f'   ← 이번에 묻는 것')

    print(f'  흐림 단계별 값 범위')
    rng = {}
    for r in BLUR:
        v = [p[idx] for p in pts if p[1] == r]
        rng[r] = (min(v), max(v))
        print(f'    흐림 {r:<4}  {min(v):10.1f} ~ {max(v):10.1f}')
    ov = [(a, b) for ii, a in enumerate(BLUR) for b in BLUR[ii + 1:]
          if rng[a][0] <= rng[b][1]]
    if dr < 0.05 and not ov:
        print('  **장면이 달라도 견줄 수 있음.** 값을 그대로 비교해도 됨')
    else:
        why = []
        if dr >= 0.05:
            why.append(f'다른 장면 위반율 {dr:.1%}')
        if ov:
            why.append(f'흐림 단계가 겹침 {len(ov)}쌍')
        print(f'  **장면 간 비교 불가** — {" · ".join(why)}')
    print()

print('이 검증이 확인 못 하는 것 — 가우시안 흐림으로 시험했으므로 **압축 흐림**은 다를 수 있음.')
print('개원중의 높은 값이 세부인지 잡음인지도 여전히 못 가름.')
