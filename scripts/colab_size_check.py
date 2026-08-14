# ===== 연기의 진짜 크기를 잼 =====
#
# `matte_manifest.json` 의 **area_ratio 는 연기 크기가 아님**을 확인했음.
# 그것은 `알파가 0 이 아닌 화소를 전부 담는 테두리` 라서, 알파 잡음이 화면 곳곳에
# 흩어져 있으면(덩어리 372\~823개) 테두리가 화면을 통째로 덮음.
# 실제로 통과분 70장의 area_ratio 중앙값이 **1.000** 이었음.
#
#   07        bbox [0, 0, 1920, 1080]   화면 그대로
#   p2        bbox [0, 0, 1920,  950]   잘라낸 뒤 화면 그대로
#   j04       bbox [0, 0, 1280,  418]   잘라낸 뒤 화면 그대로
#   kfire03   bbox [0, 0, 1024,  576]   화면 그대로
#
# 그래서 **질량99 상자**로 다시 잼. 이건 3회차가 라벨 상자로 확정한 규칙과 같은 것임.
#
# 재는 것 넷
#   (1) 통과분 70장의 질량99 상자가 **원본 화면에서 차지하는 가로·세로·넓이 비율**
#   (2) 그것을 개원중 1920x1080 에 옮기면 몇 화소인가
#   (3) 그 크기로 **겹치지 않게 몇 개나** 놓을 수 있는가  ← 개수 논거의 확인
#   (4) **제외 판정이 크기에 흔들리는가** — 3회차는 가로폭을 0.30 으로 고정해 쟀음.
#       원본 크기로 재면 상자 안 알파 평균이 달라지는지 봄
#
# 이 시험이 **못 재는 것**
#   학습이 되는가 — 학습을 돌려야 앎
#   원본 영상의 화각과 개원중 화각이 달라, 같은 비율이 같은 그림이 되는지는 모름
#
# **기준을 바꾸지 않음. 크기 논거를 세울 숫자만 잼.**
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image
from google.colab import drive

SRC  = '/content/drive/MyDrive/smoke_frames'
MAT  = f'{SRC}/matte'
OUT  = f'{SRC}/synth_trial'
KEYS = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
THR  = 0.06
BGW, BGH = 1920, 1080          # 개원중
U    = 0.30                    # 3회차가 재는 동안 고정했던 가로폭

# 잘라내기까지 반영한 **원본 화면 크기**. 아래에서 manifest 와 대조해 검산함
FRAME = {'m3': (640, 480), 'kfire03': (1024, 576), 'q1': (1920, 972),
         'j04': (1280, 418), '07': (1920, 1080), 'p2': (1920, 950)}

drive.mount('/content/drive')


def box_q(a, q=0.005):
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


man = json.load(open(f'{MAT}/matte_manifest.json'))
J = json.load(open(f'{OUT}/synth_trial3.json'))
FLOOR = J['floor']
chg = {(r['key'], r['file']): r['chg'] for r in J['records']}
print(f'문턱 {FLOOR:.3f} · 통과 {sum(1 for v in chg.values() if v >= FLOOR)}장'
      f' / {len(chg)}장')

# ---- 검산 (1) FRAME 표가 manifest 와 맞는가 ----
print('\n[검산] 적어 둔 화면 크기가 manifest 의 넓이와 맞는가')
bad = 0
for k in KEYS:
    fr = [r for r in man[k]['frames'] if r.get('file')]
    r = fr[len(fr) // 2]
    x0, y0, x1, y1 = r['bbox']
    est = (x1 - x0) * (y1 - y0) / max(r['area_ratio'], 1e-9)
    w, h = FRAME[k]
    ok = abs(est - w * h) / (w * h) < 0.01
    bad += (not ok)
    print(f'  {k:<9}적음 {w}x{h} = {w*h:>9}   manifest 추정 {est:>10.0f}   '
          f'{"통과" if ok else "**실패**"}')
if bad:
    raise SystemExit('화면 크기가 안 맞음 — 아래 숫자를 쓰면 안 됨')

# ---------------------------------------------------------------------------
# (1)(2) 통과분의 질량99 상자가 원본 화면에서 차지하는 비율
# ---------------------------------------------------------------------------
print(f'\n[진짜 크기]  통과분만 · 질량99 상자 ÷ 원본 화면')
print(f'{"출처":<9}{"통과":>5}{"가로비 중앙":>13}{"(최소~최대)":>18}'
      f'{"세로비 중앙":>13}{"넓이비 중앙":>13}')
print('-' * 74)
allw, allh, alla, per = [], [], [], {}
for k in KEYS:
    W, H = FRAME[k]
    ws, hs, ars = [], [], []
    for r in man[k]['frames']:
        f = r.get('file')
        if not f or chg.get((k, f), 0) < FLOOR:
            continue
        a = np.asarray(Image.open(f'{MAT}/{k}/{f}'))[..., 3].astype(np.float32) / 255.0
        a[a < THR] = 0
        b = box_q(a)
        if b is None:
            continue
        ws.append((b[2] - b[0]) / W)
        hs.append((b[3] - b[1]) / H)
        ars.append((b[2] - b[0]) * (b[3] - b[1]) / (W * H))
    per[k] = (ws, hs, ars)
    allw += ws;  allh += hs;  alla += ars
    if not ws:
        print(f'{k:<9}{0:>5}{"— (통과 0장)":>13}');  continue
    print(f'{k:<9}{len(ws):>5}{np.median(ws):>13.3f}'
          f'{f"({min(ws):.3f}~{max(ws):.3f})":>18}{np.median(hs):>13.3f}'
          f'{np.median(ars):>13.3f}')
print('-' * 74)
print(f'{"합":<9}{len(allw):>5}{np.median(allw):>13.3f}'
      f'{f"({min(allw):.3f}~{max(allw):.3f})":>18}{np.median(allh):>13.3f}'
      f'{np.median(alla):>13.3f}')

print(f'\n통과분 가로비 분위   ' +
      ' · '.join(f'{p}%={np.percentile(allw, p):.3f}' for p in (0, 10, 25, 50, 75, 90, 100)))
print(f'개원중 {BGW}x{BGH} 로 옮기면 가로 화소   '
      f'중앙 {BGW*np.median(allw):.0f} · 최소 {BGW*min(allw):.0f} · 최대 {BGW*max(allw):.0f}')
print(f'3회차가 재는 동안 고정했던 값 U={U} -> {int(BGW*U)}px'
      f'   (원본 크기 중앙값의 {U/np.median(allw):.2f}배)')

# ---------------------------------------------------------------------------
# (3) 겹치지 않게 몇 개를 놓을 수 있는가 — 개수 논거의 확인
# ---------------------------------------------------------------------------
mw, mh = np.median(allw), np.median(allh)
nx, ny = int(1 / mw), int(1 / mh)
print(f'\n[개수]  중앙 크기({mw:.3f} x {mh:.3f})로 겹침 없이 격자로 놓으면 '
      f'가로 {nx}개 x 세로 {ny}개 = **{nx*ny}개**')
big = sum(1 for w, h in zip(allw, allh) if w > 0.5 or h > 0.5)
print(f'  가로나 세로가 화면의 절반을 넘는 소재 {big} / {len(allw)}장'
      f'   ({big/len(allw):.0%})')
print(f'  **한 장에 둘 이상 놓으면 겹칠 확률이 이만큼임.** 원본 영상은 프레임마다')
print(f'  연기 하나이므로 `한 장에 하나` 가 자료와도 맞음')

# ---------------------------------------------------------------------------
# (4) 제외 판정이 크기에 흔들리는가
# ---------------------------------------------------------------------------
print(f'\n[검산] 3회차는 가로폭 {U} 로 **고정**해 쟀음. 원본 크기로 재면 달라지는가')
print(f'  견주는 값 — 질량99 상자 안 알파 평균 (변화는 여기에 거의 비례함)')
print(f'{"출처":<9}{"고정 U=0.30":>13}{"원본 크기":>12}{"비":>8}')
print('-' * 44)
worst = 0.0
for k in KEYS:
    fs = [r['file'] for r in man[k]['frames']
          if r.get('file') and chg.get((k, r['file']), 0) >= FLOOR]
    if not fs:
        continue
    fixed, native = [], []
    for f in fs[:8]:
        im = Image.open(f'{MAT}/{k}/{f}')
        a0 = np.asarray(im)[..., 3].astype(np.float32) / 255.0
        a0[a0 < THR] = 0
        b = box_q(a0)
        native.append(float(a0[b[1]:b[3], b[0]:b[2]].mean()))
        pw = max(int(round(BGW * U)), 8)
        ph = max(int(round(pw * im.size[1] / im.size[0])), 8)
        a1 = np.asarray(im.resize((pw, ph), Image.LANCZOS))[..., 3].astype(np.float32) / 255.0
        a1[a1 < THR] = 0
        b1 = box_q(a1)
        fixed.append(float(a1[b1[1]:b1[3], b1[0]:b1[2]].mean()))
    r = np.median(native) / max(np.median(fixed), 1e-9)
    worst = max(worst, abs(r - 1))
    print(f'{k:<9}{np.median(fixed):>13.4f}{np.median(native):>12.4f}{r:>8.2f}')
print('-' * 44)
print(f'  가장 큰 어긋남 {worst:.1%}   '
      f'{"제외 판정은 크기에 거의 안 흔들림" if worst < 0.15 else "**크기가 제외 판정을 흔듦 — 3회차 결과를 다시 봐야 함**"}')

json.dump({'w': allw, 'h': allh, 'area': alla,
           'per': {k: {'w': v[0], 'h': v[1], 'area': v[2]} for k, v in per.items()}},
          open(f'{OUT}/size.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n-> {OUT}/size.json')
print('\n**기준을 바꾸지 않았음.** 크기 논거를 세울 숫자만 쟀음.')
