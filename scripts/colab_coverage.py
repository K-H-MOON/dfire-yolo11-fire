# ===== 피복률을 잼 — 화재 저장소의 소재 게이트가 우리에게 뜻이 있는가 =====
#
# 화재 저장소 `web3d/PREREGISTER_SMOKE.md` 원문.
#
#   연기 소재 자동 게이트(사람이 영상을 고르지 않음): 테두리 휘도로 검은 배경 판정
#   → **피복률 3\~60%** → 채도 상한(컬러 스모크 배제) → 밀도·구조(빈 프레임·균질
#   안개층 배제) → 지각해시 중복 제거. 637장 중 219장이 배경 게이트에서 탈락했다.
#
# ---------------------------------------------------------------------------
# 앞서 제가 틀린 것 — **다른 양을 견줬음**
#
# `원본 넓이비 중앙값 0.715` 를 그쪽 60% 와 나란히 놓았으나, 그건 **질량99 상자 넓이**임.
# 피복률은 **알파가 0 이 아닌 화소 수 ÷ 원본 화면 화소 수** 임. 여기서 그것을 잼.
#
# ---------------------------------------------------------------------------
# **미리 못 박는 읽는 기준** — 숫자가 나오기 전에 정함
#
#   그쪽 3\~60% 를 **그대로 옮기지 않음.** 그쪽은 검은 배경에 휘도 키잉이고 우리는
#   배경 차분이라 같은 저울인지 모름. **분포를 먼저 봄.**
#
#   게이트를 넣는 뜻이 있으려면 **지금 통과한 68장 중** 걸리는 것이 있어야 함.
#       통과분 중 3\~60% 밖이 **0장이면 게이트를 안 넣음** — 넣어도 아무것도 안 바뀜
#       있으면 그 소재를 **눈으로 보고** 정함
#
#   m3 는 눈으로 확인한 **연기 없음 대조군**임.
#       **하한 3% 가 m3 를 거르면 게이트가 쓸모 있다는 증거**가 됨
#       m3 가 3\~60% 안에 들면 피복률은 잡음을 못 가르는 값임
#
# ---------------------------------------------------------------------------
# 이 시험이 **못 재는 것**
#   그쪽 `피복률` 의 분모가 정말 화면 전체인지 — 원문에 말만 있고 정의가 없음
#   채도 상한 — 우리 자료에 색 연기가 없어 무의미(시트에서 눈으로 확인한 것임)
#   밀도·구조 — 대응하는 값이 없어 **덩어리비**로 갈음함. 같은 것이 아님
#   게이트가 학습에 좋은가 — 학습을 돌려야 앎
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 2~4분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files, drive

SRC  = '/content/drive/MyDrive/smoke_frames'
MAT  = f'{SRC}/matte'
OUT  = f'{SRC}/synth_trial'
KEYS = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
THR  = 0.06
LO, HI = 0.03, 0.60            # 그쪽 값 — **견주기용이지 우리 문턱이 아님**

# 잘라내기까지 반영한 원본 화면 크기. 아래에서 manifest 와 대조해 검산함
FRAME = {'m3': (640, 480), 'kfire03': (1024, 576), 'q1': (1920, 972),
         'j04': (1280, 418), '07': (1920, 1080), 'p2': (1920, 950)}

drive.mount('/content/drive')

print('=' * 76)
print('미리 못 박은 읽는 기준 — 숫자가 나오기 전에 정함')
print('=' * 76)
print(f'  그쪽 {LO:.0%}\\~{HI:.0%} 를 **그대로 옮기지 않음** — 저울이 같은지 모름. 분포를 먼저 봄')
print('  게이트는 **지금 통과한 68장 중 걸리는 것이 있을 때만** 뜻이 있음')
print('      통과분 중 밖이 0장 → **안 넣음**')
print('      있으면 그 소재를 눈으로 보고 정함')
print('  m3 는 연기 없음 대조군 — **하한이 m3 를 거르면 게이트가 쓸모 있다는 증거**')
print('=' * 76)

man = json.load(open(f'{MAT}/matte_manifest.json'))

# ---- 검산 — 화면 크기 표가 manifest 와 맞는가 ----
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
    print(f'  {k:<9}{w}x{h} = {w*h:>9}   manifest 추정 {est:>10.0f}   '
          f'{"통과" if ok else "**실패**"}')
if bad:
    raise SystemExit('화면 크기가 안 맞음 — 아래 숫자를 쓰면 안 됨')

# ---- 지금 통과한 소재 (size_rule2) ----
p2j = f'{OUT}/size_rule2.json'
passed = set()
if os.path.exists(p2j):
    for x in json.load(open(p2j))['pieces']:
        if any(d['ok'] for d in x['draw']):
            passed.add((x['key'], x['file']))
    print(f'\nsize_rule2 통과 소재 {len(passed)}장을 읽음')
else:
    print(f'\n[주의] {p2j} 가 없어 `통과분만` 을 못 가름')

# ---- 전수 측정 ----
rows = []
for k in KEYS:
    W, H = FRAME[k]
    for p in sorted(glob.glob(f'{MAT}/{k}/*.png')):
        a = np.asarray(Image.open(p))[..., 3].astype(np.float32) / 255.0
        a[a < THR] = 0
        nz = int((a > 0).sum())
        lab, n = ndimage.label(a > 0)
        if n:
            mass = ndimage.sum(a, lab, range(1, n + 1))
            main = float(mass.max() / mass.sum())
        else:
            main = 0.0
        f = os.path.basename(p)
        rows.append({'key': k, 'file': f,
                     'cov_frame': nz / (W * H),      # **원본 화면 대비** — 그쪽과 대응
                     'cov_piece': nz / a.size,       # 조각 대비 (참고)
                     'blobs': n, 'main': main,
                     'pass': (k, f) in passed})

print(f'\n[피복률]  알파가 0 이 아닌 화소 ÷ **원본 화면** 화소')
print(f'{"출처":<9}{"장":>5}{"중앙":>9}{"최소":>9}{"최대":>9}'
      f'{"조각대비 중앙":>14}{"덩어리비 중앙":>14}')
print('-' * 70)
for k in KEYS:
    rr = [r for r in rows if r['key'] == k]
    v = [r['cov_frame'] for r in rr]
    print(f'{k:<9}{len(rr):>5}{np.median(v):>9.3f}{min(v):>9.3f}{max(v):>9.3f}'
          f'{np.median([r["cov_piece"] for r in rr]):>14.3f}'
          f'{np.median([r["main"] for r in rr]):>14.3f}')
print('-' * 70)
allv = [r['cov_frame'] for r in rows]
print(f'{"전수":<9}{len(rows):>5}{np.median(allv):>9.3f}{min(allv):>9.3f}{max(allv):>9.3f}')
pv = [r['cov_frame'] for r in rows if r['pass']]
if pv:
    print(f'{"통과분":<9}{len(pv):>5}{np.median(pv):>9.3f}{min(pv):>9.3f}{max(pv):>9.3f}')

print(f'\n분위 (전수)   ' + ' · '.join(
    f'{q}%={np.percentile(allv, q):.3f}' for q in (0, 10, 25, 50, 75, 90, 100)))
if pv:
    print(f'분위 (통과분)  ' + ' · '.join(
        f'{q}%={np.percentile(pv, q):.3f}' for q in (0, 10, 25, 50, 75, 90, 100)))

# ---- 그쪽 문턱에 걸리는 것 ----
print(f'\n[그쪽 문턱 {LO:.0%}\\~{HI:.0%} 에 걸리는 것] — **견주기용임**')
print(f'{"":<12}{"전수":>8}{"통과분":>8}')
print('-' * 30)
for tag, f in (('하한 미달', lambda r: r['cov_frame'] < LO),
               ('상한 초과', lambda r: r['cov_frame'] > HI)):
    print(f'{tag:<12}{sum(1 for r in rows if f(r)):>8}'
          f'{sum(1 for r in rows if f(r) and r["pass"]):>8}')
inside = sum(1 for r in rows if r['pass'] and LO <= r['cov_frame'] <= HI)
outside = len(pv) - inside if pv else 0
print('-' * 30)
print(f'  통과분 {len(pv)}장 중 문턱 밖 **{outside}장**')
if pv and outside == 0:
    print('  -> 미리 못 박은 기준대로 **게이트를 안 넣음** (넣어도 아무것도 안 바뀜)')
elif pv:
    print('  -> 걸리는 소재가 있음. **눈으로 보고 정할 것**')

# ---- m3 대조군 ----
m3 = [r for r in rows if r['key'] == 'm3']
if m3:
    n_lo = sum(1 for r in m3 if r['cov_frame'] < LO)
    print(f'\n[대조군] m3 {len(m3)}장 중 하한 {LO:.0%} 미달 **{n_lo}장** ({n_lo/len(m3):.0%})')
    print(f'  m3 피복률 중앙 {np.median([r["cov_frame"] for r in m3]):.3f}')
    other = [r['cov_frame'] for r in rows if r['key'] != 'm3']
    print(f'  나머지 중앙 {np.median(other):.3f}')
    if n_lo >= len(m3) * 0.8:
        print('  -> **하한이 m3 를 거름. 피복률이 잡음을 가르는 값임**')
    else:
        print('  -> **하한이 m3 를 못 거름. 피복률은 잡음을 못 가르는 값임**')

json.dump(rows, open(f'{OUT}/coverage.json', 'w'), ensure_ascii=False)
print(f'\n-> {OUT}/coverage.json')

# ---- 시트 — 피복률이 가장 큰 것과 가장 작은 것 ----
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
srt = sorted([r for r in rows if r['pass']] or rows, key=lambda r: r['cov_frame'])
pick = srt[:3] + srt[len(srt) // 2 - 1:len(srt) // 2 + 1] + srt[-3:]
CW, items = 760, []
for r in pick:
    im = Image.open(f'{MAT}/{r["key"]}/{r["file"]}')
    al = np.asarray(im)[..., 3]
    rgb = Image.new('RGB', im.size, (0, 0, 0));  rgb.paste(im, mask=im.split()[3])
    items.append((f'{r["key"]} {r["file"]}  피복률 {r["cov_frame"]:.3f} · '
                  f'덩어리 {r["blobs"]}개 · 덩어리비 {r["main"]:.3f}'
                  f'{"  [통과]" if r["pass"] else "  [제외]"}',
                  np.asarray(rgb), np.dstack([al] * 3)))
Ht = sum(round(CW * i.shape[0] / i.shape[1]) + 32 for _, i, _ in items) + 8
sh = Image.new('RGB', (CW * 2, Ht), (16, 16, 16))
d = ImageDraw.Draw(sh);  y = 0
for lab, a1, a2 in items:
    ch = round(CW * a1.shape[0] / a1.shape[1])
    d.text((6, y + 5), lab, fill=(255, 220, 0), font=F)
    sh.paste(Image.fromarray(a1).resize((CW, ch), Image.LANCZOS), (0, y + 32))
    sh.paste(Image.fromarray(a2).resize((CW, ch), Image.LANCZOS), (CW, y + 32))
    y += ch + 32
sh.save(f'{OUT}/_coverage.jpg', quality=90)
print(f'-> {OUT}/_coverage.jpg  ({len(items)}줄 · 왼쪽 소재 · 오른쪽 알파)')
files.download(f'{OUT}/_coverage.jpg')
print('\n피복률이 낮은 셋 · 가운데 둘 · 높은 셋임. **알파 칸의 모양**을 볼 것.')
