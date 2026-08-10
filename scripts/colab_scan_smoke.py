# ===== 발연 구간 자동 탐색 =====
# 드라이브에 올린 영상에서 "불이 없는 구간"을 자동으로 찾아 확대 시트로 만듦.
#
# 왜 필요한가 — 소방 재현 영상은 대부분이 화염 장면이고 점화 직전 발연은 1\~2초씩만
# 담김. kfire01(소방청 식용유 화재 실험)은 170초 중 발연이 1.4초뿐이었음.
# 사람이 재생하며 찾으면 이런 구간을 놓치기 쉬움.
#
# 원리 — 프레임마다 화염 화소 비율을 계산해 불이 없는 초를 고르고, 붙어 있는 것끼리
# 묶어 구간으로 만든 뒤 그 구간만 촘촘히 다시 뽑아 확대함.
# 영상은 드라이브에 그대로 두고 시트만 내려받으므로 큰 파일을 옮길 필요가 없음.
#
# 주의 1 — 이 스크립트가 찾는 것은 "불이 없는 구간"이지 "연기가 있는 구간"이 아님.
# 세팅 장면, 소화기 제품컷, 진화 후 수증기도 함께 걸림. 판정은 시트를 보고 사람이 함.
#
# 주의 2 — **전체 훑기 시트가 판정의 기준임. 후보 구간은 참고임.**
# 첫 판에서 주황 화소 비율로 화염을 걸렀더니 kfire03 의 발연 13초를 통째로 놓쳤음.
# 구리 팬과 빨간 소방차가 주황으로 잡혀 "불이 있는 구간"으로 분류됐기 때문임.
# 화재 저장소 `docs/PREREGISTER_F.md` 에 이미 같은 실패가 적혀 있었음 —
# "주황 화소 비율 기준이 이들에서는 작동하지 않음. kfire02는 목재 벽,
#  kfire04는 빨간 소방차와 주황 라바콘, kfire06은 마른 잔디".
# 지금은 밝은 노란 심부만 화염으로 보도록 고쳤으나, 자동 판정은 언제든 틀릴 수 있으므로
# 후보 구간이 비어 있어도 전체 훑기 시트를 반드시 눈으로 볼 것.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요.

import os, glob, json, shutil, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

SRC   = '/content/drive/MyDrive/smoke_frames'
ONLY  = []        # 특정 영상만 볼 때 파일명 일부를 넣음. 예 ['kfire02','Prevention Week']

SCAN_FPS  = 1     # 1차 훑기 (초당 1장)
HI_FPS    = 4     # 후보 구간 확대 (초당 4장)
FLAME_MAX = 0.02  # 화염 심부 화소 비율 상한 (%). 넉넉히 잡아 놓침을 줄임
PAD       = 1.0   # 후보 구간 앞뒤 여유 (초)
GAP       = 2     # 이 초 이하로 떨어진 후보는 한 구간으로 묶음
MIN_W     = 900   # 확대 시트 셀 최소 가로. 저해상도 영상을 키워서 보기 위함

drive.mount('/content/drive')
vids = sorted(sum([glob.glob(f'{SRC}/*.{e}') for e in
                   ('mp4', 'MP4', 'mov', 'MOV', 'avi', 'AVI', 'mkv', 'MKV')], []))
if ONLY:
    vids = [v for v in vids if any(k.lower() in os.path.basename(v).lower() for k in ONLY)]
assert vids, f'{SRC} 에서 영상을 찾지 못했습니다'
print(f'영상 {len(vids)}개\n')

OUT = '/content/scan'
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
except Exception:
    F = ImageFont.load_default()


def flame_stats(path):
    """화염 심부 화소 비율(%)

    붉기만 해서는 안 되고 **밝은 노란 심부**여야 화염으로 봄.
    구리 팬(R 180 G 110 B 80)과 빨간 소방차(R 190 G 30 B 40)는
    R 이 235 를 넘지 못하거나 G 가 150 을 넘지 못해 걸러짐.
    """
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return ((R > 235) & (G > 150) & (B < 150) & (G - B > 60)).mean() * 100


def sheet(items, path, cols, cw, title):
    """items = [(라벨, 이미지경로)]"""
    w0, h0 = Image.open(items[0][1]).size
    ch = round(cw * h0 / w0)
    rows = (len(items) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * ch + 36), (18, 18, 18))
    d = ImageDraw.Draw(sh)
    d.text((6, 6), title, fill=(255, 220, 0), font=F)
    for j, (lb, p) in enumerate(items):
        im = Image.open(p).convert('RGB').resize((cw, ch), Image.LANCZOS)
        x, y = (j % cols) * cw, (j // cols) * ch + 36
        sh.paste(im, (x, y))
        d.rectangle([x, y, x + 8 + 12 * len(lb), y + 30], fill=(0, 0, 0))
        d.text((x + 5, y + 3), lb, fill=(255, 220, 0), font=F)
        d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(90, 90, 90))
    sh.save(path, quality=88)


report = []
for v in vids:
    name = os.path.basename(v)
    stem = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(name)[0])[:40]
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'csv=p=0', v], capture_output=True, text=True).stdout.strip()
    try:
        dur = float(out)
    except ValueError:
        print(f'[건너뜀] 길이 확인 실패 — {name}');  continue

    tmp = '/content/_scan'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', v, '-vf', f'fps={SCAN_FPS}',
                    '-q:v', '3', f'{tmp}/%05d.jpg'], check=False)
    ps = sorted(glob.glob(f'{tmp}/*.jpg'))
    if not ps:
        print(f'[건너뜀] 프레임 추출 실패 — {name}');  continue

    secs = []
    for p in ps:
        i = int(os.path.basename(p)[:5]) - 1
        if flame_stats(p) < FLAME_MAX:
            secs.append(i / SCAN_FPS)

    # 전체 훑기 시트
    w0, _ = Image.open(ps[0]).size
    cw = max(340, min(MIN_W // 2, 460)) if w0 < 700 else 340
    for s in range(0, len(ps), 48):
        ck = [(f'{int(os.path.basename(p)[:5]) - 1}s', p) for p in ps[s:s + 48]]
        sheet(ck, f'{OUT}/{stem}__overview_{s // 48 + 1:02d}.jpg', 6, cw,
              f'{name[:80]}  ·  {dur:.0f}초  ·  전체 훑기')

    # 후보 구간 묶기
    wins = []
    for t in secs:
        if wins and t - wins[-1][1] <= GAP:
            wins[-1][1] = t
        else:
            wins.append([t, t])
    wins = [(max(0, a - PAD), min(dur, b + 1 / SCAN_FPS + PAD)) for a, b in wins]

    print(f'\n[{name[:70]}]  {dur:.0f}초  ·  확인 우선 구간 {len(wins)}개'
          f'  (전체 훑기 시트도 반드시 볼 것)')
    for k, (a, b) in enumerate(wins, 1):
        print(f'   {k:>2}. {a:6.1f} – {b:6.1f}초   ({b - a:.1f}초)')
        hi = '/content/_hi'
        shutil.rmtree(hi, ignore_errors=True); os.makedirs(hi)
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{a}', '-i', v,
                        '-t', f'{b - a}', '-vf', f'fps={HI_FPS}',
                        '-q:v', '2', f'{hi}/%04d.jpg'], check=False)
        hp = sorted(glob.glob(f'{hi}/*.jpg'))
        if not hp:
            continue
        items = [(f'{a + (int(os.path.basename(p)[:4]) - 1) / HI_FPS:.2f}s', p) for p in hp]
        w0, _ = Image.open(hp[0]).size
        sheet(items, f'{OUT}/{stem}__w{k:02d}_{a:.0f}-{b:.0f}s.jpg', 4,
              max(MIN_W, w0), f'{name[:60]}  ·  구간 {k}  ·  {a:.1f}–{b:.1f}초')
        shutil.rmtree(hi, ignore_errors=True)

    report.append({'file': name, 'dur_s': round(dur, 1),
                   'windows': [[round(a, 1), round(b, 1)] for a, b in wins]})
    shutil.rmtree(tmp, ignore_errors=True)

json.dump(report, open(f'{OUT}/scan.json', 'w'), ensure_ascii=False, indent=1)
shutil.make_archive('/content/scan', 'zip', OUT)
print(f'\n-> scan.zip  {os.path.getsize("/content/scan.zip") / 1e6:.1f}MB')
files.download('/content/scan.zip')
