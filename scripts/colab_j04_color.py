# ===== j04 의 밝은 기둥이 연기인가 버너 화염인가 — 색으로 가름 =====
#
# 가스 화염은 파랗고 연기는 무채색임. 알파가 든 자리의 원본 화소에서
# **파란 채널 − 빨간 채널(B−R)** 을 재면 갈림.
#
# **검산은 자기 영상 안에서 함.** j04 는 `발연 18-28 · 발화 29-35` 로 확정돼 있으므로,
# 같은 판으로 18\~35초를 이어 재면 **29초에서 값이 튀어야 함.**
#   튀면   지표가 작동함 → 19\~22.5초가 어느 쪽인지 읽을 수 있음
#   안 튀면 지표 실패 → 이 숫자를 근거로 쓰지 않음
#
# 대조로 q1(37\~48.5 발연 · 49\~53 발화)도 같은 방식으로 잼.
# q1 은 잡음이 가장 적은 출처라 기준선이 됨.
#
# **사전 등록 값은 안 바꿈 — 문턱 0.06 그대로 두고 재기만 함.**
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files

SRC = '/content/drive/MyDrive/smoke_frames'
FPS = 2
THR = 0.06

# (키, 파일명 조각, 판 초, 재는 구간, 발연 끝, 잘라내기)
CASE = [
    ('j04', 'シミュレーション', 18.0, (18, 35), 28, (0.00, 0.42)),
    ('q1',  '少ない油で発火',   37.0, (37, 53), 48.5, (0.10, 0.00)),
]


def norm(s):
    return unicodedata.normalize('NFC', s)


def frames_of(src, a, b, crop):
    tmp = '/content/_c'
    shutil.rmtree(tmp, ignore_errors=True);  os.makedirs(tmp)
    vf = f'fps={FPS}'
    if crop:
        vf = f'crop=iw:ih*{1 - crop[0] - crop[1]}:0:ih*{crop[0]},' + vf
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                    '-t', str(round(b - a + 1, 2)), '-vf', vf,
                    '-q:v', '2', f'{tmp}/%05d.jpg'], check=False)
    out = []
    for j, p in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
        sec = round(a + j / FPS, 1)
        if sec <= b + 1e-6:
            out.append((sec, np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)))
    return out


allf = glob.glob(f'{SRC}/*')
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception:
    F = ImageFont.load_default()

result, shots = {}, []
for key, frag, psec, (a, b), smoke_end, crop in CASE:
    hit = [p for p in allf if norm(frag) in norm(os.path.basename(p))]
    if len(hit) != 1:
        print(f'{key}: "{frag}" 에 맞는 파일 {len(hit)}개');  continue
    fr = frames_of(hit[0], a, b, crop)
    plate = fr[0][1]

    print(f'\n[{key}]  판 {psec:g}초 · {a}\\~{b}초 · 발연은 {smoke_end}초까지')
    print(f'{"초":>7}{"알파평균":>10}{"B−R":>9}{"채도":>8}   구분')
    rec = []
    for sec, img in fr:
        al = np.clip(((img - plate) / np.maximum(255.0 - plate, 8.0)).min(2), 0, 1)
        al[al < THR] = 0
        w = al.sum()
        if w < 1:
            continue
        br  = float(((img[..., 2] - img[..., 0]) * al).sum() / w)   # 파랑 − 빨강
        sat = float(((img.max(2) - img.min(2)) * al).sum() / w)     # 채도
        tag = '발연' if sec <= smoke_end + 1e-6 else '**발화**'
        rec.append((sec, float(al.mean()), br, sat, sec <= smoke_end + 1e-6))
        print(f'{sec:>7.1f}{al.mean():>10.3f}{br:>9.1f}{sat:>8.1f}   {tag}')
        if key == 'j04' and sec in (19.0, 22.5, 26.0, 31.0):
            shots.append((f'j04 {sec:g}s  B−R {br:.1f}', img, al))
    result[key] = rec

print('\n' + '=' * 56)
for key, rec in result.items():
    s = [r[2] for r in rec if r[4]]
    f = [r[2] for r in rec if not r[4]]
    if not s or not f:
        print(f'[{key}] 한쪽 구간이 비어 검산 못 함');  continue
    ms, mf = float(np.median(s)), float(np.median(f))
    print(f'[{key}] B−R 중앙값   발연 {ms:+.1f} · 발화 {mf:+.1f}   차 {mf - ms:+.1f}')
    if abs(mf - ms) < 3:
        print(f'      **지표가 두 구간을 못 가름 — 실패로 봄.** 이 숫자를 근거로 읽지 말 것')
    else:
        print(f'      지표가 두 구간을 가름')

if 'j04' in result:
    rec = result['j04']
    q = [r[2] for r in rec if r[4] and 19.0 <= r[0] <= 22.5]
    o = [r[2] for r in rec if r[4] and r[0] > 22.5]
    f = [r[2] for r in rec if not r[4]]
    if q and o and f:
        print(f'\n[j04 판정] B−R 중앙값')
        print(f'   19~22.5초 (문제 구간)  {np.median(q):+.1f}')
        print(f'   23~28초   (발연 나머지) {np.median(o):+.1f}')
        print(f'   29~35초   (확실한 화염) {np.median(f):+.1f}')
        d1 = abs(np.median(q) - np.median(o))
        d2 = abs(np.median(q) - np.median(f))
        print(f'   문제 구간이 가까운 쪽 — {"발연" if d1 < d2 else "**화염**"}'
              f'  (발연과 {d1:.1f} · 화염과 {d2:.1f})')

if shots:
    CW = 760
    H = sum(round(CW * i.shape[0] / i.shape[1]) + 28 for _, i, _ in shots) + 8
    sh = Image.new('RGB', (2 * CW, H), (16, 16, 16))
    d = ImageDraw.Draw(sh);  y = 0
    for lab, img, al in shots:
        ch = round(CW * img.shape[0] / img.shape[1])
        d.text((6, y + 4), lab, fill=(255, 220, 0), font=F)
        sh.paste(Image.fromarray(img.astype(np.uint8)).resize((CW, ch), Image.LANCZOS), (0, y + 28))
        sh.paste(Image.fromarray((al * 255).astype(np.uint8)).convert('RGB')
                 .resize((CW, ch), Image.LANCZOS), (CW, y + 28))
        y += ch + 28
    sh.save('/content/j04_color.jpg', quality=92)
    print(f'\n-> j04_color.jpg  ({len(shots)}줄 · 왼쪽 원본 · 오른쪽 알파)')
    files.download('/content/j04_color.jpg')
