# ===== 연기 오려내기 — 열다섯 출처 전수 확인 (2판) =====
#
# **소재를 만드는 스크립트가 아님. `어느 출처를 쓸 수 있는가` 를 정하는 확인용임.**
# 여기서 고른 출처만 다음 단계에서 실제로 오려냄.
#
# ---------------------------------------------------------------------------
# 1판이 틀렸던 것 — 판을 하나만 잡아 모든 구간에 썼음
#
# 1판의 코드는 이랬음.
#
#     for a, b in segs:
#         first = True
#         for sec, img in frames_of(src, a, b, crop):
#             P_MIN = ... 모든 구간에 걸쳐 최솟값 ...
#             if first:
#                 if P_FIX is None:        # ← 여기
#                     P_FIX = img.copy()
#
# `if P_FIX is None` 때문에 **판A 가 첫 구간의 첫 프레임 한 장으로 고정**되고,
# 그 한 장이 나중 구간에도 그대로 쓰였음. 판B(최솟값)도 구간을 가리지 않고
# 전 구간에 걸쳐 하나로 만들어졌음.
#
# **구간이 다르면 대개 샷이 다름.** 화각·조명·자막이 달라지므로 다른 구간의 판을
# 빼면 알파에 배경이 통째로 들어옴. 구간이 둘 이상인 일곱 개 —
# **s2 · ft · kt · j12 · 04 · 07 · 12** — 의 1판 결과는 **무효임.**
#
# 2판에서 고친 것.
#   판A·판B 를 **구간마다 따로** 만듦
#   s2 는 한 구간(0\~170초)인데도 **카메라가 줌·이동함**. 한 구간 안에서도
#   **창(window) 단위**로 판을 새로 잡을 수 있게 `win` 을 둠 (s2 만 20초)
#   프레임을 쌓지 않음. 판은 흘려보내며 만들고 **필요한 표본·끝 프레임만** 남김
#
# ---------------------------------------------------------------------------
# 방법 — 배경 차분. 다섯 번 시험해서 정한 것임
#
#     본 것 = 알파 × 연기색 + (1 − 알파) × 배경
#     알파  = (본 것 − 판) / (연기색 − 판)          연기색은 흰 연기이므로 255
#
# 세 채널에서 각각 풀고 **가장 작은 값**을 씀. 한 채널이라도 안 밝아지면 연기가
# 아니라고 보는 쪽이 안전함.
#
# **휘도 키잉은 버렸음.** 배경에 밝은 것이 하나라도 있으면 통째로 주움 —
# m3 에서 유온계·경과시간·워터마크·스토브를 다 주웠고, 07 에서 스토브와 벽을 주웠음.
# 열다섯 중 배경에 밝은 것이 없는 자료는 q1 하나뿐임.
#
# ---------------------------------------------------------------------------
# 판을 두 가지로 만들어 **둘 다 보여 줌**
#
#   판A  그 구간(창)의 **첫 프레임**. 연기가 시작되는 순간이라 가장 깨끗할 때가 많음
#   판B  그 구간(창) 프레임의 **화소별 최솟값**. 연기가 움직이면 배경만 남음
#
# 어느 쪽이 나은지는 자료마다 다름. 05 는 첫 프레임에 이미 연기가 있어 판B 가
# 나았고(알파 0.008 → 0.022), j12 는 연기가 한자리에 머물러 판B 가 무너졌음.
# **숫자로 고르면 틀림** — 05 와 07 이 똑같이 0.022 인데 07 만 연기가 제대로
# 잡혔음. **시트를 눈으로 보고 고를 것.**
#
# ---------------------------------------------------------------------------
# 구간 끝을 같은 코드로 확인함
#
# 4차 시험에서 j12 의 67.0초 프레임에 **큰 불꽃**이 있는 것을 봤음. 소재에 불꽃이
# 섞이면 **불꽃을 연기로 가르치게 됨.**
#
# 끝 점검을 따로 만들었다가 **추출과 다른 프레임을 보여 주는 것**을 발견했음
# (`-ss N -frames:v 1` 과 `-vf fps=2` 가 다른 그림을 줌).
# **그래서 이 스크립트 안에 넣었음** — 소재를 만드는 그 코드가 그 프레임을 보여 줌.
#
# 이 확인으로 q1 · j01 · j12 의 끝을 당겼음 (→ docs/TIMELINE.md 의 정정 절).
# 아래 구간은 **당긴 뒤의 값**임.
#
# ---------------------------------------------------------------------------
# 만들어지는 것
#
#   qc/{출처}_a.jpg   표본 × [원본 · 판A · 알파A · 합성A · 알파B · 합성B]
#                     줄 머리에 **어느 구간(창)의 판인지** 적힘
#   qc/{출처}_z.jpg   구간마다 끝 세 장 (끝 −1.0 · 끝 −0.5 · 끝)
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 20~40분.

import os, glob, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

SRC   = '/content/drive/MyDrive/smoke_frames'
STEAM = '/content/drive/MyDrive/smoke_frames/steam'
OUT   = '/content/qc'
FPS   = 2

# docs/TIMELINE.md 의 `제외` 와 자막 화면 시각. 고른 초가 걸리면 멈춤
EXCLUDE = {
    'q1':      [(0, 2.5), (34, 36), (58, 78)],
    'kfire02': [(1, 5), (9, 10), (15, 18), (22, 25), (30, 34), (52, 54), (56, 66), (67, 69)],
    'j12':     [(50, 54), (55, 55), (63, 63), (75, 81)],
    '05':      [(145, 147), (148, 153), (162.2, 166)],
    '07':      [(48, 54), (67, 71), (82, 88), (113, 115), (116, 124), (137, 142)],
    'm3':      [(58, 58)],
    'ft':      [(226, 268)],
    'j01':     [],
    's2':      [],
    'p2':      [],
    'kt':      [],
    'kfire03': [(23, 23), (40, 48)],
    'j04':     [],
    '04':      [(25, 25), (79, 79)],
    '12':      [(135, 140)],
}

# (이름, 파일명 조각, 구간 목록, 잘라내기(위,아래), 판 창 길이(초))
#
# scripts/colab_extract_smoke.py 의 smoke 태그와 같은 구간임.
# **여기를 고칠 일이 있으면 docs/TIMELINE.md 를 먼저 고칠 것.**
#
# 창 길이는 **s2 에만** 둠. s2 는 한 구간이지만 카메라가 줌·이동해서
# 60초에서 잰 선이 0·8·120·160초에서 어긋났음(docs/TIMELINE.md). 판도 같은 이유로
# 하나로는 못 잡음. 나머지 열넷은 구간 안에서 화각이 고정이라 창을 안 씀.
CASE = [
    ('s2',      '発生するまで',                     [(0, 170)],   None,        20.0),
    ('ft',      'ファイテック',                     [(218, 225), (269, 273)], (0.15, 0.0), None),
    # 끝을 57 → 56.5 로 당김. 사용자가 원본 재생 — `57초부터 불꽃, 팬 안에 불이 붙었다`
    ('j01',     '天ぷら油の過熱発火',                [(38, 56.5)], None,        None),
    ('m3',      '異物挟み込み',                     [(59, 92)],   None,        None),
    ('p2',      '汚れた鍋',                        [(41, 51)],   None,        None),
    ('kt',      'カンテレNEWS',                     [(354, 362), (371, 372), (377, 381)], None, None),
    ('kfire03', 'Fire Prevention Week',            [(24, 36)],   None,        None),
    # 끝을 49 → 48.5 로 당김. 사용자가 원본 재생 — `불꽃이 처음 나타나는 초 49.0`
    ('q1',      '少ない油で発火',                   [(37, 48.5)], None,        None),
    ('j04',     'シミュレーション',                 [(18, 28)],   None,        None),
    # 64~67 → 64~66.5 (67.0 에 큰 화염). 73~81 → 73~74 (75~81 은 `제외 자막 화면`)
    ('j12',     '2 東京防災',                       [(64, 66.5), (73, 74)], None, None),
    ('kfire02', '자리를 비우면',                    [(11, 26)],   None,        None),
    ('04',      'Fire Prevention - Cooking Safety', [(29, 32), (80, 89)], None, None),
    ('05',      'Chat_ Cooking Fire Prevention',    [(154, 161)], None,        None),
    ('07',      'Fire Safety_ Grease Fires',        [(39, 40), (44, 47), (55, 62),
                                                    (65, 66), (72, 77), (80, 81)], None, None),
    ('12',      'Deep-Frying',                      [(85, 89), (92, 94), (107, 110),
                                                    (116, 121), (131, 132)], None, None),
]


def norm(s):
    return unicodedata.normalize('NFC', s)


def bad(key, *secs):
    out = []
    for s in secs:
        for a, b in EXCLUDE.get(key, []):
            if a - 1e-6 <= s <= b + 1e-6:
                out.append((s, (a, b)))
    return out


def windows(segs, win):
    """구간을 판 단위로 쪼갬. win 이 없으면 구간 하나가 창 하나임.
    돌려주는 것 — (창시작, 창끝, 원구간)."""
    out = []
    for a, b in segs:
        if not win:
            out.append((a, b, (a, b)));  continue
        s = a
        while s <= b - 1e-6 or abs(s - a) < 1e-6:
            e = min(s + win - 1.0 / FPS, b)
            out.append((s, e, (a, b)))
            if e >= b - 1e-6:
                break
            s = e + 1.0 / FPS
    return out


def frames_of(src, a, b, crop=None):
    """추출 스크립트와 **같은 방식**으로 뽑음. 목록이 아니라 하나씩 내줌."""
    tmp = '/content/_q'
    shutil.rmtree(tmp, ignore_errors=True);  os.makedirs(tmp)
    vf = f'fps={FPS}'
    if crop:
        top, bot = crop
        vf = f'crop=iw:ih*{1 - top - bot}:0:ih*{top},' + vf
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                    '-t', str(round(b - a + 1, 2)), '-vf', vf,
                    '-q:v', '2', f'{tmp}/%05d.jpg'], check=False)
    for j, p in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
        sec = round(a + j / FPS, 2)
        if sec > b + 1e-6:
            continue
        yield sec, np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)


def nearest(store, want, sec, img):
    """want 에 가장 가까운 프레임 하나만 남김. 0.5초 간격이라 둘이 걸릴 수 있음."""
    cur = store.get(want)
    if cur is None or abs(sec - want) < abs(cur[0] - want):
        store[want] = (sec, img.copy())


def alpha(img, plate, smoke=255.0):
    return np.clip(((img - plate) / np.maximum(smoke - plate, 8.0)).min(2), 0, 1)


def clean(a, thr=0.06):
    a = a.copy();  a[a < thr] = 0
    return a


def comp(img, a, bg):
    h, w = a.shape
    b = np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((w, h)), dtype=np.float32)
    a3 = a[..., None]
    return np.clip(a3 * img + (1 - a3) * b, 0, 255).astype(np.uint8)


def to_img(x):
    if x.ndim == 2:
        x = np.repeat((x * 255)[..., None], 3, 2)
    return Image.fromarray(x.astype(np.uint8))


def sheet(rows, path, cw=440, hdr=34):
    h0, w0 = rows[0][0][1].shape[:2]
    ch = round(cw * h0 / w0)
    cols = max(len(r) for r in rows)
    sh = Image.new('RGB', (cols * cw, len(rows) * (ch + hdr) + 8), (16, 16, 16))
    d = ImageDraw.Draw(sh)
    for r, row in enumerate(rows):
        for c, (lab, x) in enumerate(row):
            y = r * (ch + hdr)
            d.text((c * cw + 6, y + 4), lab, fill=(255, 220, 0), font=F)
            sh.paste(to_img(x).resize((cw, ch), Image.LANCZOS), (c * cw, y + hdr))
    sh.save(path, quality=86)


drive.mount('/content/drive')
allf = glob.glob(f'{SRC}/*')
shutil.rmtree(OUT, ignore_errors=True);  os.makedirs(OUT)

cand = [p for p in glob.glob(f'{STEAM}/bg/*.jpg') if norm('로봇고_bg') in norm(os.path.basename(p))]
BG = np.asarray(Image.open(sorted(cand)[len(cand) // 2]).convert('RGB'), dtype=np.float32)
print(f'합성 배경 → {os.path.basename(sorted(cand)[len(cand) // 2])}\n')

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
except Exception:
    F = ImageFont.load_default()

# ---- 구간 경계가 제외에 걸리는지 먼저 전부 검사 ----
stop = False
for key, _, segs, _, _ in CASE:
    chk = [x for a, b in segs for x in (a, b)]
    hits = bad(key, *chk)
    if hits:
        stop = True
        for s, rng in hits:
            print(f'[{key}] {s}초 가 제외 구간 {rng} 에 걸림')
if stop:
    raise SystemExit('\n제외 구간에 걸리는 초가 있음. docs/TIMELINE.md 를 보고 고칠 것.')
print('구간 경계 검사 통과\n')

summary, missing = [], []
for key, frag, segs, crop, win in CASE:
    hit = [p for p in allf if norm(frag) in norm(os.path.basename(p))]
    if len(hit) != 1:
        missing.append((key, frag, len(hit)))
        print(f'{key:<9}[건너뜀] "{frag}" 에 맞는 파일이 {len(hit)}개', flush=True)
        continue
    src = hit[0]

    wins = windows(segs, win)
    per = 3 if len(wins) < 3 else 1          # 창이 많으면 창당 한 장만
    rows_data, end_by_seg, n, den0 = [], {}, 0, None

    for wa, wb, orig in wins:
        span = max(wb - wa, 1e-6)
        want_s = [wa + span * f for f in ((0.25, 0.55, 0.9) if per == 3 else (0.55,))]
        want_e = [max(orig[1] - 1.0, orig[0]), max(orig[1] - 0.5, orig[0]), orig[1]]
        want_e = [w for w in want_e if wa - 1e-6 <= w <= wb + 1e-6]

        P_FIX, P_MIN, got_s, got_e = None, None, {}, {}
        for sec, img in frames_of(src, wa, wb, crop):
            n += 1
            if P_FIX is None:
                P_FIX = img.copy()
            P_MIN = img.copy() if P_MIN is None else np.minimum(P_MIN, img)
            for w in want_s:
                if abs(sec - w) < 0.26:
                    nearest(got_s, w, sec, img)
            for w in want_e:
                if abs(sec - w) < 0.26:
                    nearest(got_e, w, sec, img)
        if P_FIX is None:
            continue
        if den0 is None:
            den0 = float(((255.0 - P_FIX).min(2) < 40).mean())

        lab = f'{wa:g}~{wb:g}'
        for w in sorted(got_s):
            sec, img = got_s[w]
            rows_data.append((lab, sec, img, P_FIX, P_MIN))
        for w in sorted(got_e):
            end_by_seg.setdefault(orig, []).append(got_e[w])

    if not n or not rows_data:
        missing.append((key, 'frames', 0))
        print(f'{key:<9}[건너뜀] 프레임을 못 뽑음', flush=True)
        continue

    rows, mA, mB = [], [], []
    for lab, sec, img, P_FIX, P_MIN in rows_data:
        aA = clean(alpha(img, P_FIX))
        aB = clean(alpha(img, P_MIN))
        mA.append(aA.mean());  mB.append(aB.mean())
        rows.append([('원본 %.1fs  [판 %s]' % (sec, lab), img),
                     ('판A 첫프레임 %s' % lab, P_FIX),
                     ('알파A %.3f' % aA.mean(), aA), ('합성A', comp(img, aA, BG)),
                     ('알파B %.3f' % aB.mean(), aB), ('합성B', comp(img, aB, BG))])
    sheet(rows, f'{OUT}/{key}_a.jpg')

    zr = []
    for (a, b), got in end_by_seg.items():
        zr.append([(f'{a:g}~{b:g} 끝 {s:g}s', im)
                   for s, im in sorted(got, key=lambda t: t[0])])
    if zr:
        sheet(zr, f'{OUT}/{key}_z.jpg')

    mA, mB = float(np.mean(mA)), float(np.mean(mB))
    summary.append((key, n, len(wins), mA, mB, 100 * den0))
    print(f'{key:<9}{n:>4}장 · 창 {len(wins):>2}개   알파A {mA:.3f} · 알파B {mB:.3f}'
          f'   분모40미만 {100 * den0:4.1f}%', flush=True)

print('\n' + '=' * 62)
print(f'{"출처":<10}{"프레임":>6}{"창":>4}{"알파A":>8}{"알파B":>8}{"분모40미만":>11}')
for k, n, w, a, b, d_ in summary:
    print(f'{k:<10}{n:>6}{w:>4}{a:>8.3f}{b:>8.3f}{d_:>10.1f}%')
print('\n**숫자로 고르지 말 것.** 05 와 07 이 같은 0.022 인데 07 만 연기가 잡혔음.')
print('알파 평균은 **창마다 판이 다르므로 창 사이 견주기에도 못 씀.** 시트를 볼 것.')
print('qc/{출처}_a.jpg 로 알파와 합성을, qc/{출처}_z.jpg 로 끝에 불꽃이 있는지 볼 것.')

shutil.make_archive('/content/qc', 'zip', OUT)
print(f'\n-> qc.zip  {os.path.getsize("/content/qc.zip") / 1e6:.1f}MB')
if missing:
    print('\n[확인 필요]')
    for k, m, c in missing:
        print(f'  {k:<9}"{m}" {c}개')
files.download('/content/qc.zip')
