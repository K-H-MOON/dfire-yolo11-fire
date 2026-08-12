# ===== 김(음성) 프레임 추출 =====
#
# `docs/STEAM_TIMELINE.md` 에 적은 구간만 뽑음.
# **이 스크립트의 시간대는 그 문서에서 옮긴 것임. 고칠 일이 있으면 문서를 먼저 고칠 것.**
#
# 발연 쪽 `colab_extract_smoke.py` 와 짝임. 다른 점 셋.
#
#   1. 자료가 **다른 드라이브 폴더**에 있음 — `조리 데이터 영상` (공유받은 폴더)
#      발연은 `smoke_frames` (본인 드라이브)였음
#   2. 출처가 **영상이 아니라 급식실**임. 숭곡중은 네 편이 한 출처임
#   3. 뽑는 밀도가 태그마다 다름 — 투입만 초당 4장
#
# ---------------------------------------------------------------------------
# 장수 세는 식 (2026-08-11 에 틀렸던 것)
#
#   한 구간에서 나오는 장수 = **(끝초 − 시작초) × 밀도 + 1**
#
# 이것을 `초수 × 밀도` 로 잘못 세어 발연 추출이 틀린 줄 알고 재실행을 요청한 일이
# 있었음. 추출은 맞았고 예상 공식이 틀렸던 것임.
#
# ---------------------------------------------------------------------------
# 태그와 밀도 (docs/PREREGISTER_S1.md 에 사전 등록할 것)
#
#   steam_in    투입 김. 재료를 넣은 직후 · 화면을 뒤덮음        초당 4장
#   steam_near  근접 김. 앞쪽 화구                              초당 2장
#   steam_far   원거리 김. 뒤쪽 화구이거나 카메라가 멂            초당 2장
#   bg          김이 없는 정상 조리. 배경 오탐용                  초당 1장
#
#   한 프레임에 태그는 하나. 겹치면 steam_in > steam_near > steam_far
#   상한 없음. 평가군이므로 전부 씀. 대신 급식실마다 **최대다양 순서**를 매겨 두어
#   나중에 상한을 걸고 싶으면 다시 안 뽑아도 되게 함
#
# 순서를 **급식실 단위로** 매기는 것이 발연과 다른 점임. 숭곡중은 네 편에서 나온
# 프레임을 한 뭉치로 놓고 순서를 매김 — 편이 달라도 같은 주방 같은 카메라라
# 서로 닮은 프레임이 나올 수 있기 때문임.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요.
# 큰 파일이 여럿이라(숭곡중 튀김 956MB · 원촌중 973MB) 20~40분 걸릴 수 있음.

import os, glob, json, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image
from google.colab import drive

# ---------------------------------------------------------------------------
# 자료가 있는 곳 — **공유받은 폴더임 (소유자 hankookro@gmail.com)**
#
# 공유 문서함은 Colab 에서 경로로 바로 안 보임. 드라이브 웹에서
# `조리 데이터 영상` 폴더에 마우스 오른쪽 → `바로가기 추가` 로 내 드라이브에
# 바로가기를 만들어야 아래 경로에 잡힘.
#
# 못 찾으면 스크립트가 후보 경로를 전부 훑고 무엇이 없는지 찍어 줌.
# ---------------------------------------------------------------------------
SRC_ROOTS = [
    '/content/drive/MyDrive/조리 데이터 영상',
    '/content/drive/MyDrive/smoke_frames/조리 데이터 영상',
    '/content/drive/MyDrive',                      # 마지막 수단 — 전체를 훑음
]
OUT = '/content/drive/MyDrive/smoke_frames/steam'   # ← 발연과 다른 폴더

FPS = {'steam_in': 4, 'steam_near': 2, 'steam_far': 2, 'bg': 1}

# ---------------------------------------------------------------------------
# 구간 — docs/STEAM_TIMELINE.md 에서 옮김
#
#   급식실: [ (파일명 조각, 태그, 시작초, 끝초), ... ]
#
# 파일명 조각은 드라이브 제목에 들어 있는 고유 문자열임. 한글은 저장 방식에 따라
# 자모가 갈라져 있을 수 있어(NFD) 비교 전에 NFC 로 맞춤.
# ---------------------------------------------------------------------------
STEAM = {
    '영동중': [
        ('영동중 국탕',            'steam_far',    0, 145),   # 트레이 뒤 [애매함]
        ('영동중 국탕',            'steam_near', 145, 286),   # 2:25 화각 변경 · 클로즈업
    ],
    '숭곡중': [
        ('숭곡중_튀김',            'steam_in',    13,  15),
        ('숭곡중_튀김',            'steam_in',    94,  98),
        ('숭곡중_튀김',            'steam_near',   0,  12.5),
        ('숭곡중_튀김',            'steam_near',  15.5, 38),
        ('숭곡중_튀김',            'steam_far',   38,  93.5),
        ('숭곡중_튀김',            'steam_far',   98.5, 103),
        ('숭곡중_국탕 (재료투입1)', 'steam_near',   0,  91),   # 55~79 클로즈업 [애매함]
        ('숭곡중_국탕(조리삽)',     'steam_near',   0,  63),
        ('숭곡중_국탕 (재료투입2)', 'steam_near',   0,  28),   # 옅음 [애매함]
    ],
    '원촌중':   [('원촌중_튀김(full)',      'steam_near',  24, 124)],
    '내곡중':   [('내곡중_국탕',            'steam_near',   0,  99)],
    '진선여고': [('진선여고_튀김(쉐이크)',   'steam_near',   0,  42),
                ('진선여고_튀김(배출)',     'steam_far',    0,  38)],   # 정면은 조용함
    '인화여중': [('인화여중_볶음(재료투입2)', 'steam_near',  0,  68)],
    '남일고':   [('남일고_튀김(투입)',       'steam_in',    0,  40)],   # 40초 전부 [애매함]
    '금정초':   [('금정초_튀김(투입)',       'steam_near', 20,  53)],   # 25~33 촬영자 손가락
    '로봇고':   [('로봇고_튀김(투입)',       'steam_far',   0,  13),    # 우측 솥만
                ('로봇고_튀김(투입)',       'steam_near', 14,  33)],   # 정면만
    '부산체고': [('부산체고_튀김(투입)',     'steam_near',   0,  30)],   # 옅음 [애매함]
    '울산현대차':[('울산현대차_오토틸팅',    'steam_near',   0,  13)],   # 실제 13.67초
}

# 김이 없는 정상 조리 — 배경 오탐용. 초당 1장
#
# `[검토]` 가 붙은 것은 김이 아주 미세하게 있을 수 있어 시트로 확인한 뒤 써야 함.
# 확인 전에 배경 음성으로 쓰면 **김이 있는 프레임을 김이 없다고 채점**하게 됨.
BG = {
    '개원중':   [('개원중cctv_튀김(투입)',    0, 289),
                ('개원중cctv_튀김(배출)',     0, 240)],
    '논현중':   [('논현중_튀김(full)',        0, 435)],   # [검토] 김이 조금 있음
    '로봇고_bg':[('로봇고_튀김(쉐이크)',       0,  -1),
                ('로봇고_튀김(배출)',         0,  -1)],   # [검토] 우측 솥 아주 미세
    '숭곡중_bg':[('숭곡중_볶음(재료투입2)',    0,  71)],   # [검토] 아주 미세
}
DO_BG = True          # 배경 오탐용도 함께 뽑을지

REVIEW = {'논현중', '로봇고_bg', '숭곡중_bg'}   # 시트 확인 전에는 채점에 쓰지 말 것


def norm(s):
    """한글 자모 분리(NFD)·일본어 탁점처럼 저장 방식에 따라 달라지는 글자를 맞춤."""
    return unicodedata.normalize('NFC', s)


def dhash(path, size=8):
    a = np.asarray(Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS),
                   dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def ham(a, b):
    return int(np.unpackbits(a ^ b).sum())


def farthest_order(hs):
    """최대다양 순서. 0번에서 시작해 이미 고른 것들에서 가장 먼 것을 차례로.
    더 뽑을 서로 다른 것이 없으면 멈추고 남은 것은 -1 로 둠."""
    n = len(hs)
    order, d = [0], [ham(hs[0], h) for h in hs]
    while len(order) < n:
        i = int(np.argmax(d))
        if d[i] == 0:
            break
        order.append(i)
        d = [min(x, ham(hs[i], h)) for x, h in zip(d, hs)]
    rank = [-1] * n
    for r, i in enumerate(order):
        rank[i] = r
    return rank


def duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


# ---------------------------------------------------------------------------
drive.mount('/content/drive')

allf = []
for root in SRC_ROOTS:
    if os.path.isdir(root):
        allf = [p for p in glob.glob(f'{root}/**/*', recursive=True) if os.path.isfile(p)]
        if allf:
            print(f'자료 폴더 → {root}  (파일 {len(allf)}개)')
            break
if not allf:
    raise SystemExit('자료 폴더를 못 찾음. 드라이브 웹에서 `조리 데이터 영상` 폴더에\n'
                     '  바로가기 추가 → 내 드라이브  를 한 뒤 다시 돌릴 것.')

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

PLAN = dict(STEAM)
if DO_BG:
    for k, v in BG.items():
        PLAN[k] = [(f, 'bg', a, b) for f, a, b in v]

manifest, missing = {}, []
print(f'\n{"급식실":<11}{"태그":<12}{"뽑음":>6}{"서로 다름":>10}')
print('-' * 42)

for site, items in PLAN.items():
    got = []                                    # 급식실 단위로 모음
    for item in items:
        frag, tag, a, b = item
        hit = [p for p in allf if norm(frag) in norm(os.path.basename(p))]
        if len(hit) != 1:
            missing.append((site, frag, len(hit)))
            continue
        src, fps = hit[0], FPS[tag]

        if b < 0:                               # -1 = 영상 끝까지
            b = int(duration(src))
            if b <= 0:
                missing.append((site, frag, 0));  continue

        tmp = '/content/_st'
        shutil.rmtree(tmp, ignore_errors=True);  os.makedirs(tmp)
        # 넉넉히 뽑고 끝초를 넘는 것은 초 값으로 직접 버림.
        # 경계를 ffmpeg 에 맡기지 않는 편이 나중에 헷갈리지 않음.
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                        '-t', str(round(b - a + 1, 2)), '-vf', f'fps={fps}',
                        '-q:v', '2', f'{tmp}/%05d.jpg'], check=False)

        d = f'{OUT}/{tag}'
        os.makedirs(d, exist_ok=True)
        stem = norm(os.path.splitext(os.path.basename(src))[0])[:24].replace(' ', '_')
        for j, p in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
            sec = round(a + j / fps, 2)
            if sec > b + 1e-6:
                os.remove(p);  continue
            dst = f'{d}/{site}_{stem}_{sec:08.2f}.jpg'.replace(' ', '0')
            shutil.move(p, dst)
            got.append((dst, tag, sec, os.path.basename(src)))
        shutil.rmtree(tmp, ignore_errors=True)

    if not got:
        print(f'{site:<11}{"—":<12}{"0":>6}   [추출 실패]')
        continue

    rank = farthest_order([dhash(p) for p, _, _, _ in got])
    manifest[site] = {
        'review': site in REVIEW,
        'frames': [{'file': os.path.basename(p), 'tag': t, 'sec': s, 'src': f, 'rank': r}
                   for (p, t, s, f), r in zip(got, rank)],
    }
    per = {}
    for (_, t, _, _) in got:
        per[t] = per.get(t, 0) + 1
    uniq = sum(1 for r in rank if r >= 0)
    print(f'{site:<11}{" ".join(f"{t}{c}" for t, c in per.items()):<12}'
          f'{len(got):>6}{uniq:>10}' + ('   [시트 확인 필요]' if site in REVIEW else ''))

json.dump(manifest, open(f'{OUT}/manifest_steam.json', 'w'), ensure_ascii=False, indent=1)

print('\n' + '=' * 42)
for tag in ('steam_in', 'steam_near', 'steam_far', 'bg'):
    n = len(glob.glob(f'{OUT}/{tag}/*.jpg'))
    if n:
        print(f'{tag:<12}{n:>6}장')

sites = [s for s in manifest if not manifest[s]['review'] and s in STEAM]
u = {s: sum(1 for f in manifest[s]['frames'] if f['rank'] >= 0) for s in sites}
print(f'\n김 급식실 {len(sites)}곳 · 서로 다름 합계 {sum(u.values())}장')
print(f'가장 작은 곳  {min(u, key=u.get)} {min(u.values())}장   (출처당 하한 10장)')
under = [s for s, v in u.items() if v < 10]
print('하한 미달  ' + (', '.join(under) if under else '없음'))
print(f'-> {OUT}')

if missing:
    print('\n[확인 필요] 드라이브에서 못 찾은 파일')
    for s, f, n in missing:
        print(f'  {s:<11}"{f}" 에 맞는 파일 {n}개')
