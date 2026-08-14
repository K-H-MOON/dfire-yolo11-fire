# ===== 발연 프레임 추출 =====
#
# `docs/TIMELINE.md` 에 확정한 구간만 초당 2장으로 뽑음.
# **시간대는 그 문서에서 옮긴 것임. 고칠 일이 있으면 문서를 먼저 고칠 것.**
#
# 하는 일 — 확정 구간 추출 · 전환/그래픽 구간 제외 · ft 상단 15% 잘라내기 ·
# 출처마다 **최대다양 순서**를 매겨 manifest 에 넣음.
# 순서만 매겨 두면 상한을 나중에 바꿔도 다시 안 뽑아도 됨 (`rank < 32`).
#
# 돌리기 전에 두 가지를 스스로 검사함 — 아래 `사전 검사` 절.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 10~20분.

import os, glob, json, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image
from google.colab import drive

SRC = '/content/drive/MyDrive/smoke_frames'
OUT = '/content/drive/MyDrive/smoke_frames/extracted'
FPS = 2

# ---------------------------------------------------------------------------
# 확정 구간 — docs/TIMELINE.md 에서 옮김
#
#   match   드라이브 파일명에 들어 있는 고유 문자열
#   crop    화면을 잘라낼 때만 씀. (위, 아래) 비율
#   skip    구간 안에서 빼야 하는 초 (전환 프레임 등)
#   tags    태그별 (시작초, 끝초) 목록. 초는 **재생 초**임
# ---------------------------------------------------------------------------
TIMELINE = {
    # 배속·컷 없음. 유일하게 시간 축 검증이 가능한 자료. 앞이 잘려 0초에 이미 연기
    's2': dict(match='発生するまで', tags={
        'smoke': [(0, 170)]}),

    # ×4·×6배속. 자막이 연기와 겹쳐 226~268초를 뺌. 상단 15% 를 잘라 상단 자막 제거
    # 269~273 은 연기가 상단 자막을 덮으므로 자르지 않으면 소재가 오염됨
    'ft': dict(match='ファイテック', crop=(0.15, 0.0), tags={
        'smoke': [(218, 225), (269, 273)]}),

    # 컷 5군데. 재생 44초가 실제 8분 25초에서 발췌한 것. 유온계·경과시간이 화면에
    #
    # 2026-08-11 끝을 57 → 56.5 로 당김. 사용자가 원본을 재생해
    # **57초부터 불꽃이 보이고 팬 안에 불이 붙었음**을 확인함
    'j01': dict(match='天ぷら油の過熱発火', tags={
        'smoke':       [(38, 56.5)],
        'smoke_faint': [(14, 37)]}),

    # 컷 1군데(58초는 화면이 깨진 전환 프레임). 640×480 인데 대비가 좋아 씀
    'm3': dict(match='異物挟み込み', skip=[58], tags={
        'smoke':       [(59, 92)],
        'smoke_faint': [(41, 57)]}),

    # 41초에 시간을 건너뜀. 40.2~40.9 는 와이프 전환이라 41 부터 잡음
    # 2회차(96~127초)는 자동소화가 작동해 발연이 한 프레임도 없음
    'p2': dict(match='汚れた鍋', tags={
        'smoke': [(41, 51)]}),

    # 실험 3회(500mL·100mL·IH). 방송 자막이 상시 있으나 연기와 안 겹침
    'kt': dict(match='カンテレNEWS', tags={
        'smoke':        [(354, 362), (371, 372), (377, 381)],
        'smoke_reheat': [(416, 420)]}),

    # 23초는 인터뷰에서 넘어오는 디졸브. 40~48초는 유리 뚜껑 아래에서 타는 중이라
    # 연기와 불꽃이 섞임 — 학습·평가 모두 제외
    'kfire03': dict(match='Fire Prevention Week', tags={
        'smoke':       [(24, 36)],
        'smoke_mixed': [(40, 48)]}),

    # 0~2.5초 인트로는 본편 47~49초와 같은 장면(화소로 확인). 34~36초는 경고 자막 화면
    # 53.6~54.3 은 2번 화각의 발연 1초 — 조각이라 따로 둠
    #
    # 2026-08-11 끝을 49 → 48.5 로 당김. **49.0 프레임에 팬 전체가 불타는 화염이 있음.**
    # TIMELINE 에 `49.0초 연기만 · 49.5초 불꽃` 으로 적혀 있었으나 사용자가 원본을
    # 재생해 확인한 결과 **불꽃이 처음 나타나는 초가 49.0** 이었음
    'q1': dict(match='少ない油で発火', tags={
        'smoke':      [(37, 48.5)],
        'smoke_frag': [(53.6, 54.3)]}),

    # 39~55초는 한 번 붙었다 꺼진 뒤의 연기. 물이 아니라 뚜껑으로 껐으므로 김은 아님
    'j04': dict(match='シミュレーション', tags={
        'smoke':        [(18, 28)],
        'smoke_reheat': [(39, 55)]}),

    # 55·63초에 초록 세로 띠(인코딩 결함). 그래서 발연을 64 부터 잡음
    # 56~62 는 자막이 연기가 있다고 명시하나 은박 배경이라 화면에서 거의 안 보임
    #
    # 2026-08-11 두 곳을 고침.
    #   67.0 에 큰 화염이 있어 66.5 에서 끊음 (TIMELINE 은 68 부터 발화라고 적혀 있었음)
    #   73~81 중 **75~81 은 `제외 자막 화면이 겹침`** 이라 73~74 만 씀.
    #   TIMELINE 안에서 `발연 73-81` 과 `제외 75-81` 두 줄이 어긋나 있었고
    #   이 스크립트가 발연 줄만 따라 열일곱 중 열넷이 잘못 들어가 있었음
    'j12': dict(match='2 東京防災', skip=[55, 63], tags={
        'smoke':       [(64, 66.5), (73, 74)],
        'smoke_faint': [(56, 62)]}),

    # 50초는 뚜껑을 들어올릴 때 갇혀 있던 연기. 56~66 은 K급 소화약제라 뺌
    # 자막이 하단에 있어 냄비 위 연기와 안 겹침
    #
    # 2026-08-12 끝을 26 -> 21.5 로 당김. 사용자가 원본을 재생해
    # `처음 불이 붙은 시점은 22초. 자막 사이로 냄비에 불꽃이 붙었다`
    # `23초부터는 불씨가 점점 커진다` 라고 확인했음.
    # TIMELINE 에 `27초 점화` 로 적혀 있었으나 다섯 칸 늦은 값이었음. 아홉 장 잃음
    'kfire02': dict(match='자리를 비우면', tags={
        'smoke':        [(11, 21.5)],
        'smoke_reheat': [(50, 50)]}),

    # 한 영상에 배경이 정반대인 두 구간. 25·79초는 디졸브
    #   29~32   밝은 부엌 · 33초에 발화
    #   80~89   검은 배경 · 발화로 안 이어짐. q1 다음으로 오려내기가 쉬움
    # 2026-08-12 80~89 -> 80~88.5. 89.0 프레임에 파란 옷을 입은 사람이 들어옴
    '04': dict(match='Fire Prevention - Cooking Safety', tags={
        'smoke':       [(29, 32), (80, 88.5)],
        'smoke_faint': [(26, 28)]}),

    # 슬로우모션. 1초 떨어진 프레임의 차이가 1.0~2.3 (같은 영상 화염 구간은 40~79)
    # 162.2 부터 빨간 X 그래픽이 겹치므로 161 에서 끊음
    '05': dict(match='Chat_ Cooking Fire Prevention', tags={
        'smoke': [(154, 161)]}),

    # 클로즈업과 야외 원거리가 번갈아 나옴. 아래는 클로즈업 구간만 골라낸 것
    # 143~157 의 재가열 14초는 이전 판 기록에서 통째로 빠져 있었음
    # 2026-08-12 두 곳 고침 (QC 2판 끝 시트)
    #   55~62 -> 55~61.5   62.0 은 이미 야외 원거리 컷. 문서의 제외 목록에 63-64 가 빠져 있었음
    #   80~81  통째로 뺌    80.0 부터 오른쪽 팬에 화염. 발연이 여섯 토막이 아니라 다섯임
    '07': dict(match='Fire Safety_ Grease Fires', tags={
        'smoke':        [(39, 40), (44, 47), (55, 61.5), (65, 66), (72, 77)],
        'smoke_reheat': [(94, 101), (106, 112), (143, 157)]}),

    # 셰프 설명 컷이 계속 끼어듦. 135~140 은 팔뚝 흉터 클로즈업이라 무관
    '12': dict(match='Deep-Frying', tags={
        'smoke': [(85, 89), (92, 94), (107, 110), (116, 121), (131, 132)]}),
}


# ---------------------------------------------------------------------------
# 사전 검사 — 돌리기 전에 코드가 스스로 확인하는 둘
#
#  (1) 구간 경계가 `docs/TIMELINE.md` 의 **제외** 구간에 걸리는가
#      초를 여섯 번 잘못 골랐고 여섯 번째는 이 검사가 잡았음. 소재를 만드는
#      이 스크립트에 두는 것이 가장 값어치가 큼.
#      **`표시` 는 넣지 않음** — 표시는 제외가 아니라 사유를 남기는 것임.
#
#  (2) 뽑힌 장수가 `(끝초 − 시작초) × FPS + 1` 과 맞는가
#      2026-08-11 에 이 공식을 `초수 × FPS` 로 잘못 세어 추출이 틀렸다고 단정하고
#      재실행을 요청했음. 추출은 처음부터 맞았음. 이제 코드가 대신 셈.
# ---------------------------------------------------------------------------
EXCLUDE = {
    'q1':      [(0, 2.5), (34, 36)],
    'j12':     [(50, 54), (55, 55), (63, 63), (75, 81)],
    'kfire03': [(23, 23)],                 # 40~48 은 smoke_mixed 태그로 일부러 뽑음
    '04':      [(25, 25), (79, 79)],
    '12':      [(135, 140)],
    'ft':      [(226, 268)],
    '07':      [(48, 54), (67, 71), (82, 88), (113, 115), (116, 124), (137, 142)],
    'kfire02': [(56, 66)],                 # K급 소화약제
    'm3':      [(58, 58)],
}

_bad = []
for _k, _cfg in TIMELINE.items():
    for _tag, _rs in _cfg['tags'].items():
        for _a, _b in _rs:
            for _x in (_a, _b):
                for _p, _q in EXCLUDE.get(_k, []):
                    if _p - 1e-6 <= _x <= _q + 1e-6:
                        _bad.append((_k, _tag, _x, (_p, _q)))
if _bad:
    for _k, _tag, _x, _r in _bad:
        print(f'[{_k}] {_tag} 의 {_x}초 가 제외 구간 {_r} 에 걸림')
    raise SystemExit('제외 구간에 걸리는 경계가 있음. docs/TIMELINE.md 를 보고 고칠 것.')
print('구간 경계 검사 통과')

EXPECT = {}
for _k, _cfg in TIMELINE.items():
    _skip = set(_cfg.get('skip', []))
    for _tag, _rs in _cfg['tags'].items():
        _n = 0
        for _a, _b in _rs:
            for _i in range(int(round((_b - _a) * FPS)) + 1):
                if int(round(_a * FPS + _i)) // FPS not in _skip:
                    _n += 1
        EXPECT[(_k, _tag)] = _n
print(f'예상 장수 계산 완료 — smoke 합계 '
      f'{sum(v for (k, t), v in EXPECT.items() if t == "smoke")}장\n')


def norm(s):
    """일본어 탁점·굽은 따옴표처럼 저장 방식에 따라 달라지는 글자를 맞춤."""
    return unicodedata.normalize('NFC', s)


def dhash(path, size=8):
    a = np.asarray(Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS),
                   dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def ham(a, b):
    return int(np.unpackbits(a ^ b).sum())


def farthest_order(hs):
    """최대다양 순서. 0번에서 시작해 이미 고른 것들에서 가장 먼 것을 차례로.
    더 뽑을 서로 다른 것이 없으면 거기서 멈추고, 남은 것은 순서를 -1 로 둠."""
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


drive.mount('/content/drive')
allf = glob.glob(f'{SRC}/*')
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

manifest, missing, mismatch = {}, [], []
print(f'{"출처":<9}{"태그":<14}{"뽑음":>6}{"서로 다름":>10}')
print('-' * 42)

for key, cfg in TIMELINE.items():
    hit = [p for p in allf if norm(cfg['match']) in norm(os.path.basename(p))]
    if len(hit) != 1:
        missing.append((key, cfg['match'], len(hit)))
        print(f'{key:<9}[건너뜀] "{cfg["match"]}" 에 맞는 파일이 {len(hit)}개')
        continue
    src = hit[0]
    crop = cfg.get('crop')
    skip = set(cfg.get('skip', []))
    manifest[key] = {'file': os.path.basename(src), 'tags': {}}

    for tag, ranges in cfg['tags'].items():
        d = f'{OUT}/{tag}'
        os.makedirs(d, exist_ok=True)
        tmp = '/content/_ex'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)

        got = []
        for a, b in ranges:
            vf = f'fps={FPS}'
            if crop:                       # 위·아래를 비율로 잘라냄
                top, bot = crop
                vf = f'crop=iw:ih*{1 - top - bot}:0:ih*{top},' + vf
            # 한 구간에서 나오는 장수는 **(끝초 − 시작초) × FPS + 1** 임.
            # 마지막 초는 `.0` 한 장뿐이고 `.5` 는 끝초를 넘음.
            #
            # 2026-08-11 에 이걸 `초수 × FPS` 로 잘못 세어 구간마다 한 장씩
            # 빠진 줄 알았음. 07(여섯 구간)에서 여섯 장, 12(다섯 구간)에서
            # 다섯 장이 모자라 보여 원인을 ffmpeg 로 단정했으나 **재 보니
            # 추출은 처음부터 맞았고 예상 공식이 틀린 것이었음.**
            #
            # 지금은 넉넉히 뽑고 끝초를 넘는 것을 뒤에서 버림. 결과는 같으나
            # 경계를 ffmpeg 에 맡기지 않고 초 값으로 직접 거르는 편이
            # 나중에 헷갈리지 않음.
            subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                            '-t', str(round(b - a + 1, 2)), '-vf', vf,
                            '-q:v', '2', f'{tmp}/%04d.jpg'], check=False)
            ps = sorted(glob.glob(f'{tmp}/*.jpg'))
            for j, p in enumerate(ps):
                sec = round(a + j / FPS, 1)
                if sec > b + 1e-6 or int(sec) in skip:
                    os.remove(p);  continue
                dst = f'{d}/{key}_{sec:07.1f}.jpg'.replace(' ', '0')
                shutil.move(p, dst)
                got.append((dst, sec))
            for leftover in glob.glob(f'{tmp}/*.jpg'):
                os.remove(leftover)

        if not got:
            print(f'{key:<9}{tag:<14}{"0":>6}   [추출 실패]')
            continue

        # 최대다양 순서는 학습에 쓰는 smoke 태그에만 매김
        if tag == 'smoke':
            rank = farthest_order([dhash(p) for p, _ in got])
            uniq = sum(1 for r in rank if r >= 0)
        else:
            rank = list(range(len(got)))
            uniq = len(got)

        manifest[key]['tags'][tag] = [
            {'file': os.path.basename(p), 'sec': s, 'rank': r}
            for (p, s), r in zip(got, rank)]
        exp = EXPECT.get((key, tag))
        mark = '' if exp == len(got) else f'   [예상 {exp}장과 다름]'
        if exp != len(got):
            mismatch.append((key, tag, exp, len(got)))
        print(f'{key:<9}{tag:<14}{len(got):>6}{uniq:>10}{mark}')

    shutil.rmtree('/content/_ex', ignore_errors=True)

json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), ensure_ascii=False, indent=1)

print('\n' + '=' * 42)
for tag in ('smoke', 'smoke_faint', 'smoke_reheat', 'smoke_frag', 'smoke_mixed'):
    n = len(glob.glob(f'{OUT}/{tag}/*.jpg'))
    if n:
        print(f'{tag:<14}{n:>6}장')
cap = sum(min(32, sum(1 for x in v['tags'].get('smoke', []) if x['rank'] >= 0))
          for v in manifest.values())
print(f'\n상한 32 를 걸면 학습 후보 {cap}장 (평가 출처는 상한 없이 전부 씀)')
print(f'-> {OUT}')

if mismatch:
    print('\n[확인 필요] 뽑힌 장수가 예상과 다름 — ffmpeg 가 끝 프레임을 흘렸을 수 있음')
    for k, t, e, g in mismatch:
        print(f'  {k:<9}{t:<14}예상 {e}장 · 실제 {g}장 ({g - e:+d})')
else:
    print('\n장수 검사 통과 — 모든 태그가 (끝초 − 시작초) × FPS + 1 과 맞음')

if missing:
    print('\n[확인 필요] 드라이브에서 못 찾은 파일')
    for k, m, n in missing:
        print(f'  {k:<9}"{m}" 에 맞는 파일 {n}개')
