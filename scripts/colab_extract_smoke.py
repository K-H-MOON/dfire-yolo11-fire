# ===== 발연 프레임 추출 =====
#
# `docs/TIMELINE.md` 에 확정한 구간만 초당 2장으로 뽑음.
# **이 스크립트의 시간대는 그 문서에서 옮긴 것임. 고칠 일이 있으면 문서를 먼저 고칠 것.**
#
# 2026-08-11 에 통째로 다시 씀. 이전 판은 초기 12편 심사 때 정한 시간대를 담고
# 있었고, 열다섯 개를 원본 해상도로 다시 보니 대부분 틀렸음.
#   - 04 는 25~31초를 가리키고 있었으나 그 구간은 흰 자막에서 부엌으로 넘어가는
#     디졸브였고, 진짜 발연은 80~89초(검은 배경)였음
#   - 07 은 `발연 44~79초 연속` 으로 적혀 있었으나 실제로는 여섯 토막이고
#     사이사이가 야외 원거리 컷임
#   - 12 는 `발연 85~156초 연속` 으로 적혀 있었으나 셰프 설명 컷이 계속 끼어듦
#
# ---------------------------------------------------------------------------
# 이 스크립트가 하는 일
#
#   1. 확정 구간만 초당 2장으로 뽑음
#   2. 전환 프레임·소화약제·그래픽 구간을 뺌
#   3. ft 는 상단 15% 를 잘라냄 (자막이 연기와 겹침)
#   4. 출처마다 **최대다양 순서**를 매겨 manifest 에 넣음
#
# 4번이 중요함. **순서만 매겨 두면 상한을 나중에 바꿔도 다시 안 뽑아도 됨.**
# 상한 32 는 `rank < 32` 로 거르면 되고, 늘리고 싶으면 숫자만 바꾸면 됨.
#
# ---------------------------------------------------------------------------
# 상한과 태그 규칙 (docs/PREREGISTER_S1.md 에 사전 등록할 것)
#
#   학습에 쓰는 것        `smoke` 태그만
#   평가에만 쓰는 것       smoke_faint · smoke_reheat (태그를 유지해 따로 집계)
#   아예 안 쓰는 것        smoke_frag · smoke_mixed
#   상한                 학습으로 배정된 출처만 **최대 32장** (전체 15개의 중앙값)
#                        평가 출처는 상한 없음. 하한 10장만 확인
#   고르는 방식           최대다양 — 이미 고른 것들에서 가장 먼 프레임을 차례로
#                        0번 프레임에서 시작
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
    'j01': dict(match='天ぷら油の過熱発火', tags={
        'smoke':       [(38, 57)],
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
    # **드라이브에 없음. 올려야 함**
    'kfire03': dict(match='Fire Prevention Week', tags={
        'smoke':       [(24, 36)],
        'smoke_mixed': [(40, 48)]}),

    # 0~2.5초 인트로는 본편 47~49초와 같은 장면(화소로 확인). 34~36초는 경고 자막 화면
    # 53.6~54.3 은 2번 화각의 발연 1초 — 조각이라 따로 둠
    'q1': dict(match='少ない油で発火', tags={
        'smoke':      [(37, 49)],
        'smoke_frag': [(53.6, 54.3)]}),

    # 39~55초는 한 번 붙었다 꺼진 뒤의 연기. 물이 아니라 뚜껑으로 껐으므로 김은 아님
    'j04': dict(match='シミュレーション', tags={
        'smoke':        [(18, 28)],
        'smoke_reheat': [(39, 55)]}),

    # 55·63초에 초록 세로 띠(인코딩 결함). 그래서 발연을 64 부터 잡음
    # 56~62 는 자막이 연기가 있다고 명시하나 은박 배경이라 화면에서 거의 안 보임
    'j12': dict(match='2 東京防災', skip=[55, 63], tags={
        'smoke':       [(64, 67), (73, 81)],
        'smoke_faint': [(56, 62)]}),

    # 50초는 뚜껑을 들어올릴 때 갇혀 있던 연기. 56~66 은 K급 소화약제라 뺌
    # 자막이 하단에 있어 냄비 위 연기와 안 겹침
    'kfire02': dict(match='자리를 비우면', tags={
        'smoke':        [(11, 26)],
        'smoke_reheat': [(50, 50)]}),

    # 한 영상에 배경이 정반대인 두 구간. 25·79초는 디졸브
    #   29~32   밝은 부엌 · 33초에 발화
    #   80~89   검은 배경 · 발화로 안 이어짐. q1 다음으로 오려내기가 쉬움
    '04': dict(match='Fire Prevention - Cooking Safety', tags={
        'smoke':       [(29, 32), (80, 89)],
        'smoke_faint': [(26, 28)]}),

    # 슬로우모션. 1초 떨어진 프레임의 차이가 1.0~2.3 (같은 영상 화염 구간은 40~79)
    # 162.2 부터 빨간 X 그래픽이 겹치므로 161 에서 끊음
    '05': dict(match='Chat_ Cooking Fire Prevention', tags={
        'smoke': [(154, 161)]}),

    # 클로즈업과 야외 원거리가 번갈아 나옴. 아래는 클로즈업 구간만 골라낸 것
    # 143~157 의 재가열 14초는 이전 판 기록에서 통째로 빠져 있었음
    '07': dict(match='Fire Safety_ Grease Fires', tags={
        'smoke':        [(39, 40), (44, 47), (55, 62), (65, 66), (72, 77), (80, 81)],
        'smoke_reheat': [(94, 101), (106, 112), (143, 157)]}),

    # 셰프 설명 컷이 계속 끼어듦. 135~140 은 팔뚝 흉터 클로즈업이라 무관
    '12': dict(match='Deep-Frying', tags={
        'smoke': [(85, 89), (92, 94), (107, 110), (116, 121), (131, 132)]}),
}


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

manifest, missing = {}, []
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
        print(f'{key:<9}{tag:<14}{len(got):>6}{uniq:>10}')

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

if missing:
    print('\n[확인 필요] 드라이브에서 못 찾은 파일')
    for k, m, n in missing:
        print(f'  {k:<9}"{m}" 에 맞는 파일 {n}개')
    print('  kfire03(Fire Prevention Week · DVIDS · 26.7MB)은 smoke_frames 에 올려야 함')
