# ===== 학습·평가 출처 배정 =====
# 확정 방안 — 층화 없이 시드 고정 단순 무작위. 분할 다섯.
#
# 왜 층화를 안 하는가 — 1회차에서 재려는 값이 **출처가 달라졌으면 결론이 뒤집혔을
# 것인가** 이기 때문임. 층화는 층 구성이 흔들리는 몫을 배정에서 제거하므로
# 재려던 것을 못 재게 됨. 자세한 판단 근거는 docs/PREREGISTER_S1.md 에 있음.
#
# 이 스크립트는 프레임을 뽑기 **전에** 돌려도 됨. 배정이 장수에 의존하지 않기 때문임.
# 사전 등록의 강도가 그만큼 올라감 — 프레임을 한 장도 보기 전에 분할이 확정됨.
#
# 파이썬 표준 라이브러리만 씀. 어디서 돌려도 같은 결과가 나와야 하므로
# random.Random(seed).sample 만 사용하고 numpy 를 쓰지 않음.

import json
from random import Random

# ---------------------------------------------------------------------------
# 발연 출처 15개
#
# **주의 — 이 번호는 docs/SOURCES.md 의 표 번호와 더 이상 같지 않음.**
# 2026-08-11 에 그 표를 `합계` 순위에서 `최장 토막` 순위로 다시 매기면서 순서가
# 바뀌었음. 여기 번호를 따라 바꾸지 **않음** — 배정이 이 번호로 이미 뽑혀
# split_s1.json 에 저장돼 있으므로, 번호를 바꾸면 사전 등록한 배정이 달라짐.
# 아래는 배정을 뽑을 당시의 번호를 그대로 둔 것이고, 표와의 대응은 이러함.
#
#   1 s2 / 2 ft / 3 j01 / 4 m3 / 5 p2 / 6 07 / 7 kt / 8 12 / 9 kfire03
#   10 q1 / 11 j04 / 12 05 / 13 j12 / 14 kfire02 / 15 04
#
# dur 도 옛 값임. 배정에 쓰이지 않으므로 그대로 둠.
#
# bg 는 배정에 **쓰이지 않음.** 가져갈 분할을 고를 때와 결과를 해석할 때만 씀.
# 이 값이 틀려도 배정은 유효함 (2026-08-11 에 실제로 한 번 틀렸던 값임 —
# 15번을 밝음과 어두움 양쪽에 세고 있었고 합이 15라 검산으로 안 걸렸음).
#
# **여기 bg 값 하나가 새 표와 어긋남** — 3번 j01 을 `gray` 로 두었으나 원본
# 해상도로 다시 보니 갈색 벽 · 붉은 조명임. 해석에 쓸 때는 docs/SOURCES.md 를 볼 것.
# ---------------------------------------------------------------------------
SOURCES = [
    (1,  's2',        '天ぷら油火災が発生するまで（訓練）',      170, 'dark'),
    (2,  'ft',        'ファイテック FT-02 消火実験',             52, 'gray'),
    (3,  'j01',       '札幌 こんろ 天ぷら油の過熱発火',           35, 'gray'),
    (4,  'm3',        '札幌 ＩＨ 異物挟み込み',                   35, 'dark'),
    (5,  'p2',        'NITE ガスこんろ 3.汚れた鍋',              26, 'dark'),
    (6,  'grease07',  'Fire Safety — Grease Fires in the Kitchen', 24, 'bright'),
    (7,  'kt',        'カンテレNEWS 油少なめ揚げ焼き',            22, 'gray'),
    (8,  'deepfry12', 'How to Prevent & Douse — Deep-Frying',    20, 'dark'),
    (9,  'kfire03',   'Fire Prevention Week (DVIDS)',            13, 'dark'),
    (10, 'q1',        'NITE IHこんろ 4.少ない油で発火',          13, 'dark'),
    (11, 'j04',       '天草広域連合 天ぷら油火災シミュレーション', 13, 'dark'),
    (12, 'letschat',  "Let's Chat — Cooking Fire Prevention",    13, 'bright'),
    (13, 'j12',       '2 東京防災 天ぷら油火災実験',              12, 'bright'),
    (14, 'kfire02',   '설 명절 식용유 화재 재현 (부산소방)',       12, 'bright'),
    (15, 'prev04',    'Fire Prevention - Cooking Safety',        11, 'dark'),
]

N_EVAL   = 5          # 평가군 출처 수 (학습 10 : 평가 5)
N_SPLIT  = 5          # 분할 수
SEEDS    = [1, 2, 3, 4, 5]

assert len(SOURCES) == 15
assert len(SEEDS) == N_SPLIT

ids = [s[0] for s in SOURCES]
name = {s[0]: s[1] for s in SOURCES}
bg   = {s[0]: s[4] for s in SOURCES}
dur  = {s[0]: s[3] for s in SOURCES}

# ---------------------------------------------------------------------------
# 배정 — 요건 없음. 걸러내지 않음. 나온 그대로 씀.
# ---------------------------------------------------------------------------
splits = []
for sd in SEEDS:
    ev = sorted(Random(sd).sample(ids, N_EVAL))
    tr = sorted(set(ids) - set(ev))
    splits.append({'seed': sd, 'eval': ev, 'train': tr})

# ---------------------------------------------------------------------------
# 가져갈 분할 — 2회차부터 고정해서 쓸 평가군.
# 규칙: 평가군에 밝은 배경이 1개 이상인 분할 중 시드 번호가 가장 작은 것.
#
# 이 요건은 **배정에 걸지 않고 고정에만 걸림.** 다섯 분할의 폭은 요건 없이 잰 값
# 그대로 보고함. 요건이 없으면 밝은 배경 없는 분할이 고정될 수 있고, 그러면
# 이후 모든 회차가 밝은 배경에 대해 침묵하게 되므로 이것만 막는 것임.
# ---------------------------------------------------------------------------
carry = next((s for s in splits
              if sum(bg[i] == 'bright' for i in s['eval']) >= 1), None)
assert carry is not None, '다섯 분할 모두 평가군에 밝은 배경이 없음 — 다시 뽑을 것'

# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
print(f'발연 출처 {len(ids)}개 · 학습 {len(ids) - N_EVAL} : 평가 {N_EVAL} · 분할 {N_SPLIT}\n')
print(f'{"시드":<5}{"평가군 출처":<44}{"밝":>3}{"어":>3}{"회":>3}{"평가 초":>8}')
print('-' * 69)
for s in splits:
    ev = s['eval']
    tag = ' · '.join(name[i] for i in ev)
    b = sum(bg[i] == 'bright' for i in ev)
    d = sum(bg[i] == 'dark'   for i in ev)
    g = sum(bg[i] == 'gray'   for i in ev)
    print(f'{s["seed"]:<5}{tag[:43]:<44}{b:>3}{d:>3}{g:>3}{sum(dur[i] for i in ev):>7}초')

print(f'\n가져갈 분할 — 시드 {carry["seed"]}')
print(f'  평가 {carry["eval"]}  {" · ".join(name[i] for i in carry["eval"])}')
print(f'  학습 {carry["train"]}')

# 출처별로 평가군에 몇 번 들어갔는지 — 특정 출처가 다섯 번 다 평가군이면
# 그 출처 하나가 폭을 지배하므로 결과 해석에서 반드시 봐야 함
print('\n출처별 평가군 등장 횟수 (다섯 분할 중)')
for i in ids:
    c = sum(i in s['eval'] for s in splits)
    bar = '■' * c + '·' * (N_SPLIT - c)
    print(f'  {i:>2} {name[i]:<11} {bg[i]:<7} {bar} {c}')

json.dump({'n_eval': N_EVAL, 'seeds': SEEDS, 'splits': splits,
           'carry_seed': carry['seed'],
           'sources': [{'id': i, 'key': name[i], 'bg': bg[i], 'smoke_s': dur[i]}
                       for i in ids]},
          open('split_s1.json', 'w'), ensure_ascii=False, indent=1)
print('\n-> split_s1.json')
