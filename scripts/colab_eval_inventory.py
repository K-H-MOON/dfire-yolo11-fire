import os, glob, json, unicodedata
from google.colab import drive

SRC = '/content/drive/MyDrive/smoke_frames'

ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}
TRAIN = {1: [1, 4, 6, 7, 8, 9, 11, 12, 14, 15], 2: [3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
         3: [1, 2, 5, 7, 8, 11, 12, 13, 14, 15], 4: [1, 3, 6, 8, 9, 10, 11, 13, 14, 15],
         5: [1, 2, 3, 4, 7, 8, 9, 13, 14, 15]}

drive.mount('/content/drive')


def norm(s):
    return unicodedata.normalize('NFC', s)


print('=' * 78)
print('평가 자료 점검 — 평가 스크립트를 쓰기 전에 무엇이 어디에 있는지부터 봄')
print('=' * 78)
print('  1층 보고 규칙 여섯 줄을 채우려면 아래가 다 있어야 함')
print('    E 조리 연기   분할마다 평가로 배정된 발연 출처의 프레임')
print('    D 김          급식실 11곳 2,044장')
print('    배경 오탐군    개원중 178 · 로봇고 29')
print('    논현중         포함/제외를 둘 다 보고해야 하므로 프레임이 있어야 함')
print('=' * 78)

print('\n[1] smoke_frames 아래 폴더별 jpg 장수 (두 단계까지 · 10장 이상만)')
rows = []
for d, subs, files in os.walk(SRC):
    rel = os.path.relpath(d, SRC)
    if rel.count(os.sep) > 1:
        subs[:] = []
        continue
    n = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if n >= 10:
        rows.append((rel, n))
for rel, n in sorted(rows, key=lambda r: -r[1])[:50]:
    print(f'  {n:>7}  {rel}')
print(f'  (10장 이상인 폴더 {len(rows)}개)')

print('\n[2] 배경 오탐군 · 논현중 — steam/bg 안의 접두사별 장수')
bg = glob.glob(f'{SRC}/steam/bg/*.jpg')
pre = {}
for p in bg:
    b = norm(os.path.basename(p))
    k = b.split('_')[0]
    pre[k] = pre.get(k, 0) + 1
for k, v in sorted(pre.items(), key=lambda x: -x[1]):
    print(f'  {v:>7}  {k}')
print(f'  합 {len(bg)}장')
for want, n in (('개원중', 530), ('로봇고', None), ('논현중', None)):
    got = pre.get(want, 0)
    tail = '' if n is None else f'   (1층/2층 기준 뽑음 {n})'
    print(f'  [확인] {want} {got}장{tail}'
          f'{"" if got else "   ← **없음. 보고 규칙을 못 채움**"}')

print('\n[3] 김 평가군 D — manifest_steam.json')
ms = f'{SRC}/manifest_steam.json'
if os.path.exists(ms):
    m = json.load(open(ms))
    tot = 0
    print(f'  {"시설":<12}{"프레임":>8}   태그별')
    for k, v in m.items():
        fr = v.get('frames', [])
        tags = {}
        for r in fr:
            tags[r.get('tag', '?')] = tags.get(r.get('tag', '?'), 0) + 1
        tot += len(fr)
        print(f'  {k:<12}{len(fr):>8}   ' + ' · '.join(f'{a} {b}' for a, b in sorted(tags.items())))
    print(f'  합 {tot}장   [확인] 1층의 `11곳 2,044장` 과 견줄 것')
else:
    print(f'  **{ms} 가 없음**')

print('\n[4] 발연 평가군 E — manifest.json')
mm = f'{SRC}/manifest.json'
if os.path.exists(mm):
    m = json.load(open(mm))
    print(f'  {"출처":<10}{"smoke 태그":>11}   있는 키')
    for k, v in m.items():
        tg = v.get('tags', {})
        ns = len(tg.get('smoke', []))
        print(f'  {k:<10}{ns:>11}   ' + ' · '.join(sorted(tg)))
    print(f'  manifest 에 있는 출처 {len(m)}개')
    print(f'  [확인] 배정표의 15 출처 중 manifest 에 없는 것 — '
          f'{sorted(set(ID.values()) - set(m)) or "없음"}')
else:
    print(f'  **{mm} 가 없음**')

print('\n[5] 분할마다 평가로 배정된 발연 출처')
for s in (1, 2, 3, 4, 5):
    e = [ID[i] for i in EVAL[s]]
    t = [ID[i] for i in TRAIN[s]]
    ov = sorted(set(e) & set(t))
    print(f'  분할 {s}  평가 {" · ".join(e)}')
    print(f'          [검산] 학습과 겹치는 출처 {"없음" if not ov else "**" + str(ov) + "**"}')

print('\n[6] 프레임 파일이 실제로 어디 있는가 — 출처 접두사로 훑음')
found = {}
for d, subs, files in os.walk(SRC):
    rel = os.path.relpath(d, SRC)
    if rel.startswith('ds_s1') or rel.startswith('runs_s1'):
        subs[:] = []
        continue
    for f in files:
        if not f.lower().endswith('.jpg'):
            continue
        b = norm(f)
        for k in ID.values():
            if b.startswith(norm(k) + '_'):
                found.setdefault(k, {}).setdefault(rel, 0)
                found[k][rel] += 1
                break
print(f'  {"출처":<10}{"합":>7}   폴더')
for k in ID.values():
    d = found.get(k, {})
    tot = sum(d.values())
    top = ' · '.join(f'{a} {b}' for a, b in sorted(d.items(), key=lambda x: -x[1])[:3])
    print(f'  {k:<10}{tot:>7}   {top if top else "**못 찾음**"}')

print('\n' + '=' * 78)
print('이 점검이 답하는 것')
print('=' * 78)
print('  E 를 어느 폴더에서 어떤 규칙으로 고를지')
print('  D 가 1층의 11곳 2,044장과 맞는지')
print('  배경 오탐군 개원중 178 · 로봇고 29 를 어떻게 셀지')
print('  **논현중 프레임이 있는지** — 없으면 보고 규칙의 포함/제외를 못 채움')
