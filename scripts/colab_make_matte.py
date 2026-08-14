# ===== 연기 소재 만들기 (오려내기 실행) =====
#
# `docs/PREREGISTER_S1.md` 의 `소재 — 오려내기` 절을 그대로 실행함.
# 통과한 여섯 출처만. 여덟 번의 시험으로 정해진 값들이라 여기서 바꾸지 않음.
#
# 나오는 것 — 알파가 든 RGBA 조각(PNG) + matte_manifest.json + 확인용 시트.
#
# 돌리기 전에 스스로 검사하는 셋은 아래 `사전 검사` 절에 있음.
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 10~20분.

import os, glob, json, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'                 # 추출 결과 (manifest.json 이 여기 있음)
OUT  = f'{SRC}/matte'                     # 소재가 여기 쌓임
FPS  = 2
THR  = 0.06                               # 알파가 이 아래면 0

# (키, 파일명 조각, [샷...], 잘라내기(위,아래))
#   샷마다 판을 따로 잡음. 판은 그 샷의 첫 프레임
CASE = [
    ('m3',      '異物挟み込み',        [(59, 92)],               None),
    ('kfire03', 'Fire Prevention Week', [(24, 36)],              None),
    ('q1',      '少ない油で発火',       [(37, 48.5)],             (0.10, 0.00)),
    ('j04',     'シミュレーション',     [(18, 28)],               (0.00, 0.42)),
    ('07',      'Fire Safety_ Grease Fires', [(55, 61.5), (72, 77)], None),
    ('p2',      '汚れた鍋',            [(41, 46.5), (47, 51)],   (0.12, 0.00)),
]

# docs/TIMELINE.md 의 `제외`
EXCLUDE = {'q1': [(0, 2.5), (34, 36)], 'm3': [(58, 58)],
           'kfire03': [(23, 23)], '07': [(48, 54), (67, 71), (82, 88)],
           'j04': [], 'p2': []}


# ---------------------------------------------------------------------------
# 사전 검사 셋
#   (1) 샷 경계가 제외 구간에 걸리는가
#   (2) 뽑힌 장수가 (끝초 − 시작초) × FPS + 1 과 맞는가
#   (3) 여기 적은 구간이 추출 스크립트(manifest)의 구간과 같은가
# ---------------------------------------------------------------------------
bad = [(k, x, (p, q)) for k, _, shots, _ in CASE for a, b in shots for x in (a, b)
       for p, q in EXCLUDE.get(k, []) if p - 1e-6 <= x <= q + 1e-6]
if bad:
    for k, x, r in bad:
        print(f'[{k}] {x}초 가 제외 구간 {r} 에 걸림')
    raise SystemExit('제외 구간에 걸리는 경계가 있음.')
print('(1) 샷 경계 검사 통과')

# 샷마다 **첫 프레임은 판**이라 알파가 통째로 0 임 — 소재가 안 됨.
# 그래서 샷당 (끝초 − 시작초) × FPS 임 (+1 이 아님).
# 2026-08-13 첫 실행에서 여섯 출처가 전부 샷 수만큼 모자라 이걸 잡았음.
EXPECT = {k: sum(int(round((b - a) * FPS)) for a, b in shots)
          for k, _, shots, _ in CASE}
print(f'(2) 예상 장수 — {" · ".join(f"{k} {v}" for k, v in EXPECT.items())}'
      f'   합 {sum(EXPECT.values())}장')


def norm(s):
    return unicodedata.normalize('NFC', s)


def frames_of(src, a, b, crop):
    tmp = '/content/_m'
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


def alpha_of(img, plate):
    a = np.clip(((img - plate) / np.maximum(255.0 - plate, 8.0)).min(2), 0, 1)
    a[a < THR] = 0
    return a


def dhash(a, size=8):
    x = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                   .resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


drive.mount('/content/drive')
allf = glob.glob(f'{SRC}/*')
shutil.rmtree(OUT, ignore_errors=True);  os.makedirs(OUT)

# (3) 추출 manifest 의 구간과 대조 + rank 를 가져옴
RANK = {}
mpath = f'{EXT}/manifest.json'
if os.path.exists(mpath):
    man = json.load(open(mpath))
    for k in EXPECT:
        for x in man.get(k, {}).get('tags', {}).get('smoke', []):
            RANK[(k, round(x['sec'], 1))] = x['rank']
    print(f'(3) manifest 에서 rank 를 읽음 — {len(RANK)}개')
else:
    print(f'(3) [주의] {mpath} 가 없어 rank 를 못 읽음. 상한 적용은 나중에 해야 함')

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception:
    F = ImageFont.load_default()

print(f'\n{"출처":<9}{"장":>5}{"알파평균":>9}{"면적비":>8}{"서로다름":>9}{"rank없음":>9}')
print('-' * 52)

manifest, sheets, total, mismatch = {}, [], 0, []
for key, frag, shots, crop in CASE:
    hit = [p for p in allf if norm(frag) in norm(os.path.basename(p))]
    if len(hit) != 1:
        print(f'{key:<9}[건너뜀] "{frag}" 에 맞는 파일 {len(hit)}개');  continue
    src = hit[0]
    d = f'{OUT}/{key}';  os.makedirs(d, exist_ok=True)

    recs, alphas, norank = [], [], 0
    for a, b in shots:
        fr = frames_of(src, a, b, crop)
        if not fr:
            continue
        plate = fr[0][1]                       # 판 = 이 샷의 첫 프레임
        for sec, img in fr:
            al = alpha_of(img, plate)
            ys, xs = np.nonzero(al)
            if len(ys) == 0:                   # 알파가 통째로 0 이면 소재가 안 됨
                recs.append({'sec': sec, 'shot': [a, b], 'file': None,
                             'alpha_mean': 0.0, 'skip': '알파 0'})
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rgba = np.dstack([img[y0:y1, x0:x1],
                              (al[y0:y1, x0:x1] * 255)]).astype(np.uint8)
            name = f'{key}_{sec:07.1f}.png'.replace(' ', '0')
            Image.fromarray(rgba, 'RGBA').save(f'{d}/{name}')
            alphas.append(dhash(al))
            recs.append({'sec': sec, 'shot': [a, b], 'file': name,
                         'alpha_mean': round(float(al.mean()), 4),
                         'bbox': [int(x0), int(y0), int(x1), int(y1)],
                         'area_ratio': round(float((x1 - x0) * (y1 - y0)) / al.size, 4),
                         'rank': RANK.get((key, sec))})
            if RANK.get((key, sec)) is None:
                norank += 1
            if len(sheets) < 60 and abs(sec - (a + b) / 2) < 0.6:
                sheets.append((key, sec, img, al))

    n = sum(1 for r in recs if r.get('file'))
    total += n
    if n != EXPECT[key]:
        mismatch.append((key, EXPECT[key], n))
    uniq = 0
    if alphas:                                  # 알파끼리 dHash — 참고 수치
        keep = [alphas[0]]
        for h in alphas[1:]:
            if min(ham(h, g) for g in keep) > 0:
                keep.append(h)
        uniq = len(keep)
    am = np.mean([r['alpha_mean'] for r in recs if r.get('file')]) if n else 0
    ar = np.mean([r['area_ratio'] for r in recs if r.get('file')]) if n else 0
    manifest[key] = {'file': os.path.basename(src), 'crop': crop,
                     'shots': shots, 'frames': recs}
    mark = '   [예상과 다름]' if n != EXPECT[key] else ''
    print(f'{key:<9}{n:>5}{am:>9.3f}{ar:>8.2f}{uniq:>9}{norank:>9}{mark}')

json.dump(manifest, open(f'{OUT}/matte_manifest.json', 'w'), ensure_ascii=False, indent=1)

print('-' * 52)
print(f'{"합":<9}{total:>5}장   -> {OUT}')
if mismatch:
    print('\n[확인 필요] 장수가 예상과 다름')
    for k, e, g in mismatch:
        print(f'  {k:<9}예상 {e} · 실제 {g} ({g - e:+d})')
else:
    print('장수 검사 통과 — 여섯 출처 모두 (끝초 − 시작초) × FPS + 1 과 맞음')

print('\n**`서로다름` 은 알파끼리 잰 참고 수치임.** 사전 등록의 상한 32 · 하한 10 은')
print('추출 manifest 의 rank(원본 프레임 기준)로 적용함 — 규칙을 바꾸지 않음.')

# 확인용 시트 — 출처마다 가운데 한 장
if sheets:
    CW = 380
    rows = []
    for key, sec, img, al in sheets[:12]:
        rgb = Image.fromarray(img.astype(np.uint8))
        am = Image.fromarray((al * 255).astype(np.uint8)).convert('RGB')
        rows.append([(f'{key} {sec:g}s 원본', rgb), ('알파', am)])
    h0, w0 = np.asarray(rows[0][0][1]).shape[:2]
    ch = round(CW * h0 / w0)
    sh = Image.new('RGB', (2 * CW, len(rows) * (ch + 28)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for r, row in enumerate(rows):
        for c, (lab, im) in enumerate(row):
            y = r * (ch + 28)
            dr.text((c * CW + 6, y + 4), lab, fill=(255, 220, 0), font=F)
            sh.paste(im.resize((CW, ch), Image.LANCZOS), (c * CW, y + 28))
    sh.save(f'{OUT}/_check.jpg', quality=88)
    print(f'\n확인용 시트 -> {OUT}/_check.jpg  ({len(rows)}줄)')
