# ===== 시험 합성 — 2층을 채우기 위한 사전 시험 =====
#
# **이건 학습 자료를 만드는 게 아님.** `docs/PREREGISTER_S1.md` 의 `2층에서 채울 것`
# 을 눈과 숫자로 확인하려고 몇 장만 만들어 보는 것임.
#
# 이 시험이 답하려는 것 넷
#   (가) 개원중 CCTV 화각에 **연기가 놓일 자리가 있는가**  ← 눈으로
#   (나) 알파에서 **라벨 상자를 어떻게 뽑을 것인가**        ← 세 규칙을 나란히 재
#   (다) 소재에서 **뺄 것의 기준**(알파 평균 하한)          ← 빈 소재를 일부러 넣어 봄
#   (라) 놓는 **위치·크기·개수**를 어느 범위로 둘 것인가    ← 두 가지 위치 규칙 비교
#
# 이 시험이 **답 못 하는 것**
#   소재와 배경의 **밝기·색온도 정합** — 여기선 안 맞춤. 눈으로만 보고 2층에서 정함
#   합성 자료로 학습이 되는가 — 그건 학습을 돌려야 알 수 있음
#   개원중이 흐린가 — 선명도 지표가 장면 간 비교 불가로 판정됨(기록 82)
#
# **사전 등록 위반을 하나 안고 감** — 개원중은 `배경 오탐군` 평가군에 178장 들어 있음.
# 1층에 `합성에 쓴 급식실이 평가에도 나오면 안 됨` 이라 적었으나 못 지킴.
# 다만 배경 오탐군은 **성립 판정에 안 쓰는 군**이라 오염이 판별비에 안 닿음.
# 로봇고를 쓰면 평가 D(김) 50장이 오염되어 **판별비 분모가 직접 더러워짐.**
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
BGDIR = f'{SRC}/steam/bg'
MAT   = f'{SRC}/matte'
OUT   = '/content/synth_trial'
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SEED  = 1
Q     = 0.005                       # 질량99 상자의 양끝 잘라내기
WIDTH = (0.15, 0.45)                # 소재 가로폭 ÷ 배경 가로폭
NPIECE = (1, 3)                     # 한 장에 놓을 소재 개수

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return unicodedata.normalize('NFC', s)


# ---------------------------------------------------------------------------
# 자료 모으기
# ---------------------------------------------------------------------------
bgs = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
             if norm(os.path.basename(p)).startswith(norm(BGKEY)))
if not bgs:
    raise SystemExit(f'{BGDIR} 에 {BGKEY} 프레임이 없음')

mats = []
for k in KEYS:
    for p in sorted(glob.glob(f'{MAT}/{k}/*.png')):
        mats.append((k, p))
if not mats:
    raise SystemExit(f'{MAT} 아래에 소재가 없음')

print(f'배경 {BGKEY} {len(bgs)}장 · 소재 {len(mats)}장')

# 소재별 알파 평균 — matte_manifest.json 에서 읽되, 없으면 직접 잼
amean = {}
mpath = f'{MAT}/matte_manifest.json'
if os.path.exists(mpath):
    man = json.load(open(mpath))
    for k, v in man.items():
        for r in v.get('frames', []):
            if r.get('file'):
                amean[(k, r['file'])] = r.get('alpha_mean', None)
    print(f'matte_manifest.json 에서 알파 평균 {len(amean)}개를 읽음')
else:
    print('[주의] matte_manifest.json 이 없어 알파 평균을 직접 잼')

for k, p in mats:
    key = (k, os.path.basename(p))
    if amean.get(key) is None:
        a = np.asarray(Image.open(p))[..., 3].astype(np.float32) / 255.0
        amean[key] = round(float(a.mean()), 4)


# ---------------------------------------------------------------------------
# (다) 알파 평균 분포 — 빈 소재가 어디서 갈리는가
# ---------------------------------------------------------------------------
print(f'\n[알파 평균 분포]  출처별 최솟값 ~ 최댓값')
print(f'{"출처":<9}{"장":>5}{"최소":>9}{"1사분":>9}{"중앙":>9}{"최대":>9}{"0.01미만":>10}')
print('-' * 60)
low = []
for k in KEYS:
    v = [amean[(k, os.path.basename(p))] for kk, p in mats if kk == k]
    if not v:
        continue
    v = np.array(v, dtype=np.float32)
    nlow = int((v < 0.01).sum())
    print(f'{k:<9}{len(v):>5}{v.min():>9.4f}{np.percentile(v, 25):>9.4f}'
          f'{np.median(v):>9.4f}{v.max():>9.4f}{nlow:>10}')
for kk, p in mats:
    b = os.path.basename(p)
    if amean[(kk, b)] < 0.01:
        low.append((kk, p, amean[(kk, b)]))
low.sort(key=lambda t: t[2])
print(f'\n알파 평균 0.01 미만 {len(low)}장')
for kk, p, a in low[:12]:
    print(f'  {kk:<9}{os.path.basename(p):<24}{a:.4f}')


# ---------------------------------------------------------------------------
# 상자 세 규칙
# ---------------------------------------------------------------------------
def box_all(a):
    """알파 > 0 인 화소 전부를 담는 상자."""
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_main(a):
    """가장 큰 연결 덩어리만 담는 상자."""
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return None
    mass = ndimage.sum(a, lab, range(1, n + 1))
    ys, xs = np.nonzero(lab == int(np.argmax(mass)) + 1)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_99(a):
    """알파 질량의 99% 를 담는 상자. 행·열을 따로 잘라 냄."""
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, Q)), int(np.searchsorted(c, 1 - Q)) + 1
    y0, y1 = span(a.sum(1))
    x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


RULES = [('알파0', box_all, (255, 80, 80)),
         ('최대덩어리', box_main, (80, 200, 255)),
         ('질량99', box_99, (120, 255, 120))]


def measure(a, box):
    """상자 넓이비 · 담은 질량비 · 채움(상자 안 질량 ÷ 상자 넓이)."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    inside = float(a[y0:y1, x0:x1].sum())
    tot = float(a.sum())
    area = (x1 - x0) * (y1 - y0)
    return {'area_ratio': area / a.size,
            'mass_kept': inside / max(tot, 1e-9),
            'fill': inside / max(area, 1)}


# ---------------------------------------------------------------------------
# 합성
# ---------------------------------------------------------------------------
def place(bg, piece, u, pos_rule):
    """소재를 배경 위에 얹고 (합성결과, 배경크기 알파) 를 돌려줌."""
    H, W = bg.shape[:2]
    pw = max(int(round(W * u)), 8)
    ph = max(int(round(pw * piece.shape[0] / piece.shape[1])), 8)
    if ph >= H:
        ph = H - 1
        pw = max(int(round(ph * piece.shape[1] / piece.shape[0])), 8)
    im = Image.fromarray(piece, 'RGBA').resize((pw, ph), Image.LANCZOS)
    q = np.asarray(im, dtype=np.float32)
    rgb, al = q[..., :3], q[..., 3] / 255.0

    x = int(rng.integers(0, max(W - pw, 1)))
    if pos_rule == '상반부':                 # 조리대 위쪽에만
        y = int(rng.integers(0, max(int(H * 0.55) - ph, 1)))
    else:                                    # 전면 무작위
        y = int(rng.integers(0, max(H - ph, 1)))

    A = np.zeros((H, W), np.float32)
    A[y:y + ph, x:x + pw] = al
    out = bg.copy()
    sl = out[y:y + ph, x:x + pw]
    out[y:y + ph, x:x + pw] = sl * (1 - al[..., None]) + rgb * al[..., None]
    return out, A


# ---- 검산 셋 ----
print('\n[검산]')
bg0 = np.asarray(Image.open(bgs[0]).convert('RGB'), np.float32)
pk = next(p for k, p in mats if amean[(k, os.path.basename(p))] > 0.02)
pc = np.asarray(Image.open(pk).convert('RGBA'))
o, A = place(bg0, pc, 0.3, '전면')

e0 = float(np.abs(o - bg0)[A == 0].max()) if (A == 0).any() else -1
print(f'  (1) 알파 0 인 자리가 배경과 같은가        최대 오차 {e0:.3f}'
      f'   {"통과" if e0 < 1e-3 else "**실패 — 합성식이 배경을 건드림**"}')

m1 = measure(A, box_all(A))
print(f'  (2) 알파0 상자가 질량을 다 담는가         담은 질량 {m1["mass_kept"]:.4f}'
      f'   {"통과" if m1["mass_kept"] > 0.9999 else "**실패 — 상자 밖으로 샘**"}')

b1, b2, b3 = box_all(A), box_main(A), box_99(A)
ar = [((b[2] - b[0]) * (b[3] - b[1])) for b in (b1, b2, b3)]
ok = ar[0] >= ar[1] and ar[0] >= ar[2]
print(f'  (3) 알파0 상자가 나머지 둘을 포함하는 크기인가  '
      f'{ar[0]} ≥ {ar[1]} · {ar[2]}   {"통과" if ok else "**실패 — 규칙 셈이 틀림**"}')

if not (e0 < 1e-3 and m1['mass_kept'] > 0.9999 and ok):
    print('  **검산이 하나라도 실패했으면 아래 숫자를 근거로 읽지 말 것**')


# ---- 본 시험 ----
# 대상 — 성한 소재 여섯 장(출처마다 하나) + 빈 소재 두 장(알파 평균 최소)
pick = []
for k in KEYS:
    cand = [(k, p) for kk, p in mats if kk == k
            and amean[(k, os.path.basename(p))] >= 0.01]
    if cand:
        pick.append(cand[len(cand) // 2] + ('성한 소재',))
for kk, p, a in low[:2]:
    pick.append((kk, p, '**빈 소재**'))

print(f'\n[상자 세 규칙 비교]  소재 {len(pick)}장 × 배경 {BGKEY}')
print(f'{"소재":<24}{"알파평균":>9}{"규칙":>12}{"넓이비":>9}{"담은질량":>10}{"채움":>9}')
print('-' * 74)

rows, stat = [], {r[0]: {'area': [], 'mass': [], 'fill': []} for r in RULES}
for i, (k, p, tag) in enumerate(pick):
    bg = np.asarray(Image.open(bgs[i % len(bgs)]).convert('RGB'), np.float32)
    piece = np.asarray(Image.open(p).convert('RGBA'))
    u = float(rng.uniform(*WIDTH))
    o, A = place(bg, piece, u, '전면')
    am = amean[(k, os.path.basename(p))]

    boxes = []
    for name, fn, col in RULES:
        b = fn(A)
        m = measure(A, b)
        boxes.append((name, b, col))
        if m is None:
            print(f'{os.path.basename(p):<24}{am:>9.4f}{name:>12}'
                  f'{"[상자 없음 — 알파가 통째로 0]":>28}')
            continue
        first = (name == RULES[0][0])
        print(f'{(os.path.basename(p) if first else ""):<24}'
              f'{(f"{am:.4f}" if first else ""):>9}'
              f'{name:>12}{m["area_ratio"]:>9.4f}{m["mass_kept"]:>10.4f}{m["fill"]:>9.4f}')
        stat[name]['area'].append(m['area_ratio'])
        stat[name]['mass'].append(m['mass_kept'])
        stat[name]['fill'].append(m['fill'])
    rows.append((f'{k} {os.path.basename(p)}  알파평균 {am:.4f}  {tag}', o, boxes))
    print('')

print('-' * 74)
for i, (name, _, _) in enumerate(RULES):
    s = stat[name]
    if s['area']:
        print(f'{("평균" if i == 0 else ""):<24}{"":>9}{name:>12}'
              f'{np.mean(s["area"]):>9.4f}{np.mean(s["mass"]):>10.4f}{np.mean(s["fill"]):>9.4f}')
print('\n넓이비는 작을수록 · 담은질량은 1 에 가까울수록 · 채움은 클수록 좋은 상자임.')
print('셋이 서로 맞바꿈 관계라 **하나만 보고 고르면 안 됨.**')
print('**담은질량은 빈 소재에서도 1 이 나옴** — 뺄 소재를 가리는 데는 못 씀. 채움을 볼 것.')


# ---- (라) 위치 규칙 두 가지 · 개수 ----
print(f'\n[위치·개수]  같은 배경에 규칙만 바꿔 놓아 봄')
multi = []
for pos_rule in ('전면', '상반부'):
    bg = np.asarray(Image.open(bgs[3 % len(bgs)]).convert('RGB'), np.float32)
    n = int(rng.integers(NPIECE[0], NPIECE[1] + 1))
    o, boxes, placed = bg, [], 0
    ok_mats = [(k, p) for k, p in mats if amean[(k, os.path.basename(p))] >= 0.01]
    for _ in range(n):
        k, p = ok_mats[int(rng.integers(0, len(ok_mats)))]
        piece = np.asarray(Image.open(p).convert('RGBA'))
        o, A = place(o, piece, float(rng.uniform(*WIDTH)), pos_rule)
        placed += 1
        for name, fn, col in RULES:
            b = fn(A)
            if b:
                boxes.append((name, b, col))
    print(f'  {pos_rule:<6} 놓은 소재 {placed}개 · 상자 {len(boxes)}개'
          f' (소재마다 {len(RULES)}규칙)')
    multi.append((f'위치 {pos_rule} · 소재 {placed}개  (상자는 소재마다 따로 뽑음)', o, boxes))


# ---------------------------------------------------------------------------
# 시트
# ---------------------------------------------------------------------------
# **글꼴** — DejaVu 에는 한글이 없어 앞 시트들의 한글 딱지가 네모로 나왔음.
# 나눔고딕이 없으면 한 번 받아 씀. 그래도 없으면 딱지를 로마자로 씀.
def korean_font(size):
    cand = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True) + \
           glob.glob('/usr/share/fonts/**/*Nanum*.ttf', recursive=True) + \
           glob.glob('/usr/share/fonts/**/NotoSansCJK*', recursive=True)
    if not cand:
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        cand = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
    for c in cand:
        try:
            f = ImageFont.truetype(c, size)
            # **확인** — 한글을 실제로 그려 보고 빈 그림이면 그 글꼴은 한글이 없는 것임
            t = Image.new('L', (size * 3, size * 2), 0)
            ImageDraw.Draw(t).text((2, 2), '연기', fill=255, font=f)
            if np.asarray(t).max() > 0:
                return f, True
        except Exception:
            pass
    try:
        return ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size), False
    except Exception:
        return ImageFont.load_default(), False


F, HANGUL = korean_font(26)
print(f'\n시트 글꼴 — 한글 {"됨" if HANGUL else "**안 됨 (딱지를 로마자로 씀)**"}')


def lab_fix(s):
    if HANGUL:
        return s
    for a, b in (('알파평균', 'alpha_mean'), ('성한 소재', 'usable'),
                 ('빈 소재', 'EMPTY'), ('위치', 'pos'), ('전면', 'anywhere'),
                 ('상반부', 'upper-half'), ('소재', 'pieces'), ('개', ''),
                 ('상자는 소재마다 따로 뽑음', 'one box set per piece')):
        s = s.replace(a, b)
    return s


def sheet(items, path, cw=1100):
    H = sum(round(cw * im.shape[0] / im.shape[1]) + 34 for _, im, _ in items) + 8
    sh = Image.new('RGB', (cw, H), (16, 16, 16))
    d = ImageDraw.Draw(sh);  y = 0
    for lab, im, boxes in items:
        h0, w0 = im.shape[:2]
        ch = round(cw * h0 / w0)
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS)
        dd = ImageDraw.Draw(pic)
        s = cw / w0
        for name, b, col in boxes:
            if b is None:
                continue
            dd.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=col, width=3)
        d.text((6, y + 4), lab_fix(lab), fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 34))
        y += ch + 34
    sh.save(path, quality=90)
    return path


p1 = sheet(rows, f'{OUT}/_box_rules.jpg')
p2 = sheet(multi, f'{OUT}/_place.jpg')
print(f'\n-> {p1}   (상자 색 — 빨강 알파0 · 파랑 최대덩어리 · 초록 질량99)')
print(f'-> {p2}')
files.download(p1)
files.download(p2)

print('\n이 시험은 **자료를 만들지 않음.** 결과를 보고 2층을 확정한 뒤에 본 합성을 돌림.')
