# ===== 두 번째 시험 합성 — 2층을 닫기 위한 전수 측정 =====
#
# 첫 시험(`colab_synth_trial.py`)에 **설계 결함이 있었음.**
# 성한 소재를 `알파평균 ≥ 0.01` 로 고르고 빈 소재를 최솟값에서 골라 놓고
# `0.01 이 맞는가` 를 물었음 — **문턱으로 표본을 나눠 놓고 그 문턱을 검증한 것.**
# 이번에는 **소재 175장을 전수로 잼. 고르는 일이 없으므로 순환이 없음.**
#
# ---------------------------------------------------------------------------
# 합성식을 고쳤음 — 이건 재서 정한 게 아니라 식이 정하는 것임
#
#   오려낼 때   알파 = (본 것 − 판) / (255 − 판)     ← **연기 층을 「색 255」로 놓은 것**
#   그러므로     합성 = 배경 × (1 − 알파) + 255 × 알파
#
# 첫 시험은 `배경 × (1 − 알파) + 관측색 × 알파` 로 얹었음. 관측색에는 **소재 원본의
# 어두운 배경이 섞여 있어** 그 색이 급식실 안으로 딸려 들어옴.
#   두 식의 차 = 알파 × (1 − 알파) × (255 − 판)   → 알파 0.5 · 판 60 이면 49 계조
#
# ---------------------------------------------------------------------------
# 이 시험이 **못 재는 것** — 미리 적어 둠
#
#   **연기색이 정말 255 인가**  못 잼. 판이 알파 속으로 흡수돼 소재만으로는 못 되돌림.
#                             (obs, 알파) 만으로 판을 되살리면 **항상 정확히 맞아** 떨어짐 —
#                             즉 이 되살림은 **구현 검산이지 모형 검산이 아님**
#   배경 밝기 의존             `배경 변화` 는 소재만의 성질이 아니라 **소재 × 그 자리 밝기**임.
#                             그래서 배경 3장 × 자리 9곳의 중앙값으로 잼
#   잡음 바닥의 과대추정        연속 두 프레임이 시간상 붙어 있지 않으면 움직임이 섞여 바닥이
#                             높게 나옴 → **제외가 더 엄해지는 쪽**이라 안전한 방향임
#   합성 자료로 학습이 되는가    학습을 돌려야 알 수 있음
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 10~15분.

import os, glob, json, unicodedata, itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
BGDIR = f'{SRC}/steam/bg'
MAT   = f'{SRC}/matte'
OUT   = '/content/synth_trial2'
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SEED  = 1
U     = 0.30                    # 재는 동안 소재 가로폭을 이 값으로 **고정** (소재끼리 견주려고)
NBG   = 3                       # 배경 변화를 잴 때 쓰는 배경 장수
GRID  = 3                       # 자리 3x3

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. **결과를 보기 전에** 고르는 규칙을 먼저 찍음. 보고 나서 못 바꿈
# ---------------------------------------------------------------------------
print('=' * 72)
print('미리 못 박은 규칙 — 아래 숫자가 나오기 전에 정한 것임')
print('=' * 72)
print('  상자 규칙   담은질량 평균 ≥ 0.95 를 지키는 것 중 **넓이비 평균이 가장 작은** 규칙')
print('  소재 제외   상자 안 평균 변화 < 개원중 **연속 두 프레임 차의 중앙값**(잡음 바닥)')
print('  배경        **서로 다름만 씀** — 같은 그림을 여러 장 넣으면 배경을 외우게 됨')
print('              (이건 숫자로 고른 게 아니라 논거로 정한 것임. 장수는 확인만 함)')
print('  표본        소재 **175장 전수**. 고르는 일이 없으므로 순환이 없음')
print('=' * 72)


def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(g, size=8):
    x = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size + 1, size),
                                                              Image.LANCZOS), np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


# ---------------------------------------------------------------------------
# 1. 자료
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
if not bgs_all:
    raise SystemExit(f'{BGDIR} 에 {BGKEY} 프레임이 없음')

mats = [(k, p) for k in KEYS for p in sorted(glob.glob(f'{MAT}/{k}/*.png'))]
if not mats:
    raise SystemExit(f'{MAT} 아래에 소재가 없음')
print(f'\n배경 {BGKEY} {len(bgs_all)}장(원본 전부) · 소재 {len(mats)}장(전수)')

amean = {}
mpath = f'{MAT}/matte_manifest.json'
if os.path.exists(mpath):
    man = json.load(open(mpath))
    for k, v in man.items():
        for r in v.get('frames', []):
            if r.get('file'):
                amean[(k, r['file'])] = r.get('alpha_mean')

# ---------------------------------------------------------------------------
# 2. 배경 — 서로 다름 세기. **사전 등록의 178장과 맞는지 확인**
# ---------------------------------------------------------------------------
hs = [(p, dhash(np.asarray(Image.open(p).convert('L'), np.float32))) for p in bgs_all]
for thr in (0, 3, 5):
    keep = []
    for p, h in hs:
        if all(ham(h, g) > thr for _, g in keep):
            keep.append((p, h))
    print(f'  서로 다름 (해밍 > {thr})  {len(keep)}장')
    if thr == 5:
        bgs = [p for p, _ in keep]
print(f'  사전 등록에 적힌 개원중 서로 다름 = 178장')
print(f'  -> 합성 배경으로 쓸 것 {len(bgs)}장')

# ---------------------------------------------------------------------------
# 3. 잡음 바닥 — **문턱을 제가 정하지 않고 배경 자료가 내게 함**
# ---------------------------------------------------------------------------
print('\n[잡음 바닥]  개원중 프레임 자체가 만드는 변화')
pairs = []
for a, b in zip(bgs_all[:12], bgs_all[1:13]):
    f1 = np.asarray(Image.open(a).convert('RGB'), np.float32)
    f2 = np.asarray(Image.open(b).convert('RGB'), np.float32)
    if f1.shape == f2.shape:
        pairs.append(float(np.median(np.abs(f2 - f1).mean(2))))
FLOOR = float(np.median(pairs)) if pairs else 0.0
print(f'  연속 두 프레임 차의 중앙값   {FLOOR:.3f} 계조   ({len(pairs)}쌍)   ← **제외 문턱**')

f0 = Image.open(bgs_all[0]).convert('RGB')
f0.save(f'{OUT}/_re.jpg', quality=90)
re0 = np.abs(np.asarray(Image.open(f'{OUT}/_re.jpg'), np.float32)
             - np.asarray(f0, np.float32)).mean()
print(f'  JPEG 재저장 차의 평균        {re0:.3f} 계조   (더 낮은 바닥. 참고값)')

self0 = float(np.abs(np.asarray(f0, np.float32) - np.asarray(f0, np.float32)).mean())
print(f'  [검산] 자기 자신과의 차      {self0:.3f}   '
      f'{"통과" if self0 == 0 else "**실패 — 재는 코드가 틀림**"}')

# ---------------------------------------------------------------------------
# 4. 합성식 + 구현 검산
# ---------------------------------------------------------------------------
def comp_white(bg_patch, rgb_unused, al):
    """**연기 층은 색 255.** 오려내기 식과 같은 가정."""
    return bg_patch * (1 - al[..., None]) + 255.0 * al[..., None]


def comp_obs(bg_patch, rgb, al):
    """첫 시험이 쓴 식. 소재 원본의 배경색이 딸려 들어옴 — **틀린 식**."""
    return bg_patch * (1 - al[..., None]) + rgb * al[..., None]


print('\n[구현 검산]  소재에서 판을 되살려 흰색식으로 다시 얹으면 원본이 복원되는가')
print('  (되살린 판은 항상 정확히 맞음 — 이건 **모형이 아니라 제 구현**을 확인하는 것임)')
worst = 0.0
for k, p in mats[:20]:
    q = np.asarray(Image.open(p).convert('RGBA'), np.float32)
    obs, al = q[..., :3], q[..., 3] / 255.0
    m = al < 0.999
    plate = np.zeros_like(obs)
    plate[m] = (obs[m] - 255.0 * al[..., None][m]) / (1 - al[..., None][m])
    back = comp_white(plate, obs, al)
    worst = max(worst, float(np.abs(back - obs)[m].max()))
print(f'  최대 오차 {worst:.4f} 계조   '
      f'{"통과" if worst < 0.05 else "**실패 — 합성식 구현이 식과 다름**"}')

d_obs = []
for k, p in mats[:40]:
    q = np.asarray(Image.open(p).convert('RGBA'), np.float32)
    obs, al = q[..., :3], q[..., 3] / 255.0
    d_obs.append(float((np.abs(255.0 - obs).mean(2) * al).sum() / max(al.sum(), 1)))
print(f'  두 식이 벌어지는 크기(알파 실린 평균)  {np.mean(d_obs):.1f} 계조'
      f'   ← 첫 시험이 이만큼 어둡게 얹고 있었음')

# ---------------------------------------------------------------------------
# 5. 상자 다섯 규칙 — 소재 175장 전수
# ---------------------------------------------------------------------------
def box_all(a):
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_main(a):
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return None
    mass = ndimage.sum(a, lab, range(1, n + 1))
    ys, xs = np.nonzero(lab == int(np.argmax(mass)) + 1)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_q(a, q):
    """양끝에서 질량 q 씩 잘라 낸 상자."""
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def box_clean(a, keep=0.99):
    """질량 큰 덩어리부터 누적 keep 까지만 남기고 나머지는 잡음으로 지운 뒤의 상자."""
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return None
    mass = ndimage.sum(a, lab, range(1, n + 1))
    order = np.argsort(mass)[::-1]
    c = np.cumsum(mass[order]) / mass.sum()
    take = order[:int(np.searchsorted(c, keep)) + 1] + 1
    ys, xs = np.nonzero(np.isin(lab, take))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


RULES = [('알파0',      lambda a: box_all(a),          (255,  80,  80)),
         ('최대덩어리',  lambda a: box_main(a),         ( 80, 200, 255)),
         ('질량99',     lambda a: box_q(a, 0.005),     (120, 255, 120)),
         ('질량95',     lambda a: box_q(a, 0.025),     (255, 220,   0)),
         ('잡음제거99',  lambda a: box_clean(a, 0.99),  (255, 120, 255))]


def resized_alpha(p, w_target):
    q = Image.open(p).convert('RGBA')
    pw = max(int(round(w_target)), 8)
    ph = max(int(round(pw * q.size[1] / q.size[0])), 8)
    r = np.asarray(q.resize((pw, ph), Image.LANCZOS), np.float32)
    return r[..., :3], r[..., 3] / 255.0


bg0 = np.asarray(Image.open(bgs[0]).convert('RGB'), np.float32)
H, W = bg0.shape[:2]
BGAREA = H * W

print(f'\n[상자 다섯 규칙]  소재 {len(mats)}장 전수 · 가로폭 {U:g} 고정 · 배경 {W}x{H}')
print(f'{"규칙":<12}{"넓이비":>9}{"담은질량":>10}{"채움":>9}{"상자없음":>9}')
print('-' * 50)

pieces, stat = [], {r[0]: {'a': [], 'm': [], 'f': []} for r in RULES}
incl_bad = 0
for k, p in mats:
    rgb, al = resized_alpha(p, W * U)
    boxes = {}
    for name, fn, _ in RULES:
        b = fn(al)
        boxes[name] = b
        if b is None:
            continue
        x0, y0, x1, y1 = b
        ins = float(al[y0:y1, x0:x1].sum())
        area = (x1 - x0) * (y1 - y0)
        stat[name]['a'].append(area / BGAREA)
        stat[name]['m'].append(ins / max(float(al.sum()), 1e-9))
        stat[name]['f'].append(ins / max(area, 1))
    ba = boxes['알파0']
    if ba:                                   # 검산 — 알파0 이 나머지를 다 품는가
        for name in ('최대덩어리', '질량99', '질량95', '잡음제거99'):
            b = boxes[name]
            if b and not (b[0] >= ba[0] and b[1] >= ba[1]
                          and b[2] <= ba[2] and b[3] <= ba[3]):
                incl_bad += 1
    pieces.append((k, p, rgb, al, boxes))

for name, _, _ in RULES:
    s = stat[name]
    nmiss = len(mats) - len(s['a'])
    print(f'{name:<12}{np.mean(s["a"]):>9.4f}{np.mean(s["m"]):>10.4f}'
          f'{np.mean(s["f"]):>9.4f}{nmiss:>9}')
print('-' * 50)
print(f'  [검산] 알파0 이 나머지 넷을 품는가 — 어긋난 경우 {incl_bad}건   '
      f'{"통과" if incl_bad == 0 else "**실패 — 규칙 셈이 틀림**"}')

ok = [(name, np.mean(stat[name]['a'])) for name, _, _ in RULES
      if np.mean(stat[name]['m']) >= 0.95]
if ok:
    PICK = min(ok, key=lambda t: t[1])[0]
    print(f'\n  미리 못 박은 기준(담은질량 ≥ 0.95 중 넓이비 최소) 적용 -> **{PICK}**')
else:
    PICK = '질량99'
    print(f'\n  **담은질량 0.95 를 넘는 규칙이 없음.** 기준을 못 채웠으므로 규칙을 못 고름')
    print(f'  아래는 {PICK} 로 이어 가되, 상자 규칙은 **미정으로 남김**')

PICKFN = dict((n, f) for n, f, _ in RULES)[PICK]

# ---------------------------------------------------------------------------
# 6. 소재 제외 — 배경 3장 × 자리 9곳의 중앙값. 문턱은 위에서 잰 잡음 바닥
# ---------------------------------------------------------------------------
print(f'\n[배경 변화]  소재 {len(mats)}장 전수 · 흰색식 · {PICK} 상자 안 평균')
print(f'  자리 {GRID}x{GRID} × 배경 {NBG}장의 **중앙값**. 문턱 {FLOOR:.3f} 계조')

bgimgs = [np.asarray(Image.open(bgs[i % len(bgs)]).convert('RGB'), np.float32)
          for i in range(NBG)]
recs = []
for k, p, rgb, al, boxes in pieces:
    b = boxes[PICK]
    if b is None:
        recs.append({'key': k, 'file': os.path.basename(p),
                     'alpha_mean': amean.get((k, os.path.basename(p))),
                     'chg': 0.0, 'peak': 0.0});  continue
    x0, y0, x1, y1 = b
    sub = al[y0:y1, x0:x1]
    ph, pw = sub.shape
    ch, pk = [], []
    for bi in bgimgs:
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int((bi.shape[0] - ph) * gy / max(GRID - 1, 1))
                xx = int((bi.shape[1] - pw) * gx / max(GRID - 1, 1))
                patch = bi[yy:yy + ph, xx:xx + pw]
                d = np.abs((255.0 - patch) * sub[..., None]).mean()
                ch.append(float(d))
                pk.append(float(np.abs((255.0 - patch) * sub[..., None]).max()))
    recs.append({'key': k, 'file': os.path.basename(p),
                 'alpha_mean': amean.get((k, os.path.basename(p))),
                 'chg': float(np.median(ch)), 'peak': float(np.median(pk))})

drop = [r for r in recs if r['chg'] < FLOOR]
keep = [r for r in recs if r['chg'] >= FLOOR]
print(f'\n{"출처":<9}{"장":>5}{"남음":>6}{"제외":>6}{"변화 중앙":>11}{"변화 최소":>11}')
print('-' * 50)
for k in KEYS:
    rr = [r for r in recs if r['key'] == k]
    if not rr:
        continue
    kk = [r for r in rr if r['chg'] >= FLOOR]
    v = [r['chg'] for r in rr]
    print(f'{k:<9}{len(rr):>5}{len(kk):>6}{len(rr) - len(kk):>6}'
          f'{np.median(v):>11.2f}{min(v):>11.2f}')
print('-' * 50)
print(f'{"합":<9}{len(recs):>5}{len(keep):>6}{len(drop):>6}')

print(f'\n제외된 소재 (변화 < {FLOOR:.3f})')
for r in sorted(drop, key=lambda r: r['chg'])[:15]:
    a = r['alpha_mean']
    print(f'  {r["key"]:<9}{r["file"]:<24}변화 {r["chg"]:6.2f} · 봉우리 {r["peak"]:6.1f}'
          f' · 알파평균 {a if a is None else round(a, 4)}')
if not drop:
    print('  없음 — 175장 모두 잡음 바닥 위임')

# 알파평균이 대리 지표로 쓸 만한가 — 순위 상관
va = [r['alpha_mean'] for r in recs if r['alpha_mean'] is not None]
vc = [r['chg'] for r in recs if r['alpha_mean'] is not None]
if len(va) > 2:
    ra = np.argsort(np.argsort(va));  rc = np.argsort(np.argsort(vc))
    rho = float(np.corrcoef(ra, rc)[0, 1])
    print(f'\n알파평균 ↔ 배경변화 순위상관 {rho:+.3f}   '
          f'{"알파평균을 대리 지표로 써도 됨" if rho > 0.9 else "**알파평균은 대리 지표로 못 씀**"}')

json.dump({'floor': FLOOR, 'rule': PICK, 'records': recs},
          open(f'{OUT}/synth_trial2.json', 'w'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------------------
# 7. 그림 — 흰색식 vs 관측색식, 그리고 위치 규칙
# ---------------------------------------------------------------------------
def korean_font(size):
    cand = (glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
            + glob.glob('/usr/share/fonts/**/*Nanum*.ttf', recursive=True)
            + glob.glob('/usr/share/fonts/**/NotoSansCJK*', recursive=True))
    if not cand:
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        cand = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
    for c in cand:
        try:
            f = ImageFont.truetype(c, size)
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


F, HANGUL = korean_font(24)
print(f'\n시트 글꼴 — 한글 {"됨" if HANGUL else "**안 됨**"}')


def put(bg, rgb, al, x, y, fn):
    ph, pw = al.shape
    out = bg.copy()
    out[y:y + ph, x:x + pw] = fn(out[y:y + ph, x:x + pw], rgb, al)
    return out, (x, y)


def sheet(items, path, cw=1100):
    Ht = sum(round(cw * im.shape[0] / im.shape[1]) + 40 for _, im, _ in items) + 8
    sh = Image.new('RGB', (cw, Ht), (16, 16, 16))
    d = ImageDraw.Draw(sh);  y = 0
    for lab, im, boxes in items:
        h0, w0 = im.shape[:2]
        ch2 = round(cw * h0 / w0)
        pic = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((cw, ch2), Image.LANCZOS)
        dd = ImageDraw.Draw(pic);  s = cw / w0
        for b, col in boxes:
            if b:
                dd.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=col, width=3)
        d.text((6, y + 8), lab, fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 40))
        y += ch2 + 40
    sh.save(path, quality=90)
    return path


# (가) 같은 소재를 두 식으로 — 출처마다 중앙값 소재 하나
rows = []
for k in KEYS:
    kk = [(i, r) for i, r in enumerate(recs) if r['key'] == k and r['chg'] >= FLOOR]
    if not kk:
        continue
    i = kk[len(kk) // 2][0]
    _, p, rgb, al, boxes = pieces[i]
    bg = np.asarray(Image.open(bgs[i % len(bgs)]).convert('RGB'), np.float32)
    x = (W - al.shape[1]) // 2;  y = int(H * 0.25)
    b = boxes[PICK]
    off = None if b is None else (b[0] + x, b[1] + y, b[2] + x, b[3] + y)
    o1, _ = put(bg, rgb, al, x, y, comp_white)
    o2, _ = put(bg, rgb, al, x, y, comp_obs)
    rows.append((f'{k} {os.path.basename(p)}  **흰색식(맞음)**  변화 {recs[i]["chg"]:.1f}',
                 o1, [(off, (120, 255, 120))]))
    rows.append((f'{k} {os.path.basename(p)}  관측색식(첫 시험 · 틀림)',
                 o2, [(off, (255, 80, 80))]))
p1 = sheet(rows, f'{OUT}/_formula.jpg')

# (나) 상자 다섯 규칙을 한 소재에 겹쳐
rows2 = []
for k in KEYS:
    kk = [(i, r) for i, r in enumerate(recs) if r['key'] == k and r['chg'] >= FLOOR]
    if not kk:
        continue
    i = kk[len(kk) // 2][0]
    _, p, rgb, al, boxes = pieces[i]
    bg = np.asarray(Image.open(bgs[(i + 1) % len(bgs)]).convert('RGB'), np.float32)
    x = (W - al.shape[1]) // 2;  y = int(H * 0.25)
    o, _ = put(bg, rgb, al, x, y, comp_white)
    bb = [((b[0] + x, b[1] + y, b[2] + x, b[3] + y) if b else None, col)
          for name, _, col in RULES for b in [boxes[name]]]
    rows2.append((f'{k} {os.path.basename(p)}   빨강 알파0 · 파랑 최대덩어리 · '
                  f'초록 질량99 · 노랑 질량95 · 분홍 잡음제거99', o, bb))
p2 = sheet(rows2, f'{OUT}/_boxes.jpg')

# (다) 제외된 소재 — 정말 안 보이는지
rows3 = []
for r in sorted(drop, key=lambda r: r['chg'])[:4]:
    i = next(j for j, q in enumerate(recs) if q['file'] == r['file'])
    _, p, rgb, al, boxes = pieces[i]
    bg = np.asarray(Image.open(bgs[i % len(bgs)]).convert('RGB'), np.float32)
    x = (W - al.shape[1]) // 2;  y = int(H * 0.25)
    o, _ = put(bg, rgb, al, x, y, comp_white)
    b = boxes[PICK]
    off = None if b is None else (b[0] + x, b[1] + y, b[2] + x, b[3] + y)
    rows3.append((f'제외 {r["key"]} {r["file"]}  변화 {r["chg"]:.2f} < 바닥 {FLOOR:.2f}',
                  o, [(off, (255, 80, 80))]))
p3 = sheet(rows3, f'{OUT}/_dropped.jpg') if rows3 else None

# (라) 위치 규칙 — 흰색식
rows4 = []
survivors = [i for i, r in enumerate(recs) if r['chg'] >= FLOOR]
for pos in ('전면', '상반부'):
    bg = np.asarray(Image.open(bgs[5 % len(bgs)]).convert('RGB'), np.float32)
    o, bb = bg, []
    for _ in range(2):
        i = int(rng.choice(survivors))
        _, p, rgb, al, boxes = pieces[i]
        ph, pw = al.shape
        x = int(rng.integers(0, max(W - pw, 1)))
        y = (int(rng.integers(0, max(int(H * 0.55) - ph, 1))) if pos == '상반부'
             else int(rng.integers(0, max(H - ph, 1))))
        o, _ = put(o, rgb, al, x, y, comp_white)
        b = boxes[PICK]
        if b:
            bb.append(((b[0] + x, b[1] + y, b[2] + x, b[3] + y), (120, 255, 120)))
    rows4.append((f'위치 {pos} · 소재 2개 · 흰색식 · 상자 {PICK}', o, bb))
p4 = sheet(rows4, f'{OUT}/_place2.jpg')

for q in (p1, p2, p3, p4):
    if q:
        print(f'-> {q}')
        files.download(q)

print(f'\n-> {OUT}/synth_trial2.json')
print('\n이 시험도 **학습 자료를 만들지 않음.** 여기서 2층을 닫고 나서 본 합성을 돌림.')
