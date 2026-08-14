# ===== 세 번째 시험 합성 — 두 번째 시험의 결함을 고쳐 다시 잼 =====
#
# 두 번째 시험(`colab_synth_trial2.py`)에서 **결함 열둘을 찾아 고쳤음.**
# 아래 `고친 것` 절에 전부 적음. 고치지 못한 것도 `못 고친 것` 절에 적음.
#
# ---------------------------------------------------------------------------
# 고친 것 — 무엇이 어떻게 틀렸고 어느 쪽으로 틀렸는지
#
#  1 배경 목록      해밍 > 5(30장) 을 썼음. 선언한 정의는 **해밍 > 0(178장)** 이었음
#  2 잡음 바닥 통계  바닥은 **화소 중앙값**, 견주는 값은 **상자 평균**이었음 — 다른 저울.
#                  꼬리가 긴 분포라 중앙값이 낮게 나오므로 **제외가 느슨해지는 쪽**이었음
#  3 잡음 바닥 영역  바닥은 **화면 전체**, 견주는 값은 **상자 안**이었음 — 다른 모집단
#  4 잡음 바닥 표본  앞머리 12쌍만 썼음 → 전 구간에 고르게 흩음
#  5 두 식의 차     `알파 실린 g 의 평균`(84.1)을 **실제 어두워진 양처럼** 말했음.
#                  실제 차는 화소마다 `알파 × g` 임 — **제가 부풀려 보고했음**
#  6 검산 표본      `mats[:20]`·`mats[:40]` 은 **전부 m3** 였음(m3 가 66장이라 앞을 다 채움)
#  7 검산이 항등식   `자기 자신과의 차 0` · `판 되살림` · `알파0 포함` 셋 다 **실패 불가**였음
#  8 알파0 은 규칙 아님  소재는 이미 알파 테두리로 잘려 저장돼 있어 담은질량이 **항상 1.0**.
#                  그래서 `기준 미달이면 미정으로 남김` 분기가 **죽은 코드**였음
#  9 알파평균 출처   manifest 의 alpha_mean 은 **잘라내기 전 화면 전체** 기준이었음
# 10 리사이즈 문턱   크기를 바꾼 뒤 사전 등록의 `알파 < 0.06 → 0` 이 **다시 안 걸렸음**
# 11 배경 3장       앞머리 연속 세 장이라 **밝기가 거의 같았음** → 흩어서 8장
# 12 시트 고르기     `중앙값 소재`라 적고 **파일 순서 가운데**를 골랐음.
#                  제외분 시트도 **가장 옅은 것만** 보여 문턱을 항상 옳아 보이게 했음
#
# ---------------------------------------------------------------------------
# 못 고친 것 — 이 시험에 남는 한계
#
#   알파가 채널 최소값     `colab_make_matte.py:84` 는 `.min(2)` 로 알파를 뽑고 분모를 8 로
#                        받쳤음. 그러므로 흰색식은 **채널마다 정확한 역이 아님.**
#                        알파가 낮게 잡히는 쪽이라 연기가 **옅게** 얹힘
#   연기색 255 가정        못 잼. 판이 알파에 흡수돼 소재만으로는 못 되돌림
#   모집단               175장은 이미 `알파 < 0.06 → 0` 을 지난 것들임.
#                        다만 **프레임이 통째로 버려진 것은 0장**이었음(장수 검사 통과)
#   상자 규칙의 채움       미리 못 박은 기준에 `채움` 이 없음. 지금 넣으면 사후 조정이라 안 넣음
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 15~25분.

import os, glob, json, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
BGDIR = f'{SRC}/steam/bg'
MAT   = f'{SRC}/matte'
OUT   = f'{SRC}/synth_trial'          # **드라이브에 둠** — 다음 회차와 견주려면 남아야 함
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']
SEED  = 1
THR   = 0.06                          # 사전 등록 값. 리사이즈 뒤에도 이걸 다시 걺
U     = 0.30                          # 재는 동안 소재 가로폭 고정
NBG   = 8                             # 배경 변화를 잴 때 쓰는 배경 장수 (흩어서 뽑음)
GRID  = 3
NPAIR = 24                            # 잡음 바닥을 잴 프레임 쌍 수 (전 구간에 흩음)

# 사전 등록 `배정` 표 — 분할마다 학습군에 드는 오려내기 출처
SPLIT = {1: ['m3', 'kfire03', 'j04', '07'],
         2: ['m3', 'kfire03', 'q1', 'j04', '07', 'p2'],
         3: ['p2', 'j04'],
         4: ['kfire03', 'q1', 'j04', '07'],
         5: ['m3', 'kfire03']}

rng = np.random.default_rng(SEED)
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

print('=' * 74)
print('미리 못 박은 규칙 — 아래 숫자가 나오기 전에 정한 것임 (2회차와 **같은 기준**)')
print('=' * 74)
print('  상자 규칙   담은질량 평균 ≥ 0.95 를 지키는 것 중 **넓이비 평균이 가장 작은** 규칙')
print('              단 `알파0` 은 후보에서 뺌 — 소재가 이미 그 테두리로 잘려 있어')
print('              담은질량이 항상 1.0 임. 규칙이 아니라 **이미 한 자르기**임')
print('              (2회차에서도 알파0 은 넓이비가 가장 커서 뽑히지 않았음 — 결과 안 바뀜)')
print('  소재 제외   상자 안 **평균** 변화 < 개원중 프레임 쌍의 상자 안 **평균** 차의 중앙값')
print('              (2회차는 화면 전체 화소 중앙값과 견줬음 — 저울이 달랐음)')
print('  배경        **해밍 > 0 서로 다름**만 씀')
print('  표본        소재 전수 · 검산 표본은 **출처를 골고루** 섞음')
print('=' * 74)


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
mats = [(k, p) for k in KEYS for p in sorted(glob.glob(f'{MAT}/{k}/*.png'))]
if not bgs_all or not mats:
    raise SystemExit(f'배경 {len(bgs_all)}장 · 소재 {len(mats)}장 — 자료가 없음')
print(f'\n배경 {BGKEY} {len(bgs_all)}장(원본 전부) · 소재 {len(mats)}장(전수)')

# ---------------------------------------------------------------------------
# 2. 배경 서로 다름 — 사전 등록 178장과 **코드가 직접 대조**
# ---------------------------------------------------------------------------
PRE_REG = 178
hs = [(p, dhash(np.asarray(Image.open(p).convert('L'), np.float32))) for p in bgs_all]
bgs = []
kept_h = []
for p, h in hs:
    if all(ham(h, g) > 0 for g in kept_h):
        bgs.append(p);  kept_h.append(h)
print(f'  서로 다름 (해밍 > 0)  {len(bgs)}장 · 사전 등록 {PRE_REG}장   '
      f'{"맞음" if len(bgs) == PRE_REG else "**어긋남 — 어느 쪽이 틀렸는지 봐야 함**"}')
print(f'  (추출 스크립트는 farthest_order 로 세지만 **거리 0 에서 멈추므로** 세는 값이')
print(f'   `서로 다른 해시의 개수`로 같음. 그래서 두 셈이 같은 수를 내는 것이 정상임)')

# ---------------------------------------------------------------------------
# 3. 소재 읽기 — 리사이즈 뒤 **사전 등록 문턱 0.06 을 다시 걺**
# ---------------------------------------------------------------------------
bg0 = np.asarray(Image.open(bgs[0]).convert('RGB'), np.float32)
H, W = bg0.shape[:2]
BGAREA = H * W


def load_piece(p):
    q = Image.open(p).convert('RGBA')
    pw = max(int(round(W * U)), 8)
    ph = max(int(round(pw * q.size[1] / q.size[0])), 8)
    if ph > H - 2:                     # **보호** — 배경보다 큰 소재가 있으면 높이에 맞춤
        ph = H - 2
        pw = max(int(round(ph * q.size[0] / q.size[1])), 8)
    r = np.asarray(q.resize((pw, ph), Image.LANCZOS), np.float32)
    al = r[..., 3] / 255.0
    al[al < THR] = 0                   # 사전 등록 값. 리사이즈가 만든 잔털을 없앰
    return r[..., :3], al


pieces = []
for k, p in mats:
    rgb, al = load_piece(p)
    pieces.append({'key': k, 'file': os.path.basename(p), 'path': p,
                   'rgb': rgb, 'al': al,
                   'a_piece': float(al.mean())})     # **조각 기준** 알파평균 (직접 잼)

lo = min(float(x['al'][x['al'] > 0].min()) if (x['al'] > 0).any() else 1.0 for x in pieces)
print(f'\n  [검산] 리사이즈 뒤 0 이 아닌 알파의 최솟값 {lo:.4f}   '
      f'{"통과" if lo >= THR - 1e-6 else "**실패 — 문턱이 안 걸림**"}')

# manifest 의 alpha_mean 은 **잘라내기 전 화면 전체** 기준임. 견주기용으로만 읽음
amean_frame = {}
mpath = f'{MAT}/matte_manifest.json'
if os.path.exists(mpath):
    man = json.load(open(mpath))
    for k, v in man.items():
        for r in v.get('frames', []):
            if r.get('file'):
                amean_frame[(norm(k), norm(r['file']))] = r.get('alpha_mean')
nmiss = sum(1 for x in pieces if amean_frame.get((norm(x['key']), norm(x['file']))) is None)
print(f'  manifest 의 alpha_mean(화면 전체 기준) 을 못 찾은 소재 {nmiss}장'
      f'{"" if nmiss == 0 else "   ← 이 값은 견주기에서 빠짐"}')

# ---------------------------------------------------------------------------
# 4. 상자 규칙 — 정의와 **실패할 수 있는 검산**
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
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def box_clean(a, keep=0.99):
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return None
    mass = ndimage.sum(a, lab, range(1, n + 1))
    order = np.argsort(mass)[::-1]
    c = np.cumsum(mass[order]) / mass.sum()
    take = order[:int(np.searchsorted(c, keep)) + 1] + 1
    ys, xs = np.nonzero(np.isin(lab, take))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


RULES = [('알파0',      box_all,                       (255,  80,  80), False),
         ('최대덩어리',  box_main,                      ( 80, 200, 255), True),
         ('질량99',     lambda a: box_q(a, 0.005),     (120, 255, 120), True),
         ('질량95',     lambda a: box_q(a, 0.025),     (255, 220,   0), True),
         ('잡음제거99',  lambda a: box_clean(a, 0.99),  (255, 120, 255), True)]

# **실패할 수 있는 검산** — 답을 손으로 셀 수 있는 알파를 만들어 규칙마다 대조함
t = np.zeros((40, 60), np.float32)
t[10:30, 10:40] = 0.5                  # 큰 덩어리 (질량 300)
t[2, 55] = 0.5;  t[38, 2] = 0.5        # 구석 잡음 점 둘 (질량 0.5씩)
want = {'알파0': (2, 2, 56, 39), '최대덩어리': (10, 10, 40, 30),
        '질량99': (10, 10, 40, 30), '질량95': (10, 10, 40, 30),
        '잡음제거99': (10, 10, 40, 30)}
print('\n[검산] 손으로 셀 수 있는 알파에 규칙을 걸어 예상 상자와 대조')
bad = 0
for name, fn, _, _ in RULES:
    got = fn(t)
    okk = (got == want[name])
    bad += (not okk)
    print(f'  {name:<12}얻음 {str(got):<22}예상 {str(want[name]):<22}'
          f'{"통과" if okk else "**실패**"}')
if bad:
    print('  **규칙 셈이 틀림 — 아래 숫자를 근거로 읽지 말 것**')

nfull = sum(1 for x in pieces if box_all(x['al']) == (0, 0, x['al'].shape[1], x['al'].shape[0]))
print(f'\n  알파0 상자가 조각 전체와 같은 소재 {nfull} / {len(pieces)}장'
      f'   ← 소재가 이미 그 테두리로 잘려 저장돼 있다는 뜻. 그래서 후보에서 뺌')

# ---------------------------------------------------------------------------
# 5. 배경 — 흩어서 NBG 장, 그리고 잡음 바닥을 **같은 저울로** 잼
# ---------------------------------------------------------------------------
idx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
bgimgs = [np.asarray(Image.open(bgs[i]).convert('RGB'), np.float32) for i in idx]
print(f'\n배경 {NBG}장을 {len(bgs)}장에서 **고르게 흩어** 뽑음 (index {[int(i) for i in idx]})')

# 상자 크기의 대표값 — 소재들의 질량99 상자 크기 중앙값
sizes = []
for x in pieces:
    b = box_q(x['al'], 0.005)
    if b:
        sizes.append((b[3] - b[1], b[2] - b[0]))
BH, BW = (int(np.median([s[0] for s in sizes])), int(np.median([s[1] for s in sizes]))) \
    if sizes else (200, 300)
print(f'상자 크기 중앙값 {BW}x{BH} — 잡음 바닥도 **이 크기 조각에서 평균**으로 잼')

pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
vals = []
for i in pair_idx:
    f1 = np.asarray(Image.open(bgs_all[i]).convert('RGB'), np.float32)
    f2 = np.asarray(Image.open(bgs_all[i + 1]).convert('RGB'), np.float32)
    if f1.shape != f2.shape:
        continue
    d = np.abs(f2 - f1)
    for gy in range(GRID):
        for gx in range(GRID):
            yy = int(max(f1.shape[0] - BH, 0) * gy / max(GRID - 1, 1))
            xx = int(max(f1.shape[1] - BW, 0) * gx / max(GRID - 1, 1))
            vals.append(float(d[yy:yy + BH, xx:xx + BW].mean()))
FLOOR = float(np.median(vals)) if vals else 0.0
print(f'\n[잡음 바닥]  프레임 쌍 {len(pair_idx)}개 × 자리 {GRID*GRID}곳 = 표본 {len(vals)}개')
print(f'  상자 안 평균 차의 **중앙값**  {FLOOR:.3f} 계조   ← **제외 문턱**')
print(f'  같은 표본의 1사분 {np.percentile(vals,25):.3f} · 3사분 '
      f'{np.percentile(vals,75):.3f} · 최대 {max(vals):.3f}')
print(f'  0 인 쌍이 차지하는 몫 {np.mean(np.array(vals)==0):.1%}'
      f'   (프레임이 얼어 있으면 바닥이 낮아져 **제외가 느슨해짐**)')

# ---------------------------------------------------------------------------
# 6. 합성식 — **실패할 수 있는 검산** 둘
# ---------------------------------------------------------------------------
def comp_white(bg_patch, rgb_unused, al):
    return bg_patch * (1 - al[..., None]) + 255.0 * al[..., None]


def comp_obs(bg_patch, rgb, al):
    return bg_patch * (1 - al[..., None]) + rgb * al[..., None]


def put(bg, rgb, al, x, y, fn):
    ph, pw = al.shape
    y = int(np.clip(y, 0, max(bg.shape[0] - ph, 0)))
    x = int(np.clip(x, 0, max(bg.shape[1] - pw, 0)))
    out = bg.copy()
    out[y:y + ph, x:x + pw] = fn(out[y:y + ph, x:x + pw], rgb, al)
    return out, (x, y)


print('\n[검산] 합성 경로 — 답을 미리 아는 알파로')
a0 = 0.2
flat = np.full((BH, BW), a0, np.float32)
o, (px, py) = put(bg0, np.zeros((BH, BW, 3), np.float32), flat, 400, 300, comp_white)
got = float(np.abs(o - bg0)[py:py + BH, px:px + BW].mean())
exp = float((a0 * (255.0 - bg0[py:py + BH, px:px + BW])).mean())
print(f'  (1) 상수 알파 {a0} 를 얹은 변화   얻음 {got:.4f} · 예상 {exp:.4f}   '
      f'{"통과" if abs(got - exp) < 1e-3 else "**실패 — 합성·측정 경로가 틀림**"}')

zero = np.zeros((BH, BW), np.float32)
o2, _ = put(bg0, np.zeros((BH, BW, 3), np.float32), zero, 400, 300, comp_white)
e2 = float(np.abs(o2 - bg0).max())
print(f'  (2) 알파가 전부 0 인 조각        최대 변화 {e2:.6f}   '
      f'{"통과" if e2 < 1e-6 else "**실패 — 알파 0 인데 배경이 바뀜**"}')

# 두 식의 차 — **출처를 골고루** 섞고, `알파 × g` 로 제대로 잼
samp = [x for k in KEYS for x in [q for q in pieces if q['key'] == k][:6]]
g_w, g_px = [], []
for x in samp:
    a, obs = x['al'], x['rgb']
    g = np.abs(255.0 - obs).mean(2)
    m = a > 0
    if not m.any():
        continue
    g_w.append(float((g[m] * a[m]).sum() / a[m].sum()))     # 알파 실린 g 의 평균 (2회차 값)
    g_px.append(float((a[m] * g[m]).mean()))                # **실제 화소 차의 평균**
print(f'\n[두 식의 차]  출처 여섯에서 {len(samp)}장 (출처마다 최대 6장)')
print(f'  알파 실린 g 의 평균      {np.mean(g_w):6.1f} 계조   ← 2회차가 낸 값(84.1)과 같은 셈')
print(f'  **실제 화소 차의 평균**   {np.mean(g_px):6.1f} 계조   ← 관측색식이 어둡게 얹던 양')
print(f'  둘의 비                {np.mean(g_px)/max(np.mean(g_w),1e-9):.3f}'
      f'   (알파가 작을수록 벌어짐. 2회차 보고는 이만큼 부풀려져 있었음)')

# ---------------------------------------------------------------------------
# 7. 상자 다섯 규칙 — 전수
# ---------------------------------------------------------------------------
print(f'\n[상자 규칙]  소재 {len(pieces)}장 전수 · 가로폭 {U:g} 고정 · 배경 {W}x{H}')
print(f'{"규칙":<12}{"후보":>6}{"넓이비":>9}{"담은질량":>10}{"채움":>9}{"상자없음":>9}')
print('-' * 56)
stat = {r[0]: {'a': [], 'm': [], 'f': []} for r in RULES}
for x in pieces:
    x['box'] = {}
    for name, fn, _, _ in RULES:
        b = fn(x['al'])
        x['box'][name] = b
        if b is None:
            continue
        x0, y0, x1, y1 = b
        ins = float(x['al'][y0:y1, x0:x1].sum())
        area = (x1 - x0) * (y1 - y0)
        stat[name]['a'].append(area / BGAREA)
        stat[name]['m'].append(ins / max(float(x['al'].sum()), 1e-9))
        stat[name]['f'].append(ins / max(area, 1))
for name, _, _, cand in RULES:
    s = stat[name]
    print(f'{name:<12}{("예" if cand else "**아님**"):>6}{np.mean(s["a"]):>9.4f}'
          f'{np.mean(s["m"]):>10.4f}{np.mean(s["f"]):>9.4f}{len(pieces)-len(s["a"]):>9}')
print('-' * 56)

ok = [(n, float(np.mean(stat[n]['a']))) for n, _, _, cand in RULES
      if cand and np.mean(stat[n]['m']) >= 0.95]
if ok:
    PICK = min(ok, key=lambda t: t[1])[0]
    print(f'  미리 못 박은 기준 적용 -> **{PICK}**   (통과한 후보 {[n for n,_ in ok]})')
else:
    PICK = None
    print(f'  **담은질량 0.95 를 넘는 후보가 없음 — 상자 규칙을 못 고름. 미정으로 남김**')
    print(f'  아래는 질량99 로 이어 가되 이 값들은 **규칙 선택의 근거가 아님**')
RULE = PICK or '질량99'

# ---------------------------------------------------------------------------
# 8. 소재 제외 — 같은 저울(상자 안 평균)로
# ---------------------------------------------------------------------------
print(f'\n[배경 변화]  소재 전수 · 흰색식 · {RULE} 상자 안 **평균**')
print(f'  배경 {NBG}장 × 자리 {GRID}x{GRID} 의 중앙값. 문턱 {FLOOR:.3f} 계조')
for x in pieces:
    b = x['box'][RULE]
    if b is None:
        x['chg'] = x['peak'] = 0.0;  continue
    x0, y0, x1, y1 = b
    sub = x['al'][y0:y1, x0:x1]
    ph, pw = sub.shape
    ch, pk = [], []
    for bi in bgimgs:
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(bi.shape[0] - ph, 0) * gy / max(GRID - 1, 1))
                xx = int(max(bi.shape[1] - pw, 0) * gx / max(GRID - 1, 1))
                d = np.abs((255.0 - bi[yy:yy + ph, xx:xx + pw]) * sub[..., None])
                ch.append(float(d.mean()));  pk.append(float(d.max()))
    x['chg'] = float(np.median(ch));  x['peak'] = float(np.median(pk))

drop = [x for x in pieces if x['chg'] < FLOOR]
keepp = [x for x in pieces if x['chg'] >= FLOOR]
print(f'\n{"출처":<9}{"장":>5}{"남음":>6}{"제외":>6}{"변화 중앙":>11}{"변화 최소":>11}')
print('-' * 50)
left = {}
for k in KEYS:
    rr = [x for x in pieces if x['key'] == k]
    kk = [x for x in rr if x['chg'] >= FLOOR]
    left[k] = len(kk)
    print(f'{k:<9}{len(rr):>5}{len(kk):>6}{len(rr)-len(kk):>6}'
          f'{np.median([x["chg"] for x in rr]):>11.2f}{min(x["chg"] for x in rr):>11.2f}')
print('-' * 50)
print(f'{"합":<9}{len(pieces):>5}{len(keepp):>6}{len(drop):>6}')

print(f'\n분할마다 남는 학습 소재 (상한 32장 적용)')
for s, ks in SPLIT.items():
    print(f'  {s}번  {" · ".join(ks):<44}{sum(min(left[k], 32) for k in ks):>4}장')

print(f'\n**문턱 근처** — 제외 쪽 위 다섯과 통과 쪽 아래 다섯 (문턱 {FLOOR:.3f})')
near = sorted(pieces, key=lambda x: abs(x['chg'] - FLOOR))[:10]
for x in sorted(near, key=lambda x: x['chg']):
    af = amean_frame.get((norm(x['key']), norm(x['file'])))
    print(f'  {"제외" if x["chg"] < FLOOR else "통과"}  {x["key"]:<9}{x["file"]:<22}'
          f'변화 {x["chg"]:6.2f} · 봉우리 {x["peak"]:6.1f} · 조각알파 {x["a_piece"]:.4f}'
          f' · 화면알파 {af}')

# 봉우리는 높은데 평균이 낮아 제외된 것 — 제 기준이 못 가르는 자리
odd = sorted([x for x in drop if x['peak'] > 60], key=lambda x: -x['peak'])[:6]
if odd:
    print(f'\n제외됐지만 **봉우리가 높은** 소재 {len(odd)}장 — '
          f'`작지만 진한 연기`인지 `화소 몇 개짜리 잡음`인지 이 시험은 못 가름')
    for x in odd:
        print(f'  {x["key"]:<9}{x["file"]:<22}변화 {x["chg"]:6.2f} · 봉우리 {x["peak"]:6.1f}')

# 조각 기준 알파평균이 대리 지표로 쓸 만한가 (동점을 평균순위로 처리)
def rank_avg(v):
    v = np.asarray(v, float);  o = np.argsort(v);  r = np.empty(len(v), float)
    r[o] = np.arange(len(v), dtype=float)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


rho = float(np.corrcoef(rank_avg([x['a_piece'] for x in pieces]),
                        rank_avg([x['chg'] for x in pieces]))[0, 1])
print(f'\n조각 기준 알파평균 ↔ 배경변화 순위상관 {rho:+.3f}')
print(f'  (2회차는 **화면 전체 기준** 알파평균으로 이 값을 냈음 — 조각 넓이가 섞인 값이었음)')

json.dump({'floor': FLOOR, 'rule': PICK, 'nbg': NBG, 'seed': SEED,
           'records': [{kk: x[kk] for kk in ('key', 'file', 'chg', 'peak', 'a_piece')}
                       for x in pieces]},
          open(f'{OUT}/synth_trial3.json', 'w'), ensure_ascii=False, indent=1)

prev = f'{OUT}/synth_trial2.json'
if os.path.exists(prev):
    old = {(r['key'], r['file']): r['chg'] for r in json.load(open(prev))['records']}
    fl = json.load(open(prev))['floor']
    flip = sum(1 for x in pieces
               if (old.get((x['key'], x['file']), 0) < fl) != (x['chg'] < FLOOR))
    print(f'\n2회차 결과와 대조 — 제외↔통과가 뒤바뀐 소재 {flip}장')
else:
    print(f'\n2회차 결과는 Colab 안(/content)에만 있어 사라졌음 — 대조 못 함.'
          f' 이번부터 드라이브에 남김')

# ---------------------------------------------------------------------------
# 9. 시트
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
            t2 = Image.new('L', (size * 4, size * 2), 0)
            ImageDraw.Draw(t2).text((2, 2), '연기', fill=255, font=f)
            if np.asarray(t2).max() > 0:
                return f
        except Exception:
            pass
    return ImageFont.load_default()


F = korean_font(24)


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
                dd.rectangle([b[0]*s, b[1]*s, b[2]*s, b[3]*s], outline=col, width=3)
        d.text((6, y + 8), lab, fill=(255, 220, 0), font=F)
        sh.paste(pic, (0, y + 40))
        y += ch2 + 40
    sh.save(path, quality=90)
    return path


def off(b, x, y):
    return None if b is None else (b[0] + x, b[1] + y, b[2] + x, b[3] + y)


# (가) 문턱 근처 열 장 — **양쪽을 보여야 문턱을 볼 수 있음**
rows = []
for j, x in enumerate(sorted(near, key=lambda z: z['chg'])):
    bg = bgimgs[j % len(bgimgs)]
    px, py = (W - x['al'].shape[1]) // 2, int(H * 0.25)
    o, (px, py) = put(bg, x['rgb'], x['al'], px, py, comp_white)
    col = (255, 80, 80) if x['chg'] < FLOOR else (120, 255, 120)
    rows.append((f'{"제외" if x["chg"] < FLOOR else "통과"} {x["key"]} {x["file"]}  '
                 f'변화 {x["chg"]:.2f} · 문턱 {FLOOR:.2f}', o, [(off(x['box'][RULE], px, py), col)]))
p1 = sheet(rows, f'{OUT}/_threshold.jpg')

# (나) 상자 규칙 — 출처마다 **변화의 중앙값** 소재 (2회차는 파일 순서 가운데를 골랐음)
rows2 = []
for k in KEYS:
    kk = sorted([x for x in pieces if x['key'] == k and x['chg'] >= FLOOR],
                key=lambda z: z['chg'])
    if not kk:
        continue
    x = kk[len(kk) // 2]
    bg = bgimgs[KEYS.index(k) % len(bgimgs)]
    px, py = (W - x['al'].shape[1]) // 2, int(H * 0.25)
    o, (px, py) = put(bg, x['rgb'], x['al'], px, py, comp_white)
    rows2.append((f'{k} {x["file"]}  변화 {x["chg"]:.1f} (이 출처 통과분의 중앙값)   '
                  f'빨강 알파0 · 파랑 최대덩어리 · 초록 질량99 · 노랑 질량95 · 분홍 잡음제거99',
                  o, [(off(x['box'][n], px, py), c) for n, _, c, _ in RULES]))
p2 = sheet(rows2, f'{OUT}/_boxes.jpg')

# (다) 위치 규칙 — **같은 소재**를 두 규칙에 넣어 짝지음 (2회차는 서로 다른 소재였음)
surv = [x for x in pieces if x['chg'] >= FLOOR]
sel = [surv[int(i)] for i in rng.choice(len(surv), size=min(2, len(surv)), replace=False)]
xs_ = [int(rng.integers(0, max(W - x['al'].shape[1], 1))) for x in sel]
rows3 = []
for pos in ('전면', '상반부'):
    o, bb = bgimgs[3 % len(bgimgs)], []
    for x, xx in zip(sel, xs_):
        ph = x['al'].shape[0]
        yy = (int(rng.integers(0, max(int(H * 0.55) - ph, 1))) if pos == '상반부'
              else int(rng.integers(0, max(H - ph, 1))))
        o, (ax, ay) = put(o, x['rgb'], x['al'], xx, yy, comp_white)
        bb.append((off(x['box'][RULE], ax, ay), (120, 255, 120)))
    rows3.append((f'위치 {pos} · **같은 소재 둘** · 흰색식 · 상자 {RULE}', o, bb))
p3 = sheet(rows3, f'{OUT}/_place.jpg')

for q in (p1, p2, p3):
    print(f'-> {q}')
    files.download(q)
print(f'-> {OUT}/synth_trial3.json')
print('\n이 시험도 **학습 자료를 만들지 않음.** 여기서 2층을 닫고 나서 본 합성을 돌림.')
