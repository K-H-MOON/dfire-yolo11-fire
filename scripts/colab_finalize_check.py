# ===== 마무리(finalize)를 채울 것인가 — 재고 정함 =====
#
# 화재 저장소 `scripts/synthesize_smoke.py` 원문 (165\~172줄).
#
#   def finalize(img, bg, rng):
#       """센서 특성 정합 + JPEG 재압축 — 배경과 같은 입자감·압축 이력을 갖게 한다."""
#       sig = bg_noise(bg) / 255 * rng.uniform(.8, 1.4)
#       img = np.clip(img + rng.normal(0, sig, img.shape).astype(np.float32), 0, 1)
#       out = (img * 255).astype(np.uint8)
#       q = int(rng.integers(55, 93))
#       return cv2.imdecode(cv2.imencode('.jpg', out, [IMWRITE_JPEG_QUALITY, q])[1], 1)
#
#   def bg_noise(img):                                          # 25\~27줄
#       g = cvtColor(img, BGR2GRAY).astype(float32)
#       return float(np.median(np.abs(cv2.Laplacian(g, CV_32F))) * 0.9)
#
# 마무리는 **둘**임 — (1) 잡음 더하기 (2) JPEG 재압축. 갈라서 잼.
#
# ---------------------------------------------------------------------------
# 왜 다시 재는가 — 제가 놓쳤던 것
#
#   `bg_noise` 는 **라플라시안 절대값의 중앙값**임. 기록 82 에서 우리는 이미
#   `라플라시안은 장면 사이에 못 견줌 · 개원중이 흐리다는 근거가 없었음` 으로
#   한 번 뒤집힌 자리임. 그 지표를 그대로 가져와 잡음 세기를 정하면
#   `R <= 1.2` · `틴트` 와 같은 실패(그쪽 값을 재지 않고 옮김)를 반복하는 것임.
#
#   지금 이 작업 공간에서 급식실 프레임 다섯 장으로 미리 재 본 값 —
#       논현중_01 7.20 · 논현중_05 5.40 · 로봇고_01 **16.20** · 로봇고_02 10.80 · 숭곡중_01 8.10
#       흐림 시그마 1 을 걸면 전부 1.80\~4.50 으로 무너짐
#       실제 잡음 시그마 2 를 더해도 로봇고_01 은 16.20 -> 17.10 (거의 안 움직임)
#   -> **무늬가 지표를 지배함.** 개원중 178장으로 다시 재서 정함.
#
# ---------------------------------------------------------------------------
# **미리 못 박는 읽는 기준** — 숫자가 나오기 전에 정함. 어긋났을 때 순서까지 적음
#
#   [잡음 더하기]  아래 셋을 **전부** 통과해야 넣음. 하나라도 어긋나면 **안 넣음.**
#     가) bg_noise 중앙값이 실측 센서 잡음 시그마의 **0.5\~2.0배** 안에 있을 것
#         (실측 = 연속 프레임 쌍 차의 MAD 기반 강건 추정 / 루트2)
#     나) 같은 카메라·같은 자리 178장 사이에서 bg_noise 의 사분위 폭이
#         중앙값의 **50% 이하**일 것 (흩어지면 잡음이 아니라 장면 내용을 재는 것)
#     다) 얹은 자리의 **매끈해짐이 실재**할 것 —
#         알파 >= 0.3 자리에서 **배경 알갱이의 눌림**이 잡음 바닥 이상
#         (실재하지 않으면 메울 것이 없으므로 목적 자체가 없음)
#
#     [다] 를 재는 법을 한 번 고쳤음 — 처음에 국소 표준편차로 재려 했으나
#     가짜 자료로 미리 돌려 보니 `255 x 알파` 자체의 기울기가 섞여 눌림이
#     **거꾸로**(배경 1.440 -> 합성 2.302) 나왔음. 같은 소재를 연속 두 프레임에
#     얹으면 합성물의 차가 `(f2 - f1) x (1 - 알파)` 라 알파 기울기가 정확히
#     지워지므로 그것으로 바꿈. **문턱(잡음 바닥)은 안 바꿨고 재는 법만 바꿨음.**
#
#   [JPEG 재압축]  개원중 배경의 **추정 품질 분포**를 재고 그 10\~90% 구간을 씀.
#     그쪽 55\~93 을 그대로 옮기지 않음 — 그쪽 배경은 여러 출처가 섞여 이력이
#     제각각이었고 우리 배경은 개원중 한 곳임.
#     재압축이 합성물의 상자 안 변화를 잡음 바닥 밑으로 떨어뜨리는 장이
#         **0장** -> 절차 7단계(가시성 검사) 자리를 안 바꿈
#         **1장 이상** -> 검사를 **재압축 뒤로** 옮기고, 떨어진 장은 버림
#
#   [순서 문제]  화재 저장소는 `scene()` 202\~203줄에서 라벨을 확정한 **뒤에**
#     finalize 를 검. 즉 재압축이 지운 연기도 라벨은 남음. 그쪽은 이것을 안 적었음.
#     우리가 잡음 더하기를 넣으면 검사를 뒤로 옮길 수 없음 —
#     더한 잡음이 `배경 대비 변화` 에 그대로 실려 **연기 없이도 검사를 통과**시킴.
#     이 시험의 [D] 가 그것을 숫자로 확인함.
#
# ---------------------------------------------------------------------------
# 이 시험이 **못 재는 것**
#   개원중 프레임의 압축 이력이 **카메라 것인지 우리가 잘라 저장한 것인지** —
#     우리가 저장했다면 여기서 재는 품질은 우리 저장 품질임. 그래도 `배경과 같은
#     이력` 이라는 목적에는 그 값이 맞는 값임 (배경도 같은 손을 거쳤으므로)
#   마무리가 학습에 좋은가 — 학습을 돌려야 앎. 여기서는 **근거가 있는가**만 봄
#   센서 잡음이 채널마다 다른지 — 밝기 평균으로만 봄
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 4~7분.

import os, glob, json, unicodedata, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
MAT   = f'{SRC}/matte'
OUT   = f'{SRC}/synth_trial'
BGDIR = f'{SRC}/steam/bg'
BGKEY = '개원중'
KEYS  = ['m3', 'kfire03', 'q1', 'j04', '07', 'p2']

THR    = 0.06          # 알파 문턱 — 2층 확정값
IMGSZ, STRIDE = 640, 8
UMIN   = 0.30          # 크기 하한 가로폭 — 2층 확정값
GRID   = 3             # 자리 3x3
NBG    = 4             # 합성 시험에 쓸 배경 장수
NPAIR  = 40            # 잡음 바닥에 쓸 프레임 쌍
NSAMP  = 24            # 합성 시험에 쓸 소재 장수
QS     = [55, 70, 85, 93]
SEED   = 1

drive.mount('/content/drive')
rng = np.random.default_rng(SEED)

print('=' * 78)
print('미리 못 박은 읽는 기준 — 숫자가 나오기 전에 정함')
print('=' * 78)
print('  [잡음 더하기] 가·나·다 **전부** 통과해야 넣음. 하나라도 어긋나면 안 넣음')
print('     가) bg_noise 중앙값 / 실측 센서 잡음 시그마 가 **0.5~2.0** 안')
print('     나) 178장 사이 bg_noise 사분위 폭 <= 중앙값의 **50%**')
print('     다) 알파>=0.3 자리의 **배경 알갱이 눌림 >= 잡음 바닥** (프레임 쌍으로 잼)')
print('  [재압축] 개원중 추정 품질의 **10~90% 구간**을 씀 (그쪽 55~93 을 안 옮김)')
print('     재압축에 떨어지는 장이 1장이라도 있으면 **검사를 재압축 뒤로** 옮김')
print('=' * 78)


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(g, size=8):
    x = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size + 1, size),
                                                              Image.LANCZOS), np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


def box_q(a, q=0.005):
    if a.sum() <= 0:
        return None
    def span(v):
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, q)), int(np.searchsorted(c, 1 - q)) + 1
    y0, y1 = span(a.sum(1));  x0, x1 = span(a.sum(0))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def lap_noise(g):
    """화재 저장소 bg_noise — 밝기 화면 하나를 받음 (계조 단위)"""
    import cv2
    return float(np.median(np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))) * 0.9)


def jpeg_round(rgb_u8, q):
    """PIL 로 JPEG 왕복 — Colab 에 cv2 가 있어도 저장 경로를 하나로 둠"""
    b = io.BytesIO()
    Image.fromarray(rgb_u8).save(b, 'JPEG', quality=int(q))
    b.seek(0)
    return np.asarray(Image.open(b).convert('RGB'), np.float32)


# JPEG 표준 휘도 양자화표 (Annex K) — **자연순(행 우선)**
# PIL 의 `.quantization` 이 자연순으로 돌려줌. 지그재그 표를 쓰면 잔차가 크게 남고
# 역산이 1\~2 어긋남 — 여기서 확인함 (q 55/70/88/93 저장분에서 자연순 잔차 0)
NAT_BASE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,   12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,   14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    np.float32)


def est_quality(path):
    """양자화표에서 품질을 역산. 못 읽으면 None"""
    try:
        t = Image.open(path).quantization
    except Exception:
        return None
    if not t or 0 not in t:
        return None
    a = np.asarray(t[0], np.float32)
    if a.size != 64:
        return None
    best, bq = None, None
    for Q in range(1, 101):
        s = 5000.0 / Q if Q < 50 else 200.0 - 2.0 * Q
        pred = np.clip(np.floor((NAT_BASE * s + 50) / 100), 1, 255)
        e = float(np.abs(pred - a).sum())
        if best is None or e < best:
            best, bq = e, Q
    return bq, best


# ---------------------------------------------------------------------------
# 0. 배경 — 2층이 못 박은 `해밍 > 0 서로 다름` 을 그대로 되살림
# ---------------------------------------------------------------------------
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(np.asarray(Image.open(p).convert('L'), np.float32))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = np.asarray(Image.open(bgs[0]).convert('RGB')).shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ))
PWMIN = int(round(W * UMIN))
print(f'\n배경 {len(bgs_all)}장 -> 서로 다름 {len(bgs)}장 · {W}x{H}')
print(f'  [검산] 2층 문서의 **178장**과 {"일치" if len(bgs) == 178 else f"**불일치 — {len(bgs)}장**"}')
if len(bgs) != 178:
    print('  -> 아래 숫자를 2층과 나란히 읽으면 안 됨')


# ---------------------------------------------------------------------------
# A. bg_noise 가 잡음을 재는가 무늬를 재는가
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('[A] bg_noise 는 무엇을 재는가 — 개원중 178장')
print('=' * 78)

GM = {}                       # 밝기 평균 화면 (필요한 것만 담음)
def gray(p):
    if p not in GM:
        GM[p] = np.asarray(Image.open(p).convert('RGB'), np.float32).mean(2)
    return GM[p]

lap = np.array([lap_noise(gray(p)) for p in bgs], np.float32)
q1, q2, q3 = np.percentile(lap, [25, 50, 75])
iqr_ratio = (q3 - q1) / max(q2, 1e-9)
print(f'  bg_noise   중앙 {q2:.3f} · 25% {q1:.3f} · 75% {q3:.3f} · '
      f'최소 {lap.min():.3f} · 최대 {lap.max():.3f}')
print(f'  사분위 폭 / 중앙 = **{iqr_ratio:.1%}**   (기준 나: 50% 이하여야 넣음)')

# 실측 센서 잡음 — 연속 프레임 쌍. 움직임에 안 흔들리게 MAD 로
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
sig_true, diffs = [], []
for i in pair_idx:
    a, b = gray(bgs_all[i]), gray(bgs_all[i + 1])
    if a.shape != b.shape:
        continue
    d = b - a
    sig_true.append(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)))
    diffs.append(np.abs(d))
sig_true = float(np.median(sig_true))
ratio = q2 / max(sig_true, 1e-9)
print(f'  실측 센서 잡음 시그마 (프레임 쌍 {len(diffs)}개 · MAD/루트2) = **{sig_true:.3f} 계조**')
print(f'  bg_noise 중앙 / 실측 = **{ratio:.2f}배**   (기준 가: 0.50~2.00 이어야 넣음)')

# 흐림·잡음 반응 — 지표가 무엇에 움직이는가
import cv2
sub = [bgs[i] for i in np.linspace(0, len(bgs) - 1, 5).round().astype(int)]
print(f'\n  {"프레임":<26}{"원본":>8}{"흐림1":>8}{"흐림2":>8}{"+잡음2":>9}{"+잡음5":>9}')
print('  ' + '-' * 68)
for p in sub:
    g = gray(p)
    n2 = g + rng.normal(0, 2, g.shape)
    n5 = g + rng.normal(0, 5, g.shape)
    print(f'  {os.path.basename(p)[:24]:<26}{lap_noise(g):>8.2f}'
          f'{lap_noise(cv2.GaussianBlur(g, (0, 0), 1)):>8.2f}'
          f'{lap_noise(cv2.GaussianBlur(g, (0, 0), 2)):>8.2f}'
          f'{lap_noise(n2):>9.2f}{lap_noise(n5):>9.2f}')
print('  ' + '-' * 68)
print('  읽는 법 — 잡음 지표라면 `+잡음2` 에서 실제 더한 만큼 올라야 하고,')
print('           같은 카메라 178장 사이에서 크게 흩어지면 안 됨')

ok_a = 0.5 <= ratio <= 2.0
ok_b = iqr_ratio <= 0.50
print(f'\n  기준 가 {"통과" if ok_a else "**어긋남**"} · 기준 나 {"통과" if ok_b else "**어긋남**"}')


# ---------------------------------------------------------------------------
# B. 압축 이력
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('[B] 개원중 배경의 JPEG 압축 이력 — 양자화표에서 역산')
print('=' * 78)
qs, errs = [], []
for p in bgs:
    r = est_quality(p)
    if r:
        qs.append(r[0]);  errs.append(r[1])
if qs:
    qs = np.array(qs, np.float32)
    lo, hi = np.percentile(qs, [10, 90])
    print(f'  {len(qs)}장 · 추정 품질 중앙 **{np.median(qs):.0f}** · '
          f'최소 {qs.min():.0f} · 최대 {qs.max():.0f}')
    print(f'  10~90% 구간 = **{lo:.0f} ~ {hi:.0f}**    (그쪽 값 55~93 과 견줄 것)')
    print(f'  표 맞춤 잔차 중앙 {np.median(errs):.1f} '
          f'(0 에 가까울수록 표준표를 그대로 쓴 것 — 크면 역산을 믿지 말 것)')
    QLO, QHI = int(round(lo)), int(round(hi))
    # **우리 구간을 실제로 재지 않으면 [F] 가 아무 말도 못 함** — 시험 품질에 넣음
    QS = sorted(set(QS + [QLO, QHI]))
    print(f'  -> 재압축 시험 품질에 우리 구간 {QLO}·{QHI} 를 넣음 : {QS}')
else:
    print('  양자화표를 못 읽음 — 재압축 품질을 정할 근거가 없음')
    QLO, QHI = None, None
print('  **한계** — 이 값은 카메라의 이력이 아니라 우리가 프레임을 잘라 저장한')
print('           품질일 수 있음. 배경도 같은 손을 거쳤으므로 `배경과 같게` 라는')
print('           목적에는 맞으나, `센서 이력을 흉내낸다` 는 뜻은 아님')


# ---------------------------------------------------------------------------
# C. 잡음 바닥 — 2층이 쓰는 것과 같은 방식으로 되살림
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('[C] 잡음 바닥 — 크기별 (2층 절차 7단계가 쓰는 값)')
print('=' * 78)
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
FA, FV = [], []
print(f'  {"상자":>13}{"바닥(계조)":>12}')
print('  ' + '-' * 25)
for ar in AREAS:
    bw = int(min(W, max(MINSIDE, round(np.sqrt(ar * 16 / 9)))))
    bh = int(min(H, max(MINSIDE, round(ar / bw))))
    v = []
    for d in diffs:
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(H - bh, 0) * gy / max(GRID - 1, 1))
                xx = int(max(W - bw, 0) * gx / max(GRID - 1, 1))
                v.append(float(d[yy:yy + bh, xx:xx + bw].mean()))
    FA.append(float(np.log(bw * bh)));  FV.append(float(np.median(v)))
    print(f'  {f"{bw}x{bh}":>13}{FV[-1]:>12.3f}')
print('  ' + '-' * 25)
print(f'  {min(FV):.2f} ~ {max(FV):.2f} 계조 · 폭이 중앙의 {(max(FV)-min(FV))/np.median(FV):.1%}')
print(f'  [검산] 2층 기록의 **2.85~5.41** 과 '
      f'{"일치" if abs(min(FV)-2.85) < 0.3 and abs(max(FV)-5.41) < 0.3 else "**불일치 — 아래를 2층과 나란히 읽지 말 것**"}')


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


# ---------------------------------------------------------------------------
# D. 더한 잡음이 7단계 검사를 무력화하는가 — 연기 없이 배경에만 걸어 봄
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('[D] 마무리의 잡음만 걸었을 때 — 연기가 **없어도** 검사를 통과하는가')
print('=' * 78)
print(f'  {"상자":>13}{"바닥":>9}{"더한잡음 평균|변화|":>22}{"판정":>10}')
print('  ' + '-' * 56)
bidx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
n_vac = 0
for ar, fv in zip(AREAS, FV):
    bw = int(min(W, max(MINSIDE, round(np.sqrt(ar * 16 / 9)))))
    bh = int(min(H, max(MINSIDE, round(ar / bw))))
    v = []
    for i in bidx:
        g = gray(bgs[i])
        sg = lap_noise(g) * rng.uniform(0.8, 1.4)          # 원문 식 (계조 단위)
        d = np.abs(rng.normal(0, sg, (bh, bw)))
        v.append(float(d.mean()))
    mv = float(np.median(v))
    vac = mv >= fv
    n_vac += vac
    print(f'  {f"{bw}x{bh}":>13}{fv:>9.3f}{mv:>22.3f}'
          f'{"**통과시킴**" if vac else "못 통과":>10}')
print('  ' + '-' * 56)
print(f'  {len(AREAS)}개 크기 중 **{n_vac}개**에서 연기 없이도 검사를 통과함')
if n_vac:
    print('  -> 잡음을 더하면 검사를 **재압축 뒤로 옮길 수 없음** (검사가 빈 껍데기가 됨)')
else:
    print('  -> 잡음을 더해도 검사는 살아 있음')


# ---------------------------------------------------------------------------
# E. 얹은 자리가 실제로 매끈해지는가 — 마무리의 목적이 실재하는가
# F. 재압축이 옅은 연기를 지우는가
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('[E][F] 실제로 얹어 봄 — 매끈해짐 · 재압축')
print('=' * 78)

p2j = f'{OUT}/size_rule2.json'
covj = f'{OUT}/coverage.json'
passed = []
if os.path.exists(p2j):
    for x in json.load(open(p2j))['pieces']:
        if any(d['ok'] for d in x['draw']):
            passed.append((x['key'], x['file']))
if os.path.exists(covj):
    over = {(r['key'], r['file']) for r in json.load(open(covj)) if r['cov_frame'] > 0.60}
    passed = [t for t in passed if t not in over]
print(f'  쓸 수 있는 소재 {len(passed)}장'
      f'  [검산] 2층의 **67장**과 {"일치" if len(passed) == 67 else f"**불일치**"}')
if not passed:
    raise SystemExit('소재 목록을 못 읽음 — size_rule2.json / coverage.json 을 먼저 만들 것')

pick = [passed[i] for i in np.linspace(0, len(passed) - 1, min(NSAMP, len(passed)))
        .round().astype(int)]

# **프레임 쌍**으로 씀 — [E] 가 알파 기울기에 오염되지 않게 하려는 것.
# 국소 표준편차로 재면 `255 x 알파` 자체의 기울기가 섞여 눌림이 음수로도 나옴
# (가짜 자료 시험에서 배경 1.440 -> 합성 2.302 로 **거꾸로** 나왔음).
# 같은 소재를 연속 두 프레임에 얹으면 합성물의 차 = (f2 - f1) x (1 - 알파) 로
# 알파 기울기가 정확히 지워지고 **배경 알갱이만** 남음.
PAIRS = []
for i in np.linspace(0, len(bgs_all) - 2, NBG).round().astype(int):
    a = np.asarray(Image.open(bgs_all[i]).convert('RGB'), np.float32)
    b = np.asarray(Image.open(bgs_all[i + 1]).convert('RGB'), np.float32)
    if a.shape == b.shape and a.shape[:2] == (H, W):
        PAIRS.append((a, b))
if not PAIRS:
    raise SystemExit('프레임 쌍을 못 만듦')
BG = [p[0] for p in PAIRS]

rows = []
for k, f in pick:
    a8 = np.asarray(Image.open(f'{MAT}/{k}/{f}'))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    if a8.max() == 0:
        continue
    ph0, pw0 = a8.shape
    smax = min(1.0, W / pw0, H / ph0)
    smin = min(PWMIN / pw0, smax)
    s = float(np.exp(rng.uniform(np.log(smin), np.log(smax)))) if smin < smax else smax
    pw, ph = max(int(round(pw0 * s)), 4), max(int(round(ph0 * s)), 4)
    if pw > W or ph > H:
        continue
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS), np.float32) / 255.0
    al[al < THR] = 0
    b = box_q(al)
    if b is None or min(b[2] - b[0], b[3] - b[1]) < MINSIDE:
        continue
    yy = int(rng.integers(0, max(H - ph, 0) + 1))
    xx = int(rng.integers(0, max(W - pw, 0) + 1))
    f1, f2 = PAIRS[int(rng.integers(0, len(PAIRS)))]
    bg = f1

    img = bg.copy()
    A = np.zeros((H, W), np.float32)
    A[yy:yy + ph, xx:xx + pw] = al
    img = img * (1 - A[..., None]) + 255.0 * A[..., None]      # 2층 확정 흰색식

    x0, y0, x1, y1 = xx + b[0], yy + b[1], xx + b[2], yy + b[3]
    area = (x1 - x0) * (y1 - y0)
    fl = floor_at(area)
    base = float(np.abs(img - bg).mean(2)[y0:y1, x0:x1].mean())

    # [E] 매끈해짐 — 프레임 쌍의 **실제 알갱이**가 얼마나 눌리는가
    #     합성물 사이의 차 = (f2 - f1) x (1 - 알파)  → 알파 기울기가 안 섞임
    m = A >= 0.30
    if m.sum() >= 200:
        dn = np.abs(f2 - f1).mean(2)
        e_bg = float(dn[m].mean())                      # 배경이 스스로 만드는 알갱이
        e_im = float((dn * (1 - A))[m].mean())          # 합성물에 남는 알갱이
        drop = e_bg - e_im
    else:
        e_bg = e_im = drop = float('nan')

    # [F] 재압축이 지우는가
    reQ = {}
    for q in QS:
        r = jpeg_round(np.clip(img, 0, 255).astype(np.uint8), q)
        reQ[q] = float(np.abs(r - bg).mean(2)[y0:y1, x0:x1].mean())

    rows.append({'key': k, 'file': f, 's': s, 'area': int(area), 'floor': fl,
                 'base': base, 'std_bg': e_bg, 'std_img': e_im, 'drop': drop,
                 're': reQ})

print(f'\n  표본 {len(rows)}장 (소재 {len(pick)}장에서 놓기에 성공한 것)')

print(f'\n  [E] 알파>=0.3 자리 — 배경 알갱이가 얼마나 눌리는가 (프레임 쌍)')
good = [r for r in rows if r['drop'] == r['drop']]
d = np.array([r['drop'] for r in good], np.float32)
fls = np.array([r['floor'] for r in good], np.float32)
if d.size:
    print(f'      배경 알갱이 {np.mean([r["std_bg"] for r in good]):.3f}'
          f' -> 합성에 남음 {np.mean([r["std_img"] for r in good]):.3f}'
          f'  눌림 중앙 **{np.median(d):.3f} 계조**')
    n_big = int(np.greater_equal(d, fls).sum())
    print(f'      눌림 >= 그 장의 잡음 바닥인 장 **{n_big}/{d.size}장**')
    ok_c = n_big >= d.size * 0.5
    print(f'      기준 다 {"통과 — 메울 것이 실재함" if ok_c else "**어긋남 — 메울 것이 크지 않음**"}')
else:
    ok_c = False
    print('      알파>=0.3 자리가 모자라 못 잼')

print(f'\n  [F] 재압축 뒤 상자 안 평균 변화 — 바닥 밑으로 떨어지는 장')
print(f'      {"품질":>6}{"변화 중앙":>12}{"원본 대비":>11}{"바닥 밑":>10}')
print('      ' + '-' * 39)
FLR = np.array([r['floor'] for r in rows], np.float32)
BASE = np.array([r['base'] for r in rows], np.float32)


def n_below(q):
    """품질 q 로 재압축한 뒤 상자 안 변화가 그 장의 잡음 바닥 밑으로 내려간 장 수"""
    v = np.array([r['re'][q] for r in rows], np.float32)
    return int(np.less(v, FLR).sum())


for q in QS:
    v = np.array([r['re'][q] for r in rows], np.float32)
    mark = ' <- 우리 구간' if (QLO and QLO <= q <= QHI) else ''
    print(f'      {q:>6}{np.median(v):>12.3f}'
          f'{np.median(v / np.maximum(BASE, 1e-9)):>11.3f}'
          f'{n_below(q):>7}/{len(v)}{mark}')
print('      ' + '-' * 39)
print(f'      (재압축 전 변화 중앙 {np.median(BASE):.3f} · '
      f'바닥 중앙 {np.median(FLR):.3f})')
if QLO:
    vq = [n_below(q) for q in QS if QLO <= q <= QHI]
    worst = max(vq) if vq else 0
    print(f'      우리 구간 {QLO}~{QHI} 안에서 떨어지는 장 **{worst}장**')
    if worst:
        print('      -> 절차 7단계의 가시성 검사를 **재압축 뒤로** 옮기고 떨어진 장은 버림')
    else:
        print('      -> 검사 자리를 안 바꿔도 됨')

json.dump({'lap_median': float(q2), 'lap_iqr_ratio': float(iqr_ratio),
           'sigma_true': sig_true, 'ratio': float(ratio),
           'q_lo': QLO, 'q_hi': QHI, 'floor': list(map(float, FV)),
           'vacuous_sizes': int(n_vac), 'rows': rows},
          open(f'{OUT}/finalize_check.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {OUT}/finalize_check.json')


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('판정 — 미리 못 박은 기준을 그대로 적용')
print('=' * 78)
print(f'  가) bg_noise / 실측 잡음 = {ratio:.2f}배          {"통과" if ok_a else "**어긋남**"}')
print(f'  나) 사분위 폭 / 중앙 = {iqr_ratio:.1%}            {"통과" if ok_b else "**어긋남**"}')
print(f'  다) 매끈해짐 >= 바닥                  {"통과" if ok_c else "**어긋남**"}')
print('  ' + '-' * 60)
if ok_a and ok_b and ok_c:
    print('  -> **잡음 더하기를 넣음.** 단 [D] 에서 검사가 빈 껍데기가 되는 크기가')
    print(f'     {n_vac}개이므로, 가시성 검사는 **잡음 더하기 앞**에 두고 그 한계를 적음')
else:
    print('  -> **잡음 더하기를 안 넣음.** 어긋난 기준을 그대로 적음')
print(f'  -> 재압축: 우리 구간 **{QLO}~{QHI}**' if QLO else '  -> 재압축: 근거 없음 — 안 넣음')


# ---------------------------------------------------------------------------
# 시트 — 눈으로 대조 (원본 배율, 잘라서)
# ---------------------------------------------------------------------------
def korean_font(size):
    c = (glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
         + glob.glob('/usr/share/fonts/**/*Nanum*.ttf', recursive=True)
         + glob.glob('/usr/share/fonts/**/NotoSansCJK*', recursive=True))
    if not c:
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        c = glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True)
    for f in c:
        try:
            ft = ImageFont.truetype(f, size)
            t = Image.new('L', (size * 4, size * 2), 0)
            ImageDraw.Draw(t).text((2, 2), '연기', fill=255, font=ft)
            if np.asarray(t).max() > 0:
                return ft
        except Exception:
            pass
    return ImageFont.load_default()


F = korean_font(20)
srt = sorted(rows, key=lambda r: r['base'])[:4]        # 가장 옅은 넷
CH, items = 300, []
for r in srt:
    a8 = np.asarray(Image.open(f'{MAT}/{r["key"]}/{r["file"]}'))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    pw = max(int(round(a8.shape[1] * r['s'])), 4)
    ph = max(int(round(a8.shape[0] * r['s'])), 4)
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS), np.float32) / 255.0
    al[al < THR] = 0
    bg = BG[0][:ph, :pw]
    im = bg * (1 - al[..., None]) + 255.0 * al[..., None]
    lo = jpeg_round(np.clip(im, 0, 255).astype(np.uint8), QS[0])
    items.append((f'{r["key"]} {r["file"]}  배율 {r["s"]:.3f} · 변화 {r["base"]:.2f} · '
                  f'바닥 {r["floor"]:.2f} · q{QS[0]} 뒤 {r["re"][QS[0]]:.2f}',
                  [np.clip(bg, 0, 255).astype(np.uint8),
                   np.clip(im, 0, 255).astype(np.uint8),
                   np.clip(lo, 0, 255).astype(np.uint8),
                   np.clip(np.abs(lo - bg).mean(2) * 12, 0, 255).astype(np.uint8)]))
if items:
    CW = max(i[1][0].shape[1] for i in items)
    Ht = sum(i[1][0].shape[0] + 28 for i in items) + 8
    sh = Image.new('RGB', (CW * 4, Ht), (16, 16, 16))
    dr = ImageDraw.Draw(sh);  y = 0
    for lab, arrs in items:
        dr.text((6, y + 4), lab + '   (배경 · 합성 · q재압축 · 차이x12)',
                fill=(255, 220, 0), font=F)
        for j, a in enumerate(arrs):
            a = a if a.ndim == 3 else np.dstack([a] * 3)
            sh.paste(Image.fromarray(a), (CW * j, y + 28))
        y += arrs[0].shape[0] + 28
    sh.save(f'{OUT}/_finalize.jpg', quality=92)
    print(f'-> {OUT}/_finalize.jpg  ({len(items)}줄 · **원본 배율**)')
    files.download(f'{OUT}/_finalize.jpg')
    print('\n가장 옅은 넷임. 셋째 칸에서 연기가 사라졌는지 **눈으로** 볼 것.')
