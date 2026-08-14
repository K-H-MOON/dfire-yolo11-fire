import os, glob, json, io, unicodedata
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
MAT   = f'{SRC}/matte'
OUT   = f'{SRC}/synth_trial'
BGDIR = f'{SRC}/steam/bg'
BGKEY = '개원중'

THR    = 0.06
IMGSZ, STRIDE = 640, 8
UMIN   = 0.30
GRID   = 3             # 자리 3x3   — size_rule2 와 같게
NBG    = 4             # 배경 장수  — size_rule2 와 같게
NPAIR  = 24            # 프레임 쌍  — size_rule2 와 같게
NSAMP  = 24            # [E][F] 표본 소재
NQBG   = 8             # 압축 이력을 직접 잴 배경 장수
QS     = [55, 70, 85, 92]     # 원문 `integers(55, 93)` = **55~92**
QSCAN  = list(range(60, 100, 2))
SEED   = 1

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

print('=' * 78)
print('다시 못 박은 읽는 기준 — 숫자가 나오기 전에 정함 (원문 식과 산술에서 끌어냄)')
print('=' * 78)
print('  [잡음] 가·나·다 **전부** 통과해야 넣음')
print('     가) 실측 센서 잡음 시그마가 **[0.8 x bg_noise, 1.4 x bg_noise]** 안에 들어올 것')
print('     나) 178장 사이 bg_noise 사분위 폭 <= 중앙값의 **50%**')
print('     다) **눌림 중앙 >= 바닥 중앙** · 장별 비율도 적음 · 어긋나면 안 넣음')
print('  [재압축] 품질을 **직접 재서** 정함 (표 역산은 잔차 0 일 때만)')
print('     바닥 위에 있던 장을 밑으로 떨어뜨리면 1장이라도 -> 검사를 재압축 뒤로')
print('=' * 78)


r'''공통'''
def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(g, size=8):
    x = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size + 1, size),
                                                              Image.LANCZOS), np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


def rgbf(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float32)


def lumaf(p):
    """원문 bg_noise 가 쓰는 것과 같은 휘도 (PIL 'L' = ITU-R 601 luma)"""
    return np.asarray(Image.open(p).convert('L'), np.float32)


def lap_noise(g):
    """화재 저장소 bg_noise — 계조 단위. 순수 흰 잡음이면 2.715 x 시그마"""
    return float(np.median(np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))) * 0.9)


LAP_WHITE = 0.9 * 0.6745 * np.sqrt(20.0)      # = 2.715


def jpeg_round(rgb_u8, q):
    b = io.BytesIO()
    Image.fromarray(rgb_u8).save(b, 'JPEG', quality=int(q))
    b.seek(0)
    return np.asarray(Image.open(b).convert('RGB'), np.float32)


r'''JPEG 표준 휘도 양자화표 (Annex K) — **자연순(행 우선)**. / PIL 의 `.quantization` 이 자연순으로 돌려줌 (q 55/70/88/93 저장분에서 잔차 0 확인)'''
NAT_BASE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,   12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,   14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    np.float32)


def est_quality(path):
    try:
        t = Image.open(path).quantization
    except Exception:
        return None
    if not t or 0 not in t:
        return None
    a = np.asarray(t[0], np.float32)
    if a.size != 64:
        return None
    best = None
    for Q in range(1, 101):
        s = 5000.0 / Q if Q < 50 else 200.0 - 2.0 * Q
        pred = np.clip(np.floor((NAT_BASE * s + 50) / 100), 1, 255)
        e = float(np.abs(pred - a).sum())
        if best is None or e < best[1]:
            best = (Q, e)
    return best


r'''0. 배경'''
bgs_all = sorted(p for p in glob.glob(f'{BGDIR}/*.jpg')
                 if norm(os.path.basename(p)).startswith(norm(BGKEY)))
bgs, kh = [], []
for p in bgs_all:
    h = dhash(lumaf(p))
    if all(ham(h, g) > 0 for g in kh):
        bgs.append(p);  kh.append(h)
H, W = rgbf(bgs[0]).shape[:2]
MINSIDE = int(round(STRIDE * W / IMGSZ))
PWMIN = int(round(W * UMIN))
print(f'\n배경 {len(bgs_all)}장 -> 서로 다름 {len(bgs)}장 · {W}x{H}')
print(f'  [검산] 2층의 **178장**과 {"일치" if len(bgs) == 178 else f"**불일치 — {len(bgs)}장**"}')


r'''C. 잡음 바닥 — **size_rule2 와 한 글자도 다르지 않게**'''
print('\n' + '=' * 78)
print('[C] 잡음 바닥 — size_rule2 를 그대로 되살림 (채널마다 절대값 -> 평균 · 쌍 24개)')
print('=' * 78)
AREAS = np.unique(np.round(np.geomspace(MINSIDE * MINSIDE, 1900 * 1000, 8)).astype(int))
pair_idx = np.linspace(0, len(bgs_all) - 2, NPAIR).round().astype(int)
diffs, zero_pairs = [], 0
for i in pair_idx:
    f1, f2 = rgbf(bgs_all[i]), rgbf(bgs_all[i + 1])
    if f1.shape != f2.shape:
        continue
    d = np.abs(f2 - f1).mean(2)               # 채널마다 절대값 -> 평균
    zero_pairs += (float(d.mean()) < 0.01)
    diffs.append(d)
print(f'  쌍 {len(diffs)}개 · 그중 **사실상 같은 프레임 {zero_pairs}개**'
      f'{"" if not zero_pairs else "   ← 바닥을 끌어내림. 아래 [A] 에서 따로 다룸"}')

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
FLO, FHI = min(FV), max(FV)
ok_floor = abs(FLO - 2.85) < 0.15 and abs(FHI - 5.41) < 0.15
print(f'  {FLO:.2f} ~ {FHI:.2f} 계조')
print(f'  [검산] 2층의 **2.85~5.41** 과 '
      f'{"일치 — 3회차 식을 되살렸음" if ok_floor else "**불일치. 아래 판정을 2층과 나란히 읽지 말 것**"}')


def floor_at(area):
    return float(np.interp(np.log(max(area, 1)), FA, FV))


r'''A. bg_noise — 넣을 잡음이 배경의 알갱이와 맞는가'''
print('\n' + '=' * 78)
print('[A] 넣을 잡음이 배경의 실제 알갱이와 맞는가')
print('=' * 78)
lap = np.array([lap_noise(lumaf(p)) for p in bgs], np.float32)
q1, q2, q3 = np.percentile(lap, [25, 50, 75])
iqr_ratio = (q3 - q1) / max(q2, 1e-9)
print(f'  bg_noise (휘도)  중앙 {q2:.3f} · 25% {q1:.3f} · 75% {q3:.3f} · '
      f'최소 {lap.min():.3f} · 최대 {lap.max():.3f}')

r'''실측 센서 잡음 — **같은 프레임 쌍은 뺌** (1판이 안 뺐음)'''
sig_list, nskip = [], 0
for i in pair_idx:
    a, b = lumaf(bgs_all[i]), lumaf(bgs_all[i + 1])
    if a.shape != b.shape:
        continue
    d = b - a
    s = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    if s < 0.05:
        nskip += 1;  continue
    sig_list.append(s)
sig_true = float(np.median(sig_list)) if sig_list else float('nan')
print(f'  실측 센서 잡음 시그마 = **{sig_true:.3f} 계조**  '
      f'(쌍 {len(sig_list)}개 · 사실상 같은 쌍 {nskip}개 뺌)')

r'''순수 흰 잡음이었다면 bg_noise 가 얼마로 나왔어야 하는가 — 직접 만들어 잼'''
ref = lap_noise(rng.normal(0, sig_true, (700, 700)))
print(f'  같은 시그마의 **순수 흰 잡음** 그림에서 잰 bg_noise = {ref:.3f} '
      f'(산술값 {LAP_WHITE * sig_true:.3f})')
print(f'  실제 배경의 bg_noise {q2:.3f} 은 그보다 '
      f'{"작음 -> 잡음이 흰 잡음이 아님(압축으로 이웃이 묶임)" if q2 < ref else "큼 -> 무늬가 섞여 있음"}')

sig_lo, sig_hi = 0.8 * q2, 1.4 * q2
print(f'\n  원문이 넣을 수 있는 시그마 범위 = [0.8 x bg_noise, 1.4 x bg_noise] '
      f'= **{sig_lo:.3f} ~ {sig_hi:.3f} 계조**')
print(f'  실측 알갱이 {sig_true:.3f} 계조가 그 안에 '
      f'{"들어옴" if sig_lo <= sig_true <= sig_hi else "**안 들어옴**"}')
if sig_true < sig_lo:
    print(f'  -> 어떤 뽑기를 해도 최소 **{sig_lo / sig_true:.2f}배** 과하게 넣음')
elif sig_true > sig_hi:
    print(f'  -> 어떤 뽑기를 해도 최소 **{sig_true / sig_hi:.2f}배** 모자라게 넣음')

ok_a = sig_lo <= sig_true <= sig_hi
ok_b = iqr_ratio <= 0.50
print(f'\n  사분위 폭 / 중앙 = {iqr_ratio:.1%}')
print(f'  기준 가 {"통과" if ok_a else "**어긋남**"} · 기준 나 {"통과" if ok_b else "**어긋남**"}')

print(f'\n  참고 — 지표가 무엇에 움직이는가')
print(f'  {"프레임":<24}{"원본":>8}{"흐림1":>8}{"흐림2":>8}{"+잡음2":>9}{"+잡음5":>9}')
print('  ' + '-' * 66)
for p in [bgs[i] for i in np.linspace(0, len(bgs) - 1, 5).round().astype(int)]:
    g = lumaf(p)
    print(f'  {os.path.basename(p)[:22]:<24}{lap_noise(g):>8.2f}'
          f'{lap_noise(cv2.GaussianBlur(g, (0, 0), 1)):>8.2f}'
          f'{lap_noise(cv2.GaussianBlur(g, (0, 0), 2)):>8.2f}'
          f'{lap_noise(g + rng.normal(0, 2, g.shape)):>9.2f}'
          f'{lap_noise(g + rng.normal(0, 5, g.shape)):>9.2f}')
print('  ' + '-' * 66)
print(f'  읽는 법 — 순수 흰 잡음이면 시그마의 {LAP_WHITE:.2f}배가 나옴. '
      f'`+잡음2` 는 원본에 {LAP_WHITE * 2:.2f} 어치가 더해진 것')


r'''B. 압축 이력 — 표 역산 + **직접 재기**'''
print('\n' + '=' * 78)
print('[B] 개원중 배경의 JPEG 압축 이력')
print('=' * 78)
est = [est_quality(p) for p in bgs]
est = [e for e in est if e]
if est:
    eq = np.array([e[0] for e in est], np.float32)
    er = np.array([e[1] for e in est], np.float32)
    print(f'  표 역산  {len(eq)}장 · 품질 중앙 {np.median(eq):.0f} · '
          f'잔차 중앙 **{np.median(er):.1f}**')
    trust_tbl = float(np.median(er)) < 1e-6
    print(f'  -> 잔차가 0 이 {"맞음 — 역산을 씀" if trust_tbl else "**아님. 표준 양자화표로 저장된 파일이 아니므로 역산을 안 씀**"}')
else:
    trust_tbl = False
    print('  표 역산  양자화표를 못 읽음')

qbg = [bgs[i] for i in np.linspace(0, len(bgs) - 1, NQBG).round().astype(int)]
print(f'\n  직접 재기 — 배경 {len(qbg)}장을 품질마다 되감아 `|되감은 것 - 원본|` 을 잼')
print(f'  {"품질":>6}{"변화 평균(계조)":>18}')
print('  ' + '-' * 24)
VQ = np.zeros((len(QSCAN), len(qbg)), np.float32)       # 한 번만 잼
for j, p in enumerate(qbg):
    a = rgbf(p);  a8 = a.astype(np.uint8)
    for i, q in enumerate(QSCAN):
        VQ[i, j] = float(np.abs(jpeg_round(a8, q) - a).mean())
curve = VQ.mean(1)
for q, c in zip(QSCAN, curve):
    print(f'  {q:>6}{c:>18.4f}{"   <- 가장 작음" if c == curve.min() else ""}')
print('  ' + '-' * 24)
best_q = [QSCAN[int(np.argmin(VQ[:, j]))] for j in range(len(qbg))]
QLO, QHI = int(min(best_q)), int(max(best_q))
print(f'  배경별 최소 지점 {best_q}')
print(f'  -> 채택 구간 **{QLO} ~ {QHI}**    (원문은 55~92)')
QS = sorted(set(QS + list(range(QLO, QHI + 1, max(1, (QHI - QLO) // 2 or 1)))))
print(f'  재압축 시험 품질 : {QS}')
print('  **한계** — 이 값은 카메라 이력이 아니라 우리가 잘라 저장한 품질일 수 있음')


r'''D. 마무리를 **그대로** 배경에만 걸었을 때 — 연기 없이도 검사를 통과하는가'''
print('\n' + '=' * 78)
print('[D] 마무리(잡음 + 자르기 + 재압축)를 **연기 없는 배경**에 그대로 걸어 봄')
print('=' * 78)
bidx = np.linspace(0, len(bgs) - 1, NBG).round().astype(int)
FIN = []
for i in bidx:
    a = rgbf(bgs[i])
    sg = lap_noise(lumaf(bgs[i])) * rng.uniform(0.8, 1.4)
    n = np.clip(a + rng.normal(0, sg, a.shape), 0, 255)      # 원문의 clip 포함
    qq = int(rng.integers(QLO, QHI + 1))
    FIN.append((a, np.clip(n, 0, 255), jpeg_round(n.astype(np.uint8), qq), sg, qq))
print(f'  쓴 시그마 {[f"{f[3]:.2f}" for f in FIN]} · 품질 {[f[4] for f in FIN]}')
print(f'  {"상자":>13}{"바닥":>9}{"잡음만":>10}{"잡음+재압축":>14}{"판정":>12}')
print('  ' + '-' * 60)
n_vac = 0
for ar, fv in zip(AREAS, FV):
    bw = int(min(W, max(MINSIDE, round(np.sqrt(ar * 16 / 9)))))
    bh = int(min(H, max(MINSIDE, round(ar / bw))))
    v1, v2 = [], []
    for a, n, f, _, _ in FIN:
        d1 = np.abs(n - a).mean(2);  d2 = np.abs(f - a).mean(2)
        for gy in range(GRID):
            for gx in range(GRID):
                yy = int(max(H - bh, 0) * gy / max(GRID - 1, 1))
                xx = int(max(W - bw, 0) * gx / max(GRID - 1, 1))
                v1.append(float(d1[yy:yy + bh, xx:xx + bw].mean()))
                v2.append(float(d2[yy:yy + bh, xx:xx + bw].mean()))
    m1, m2 = float(np.median(v1)), float(np.median(v2))
    vac = m2 >= fv
    n_vac += vac
    print(f'  {f"{bw}x{bh}":>13}{fv:>9.3f}{m1:>10.3f}{m2:>14.3f}'
          f'{"**통과시킴**" if vac else "못 통과":>12}')
print('  ' + '-' * 60)
print(f'  {len(AREAS)}개 크기 중 **{n_vac}개**에서 연기 없이도 검사를 통과함')


r'''E·F. 소재 — **게이트가 고른 크기 그대로**'''
print('\n' + '=' * 78)
print('[E][F] 소재 — 3회차 게이트가 통과시킨 크기·상자를 그대로 씀')
print('=' * 78)
sr = json.load(open(f'{OUT}/size_rule2.json'))['pieces']
cov = {(r['key'], r['file']): r['cov_frame'] for r in json.load(open(f'{OUT}/coverage.json'))}
pass1 = [x for x in sr if any(d['ok'] for d in x['draw'])]
pass2 = [x for x in pass1 if cov.get((x['key'], x['file']), 0.0) <= 0.60]
print(f'  크기 규칙 통과 **{len(pass1)}장** -> 피복률 상한 60% 적용 -> **{len(pass2)}장**')
print(f'  [검산] 2층의 **68 -> 67** 과 '
      f'{"일치" if (len(pass1), len(pass2)) == (68, 67) else "**불일치**"}')

pick = [pass2[i] for i in np.linspace(0, len(pass2) - 1, min(NSAMP, len(pass2)))
        .round().astype(int)]

r'''배경(게이트와 같은 4장) · 프레임 쌍(알갱이용)'''
BGM = [rgbf(bgs[i]).mean(2) for i in bidx]
BGR = [rgbf(bgs[i]) for i in bidx]
DN = []
for i in np.linspace(0, len(bgs_all) - 2, NBG).round().astype(int):
    f1, f2 = rgbf(bgs_all[i]), rgbf(bgs_all[i + 1])
    DN.append(np.abs(f2 - f1).mean(2) if f1.shape == f2.shape else np.zeros((H, W), np.float32))

JBG = {}


def jbg(bi, q):
    if (bi, q) not in JBG:
        JBG[(bi, q)] = jpeg_round(BGR[bi].astype(np.uint8), q)
    return JBG[(bi, q)]


rows = []
for x in pick:
    ok = [d for d in x['draw'] if d['ok']]
    d0 = ok[len(ok) // 2]                       # 통과한 크기 중 가운데
    s, b = d0['s'], d0['box']
    a8 = np.asarray(Image.open(f'{MAT}/{x["key"]}/{x["file"]}'))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    pw = max(int(round(a8.shape[1] * s)), 4);  ph = max(int(round(a8.shape[0] * s)), 4)
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS), np.float32) / 255.0
    al[al < THR] = 0
    sub = al[b[1]:b[3], b[0]:b[2]]
    bh, bw = sub.shape                 # 상자가 조각 밖으로 나가면 잘리므로 **잘린 뒤 크기**를 씀
    if bh < 1 or bw < 1:
        continue
    area = bw * bh
    fl = d0['floor']

    chg, drop, req = [], [], {q: [] for q in QS}
    for gy in range(GRID):
        for gx in range(GRID):
            yy = int(max(H - ph, 0) * gy / max(GRID - 1, 1))
            xx = int(max(W - pw, 0) * gx / max(GRID - 1, 1))
            y0, x0 = yy + b[1], xx + b[0]
            for bi in range(NBG):
                chg.append(float(((255.0 - BGM[bi][y0:y0 + bh, x0:x0 + bw]) * sub).mean()))
                drop.append(float((DN[bi][y0:y0 + bh, x0:x0 + bw] * sub).mean()))
            r'''재압축은 배경 1장 · 자리마다 (되감기 비용 때문)'''
            A = np.zeros((H, W), np.float32);  A[yy:yy + ph, xx:xx + pw] = al
            img = BGR[0] * (1 - A[..., None]) + 255.0 * A[..., None]
            for q in QS:
                r = jpeg_round(np.clip(img, 0, 255).astype(np.uint8), q)
                req[q].append(float(np.abs(r - jbg(0, q)).mean(2)[y0:y0 + bh, x0:x0 + bw].mean()))
    rows.append({'key': x['key'], 'file': x['file'], 's': s, 'box': b,
                 'bw': int(bw), 'bh': int(bh), 'area': int(area),
                 'floor': fl, 'chg_rec': d0['chg'], 'chg': float(np.median(chg)),
                 'drop': float(np.median(drop)),
                 're': {q: float(np.median(req[q])) for q in QS}})

print(f'\n  표본 {len(rows)}장')
rr = np.array([r['chg'] / max(r['chg_rec'], 1e-9) for r in rows], np.float32)
print(f'  [검산] 다시 잰 변화 / size_rule2 가 적어 둔 변화 중앙 **{np.median(rr):.3f}** '
      f'{"통과" if abs(np.median(rr) - 1) <= 0.02 else "**실패 — 아래를 못 씀**"}')

CH = np.array([r['chg'] for r in rows], np.float32)
FL = np.array([r['floor'] for r in rows], np.float32)
DR = np.array([r['drop'] for r in rows], np.float32)

print(f'\n  [E] 알파가 배경 알갱이를 누르는 양 (= 평균(알갱이 x 알파) · 항등식)')
print(f'      눌림 중앙 **{np.median(DR):.3f}** · 바닥 중앙 **{np.median(FL):.3f}** 계조')
print(f'      장별로 눌림 >= 바닥 인 장 **{int(np.greater_equal(DR, FL).sum())}/{len(DR)}장**')
ok_c1 = float(np.median(DR)) >= float(np.median(FL))
ok_c2 = int(np.greater_equal(DR, FL).sum()) >= len(DR) * 0.5
ok_c = ok_c1 and ok_c2
print(f'      대표값 {"통과" if ok_c1 else "**어긋남**"} · 장별 비율 '
      f'{"통과" if ok_c2 else "**어긋남**"}'
      f'{"" if ok_c1 == ok_c2 else "   ← 둘이 어긋남. 미리 정한 대로 안 넣음"}')

print(f'\n  [F] 재압축 — 합성물과 배경을 **같은 품질로** 되감아 견줌')
print(f'      {"품질":>6}{"변화 중앙":>12}{"원본 대비":>11}{"떨어진 장":>11}')
print('      ' + '-' * 41)
above = np.greater_equal(CH, FL)


def n_drop(q):
    v = np.array([r['re'][q] for r in rows], np.float32)
    return int((above & np.less(v, FL)).sum())


for q in QS:
    v = np.array([r['re'][q] for r in rows], np.float32)
    mark = ' <- 우리 구간' if QLO <= q <= QHI else ''
    print(f'      {q:>6}{np.median(v):>12.3f}{np.median(v / np.maximum(CH, 1e-9)):>11.3f}'
          f'{n_drop(q):>8}/{int(above.sum())}{mark}')
print('      ' + '-' * 41)
print(f'      (재압축 전 변화 중앙 {np.median(CH):.3f} · 바닥 중앙 {np.median(FL):.3f} · '
      f'바닥 위 {int(above.sum())}/{len(rows)}장)')
worst = max([n_drop(q) for q in QS if QLO <= q <= QHI] or [0])
print(f'      우리 구간 {QLO}~{QHI} 에서 **바닥 위에 있던 장이 밑으로 떨어진 수 {worst}장**')

json.dump({'floor': list(map(float, FV)), 'floor_ok': bool(ok_floor),
           'lap_median': float(q2), 'lap_iqr_ratio': float(iqr_ratio),
           'sigma_true': sig_true, 'sig_lo': float(sig_lo), 'sig_hi': float(sig_hi),
           'q_lo': QLO, 'q_hi': QHI, 'trust_table': bool(trust_tbl),
           'vacuous_sizes': int(n_vac), 'worst_drop': int(worst), 'rows': rows},
          open(f'{OUT}/finalize_check2.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {OUT}/finalize_check2.json')


r'''판정 — 미리 못 박은 대로. [F] 도 함께 읽음'''
print('\n' + '=' * 78)
print('판정')
print('=' * 78)
if not ok_floor:
    print('  **[C] 검산이 실패했으므로 아래를 2층과 나란히 읽지 말 것**')
print(f'  가) 실측 알갱이 {sig_true:.3f} 가 [{sig_lo:.3f}, {sig_hi:.3f}] 안'
      f'{"":>4}{"통과" if ok_a else "**어긋남**"}')
print(f'  나) 사분위 폭 {iqr_ratio:.1%} <= 50%{"":>18}{"통과" if ok_b else "**어긋남**"}')
print(f'  다) 눌림 {np.median(DR):.3f} vs 바닥 {np.median(FL):.3f}{"":>14}'
      f'{"통과" if ok_c else "**어긋남**"}')
print('  ' + '-' * 62)
add_noise = ok_a and ok_b and ok_c
if add_noise:
    print('  -> **잡음 더하기를 넣음**')
else:
    print('  -> **잡음 더하기를 안 넣음** (어긋난 기준을 그대로 적음)')
print(f'  -> 재압축 품질 구간 **{QLO}~{QHI}**')
if worst:
    print(f'  -> 재압축이 {worst}장을 바닥 밑으로 떨어뜨림. '
          f'절차 7단계 검사를 **재압축 뒤로** 옮기고 떨어진 장은 버림')
else:
    print('  -> 재압축이 바닥 위의 장을 떨어뜨리지 않음. 검사 자리를 안 바꿔도 됨')
if add_noise and n_vac:
    print(f'  -> 다만 [D] 에서 {n_vac}개 크기가 빈 껍데기가 되므로 검사는 '
          f'**잡음 앞**에 둘 수밖에 없음. 위 두 줄이 부딪히므로 그 한계를 적을 것')


r'''시트 — **우리 구간의 품질로** · 잰 자리 그대로 · 원본 배율'''
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


QSHEET = QLO
F = korean_font(20)
items = []
for r in sorted(rows, key=lambda r: r['chg'])[:4]:
    a8 = np.asarray(Image.open(f'{MAT}/{r["key"]}/{r["file"]}'))[..., 3].copy()
    a8[a8 < int(round(THR * 255))] = 0
    pw = max(int(round(a8.shape[1] * r['s'])), 4)
    ph = max(int(round(a8.shape[0] * r['s'])), 4)
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS), np.float32) / 255.0
    al[al < THR] = 0
    yy = int(max(H - ph, 0) // 2);  xx = int(max(W - pw, 0) // 2)      # 가운데 자리
    A = np.zeros((H, W), np.float32);  A[yy:yy + ph, xx:xx + pw] = al
    img = np.clip(BGR[0] * (1 - A[..., None]) + 255.0 * A[..., None], 0, 255)
    rec = jpeg_round(img.astype(np.uint8), QSHEET)
    bq = jbg(0, QSHEET)
    b = r['box'];  y0, x0 = yy + b[1], xx + b[0]
    y1, x1 = y0 + r['bh'], x0 + r['bw']
    cut = lambda a: np.clip(a[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    dif = np.clip(np.abs(rec - bq).mean(2)[y0:y1, x0:x1] * 12, 0, 255).astype(np.uint8)
    items.append((f'{r["key"]} {r["file"]}  배율 {r["s"]:.3f} · 변화 {r["chg"]:.2f} · '
                  f'바닥 {r["floor"]:.2f} · q{QSHEET} 뒤 {r["re"].get(QSHEET, float("nan")):.2f}',
                  [cut(BGR[0]), cut(img), cut(rec), np.dstack([dif] * 3)]))
if items:
    CW = max(i[1][0].shape[1] for i in items)
    Ht = sum(i[1][0].shape[0] + 28 for i in items) + 8
    sh = Image.new('RGB', (CW * 4, Ht), (16, 16, 16))
    dr = ImageDraw.Draw(sh);  y = 0
    for lab, arrs in items:
        dr.text((6, y + 4), lab + f'   (배경 · 합성 · q{QSHEET}재압축 · 차이x12) '
                                  f'— 넷 다 **라벨 상자 안**만 잘랐음',
                fill=(255, 220, 0), font=F)
        for j, a in enumerate(arrs):
            sh.paste(Image.fromarray(a), (CW * j, y + 28))
        y += arrs[0].shape[0] + 28
    sh.save(f'{OUT}/_finalize2.jpg', quality=95)
    print(f'\n-> {OUT}/_finalize2.jpg  ({len(items)}줄 · 원본 배율 · **우리 구간 q{QSHEET}**)')
    files.download(f'{OUT}/_finalize2.jpg')
    print('가장 옅은 넷임. 셋째 칸(왼쪽에서 세 번째)에서 연기가 사라졌는지 눈으로 볼 것.')
