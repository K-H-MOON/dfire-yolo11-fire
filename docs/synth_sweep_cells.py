# 합성 파이프라인 step4 — 판별력 있는 지표(스케일·열화 sweep). base=ptrain_b79(D-Fire).
# 상위: docs/HANDOFF_DFIRE_YOLO11.md · 기준: docs/PREREGISTER_DFIRE_QC.md (§다음 진짜 step4)
#
# ★설계 원칙(이 세션 합의):
#   - 지금은 "학습"이 아니라 "불꽃을 조정하고 base 반응을 측정"하는 단계.
#   - 그래서 불꽃은 실사(NIST)만 사용. 생성형 안 섞음 = 인과 깨끗하게(작은/열화 불꽃을 못 잡는
#     원인이 '크기·화질' 때문인지 '가짜 불꽃' 때문인지 섞이지 않도록). 생성형 다양성은 학습 단계에.
#   - 통제 변수 = 불꽃 크기(CELL24)·화질(CELL25). 불꽃 정체·배경·위치는 다양화하여
#     각 지점 recall을 여러 합성본의 평균 + 불꽃별 분산으로 보고(= break-point이 불꽃 특정적인지 노출).
#   - base=frozen. recall 프록시(리얼리즘·envelope). 위치는 frozen-base recall에 무관(translation-equivariant).
#
# ★Colab 콜드 재연결 대비: 각 셀 자립(마운트·설치 가드·경로 탐색). SEED 고정 재현.
# 데이터: 실배경=Drive realneg_frames/synth · 불꽃소스=Drive firecrop_src/nist_stovetop_cornoil · base=Drive dfire_runs/...


# ========== CELL 20c: NIST 유류불 실사 뱅크 확장 (4시험 × 불꽃후보 이벤트 → 품질필터) ==========
# 검증된 NIST FCD Stovetop Cooking Pan Fire(퍼블릭도메인) 4개 '유류(corn oil)' 시험.
#   이벤트 번호↔라벨은 시험마다 다름(예: calphalon 점화=Evt3, massloss13 점화=Evt2) →
#   불꽃 후보 이벤트(Evt2~5,EvtP)를 전부 받고, extract_flame 면적필터가 비불꽃(기름붓기/가열)을 자동 컬링.
#   Gasoline_50g 은 도메인(식용유) 아님 → 제외. Kitchen Room Fire 는 광각 룸화재라 마스킹 난망 → 제외.
import os, urllib.request, urllib.parse, glob
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'; os.makedirs(OUT, exist_ok=True)
BASE = 'https://nist-el-nfrlhrr.s3.amazonaws.com/HRR/ASSET_FILES/'

# 검증된 시험 id (2026-08 웹확인 · 전부 corn oil 유류불)
TESTS = {
    'cornoil_calphalon':  'Corn Oil/video/1574198232',       # 50g 20cm Calphalon (peak 34.0 kW)
    'cornoil_alumipan2':  'Corn Oil/video/1574199884',       # 50g 20cm AlumiPan Repeat2 (28.2 kW)
    'cornoil_massloss13': 'Hamins Kitchen/video/1508954077', # MassLoss_13 (peak 73.7 kW)
    'cornoil_massloss14c':'Hamins Kitchen/video/1508958465', # MassLoss_14c (18.5 kW)
}
EVENTS = ['Evt2', 'Evt3', 'Evt4', 'Evt5', 'EvtP']   # 불꽃 후보(발화~최성기~소화). Evt1(기름붓기)=비불꽃 제외.

n_ok = 0
for tag, rel in TESTS.items():
    for evt in EVENTS:
        url = urllib.parse.quote(f'{BASE}{rel}-{evt}.jpg', safe=':/')
        dst = f'{OUT}/{tag}_{evt}_FIRE__{os.path.basename(rel)}-{evt}.jpg'
        if os.path.exists(dst):
            n_ok += 1; continue
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60).read()
            if data[:2] == b'\xff\xd8':
                open(dst, 'wb').write(data); n_ok += 1; print('OK  ', tag, evt, f'{len(data)//1024}KB')
            else:
                print('NOT-JPEG', tag, evt)
        except Exception as e:
            print('skip', tag, evt, str(e)[:40])   # 없는 이벤트(404)는 정상 스킵
print(f'\n다운로드/기존 {n_ok}개. 다음 CELL(24/25)의 load_flames 가 md5중복·면적<0.30·<60px 로 컬링해 실사 뱅크 확정.')
print('→ 뱅크 규모·크기범위는 CELL 24 첫 출력(flame bank)에서 확인.')


# ========== CELL 24: 스케일 sweep — base가 불꽃을 놓치기 시작하는 크기(envelope) ==========
# 불꽃/배경/위치는 다양화, 크기만 통제. 각 스케일 recall = 여러 합성본 평균 + 불꽃별 분산.
import os, glob, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
assert os.path.exists(W), 'best.pt 없음: ' + W
assert glob.glob(f'{SRC}/*FIRE*.jpg'), '불꽃 소스 없음 — CELL 20c 먼저'
assert glob.glob(f'{BG}/**/*.jpg', recursive=True), 'synth 배경 없음 — CELL 17 먼저'

SEED = 0
MIN_COVER, MIN_PX = 0.30, 60           # 소스 품질필터(v2와 동일)
SCALES = [0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.50]   # 불꽃높이/이미지높이 (초소~대)
N_BG_PER = 6                            # (불꽃,스케일)당 배경 표본 수 → 위치/배경 다양화
CONFS = (0.05, 0.25, 0.50)
rng = np.random.default_rng(SEED)

def extract_flame(path):               # 어두운 배경 → 주황∪백열 + 최대연결성분 + 소프트알파 (v2와 동일)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None, 0.0
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    rgba = np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8)
    return Image.fromarray(rgba), float(mm.mean())

def load_flames():                     # 클린 실사 불꽃만(중복·저면적·너무작음 제외)
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl, cov = extract_flame(p)
        if fl is None or cov < MIN_COVER or max(fl.size) < MIN_PX: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):    # 소프트알파 합성 + GT bbox
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def detect(m, pil, gt):                # base 추론 → (top_conf, box정오 dict per conf)
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    has = (r.boxes is not None and len(r.boxes))
    tc = float(r.boxes.conf.max().cpu()) if has else 0.0
    xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
    cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
    boxok = {}
    for c in CONFS:
        ok = any((cf[i] >= c and iou(xy[i], gt) >= 0.5) for i in range(len(cf))) if gt else False
        boxok[c] = ok
    return tc, boxok

flames = load_flames()
print(f'=== flame bank (실사) {len(flames)}종 ===')
for nm, fl in flames: print(f'  {nm:24} src {fl.size}')
assert flames, '클린 불꽃 0 — 소스/필터 확인'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True))
m = YOLO(W)

# sweep
rows = []          # (scale, flame, img_rec@c dict, box_rec@c dict, mean_px)
sample_preview = []
for s in SCALES:
    for nm, fl in flames:
        img_hit = {c: 0 for c in CONFS}; box_hit = {c: 0 for c in CONFS}; pxs = []
        for k in range(N_BG_PER):
            bp = bgs[int(rng.integers(len(bgs)))]; bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
            th = max(1, int(Hd*s)); tw = max(1, int(fl.width*th/fl.height))
            fl_r = fl.resize((tw, th))
            px = int(Wd*rng.uniform(0.15, 0.85) - tw/2); py = int(Hd*rng.uniform(0.35, 0.75) - th/2)
            comp, box = paste(bg, fl_r, px, py)
            if box is None: continue
            pxs.append(max(box[2]-box[0], box[3]-box[1]))
            tc, boxok = detect(m, comp, box)
            for c in CONFS:
                if tc >= c: img_hit[c] += 1
                if boxok[c]: box_hit[c] += 1
            if k == 0 and s in (0.06, 0.13, 0.35):
                sample_preview.append((comp, box, f'{nm[:10]} s{s} tc{tc:.2f}'))
        n = max(1, len(pxs))
        rows.append({'scale': s, 'flame': nm,
                     'img': {c: img_hit[c]/n for c in CONFS},
                     'box': {c: box_hit[c]/n for c in CONFS},
                     'px': float(np.mean(pxs)) if pxs else 0.0, 'n': n})

# 집계: 스케일별 = 불꽃 평균 + 분산
print(f'\n=== 스케일 sweep · base recall (불꽃 {len(flames)}종 × 배경 {N_BG_PER}) ===')
print(f'{"scale":>6}{"px~":>6}' + ''.join(f'{f"img@{c}":>9}' for c in CONFS)
      + ''.join(f'{f"box@{c}":>9}' for c in CONFS) + f'{"img@.25±":>10}')
scale_summary = []
for s in SCALES:
    sr = [r for r in rows if r['scale'] == s]
    px = np.mean([r['px'] for r in sr])
    img_c = {c: np.mean([r['img'][c] for r in sr]) for c in CONFS}
    box_c = {c: np.mean([r['box'][c] for r in sr]) for c in CONFS}
    spread = np.std([r['img'][0.25] for r in sr])   # 불꽃별 recall 표준편차 = break-point 불꽃의존성
    scale_summary.append((s, px, img_c, box_c, spread))
    print(f'{s:>6.2f}{px:>6.0f}' + ''.join(f'{img_c[c]:>9.3f}' for c in CONFS)
          + ''.join(f'{box_c[c]:>9.3f}' for c in CONFS) + f'{spread:>10.3f}')
print('\n※ 읽기: img@c = 이미지경보 recall(배포 관점) · box@c = 위치까지 맞은 recall · ±=불꽃별 std.')
print('  recall이 급락하는 px/scale = base envelope. std 크면 break-point이 불꽃 특정적(다양성 필요 신호).')

# 곡선 PNG
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
xs = [px for _, px, *_ in scale_summary]
for c in CONFS:
    ax[0].plot(xs, [ic[c] for _, _, ic, _, _ in scale_summary], 'o-', label=f'img@{c}')
ax[0].set_xlabel('flame box size (px, longer side)'); ax[0].set_ylabel('image recall')
ax[0].set_title('Scale sweep — image-alarm recall vs flame px'); ax[0].legend(); ax[0].grid(alpha=.3)
for r in rows:  # 불꽃별 산점(분산 시각화)
    ax[1].scatter(r['px'], r['img'][0.25], s=14, alpha=.4, color='tab:red')
ax[1].plot(xs, [ic[0.25] for _, _, ic, _, _ in scale_summary], 'k-o', label='mean img@0.25')
ax[1].set_xlabel('flame box size (px)'); ax[1].set_ylabel('recall@0.25 (per-flame dots)')
ax[1].set_title('Per-flame spread (break-point flame-specific?)'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/scale_sweep.png', dpi=95, bbox_inches='tight'); plt.show()

# 합성본 미리보기(육안 sanity)
if sample_preview:
    import matplotlib.patches as patches
    K = min(9, len(sample_preview)); cols = 3; rr = (K+cols-1)//cols
    fig, ax = plt.subplots(rr, cols, figsize=(4*cols, 3*rr)); ax = np.array(ax).reshape(rr, cols)
    for i in range(rr*cols):
        a = ax[i//cols, i%cols]; a.axis('off')
        if i >= K: continue
        comp, box, cap = sample_preview[i]; a.imshow(comp)
        a.add_patch(patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='lime', lw=2))
        a.set_title(cap, fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/scale_preview.png', dpi=90, bbox_inches='tight'); plt.show()
print(f'\n저장 → {OUT}/scale_sweep.png · scale_preview.png')


# ========== CELL 25: 열화 sweep — CCTV 화질열화에 base recall이 언제 깨지나 ==========
# 스케일은 '충분히 잡히는' 고정값(FIX_SCALE)에 두고, 화질만 통제(JPEG·블러·저해상·복합).
# aug/noise 유의미성(PREREGISTER §다음 step4) = 어떤 열화가 검출을 죽이나 = 학습 aug 우선순위 근거.
import os, glob, io, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)

SEED = 0
MIN_COVER, MIN_PX = 0.30, 60
FIX_SCALE = 0.20        # 무열화서 충분히 잡히는 크기(CELL24 곡선서 recall 높은 지점)
N_COMP = 40             # 열화 지점당 합성본 수(불꽃×배경×위치 랜덤) → 평균 recall
CONFS = (0.05, 0.25, 0.50)
rng = np.random.default_rng(SEED)

# 열화 정의 (각 유형 독립 severity + 복합 CCTV 1점)
DEGRADE = {
    'jpeg_q':    [95, 70, 50, 30, 15, 8],        # JPEG 품질(낮을수록 열화)
    'blur_sig':  [0.0, 1.0, 2.0, 3.0, 5.0],      # 가우시안 블러 σ(px)
    'downscale': [1.0, 0.5, 0.33, 0.25, 0.15],   # 저해상 CCTV: 축소→확대 비율(작을수록 열화)
}

def extract_flame(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None, 0.0
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8)), float(mm.mean())

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl, cov = extract_flame(p)
        if fl is None or cov < MIN_COVER or max(fl.size) < MIN_PX: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def apply_degrade(pil, kind, val):
    if kind == 'jpeg_q':
        b = io.BytesIO(); pil.save(b, 'JPEG', quality=int(val)); b.seek(0); return Image.open(b).convert('RGB')
    if kind == 'blur_sig':
        if val <= 0: return pil
        arr = np.asarray(pil).astype(np.float32)
        for ch in range(3): arr[..., ch] = gaussian_filter(arr[..., ch], sigma=val)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == 'downscale':
        if val >= 1.0: return pil
        W_, H_ = pil.size; sw, sh = max(1, int(W_*val)), max(1, int(H_*val))
        return pil.resize((sw, sh), Image.BILINEAR).resize((W_, H_), Image.BILINEAR)
    return pil

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def detect(m, pil, gt):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    has = (r.boxes is not None and len(r.boxes)); tc = float(r.boxes.conf.max().cpu()) if has else 0.0
    xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
    cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
    boxok = {c: (any((cf[i] >= c and iou(xy[i], gt) >= 0.5) for i in range(len(cf))) if gt else False) for c in CONFS}
    return tc, boxok

flames = load_flames(); assert flames, '클린 불꽃 0 — CELL 20c 먼저'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); m = YOLO(W)
print(f'flame bank {len(flames)}종 · FIX_SCALE {FIX_SCALE} · 지점당 합성 {N_COMP}')

def make_batch(n):    # 고정 스케일 합성본 n개(불꽃/배경/위치 랜덤·재현 SEED) + GT
    out = []
    r2 = np.random.default_rng(SEED)
    for _ in range(n):
        nm, fl = flames[int(r2.integers(len(flames)))]
        bp = bgs[int(r2.integers(len(bgs)))]; bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
        th = max(1, int(Hd*FIX_SCALE)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
        px = int(Wd*r2.uniform(0.15, 0.85) - tw/2); py = int(Hd*r2.uniform(0.35, 0.75) - th/2)
        comp, box = paste(bg, fl_r, px, py)
        if box: out.append((comp, box))
    return out

batch = make_batch(N_COMP)
print(f'합성본 {len(batch)}개 준비.')

# baseline(무열화)
base_img = {c: np.mean([detect(m, comp, box)[0] >= c for comp, box in batch]) for c in CONFS}
print(f'\n[무열화 baseline] ' + ' · '.join(f'img@{c} {base_img[c]:.3f}' for c in CONFS))

results = {}
for kind, vals in DEGRADE.items():
    print(f'\n=== 열화: {kind} ===')
    print(f'{"val":>8}' + ''.join(f'{f"img@{c}":>9}' for c in CONFS) + ''.join(f'{f"box@{c}":>9}' for c in CONFS))
    results[kind] = []
    for v in vals:
        img_hit = {c: 0 for c in CONFS}; box_hit = {c: 0 for c in CONFS}
        for comp, box in batch:
            d = apply_degrade(comp, kind, v); tc, boxok = detect(m, d, box)
            for c in CONFS:
                if tc >= c: img_hit[c] += 1
                if boxok[c]: box_hit[c] += 1
        n = len(batch)
        ic = {c: img_hit[c]/n for c in CONFS}; bc = {c: box_hit[c]/n for c in CONFS}
        results[kind].append((v, ic, bc))
        print(f'{v:>8}' + ''.join(f'{ic[c]:>9.3f}' for c in CONFS) + ''.join(f'{bc[c]:>9.3f}' for c in CONFS))

# 복합(현실 CCTV): downscale 0.33 + jpeg 30 + blur 1.0
comp_hit = {c: 0 for c in CONFS}
for comp, box in batch:
    d = apply_degrade(apply_degrade(apply_degrade(comp, 'downscale', 0.33), 'jpeg_q', 30), 'blur_sig', 1.0)
    tc, _ = detect(m, d, box)
    for c in CONFS:
        if tc >= c: comp_hit[c] += 1
print('\n[복합 CCTV = downscale0.33+jpeg30+blur1.0] '
      + ' · '.join(f'img@{c} {comp_hit[c]/len(batch):.3f}' for c in CONFS))

# 곡선
fig, ax = plt.subplots(1, len(DEGRADE), figsize=(6*len(DEGRADE), 4.5))
for j, (kind, res) in enumerate(results.items()):
    xs = [v for v, _, _ in res]
    for c in CONFS: ax[j].plot(xs, [ic[c] for _, ic, _ in res], 'o-', label=f'img@{c}')
    ax[j].set_title(f'Degrade: {kind}'); ax[j].set_xlabel(kind); ax[j].set_ylabel('image recall')
    ax[j].legend(); ax[j].grid(alpha=.3)
    if kind in ('jpeg_q', 'downscale'): ax[j].invert_xaxis()   # 왼→오 = 열화 심해지는 방향
plt.tight_layout(); plt.savefig(f'{OUT}/degrade_sweep.png', dpi=95, bbox_inches='tight'); plt.show()
print(f'\n저장 → {OUT}/degrade_sweep.png')
print('※ 읽기: recall 급락 지점 = base가 그 열화에 취약 = 학습 시 그 aug를 넣어야 할 근거(유의미).')
print('  ⚠️ frozen-base 프록시 = "학습된 base가 열화 불꽃을 인식하나"이지 "그 aug로 학습하면 좋아지나"의 직접 증거 아님.')


# ========== CELL 26: 진단·검정력 — plateau 원인(놓친 합성본 육안) + 크기×화질 상호작용 ==========
# 자기감사 반영: (1) N=40 저검정력 → N_COMP=160. (2) plateau<1.0(~15% 놓침) 원인=base한계 vs 합성아티팩트
#   → 놓친 합성본 몽타주로 육안 진단. (3) 무릎(0.11)·plateau(0.25) 두 스케일서 열화 → 크기×화질 상호작용.
# 앞 CELL(20c)로 불꽃 뱅크 확보 전제. 자립(마운트·설치가드).
import os, glob, io, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as patches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)

SEED = 0
MIN_COVER, MIN_PX = 0.30, 60
N_COMP = 160                    # 검정력↑ (앞 40 → 160)
FIX_SCALES = [0.11, 0.25]       # 무릎(marginal) · plateau
CONFS = (0.05, 0.25, 0.50)
DEGRADE = {'jpeg_q': [95, 50, 30, 15, 8], 'blur_sig': [0.0, 2.0, 3.0, 5.0], 'downscale': [1.0, 0.33, 0.25, 0.15]}

def extract_flame(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None, 0.0
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8)), float(mm.mean())

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl, cov = extract_flame(p)
        if fl is None or cov < MIN_COVER or max(fl.size) < MIN_PX: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def apply_degrade(pil, kind, val):
    if kind == 'jpeg_q':
        b = io.BytesIO(); pil.save(b, 'JPEG', quality=int(val)); b.seek(0); return Image.open(b).convert('RGB')
    if kind == 'blur_sig':
        if val <= 0: return pil
        arr = np.asarray(pil).astype(np.float32)
        for ch in range(3): arr[..., ch] = gaussian_filter(arr[..., ch], sigma=val)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == 'downscale':
        if val >= 1.0: return pil
        W_, H_ = pil.size; sw, sh = max(1, int(W_*val)), max(1, int(H_*val))
        return pil.resize((sw, sh), Image.BILINEAR).resize((W_, H_), Image.BILINEAR)
    return pil

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def detect(m, pil, gt):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    has = (r.boxes is not None and len(r.boxes)); tc = float(r.boxes.conf.max().cpu()) if has else 0.0
    xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
    cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
    boxok = {c: (any((cf[i] >= c and iou(xy[i], gt) >= 0.5) for i in range(len(cf))) if gt else False) for c in CONFS}
    return tc, boxok

flames = load_flames(); assert flames, '클린 불꽃 0 — CELL 20c 먼저'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); m = YOLO(W)
print(f'flame bank {len(flames)}종 · N_COMP {N_COMP} · scales {FIX_SCALES}')

def make_batch(n, scale):
    out = []; r2 = np.random.default_rng(SEED)     # 재현·스케일 무관 동일 배치 구성
    for _ in range(n):
        nm, fl = flames[int(r2.integers(len(flames)))]
        bp = bgs[int(r2.integers(len(bgs)))]; bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
        th = max(1, int(Hd*scale)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
        px = int(Wd*r2.uniform(0.15, 0.85) - tw/2); py = int(Hd*r2.uniform(0.35, 0.75) - th/2)
        comp, box = paste(bg, fl_r, px, py)
        if box: out.append((comp, box, nm))
    return out

for FS in FIX_SCALES:
    batch = make_batch(N_COMP, FS)
    print(f'\n############### FIX_SCALE {FS} · 합성 {len(batch)} ###############')
    base = []; missed = []
    for comp, box, nm in batch:
        tc, _ = detect(m, comp, box); base.append(tc)
        if tc < 0.25: missed.append((comp, box, nm, tc))
    base = np.array(base); n = len(base)
    for c in CONFS:
        p = (base >= c).mean(); se = (p*(1-p)/n)**0.5
        print(f'  [무열화] img@{c}: {p:.3f}  (95% CI ±{1.96*se:.3f}, n={n})')
    print(f'  놓침(top<0.25): {len(missed)}/{n} = {len(missed)/n:.1%} → 몽타주로 원인 육안')
    if missed:
        K = min(24, len(missed)); cols = 4; rr = (K+cols-1)//cols
        fig, ax = plt.subplots(rr, cols, figsize=(4*cols, 3*rr)); ax = np.array(ax).reshape(rr, cols)
        for i in range(rr*cols):
            a = ax[i//cols, i%cols]; a.axis('off')
            if i >= K: continue
            comp, box, nm, tc = missed[i]; a.imshow(comp)
            a.add_patch(patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor='lime', lw=2))
            a.set_title(f'{nm[:10]} tc{tc:.2f}', fontsize=8)
        op = f'{OUT}/missed_scale{FS}.png'
        plt.suptitle(f'놓친 합성본 (FIX_SCALE {FS}) — 왜 안 잡혔나(저대비/창문/작음/마스킹?)', fontsize=12)
        plt.tight_layout(); plt.savefig(op, dpi=90, bbox_inches='tight'); plt.show()
        print(f'  → {op}')
    for kind, vals in DEGRADE.items():
        cells = []
        for v in vals:
            rec = np.mean([detect(m, apply_degrade(comp, kind, v), box)[0] >= 0.25 for comp, box, nm in batch])
            cells.append(f'{v}:{rec:.3f}')
        print(f'  [{kind}] ' + ' | '.join(cells) + '   (img@0.25)')
print('\n※ missed_scale*.png 에서 놓친 불꽃이 (a)창문/밝은벽 위 저대비·(b)마스킹 불량·(c)너무 작음 중')
print('  무엇 때문인지 육안 판정 → base 한계 vs 합성 아티팩트 구분. 후자면 배치/대비 규칙이 step4 레버.')


# ========== CELL 27 (★권위·최종): 클린 뱅크(진짜 불꽃 6종) 재측정 ==========
# ★CELL 24/25/26 은 뱅크 미큐레이션이라 비불꽃(가열코일·금속자=Evt2/4/5)이 섞여 recall 오염됨(0.994→0.369).
#   진짜 결과는 이 CELL 27 (KEEP 화이트리스트로 NIST id+event 진짜 불꽃 6종만). 추출=largest-CC(원복).
# 결과(2026-08-29·N=160): scale0.25 img@0.25=0.994(≈천장)/블라인드 0.6% · scale0.11=0.694 · 열화는 충분크기서 견고·작은불꽃×JPEG만 유해.
import os, glob, io, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 0; N_COMP = 160; SCALES = [0.11, 0.25]; CONFS = (0.05, 0.25, 0.50)
DEGRADE = {'jpeg_q': [95, 50, 30, 15, 8], 'blur_sig': [0.0, 2.0, 3.0, 5.0], 'downscale': [1.0, 0.33, 0.25, 0.15]}
# ★진짜 불꽃만(뱅크 이미지 육안 확정): calphalon·alumipan2 점화(Evt3)+최성기(EvtP), massloss13·14c 최성기(EvtP).
KEEP = ['1574198232-Evt3', '1574198232-EvtP', '1574199884-Evt3', '1574199884-EvtP',
        '1508954077-EvtP', '1508958465-EvtP']

def extract_flame(path):                        # largest-CC (원복)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None, 0.0
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8)), float(mm.mean())

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        if not any(k in os.path.basename(p) for k in KEEP): continue   # ★진짜 불꽃만
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl, cov = extract_flame(p)
        if fl is None or max(fl.size) < 60: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def apply_degrade(pil, kind, val):
    if kind == 'jpeg_q':
        b = io.BytesIO(); pil.save(b, 'JPEG', quality=int(val)); b.seek(0); return Image.open(b).convert('RGB')
    if kind == 'blur_sig':
        if val <= 0: return pil
        arr = np.asarray(pil).astype(np.float32)
        for ch in range(3): arr[..., ch] = gaussian_filter(arr[..., ch], sigma=val)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == 'downscale':
        if val >= 1.0: return pil
        W_, H_ = pil.size; sw, sh = max(1, int(W_*val)), max(1, int(H_*val))
        return pil.resize((sw, sh), Image.BILINEAR).resize((W_, H_), Image.BILINEAR)
    return pil

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def detect(m, pil, gt):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    has = (r.boxes is not None and len(r.boxes)); tc = float(r.boxes.conf.max().cpu()) if has else 0.0
    xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4)); cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
    boxok = {c: (any((cf[i] >= c and iou(xy[i], gt) >= 0.5) for i in range(len(cf))) if gt else False) for c in CONFS}
    return tc, boxok

flames = load_flames(); assert flames, '클린 불꽃 0 — KEEP/경로 확인'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); m = YOLO(W)
print(f'=== 클린 뱅크 {len(flames)}종 (금속·자 제외) ===')
for nm, fl in flames: print(f'  {nm:26} {fl.size}')

def make_batch(n, scale):
    out = []; r2 = np.random.default_rng(SEED)
    for _ in range(n):
        nm, fl = flames[int(r2.integers(len(flames)))]
        bp = bgs[int(r2.integers(len(bgs)))]; bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
        th = max(1, int(Hd*scale)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
        px = int(Wd*r2.uniform(0.15, 0.85) - tw/2); py = int(Hd*r2.uniform(0.35, 0.75) - th/2)
        comp, box = paste(bg, fl_r, px, py)
        if box: out.append((comp, box, nm))
    return out

for FS in SCALES:
    batch = make_batch(N_COMP, FS)
    print(f'\n############### FIX_SCALE {FS} · 합성 {len(batch)} ###############')
    base = np.array([detect(m, comp, box)[0] for comp, box, nm in batch]); n = len(base)
    for c in CONFS:
        p = (base >= c).mean(); se = (p*(1-p)/n)**0.5
        print(f'  [무열화] img@{c}: {p:.3f}  (95% CI ±{1.96*se:.3f}, n={n})')
    print(f'  놓침(top<0.25): {int((base < 0.25).sum())}/{n} = {(base < 0.25).mean():.1%}')
    for kind, vals in DEGRADE.items():
        cells = [f'{v}:{np.mean([detect(m, apply_degrade(comp, kind, v), box)[0] >= 0.25 for comp, box, nm in batch]):.3f}' for v in vals]
        print(f'  [{kind}] ' + ' | '.join(cells) + '   (img@0.25)')


# ========== CELL 28: 합성유발 FP — 같은 synth 배경에 불꽃 있음/없음 → base FP 비교 (한 변수=불꽃 유무) ==========
# 질문: 합성(불꽃 붙이기) 행위가 헛불(FP)을 *유발*하나, 아니면 FP는 배경 자체(주황물체 색혼동)에서 오나?
# 설계(paired·한 변수): 같은 배경 N장을 (1)맨 배경 (2)깨끗한 불꽃 붙임(scale 0.25) 둘로 → base 검출 비교.
#   ★ΔFP = (불꽃있을때 '불꽃 아닌 곳' FP) − (맨 배경 FP). ≈0 → 합성 무해 · >0 → 합성이 FP 유발.
# 클린 뱅크(KEEP 6종)·largest-CC. 프록시(frozen-base). 진행 전제=CELL 27 뱅크.
import os, glob, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as patches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 0; FS = 0.25; N_BG = 300; CONFS = (0.05, 0.25, 0.50)
KEEP = ['1574198232-Evt3', '1574198232-EvtP', '1574199884-Evt3', '1574199884-EvtP',
        '1508954077-EvtP', '1508958465-EvtP']

def extract_flame(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None, 0.0
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8)), float(mm.mean())

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        if not any(k in os.path.basename(p) for k in KEEP): continue
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl, cov = extract_flame(p)
        if fl is None or max(fl.size) < 60: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def boxes(m, pil):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return np.zeros((0, 4)), np.zeros(0)
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

flames = load_flames(); assert flames, '클린 불꽃 0'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); m = YOLO(W)
rng = np.random.default_rng(SEED)
sel = [bgs[i] for i in rng.choice(len(bgs), size=min(N_BG, len(bgs)), replace=False)]
print(f'뱅크 {len(flames)}종 · 배경 {len(sel)}장 (paired: 맨배경 vs 불꽃scale{FS})')

rows = []
for bp in sel:
    bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
    xyb, cfb = boxes(m, bg)                                   # (1) 맨 배경
    nm, fl = flames[int(rng.integers(len(flames)))]
    th = max(1, int(Hd*FS)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
    px = int(Wd*rng.uniform(0.15, 0.85) - tw/2); py = int(Hd*rng.uniform(0.35, 0.75) - th/2)
    comp, gt = paste(bg, fl_r, px, py)                       # (2) 불꽃 붙임
    xyc, cfc = boxes(m, comp)
    rows.append((bp, xyb, cfb, comp, gt, xyc, cfc, nm))

print(f'\n=== 합성유발 FP (paired · n={len(rows)}) ===')
print(f'{"conf":>6}{"맨배경FP":>10}{"불꽃recall":>11}{"불꽃외 FP(합성후)":>17}{"ΔFP(합성유발)":>14}')
for c in CONFS:
    bare_fp = np.array([(cfb >= c).any() for (_, xyb, cfb, _, _, _, _, _) in rows])
    recall, spur = [], []
    for (_, _, _, comp, gt, xyc, cfc, _) in rows:
        on = any(cfc[i] >= c and iou(xyc[i], gt) >= 0.3 for i in range(len(cfc))) if gt else False
        sp = any(cfc[i] >= c and iou(xyc[i], gt) < 0.3 for i in range(len(cfc)))
        recall.append(on); spur.append(sp)
    recall = np.array(recall); spur = np.array(spur)
    dfp = spur.mean() - bare_fp.mean()
    n = len(rows); se = ((spur.mean()*(1-spur.mean()) + bare_fp.mean()*(1-bare_fp.mean()))/n)**0.5
    print(f'{c:>6.2f}{bare_fp.mean():>10.3f}{recall.mean():>11.3f}{spur.mean():>17.3f}{dfp:>+11.3f}±{1.96*se:.3f}')
print('※ ΔFP≈0 → 합성(붙여넣기)이 FP 안 더함·FP는 배경 주황물체서 옴. ΔFP>0 → 합성이 헛불 유발(합성방식 고쳐야).')

# 몽타주: (A) 맨배경 헛불(conf0.25) · (B) 불꽃 붙인 뒤 '불꽃 외' 헛불(conf0.25) → 트리거 육안(주황물체 vs 붙여넣기 아티팩트)
C0 = 0.25
bareFP = [(bp, xyb, cfb) for (bp, xyb, cfb, _, _, _, _, _) in rows if (cfb >= C0).any()]
compFP = [(comp, gt, xyc, cfc, nm) for (_, _, _, comp, gt, xyc, cfc, nm) in rows
          if any(cfc[i] >= C0 and iou(xyc[i], gt) < 0.3 for i in range(len(cfc)))]
print(f'\nconf{C0}: 맨배경 헛불 {len(bareFP)}장 · 불꽃외 헛불(합성후) {len(compFP)}장 → 몽타주로 트리거 확인')

def montage(items, drawer, title, path, K=16):
    if not items: print('  (0장)'); return
    K = min(K, len(items)); cols = 4; rr = (K+cols-1)//cols
    fig, ax = plt.subplots(rr, cols, figsize=(4*cols, 3*rr)); ax = np.array(ax).reshape(rr, cols)
    for i in range(rr*cols):
        a = ax[i//cols, i%cols]; a.axis('off')
        if i >= K: continue
        drawer(a, items[i])
    fig.suptitle(title, fontsize=12); plt.tight_layout(); plt.savefig(path, dpi=90, bbox_inches='tight'); plt.show(); print('  →', path)

def draw_bare(a, it):
    bp, xy, cf = it; a.imshow(Image.open(bp).convert('RGB'))
    for i in range(len(cf)):
        if cf[i] >= C0: a.add_patch(patches.Rectangle((xy[i][0], xy[i][1]), xy[i][2]-xy[i][0], xy[i][3]-xy[i][1], fill=False, edgecolor='red', lw=2))
    a.set_title(f'top{cf.max():.2f}', fontsize=8)

def draw_comp(a, it):
    comp, gt, xy, cf, nm = it; a.imshow(comp)
    if gt: a.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1], fill=False, edgecolor='lime', lw=1.5))  # 진짜 불꽃(초록)
    for i in range(len(cf)):
        if cf[i] >= C0 and iou(xy[i], gt) < 0.3: a.add_patch(patches.Rectangle((xy[i][0], xy[i][1]), xy[i][2]-xy[i][0], xy[i][3]-xy[i][1], fill=False, edgecolor='red', lw=2))  # 불꽃외 헛불(빨강)
    a.set_title(f'{nm[:10]}', fontsize=8)

montage(bareFP, draw_bare, f'맨배경 헛불 (conf{C0}) — 무엇에 오탐? (주황물체 색혼동?)', f'{OUT}/fp_bare.png')
montage(compFP, draw_comp, f'불꽃 붙인 뒤 불꽃외 헛불 (conf{C0}·초록=진짜불꽃·빨강=헛불) — 배경물체? 붙여넣기?', f'{OUT}/fp_synth_induced.png')


# ========== CELL 29: AI-Hub ENB 불꽃 vs NIST 유류불 — base recall 실측 (paired) ==========
# objective (A) 직답: D-Fire base가 'AI-Hub 실내 불꽃' 합성을 얼마나 인식하나? NIST 유류불(CELL27=0.994)과 비교.
#   ★비교 = 불꽃 소스(NIST corn-oil KEEP6 [색마스크추출]  vs  AI-Hub ENB 수동박스16 [페더알파]).
#   ⚠️추출법은 소스별 최적: NIST=색마스크 자동 · AI-Hub=사람이 박스+가장자리 페더(색마스크가 AI-Hub엔 실패해 v1/v2/QC 탈락 → 수동).
#      NIST는 참조 천장(CELL27 0.994 재현 확인용). 배경·스케일·위치·N·SEED·합성(paste)·평가는 두 뱅크 동일.
#   ★paired: 배경/위치 '계획'을 SEED로 1회 생성해 두 뱅크 공유 → 소스 외 조건 동일(다른세션 비교 아님).
#   예측 안 함 — 숫자로 답. 준비: 로컬 manual_flame_box.py→manual_crops → Drive firecrop_src/aihub_enb_manual
#      (또는 aihub_enb_manual_crops.zip 업로드시 자동 언집).
#   출처: AI-Hub 71751(불꽃추출·비배포·출처표기) · NIST FCD(퍼블릭도메인).
import os, glob, subprocess, sys, hashlib, zipfile, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image

def robust_open_rgb(p):   # ★Drive FUSE 한글파일명(NFC/NFD 불일치) 대응 — 여러 형태 시도, 실패시 None
    for q in (p, unicodedata.normalize('NFC', p), unicodedata.normalize('NFD', p)):
        try: return Image.open(q).convert('RGB')
        except Exception: pass
    return None

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W    = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
BG   = '/content/drive/MyDrive/realneg_frames/synth'
FSRC = '/content/drive/MyDrive/firecrop_src'
NIST = f'{FSRC}/nist_stovetop_cornoil'
AIH  = f'{FSRC}/aihub_enb_manual'                # ★사람이 박스 친 수동 크롭(색마스크 실패 대체)
OUT  = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 0; N_COMP = 160; SCALES = [0.11, 0.25]; CONFS = (0.05, 0.25, 0.50)
KEEP_NIST = ['1574198232-Evt3', '1574198232-EvtP', '1574199884-Evt3', '1574199884-EvtP',
             '1508954077-EvtP', '1508958465-EvtP']

# AI-Hub 수동 크롭: 폴더 없으면 zip 자동 언집
if not os.path.isdir(AIH) or not glob.glob(f'{AIH}/*.jpg'):
    zp = f'{FSRC}/aihub_enb_manual_crops.zip'
    if os.path.exists(zp):
        os.makedirs(AIH, exist_ok=True)
        with zipfile.ZipFile(zp) as zf: zf.extractall(AIH)
        print(f'[unzip] {zp} → {AIH} ({len(glob.glob(f"{AIH}/*.jpg"))} jpg)')

def extract_flame(path):                        # ★NIST용: 색마스크 largest-CC (CELL27과 동일)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    mm = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(mm); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; m2 = mm[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*m2*255]).astype(np.uint8))

def alpha_feather(path, fr=0.08):               # ★AI-Hub 수동크롭용: 카빙0·가장자리만 소프트(불꽃 보존·QC 채택 B안)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    h, w = im.shape[:2]; fpx = max(2, int(min(h, w)*fr))
    inner = np.zeros((h, w), np.float32); inner[fpx:h-fpx, fpx:w-fpx] = 1.0
    a = np.clip(ndimage.gaussian_filter(inner, fpx*0.6), 0, 1)
    return Image.fromarray(np.dstack([im, a*255]).astype(np.uint8))

def load_bank(src, glob_pat, keep, extractor):
    out, seen = [], set()
    for p in sorted(glob.glob(f'{src}/{glob_pat}')):
        bn = os.path.basename(p)
        if keep is not None and not any(k in bn for k in keep): continue
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl = extractor(p)
        if fl is None or max(fl.size) < 60: continue
        seen.add(md5); out.append((bn[:26], fl))
    return out

def paste(bg_img, fl_rgba, px, py):             # ★CELL27과 동일
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def detect_top(m, pil, gt):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return 0.0, 0.0
    xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    top = float(cf.max())
    box_ok = max([cf[i] for i in range(len(cf)) if gt and iou(xy[i], gt) >= 0.5] + [0.0])   # bbox-정합 최고conf
    return top, box_ok

banks = {'NIST_cornoil': load_bank(NIST, '*FIRE*.jpg', KEEP_NIST, extract_flame),
         'AIHub_ENB':    load_bank(AIH,  '*.jpg', None, alpha_feather)}
for nm, fl in banks.items(): print(f'  뱅크 {nm:14}: {len(fl)}종')
assert banks['NIST_cornoil'] and banks['AIHub_ENB'], '뱅크 비었음 — 경로/업로드 확인'
m = YOLO(W); bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True))
_probe = sum(1 for p in bgs[:40] if robust_open_rgb(p) is not None)
print(f'배경 {len(bgs)}개 · 앞40개 중 열림 {_probe}/40')
if _probe == 0:
    print("⚠️ 배경이 하나도 안 열림 = Drive FUSE 미동기화. 아래를 '별도 셀'에 먼저 실행 후 CELL29 재실행:")
    print("   from google.colab import drive; drive.mount('/content/drive', force_remount=True)")
    raise SystemExit('배경 open 실패 — force_remount 후 재시도')

# ★paired 계획: (배경, ux, uy) 를 SEED로 1회 생성 → 두 뱅크에 공유. 불꽃 index만 뱅크별 RNG.
def make_plan(n):
    r = np.random.default_rng(SEED); pl = []
    for _ in range(n):
        pl.append((bgs[int(r.integers(len(bgs)))], float(r.uniform(0.15, 0.85)), float(r.uniform(0.35, 0.75))))
    return pl

print(f'\n{"뱅크":14}{"scale":>7}  ' + '  '.join(f'img@{c}' for c in CONFS) + '   box@0.25   miss')
results = {}
for FS in SCALES:
    plan = make_plan(N_COMP)
    for bank_nm, flames in banks.items():
        fr = np.random.default_rng(SEED + 12345)     # 불꽃 선택 RNG(뱅크별 동일 시드→같은 순번, 뱅크 크기만 다름)
        tops, boxoks = [], []
        for bp, ux, uy in plan:
            bg = robust_open_rgb(bp)
            if bg is None: continue          # ★Drive FUSE 못여는 배경은 양뱅크 동일하게 건너뜀(paired 유지)
            Wd, Hd = bg.size
            nm, fl = flames[int(fr.integers(len(flames)))]
            th = max(1, int(Hd*FS)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
            px = int(Wd*ux - tw/2); py = int(Hd*uy - th/2)
            comp, gt = paste(bg, fl_r, px, py)
            if gt is None: continue
            t, b = detect_top(m, comp, gt); tops.append(t); boxoks.append(b)
        tops = np.array(tops); boxoks = np.array(boxoks); n = len(tops)
        cells = '  '.join(f'{(tops>=c).mean():.3f}' for c in CONFS)
        box025 = (boxoks >= 0.25).mean(); miss = (tops < 0.25).mean()
        print(f'{bank_nm:14}{FS:>7}  {cells}   {box025:.3f}     {miss:.1%}  (n={n})')
        results[(bank_nm, FS)] = (tops, boxoks)

print('\n※ 판정: 두 뱅크 img@0.25 를 직접 비교(같은 배경/위치/N/SEED). NIST≈0.994(CELL27 재현) 대비 AIHub 값이 실측 답.')
print('  img@ = 프레임에 conf≥임계 박스 하나라도(=화재 프레임 인식) · box@0.25 = 진짜 불꽃 위치와 IoU≥0.5 정합(위치까지 맞음).')


# ========== CELL 30: AI-Hub 0.775 원인 분리 — 알파(추출) 한 변수 {feather_B, lumfade_A, colormask} ==========
# 질문: AIHub 0.775(scale0.25)가 (a)불꽃 자체 약함 vs (b)페더 다크헤일로 방해? → 같은 수동크롭에 알파만 바꿔 recall 비교.
#   colormask = NIST와 동일 추출(불꽃모양만·다크코너 없음) · lumfade = 어두운코너 제거 · feather = 현행(박스 그대로).
#   ★lumfade/colormask 가 feather_B(0.775)보다 유의하게 높으면 → (b) 다크헤일로가 주범(불꽃은 멀쩡).
#     비슷하면 → (a) 불꽃 자체 약함(CCTV 블러/과노출/D-Fire 분포밖). + 놓친/잡은 합성 몽타주로 육안 교차확인.
#   같은 배경/계획/SEED. scale 0.25 고정(그 지점서 차이 큼).
import os, glob, subprocess, sys, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
AIH = '/content/drive/MyDrive/firecrop_src/aihub_enb_manual'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 0; N_COMP = 160; FS = 0.25; CONFS = (0.05, 0.25, 0.50)

def ropen(p):
    for q in (p, unicodedata.normalize('NFC', p), unicodedata.normalize('NFD', p)):
        try: return Image.open(q).convert('RGB')
        except Exception: pass
    return None

def a_feather(path, fr=0.08):        # B: 박스 그대로(다크코너 포함)·가장자리만 소프트
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    h, w = im.shape[:2]; fpx = max(2, int(min(h, w)*fr))
    inner = np.zeros((h, w), np.float32); inner[fpx:h-fpx, fpx:w-fpx] = 1.0
    a = np.clip(ndimage.gaussian_filter(inner, fpx*0.6), 0, 1)
    return Image.fromarray(np.dstack([im, a*255]).astype(np.uint8))

def a_lumfade(path, lo=20, hi=65):   # A: 밝기 페이드(어두운 코너 제거)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    lum = 0.299*im[...,0]+0.587*im[...,1]+0.114*im[...,2]
    a = np.clip((lum-lo)/(hi-lo), 0, 1); a = ndimage.gaussian_filter(a, 0.8)
    return Image.fromarray(np.dstack([im, a*255]).astype(np.uint8))

def a_colormask(path):               # NIST와 동일: 색마스크 largest-CC(불꽃모양만)
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[...,0], im[...,1], im[...,2]; lum = 0.299*R+0.587*G+0.114*B
    mask = ((R > B+30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    mm = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(mm); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad); x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; m2 = mm[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*m2*255]).astype(np.uint8))

def load(extractor):
    out = []
    for p in sorted(glob.glob(f'{AIH}/*.jpg')):
        fl = extractor(p)
        if fl is None or max(fl.size) < 60: continue
        out.append((os.path.basename(p)[:20], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, Wd = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(Wd, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, Wd), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def det(m, pil):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return 0.0, np.zeros((0, 4)), np.zeros(0)
    xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    return float(cf.max()), xy, cf

m = YOLO(W); bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True))
r = np.random.default_rng(SEED)
plan = [(bgs[int(r.integers(len(bgs)))], float(r.uniform(0.15, 0.85)), float(r.uniform(0.35, 0.75))) for _ in range(N_COMP)]

variants = {'feather_B': load(a_feather), 'lumfade_A': load(a_lumfade), 'colormask': load(a_colormask)}
print(f'{"알파":12}{"종":>4}  ' + '  '.join(f'img@{c}' for c in CONFS) + '   miss   (scale 0.25)')
for vn, flames in variants.items():
    fr2 = np.random.default_rng(SEED + 12345); tops = []
    for bp, ux, uy in plan:
        bg = ropen(bp)
        if bg is None: continue
        Wd, Hd = bg.size; nm, fl = flames[int(fr2.integers(len(flames)))]
        th = max(1, int(Hd*FS)); tw = max(1, int(fl.width*th/fl.height)); flr = fl.resize((tw, th))
        px = int(Wd*ux - tw/2); py = int(Hd*uy - th/2)
        comp, gt = paste(bg, flr, px, py)
        if gt is None: continue
        t, _, _ = det(m, comp); tops.append(t)
    tops = np.array(tops); n = len(tops)
    se = (tops >= 0.25).std()/max(n, 1)**0.5
    print(f'{vn:12}{len(flames):>4}  ' + '  '.join(f'{(tops>=c).mean():.3f}' for c in CONFS) + f'   {(tops<0.25).mean():.1%} (n={n}, ±{1.96*se:.3f}@.25)')

# 몽타주: feather_B 합성 중 놓친 것 vs 잡은 것 (육안 원인 확인)
fl_b = variants['feather_B']; fr2 = np.random.default_rng(SEED + 12345)
miss_items, hit_items = [], []
for bp, ux, uy in plan:
    bg = ropen(bp)
    if bg is None: continue
    Wd, Hd = bg.size; nm, fl = fl_b[int(fr2.integers(len(fl_b)))]
    th = max(1, int(Hd*FS)); tw = max(1, int(fl.width*th/fl.height)); flr = fl.resize((tw, th))
    px = int(Wd*ux - tw/2); py = int(Hd*uy - th/2)
    comp, gt = paste(bg, flr, px, py)
    if gt is None: continue
    top, xy, cf = det(m, comp)
    tgt = miss_items if top < 0.25 else hit_items
    if len(tgt) < 8: tgt.append((comp, gt, xy, cf, nm, top))

def draw(items, title, path):
    if not items: print('  (0장)'); return
    K = len(items); cols = 4; rr = (K+cols-1)//cols
    fig, ax = plt.subplots(rr, cols, figsize=(4*cols, 3*rr)); ax = np.array(ax).reshape(rr, cols)
    for i in range(rr*cols):
        a = ax[i//cols, i%cols]; a.axis('off')
        if i >= K: continue
        comp, gt, xy, cf, nm, top = items[i]; a.imshow(comp)
        a.add_patch(mpatches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1], fill=False, edgecolor='lime', lw=1.5))
        for j in range(len(cf)):
            if cf[j] >= 0.25: a.add_patch(mpatches.Rectangle((xy[j][0], xy[j][1]), xy[j][2]-xy[j][0], xy[j][3]-xy[j][1], fill=False, edgecolor='red', lw=1))
        a.set_title(f'{nm[:12]} top{top:.2f}', fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=90, bbox_inches='tight'); plt.show(); print('  →', path)

print('\n[놓친 합성 top<0.25] 초록=붙인 불꽃 위치 · 빨강=검출박스')
draw(miss_items, 'AIHub feather_B 놓침 (다크헤일로? 흐린불꽃?)', f'{OUT}/aihub_missed.png')
print('[잡은 합성]')
draw(hit_items, 'AIHub feather_B 검출', f'{OUT}/aihub_hit.png')
print('\n※ 판정: lumfade/colormask ≫ feather_B(0.775) → (b) 다크헤일로가 주범(불꽃 멀쩡) · 비슷 → (a) 불꽃 자체 약함.')


# ========== CELL 31 (DRY-RUN): ablation 관통 테스트 — 2소스 × 2배경 × L1-L4 (scale 128) ==========
# 조언자 권고: 전체 24셀 전에 파이프라인·GT기록·집계·합성함수를 소규모로 관통 검증.
#   계단 L1(over·랜덤위치)→L2(→screen·랜덤)→L3(screen·컨텍스트배치)→L4(+스필). scale 128px 고정(dry-run).
#   페어링: L1↔L2 같은 랜덤점(블렌딩만). L3↔L4 같은 컨텍스트점(스필만). L2→L3은 위치=조작변수.
#   접지선 anchor_frac 정렬 · GT=불꽃 알파 bbox(스필 제외·고정) · 지표=tp_conf/fp_conf. 몽타주로 육안 버그체크.
import os, glob, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR = '/content/drive/MyDrive'; FSRC = f'{DR}/firecrop_src'
W = f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'
BG_ROOT = f'{DR}/realneg_frames/synth'
VFX = f'{FSRC}/vfx_bank'; NIST = f'{FSRC}/nist_stovetop_cornoil'; OUT = f'{DR}/synth_sweep'; os.makedirs(OUT, exist_ok=True)
KEEP_NIST = ['1574198232-EvtP', '1574199884-EvtP']   # calphalon·alumipan2 peak (256px 가능)
SCALE = 128; CONFS = (0.05, 0.25, 0.50)

def screen(a, b): return 255.0 - (255.0-a)*(255.0-b)/255.0
def ropen(p):
    for q in (p, unicodedata.normalize('NFC', p), unicodedata.normalize('NFD', p)):
        try: return Image.open(q).convert('RGB')
        except Exception: pass
    return None
def nist_extract(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R+0.587*G+0.114*B
    mask = ((R > B+30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    mm = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(mm); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad); x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; m2 = mm[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return np.dstack([crop, np.clip(l/160., 0, 1)*m2*255]).astype(np.uint8)
def core_lum(rgba):
    a = rgba[..., 3]/255.; lum = 0.299*rgba[...,0]+0.587*rgba[...,1]+0.114*rgba[...,2]; m = a > 0.5
    return float(np.percentile(lum[m], 98)) if m.any() else 200.0
def composite(bg_pil, flame, point, scale_px, anchor_frac, blend='screen', spill=True):
    bg = np.asarray(bg_pil).astype(np.float32); H, Wd = bg.shape[:2]
    fr = Image.fromarray(flame, 'RGBA'); tw = max(1, int(fr.width*scale_px/fr.height))
    flr = np.asarray(fr.resize((tw, scale_px))).astype(np.float32); fh, fw = flr.shape[:2]
    ax = fw//2; ay = int(anchor_frac*(fh-1)); px = int(point[0]-ax); py = int(point[1]-ay)
    cx, cy = px+fw//2, py+fh//2; out = bg.copy()
    if spill:
        yy, xx = np.mgrid[0:H, 0:Wd]; d2 = (xx-cx)**2 + (yy-cy)**2; r0 = fh*0.3
        inten = (core_lum(flame)/255.)*(r0*r0/(d2+r0*r0))*0.7            # 코어휘도∝·역제곱(코어반경 clamp)
        out = screen(out, np.dstack([inten*255, inten*140, inten*40]).astype(np.float32))
    x0c, y0c = max(0, px), max(0, py); xe = min(Wd, px+fw); ye = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = xe-x0c, ye-y0c; A = np.zeros((H, Wd), np.float32)
    if rw > 0 and rh > 0:
        reg = flr[fy0:fy0+rh, fx0:fx0+rw]; al = reg[..., 3:4]/255.; rgb = reg[..., :3]
        dst = out[y0c:y0c+rh, x0c:x0c+rw]
        blended = screen(dst, rgb) if blend == 'screen' else rgb
        out[y0c:y0c+rh, x0c:x0c+rw] = dst*(1-al) + blended*al
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    gt = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), gt
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih; ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0
def detect(m, pil, gt):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return 0., 0., []
    xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    tp = max([cf[i] for i in range(len(cf)) if gt and iou(xy[i], gt) >= 0.5] + [0.])
    fp = max([cf[i] for i in range(len(cf)) if (not gt) or iou(xy[i], gt) < 0.3] + [0.])
    return tp, fp, list(zip(xy.tolist(), cf.tolist()))

BG_MAN = json.load(open(f'{FSRC}/manifest.json'))        # bgNN.jpg → rel path
PLACE  = json.load(open(f'{FSRC}/placement.json'))       # bgNN.jpg → [x1,y1,x2,y2]
bg_names = sorted(PLACE.keys())[:2]
vrows = list(csv.DictReader(open(f'{VFX}/manifest.csv')))
vp = [r for r in vrows if r['orientation'] == 'vertical'][0]
vfx_flame = np.asarray(Image.open(f'{VFX}/{vp["matte"]}').convert('RGBA')); vfx_anchor = float(vp['anchor_frac'])
nfiles = [p for p in sorted(glob.glob(f'{NIST}/*FIRE*.jpg')) if any(k in os.path.basename(p) for k in KEEP_NIST)]
nist_flame = nist_extract(nfiles[0]); nist_anchor = 1.0
sources = [('VFX', vp['scene_id'], vfx_flame, vfx_anchor), ('NIST', 'peak', nist_flame, nist_anchor)]
print(f'소스: VFX {vp["matte"][:20]}(anchor {vfx_anchor}) · NIST {os.path.basename(nfiles[0])[:24]}')
print(f'배경: {bg_names}')

m = YOLO(W)
LEVELS = [('L1_over_rand','over',False,'random'), ('L2_screen_rand','screen',False,'random'),
          ('L3_screen_ctx','screen',False,'context'), ('L4_screen_ctx_spill','screen',True,'context')]
rng = np.random.default_rng(0); results = []; montage = []
for bgname in bg_names:
    bg = ropen(f'{BG_ROOT}/{BG_MAN[bgname]}')
    if bg is None: print(f'[skip] 배경 못열음 {bgname}'); continue
    Wd, Hd = bg.size; box = PLACE[bgname]
    ctx = ((box[0]+box[2])//2, box[3])                                  # 컨텍스트=박스 하단중앙(조리표면)
    rpt = (int(rng.uniform(0.2, 0.8)*Wd), int(rng.uniform(0.4, 0.7)*Hd))
    for sname, scene, flame, anchor in sources:
        for lname, blend, spill, pos in LEVELS:
            point = ctx if pos == 'context' else rpt
            comp, gt = composite(bg, flame, point, SCALE, anchor, blend, spill)
            tp, fp, dets = detect(m, comp, gt)
            results.append((sname, bgname, lname, tp, fp, gt is not None))
            if len(montage) < 16: montage.append((comp, gt, dets, f'{sname} {bgname[:5]} {lname}'))

print(f'\n{"src":5}{"bg":8}{"level":22}{"tp_conf":>8}{"fp_conf":>8}{"gt":>4}')
for sname, bgname, lname, tp, fp, hasgt in results:
    print(f'{sname:5}{bgname:8}{lname:22}{tp:>8.3f}{fp:>8.3f}{"Y" if hasgt else "N":>4}')

K = len(montage); cols = 4; rr = (K+cols-1)//cols
fig, ax = plt.subplots(rr, cols, figsize=(5*cols, 3.0*rr)); ax = np.array(ax).reshape(rr, cols)
for i in range(rr*cols):
    a = ax[i//cols, i%cols]; a.axis('off')
    if i >= K: continue
    comp, gt, dets, lab = montage[i]; a.imshow(comp)
    if gt: a.add_patch(mpatches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1], fill=False, edgecolor='lime', lw=2))
    for xy, cf in dets:
        if cf >= 0.25: a.add_patch(mpatches.Rectangle((xy[0], xy[1]), xy[2]-xy[0], xy[3]-xy[1], fill=False, edgecolor='red', lw=1))
    a.set_title(lab, fontsize=8)
fig.suptitle('CELL31 DRY-RUN: 초록=GT불꽃 · 빨강=base검출(conf>=0.25)')
plt.tight_layout(); plt.savefig(f'{OUT}/dryrun_montage.png', dpi=85, bbox_inches='tight'); plt.show()
print('\n몽타주:', f'{OUT}/dryrun_montage.png')
print('※ 버그체크: (1)L3/L4 불꽃이 박스하단(조리면)에 앉았나 (2)L1 over→L2 screen 밝아짐 (3)L4 스필 글로우 (4)GT초록=불꽃만(스필 제외)')


# ========== CELL 32-nist (재빌드): NIST 뱅크 재구축 — ign rescue + 패딩·anchor 교정 + 빌더 커밋 ==========
# 근거: CELL32-pre 실측 = ign 원본 실크기 alumipan2 164x131·calphalon 152x89 (표준임계 T0·응집 fill0.41/0.58).
#   옛 뱅크 57px = (미커밋)빌더 아티팩트. 표준 extract_flame(line826)이 이미 전체불꽃 포착 → 임계 안 느슨(T2/T3=반사/리그 노이즈 실측).
# 수정: (1)pad 고정10 → 비례(불꽃높이 5%·[3,10] 클램프: peak는 옛값 유지·ign 과패딩 방지)
#       (2)anchor=알파 접지행/매트높이(pad 무관하게 올바름) (3)manifest에 stage·source(NIST_ig) 태깅.
# 집계: ign=source 'NIST_ig'로 peak('NIST')와 분리 → 이벤트 여전히 2개(4장면 아님·pseudoreplication 회피).
#   ign = "소형·저대비 유류불" 별도 스트레스행. massloss13/14c = 금속리그 제외(사전등록 육안).
import os, csv, numpy as np
from scipy import ndimage
from PIL import Image
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
SRC='/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
NISTB='/content/drive/MyDrive/firecrop_src/nist_bank'; os.makedirs(NISTB, exist_ok=True)
FRAMES=[  # (event, stage, source, path) — CELL20c 명명. massloss 제외(금속리그).
 ('alumipan2','peak','NIST',    f'{SRC}/cornoil_alumipan2_EvtP_FIRE__1574199884-EvtP.jpg'),
 ('calphalon','peak','NIST',    f'{SRC}/cornoil_calphalon_EvtP_FIRE__1574198232-EvtP.jpg'),
 ('alumipan2','ign', 'NIST_ig', f'{SRC}/cornoil_alumipan2_Evt3_FIRE__1574199884-Evt3.jpg'),
 ('calphalon','ign', 'NIST_ig', f'{SRC}/cornoil_calphalon_Evt3_FIRE__1574198232-Evt3.jpg'),
]
def extract_flame(path):                 # ★표준(CELL29 line826) + pad 비례화
    im=np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R,G,B=im[...,0],im[...,1],im[...,2]; lum=0.299*R+0.587*G+0.114*B
    mask=((R>B+30)&(R>90))|(lum>210)
    if mask.sum()<50: return None
    lbl,_=ndimage.label(mask); c=np.bincount(lbl.ravel()); c[0]=0
    mm=ndimage.binary_dilation(lbl==c.argmax(), iterations=3)
    ys,xs=np.where(mm); fh=ys.max()-ys.min()+1
    pad=int(np.clip(fh*0.05, 3, 10))     # ★고정10 → 5%비례·상한10(peak=10 유지·ign 과패딩방지)
    x0=max(0,xs.min()-pad); y0=max(0,ys.min()-pad)
    x1=min(im.shape[1]-1,xs.max()+pad); y1=min(im.shape[0]-1,ys.max()+pad)
    crop=im[y0:y1,x0:x1]; m2=mm[y0:y1,x0:x1].astype(np.float32); l=lum[y0:y1,x0:x1]
    a=np.clip(l/160.,0,1)*m2*255
    return Image.fromarray(np.dstack([crop,a]).astype(np.uint8))
def anchor_of(rgba):                     # ★알파 접지행/매트높이 (pad 무관 교정)
    a=np.asarray(rgba)[...,3].astype(np.float32)/255.; rc=(a>0.3).sum(axis=1)
    if rc.max()==0: return 1.0
    return float(np.where(rc>=0.05*rc.max())[0].max()/(a.shape[0]-1))
rows=[]
print(f'{"event":10}{"stage":6}{"source":8}{"matte WxH":>12}{"anchor":>8}   scale적용')
for event,stage,source,path in FRAMES:
    if not os.path.exists(path): print(f'{event:10}{stage:6} FILE MISSING'); continue
    fl=extract_flame(path)
    if fl is None: print(f'{event:10}{stage:6} extract 실패'); continue
    W,H=fl.size; anc=anchor_of(fl); fn=f'{event}_{stage}.png'; fl.save(f'{NISTB}/{fn}')
    rows.append(dict(source=source,scene_id=event,matte=fn,anchor_frac=round(anc,3),h=H,stage=stage,equip_flag='N'))
    print(f'{event:10}{stage:6}{source:8}{f"{W}x{H}":>12}{anc:>8.3f}   {[s for s in (64,128,256) if H>=s]}')
with open(f'{NISTB}/manifest.csv','w',newline='') as f:
    wr=csv.DictWriter(f, fieldnames=['source','scene_id','matte','anchor_frac','h','stage','equip_flag']); wr.writeheader()
    for r in rows: wr.writerow(r)
print(f'\n저장: {NISTB}/manifest.csv ({len(rows)}매트) + *.png')
print('※ 확인: ign h>=64(scale64 다운스케일 OK)·peak≈233x657/283x620(옛뱅크 정합)·anchor peak~0.98/ign~0.96. massloss 미포함=금속리그.')
print('  이 셀=이전 미커밋 빌더 대체(재현성 확보).')

# --- 팬 림/anchor 확인(리뷰 요청): 접지선이 불꽃 base인가 팬림 반사인가 · base zone 색분석 ---
import matplotlib.pyplot as plt
mats=[(r['scene_id'],r['stage'],r['matte'],r['anchor_frac']) for r in rows]
fig,ax=plt.subplots(len(mats),3,figsize=(10,3*len(mats))); ax=np.atleast_2d(ax)
for i,(sc,stg,fn,anc) in enumerate(mats):
    rgba=np.asarray(Image.open(f'{NISTB}/{fn}')); H=rgba.shape[0]; gl=int(anc*(H-1))
    rgb=rgba[...,:3].copy(); rgb[max(0,gl-1):gl+2,:]=[255,0,0]          # 접지선 빨강
    ax[i,0].imshow(rgb); ax[i,0].axis('off'); ax[i,0].set_title(f'{sc}_{stg} anchor={anc:.2f}',fontsize=8)
    ax[i,1].imshow(rgba[...,3],cmap='gray'); ax[i,1].axhline(gl,color='r'); ax[i,1].axis('off'); ax[i,1].set_title('alpha',fontsize=8)
    lo=max(0,int(gl-0.12*H)); reg=rgba[lo:gl+1]; msk=reg[...,3]>76        # 하단12% · alpha>0.3
    ax[i,2].imshow(reg[...,:3].astype(np.uint8)); ax[i,2].axis('off')
    if msk.any():
        sub=reg[...,:3][msk].astype(int); rr,gg,bb=sub[:,0],sub[:,1],sub[:,2]; lm=0.299*rr+0.587*gg+0.114*bb
        flame=float(((rr-bb)>15).mean()); metal=float(((np.abs(rr-bb)<20)&(lm>200)).mean())
    else: flame=metal=0.0
    ax[i,2].set_title(f'base zone: flame {flame:.0%} / achroma {metal:.0%}',fontsize=8)
    print(f'{sc}_{stg}: H={H} anchor_row={gl}/{H-1}  base zone(하단12%): 불꽃색 {flame:.0%} · 무채색밝음(팬림?) {metal:.0%}')
fig.suptitle('NIST rebuild anchor check: red=grounding line | base zone flame-colored(OK) vs achromatic-bright(pan-rim specular?)',fontsize=9)
plt.tight_layout(); plt.savefig(f'{NISTB}/rebuild_anchor_check.png',dpi=90,bbox_inches='tight'); plt.show()
print('※ 판단: base zone 불꽃색↑·무채색↓ → 접지선=불꽃 base(정상·GT 깨끗). 무채색밝음↑ → 팬림 specular가 alpha/GT에 낌 → 말해줘(접지선아래 alpha 컷 1줄).')


# ========== CELL 32 (본실험 ablation): over base + 수정4 반영 ==========
# 수정: (1)spill=additive (2)0-c 배경FP 제외 (3)랜덤 다중시드(bg별 공유=페어링) (4)NIST 파일뱅크.
# 조건0: 0a_hard(불투명 사각형·바닥) · 0-c(무불꽃 배경FP·아래서 측정) · 0-b(생성셋 0.809·CELL 확인·여기 미포함).
# 계단: over_rand → over_ctx(배치) → over_ctx_spill(스필) + screen_ctx(washout 다배경 확정).
# SCALE 64/128/256(무업스케일) × SOURCE(VFX/NIST) × 18배경. 장면단위 집계. 지표 recall/tp_conf/synth_FP.
import os, glob, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
from collections import defaultdict

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR = '/content/drive/MyDrive'; FSRC = f'{DR}/firecrop_src'
W = f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; BG_ROOT = f'{DR}/realneg_frames/synth'
VFXB = f'{FSRC}/vfx_bank'; NISTB = f'{FSRC}/nist_bank'; OUT = f'{DR}/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SCALES = [64, 128, 256]; SEEDS = 3; CONF = 0.25

def screen(a, b): return 255.0 - (255.0-a)*(255.0-b)/255.0
def ropen(p):
    for q in (p, unicodedata.normalize('NFC', p), unicodedata.normalize('NFD', p)):
        try: return Image.open(q).convert('RGB')
        except Exception: pass
    return None
def core_lum(rgba):
    a = rgba[..., 3]/255.; lum = 0.299*rgba[...,0]+0.587*rgba[...,1]+0.114*rgba[...,2]; m = a > 0.5
    return float(np.percentile(lum[m], 98)) if m.any() else 200.0
def composite(bg_pil, flame, point, scale_px, anchor_frac, blend='over', spill=False, hard=False):
    bg = np.asarray(bg_pil).astype(np.float32); H, Wd = bg.shape[:2]
    fr = Image.fromarray(flame, 'RGBA'); tw = max(1, int(fr.width*scale_px/fr.height))
    flr = np.asarray(fr.resize((tw, scale_px))).astype(np.float32); fh, fw = flr.shape[:2]
    ax = fw//2; ay = int(anchor_frac*(fh-1)); px = int(point[0]-ax); py = int(point[1]-ay)
    cx, cy = px+fw//2, py+fh//2; out = bg.copy()
    if spill:                                                            # ★수정1: additive 반사광
        yy, xx = np.mgrid[0:H, 0:Wd]; d2 = (xx-cx)**2+(yy-cy)**2; r0 = fh*0.3
        inten = (core_lum(flame)/255.)*(r0*r0/(d2+r0*r0))*0.7
        out = np.clip(out + np.dstack([inten*255, inten*140, inten*40]).astype(np.float32), 0, 255)
    x0c, y0c = max(0, px), max(0, py); xe = min(Wd, px+fw); ye = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = xe-x0c, ye-y0c; A = np.zeros((H, Wd), np.float32)
    if rw > 0 and rh > 0:
        reg = flr[fy0:fy0+rh, fx0:fx0+rw]; rgb = reg[..., :3]; flame_a = reg[..., 3:4]/255.
        al = np.ones_like(flame_a) if hard else flame_a   # hard=불투명 사각형 붙임(0-a) · 단 GT는 flame_a로 통일
        dst = out[y0c:y0c+rh, x0c:x0c+rw]; blended = screen(dst, rgb) if blend == 'screen' else rgb
        out[y0c:y0c+rh, x0c:x0c+rw] = dst*(1-al) + blended*al; A[y0c:y0c+rh, x0c:x0c+rw] = flame_a[..., 0]
    ys, xs = np.where(A > 0.1)
    gt = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), gt
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih; ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0
def dets_of(m, pil):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    return list(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))

def load_bank(bd, filt=None):
    out = []
    for r in csv.DictReader(open(f'{bd}/manifest.csv')):
        if filt and not filt(r): continue
        out.append((r['source'], r['scene_id'], f'{bd}/{r["matte"]}', float(r['anchor_frac']), int(r['h'])))
    return out
vfx = load_bank(VFXB); byscene = {}
for s in vfx: byscene.setdefault(s[1], s)                                 # 장면당 1매트
vfx = list(byscene.values())
nist = load_bank(NISTB, filt=lambda r: r.get('equip_flag') != 'Y')   # peak+ign(rescue) · 무업스케일은 아래 mh<scale가 처리 · ign은 source=NIST_ig로 분리집계
sources = vfx + nist
print(f'소스: VFX {len(vfx)}장면 + NIST {len(nist)}장면 = {len(sources)}')

BG_MAN = json.load(open(f'{FSRC}/manifest.json')); PLACE = json.load(open(f'{FSRC}/placement.json'))
bgs = []
for n in sorted(PLACE):
    im = ropen(f'{BG_ROOT}/{BG_MAN[n]}')
    if im is not None: bgs.append((n, im, PLACE[n]))
print(f'배경: {len(bgs)}장')
m = YOLO(W)

# ★수정2: 0-c 배경 FP 박스 집합 (무불꽃 배경)
bgfp = {}
for n, im, box in bgs:
    bgfp[n] = [xy for xy, cf in dets_of(m, im) if cf >= CONF]
print(f'0-c 배경FP: ' + ', '.join(f'{n[:5]}:{len(bgfp[n])}' for n, _, _ in bgs))

# ★수정3: bg별 랜덤위치 다중시드(소스 공유=페어링)
randpts = {}
for n, im, box in bgs:
    Wd, Hd = im.size; pl = []
    for sd in range(SEEDS):
        r = np.random.default_rng(1000*sd + sum(map(ord, n)))
        pl.append((int(r.uniform(0.2, 0.8)*Wd), int(r.uniform(0.4, 0.7)*Hd)))
    randpts[n] = pl

def synth_fp(dets, gt, n):                                               # 배경FP 제외한 합성유발 FP
    best = 0.0
    for xy, cf in dets:
        if cf < CONF: continue
        if gt and iou(xy, gt) >= 0.3: continue
        if any(iou(xy, b) >= 0.5 for b in bgfp[n]): continue
        best = max(best, cf)
    return best

LEVELS = [('0a_hard_ctx', 'over', False, 'context', True), ('over_rand', 'over', False, 'random', False),
          ('over_ctx', 'over', False, 'context', False), ('screen_ctx', 'screen', False, 'context', False),
          ('over_ctx_spill', 'over', True, 'context', False)]
rows = []
for si, (src, scene, mpath, anchor, mh) in enumerate(sources):
    flame = np.asarray(Image.open(mpath).convert('RGBA'))
    for scale in SCALES:
        if mh < scale: continue                                          # 무업스케일
        for lname, blend, spill, pos, hard in LEVELS:
            hit = []; tpc = []; sfp = []
            for n, im, box in bgs:
                pts = [((box[0]+box[2])//2, box[3])] if pos == 'context' else randpts[n]
                for point in pts:
                    comp, gt = composite(im, flame, point, scale, anchor, blend, spill, hard)
                    if gt is None: continue
                    dts = dets_of(m, comp)
                    tp = max([cf for xy, cf in dts if iou(xy, gt) >= 0.5] + [0.0])
                    hit.append(1 if tp >= CONF else 0); tpc.append(tp); sfp.append(synth_fp(dts, gt, n))
            if hit:
                rows.append((src, scene, scale, lname, float(np.mean(hit)), float(np.mean(tpc)), float(np.mean(sfp))))
    if (si+1) % 5 == 0: print(f'  ...{si+1}/{len(sources)} 소스 처리')

# --- 장면단위 집계: (source,scale,level) 위에서 장면 평균 ---
agg = defaultdict(list)
for src, scene, scale, lname, rec, tp, sfp in rows: agg[(src, scale, lname)].append((rec, tp, sfp))
print(f'\n{"src":5}{"scale":>6}  {"level":16}{"recall":>8}{"tp_conf":>8}{"synFP":>7}{"scenes":>7}')
for (src, scale, lname) in sorted(agg):
    v = agg[(src, scale, lname)]
    print(f'{src:5}{scale:>6}  {lname:16}{np.mean([x[0] for x in v]):>8.3f}{np.mean([x[1] for x in v]):>8.3f}{np.mean([x[2] for x in v]):>7.3f}{len(v):>7}')
json.dump(rows, open(f'{OUT}/ablation_rows.json', 'w'))
print('\n저장: synth_sweep/ablation_rows.json (장면별 원자료)')
print('※ 핵심: over_ctx vs screen_ctx(washout 다배경 확정) · over_rand vs over_ctx(발견2 배치효과) · over_ctx vs over_ctx_spill(스필) · scale별 · VFX vs NIST(NIST=경향).')


# ========== CELL 33: VFX 장면별 분포 — ⑤교락·③스필 재판정·n=26 견고성·⑥종횡비 ==========
# CELL 32의 ablation_rows.json(장면별 원자료)을 뜯어봄. 목적: 평균에 묻힌 것들.
#   (a)VFX recall 분산(전반저조 vs 소수약함=⑤교락) (b)스필 per-scene(돕나 해치나=③재판정) (c)over>screen 장면별 (d)orientation별(⑥종횡비).
import os, json, csv, numpy as np
from collections import defaultdict
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'
rows=json.load(open(f'{DR}/synth_sweep/ablation_rows.json'))   # [src,scene,scale,level,recall,tp_conf,synFP]
V=[r for r in rows if r[0]=='VFX']
def cell(scale, level): return {r[1]:(r[4],r[5],r[6]) for r in V if r[2]==scale and r[3]==level}

print('=== (a) VFX over_ctx recall 장면분포 — 전반저조 vs 소수약함 ===')
for sc in (64,128,256):
    vals=np.array([v[0] for v in cell(sc,'over_ctx').values()]); q=np.percentile(vals,[0,25,50,75,100])
    print(f'  s{sc}: mean {vals.mean():.3f} std {vals.std():.3f} | min/Q1/med/Q3/max {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}/{q[3]:.2f}/{q[4]:.2f} | rec0 {int((vals==0).sum())}/26 · rec>=0.8 {int((vals>=0.8).sum())}/26')

print('\n=== (b) 스필 per-scene 효과 (spill - over_ctx) ===')
for sc in (64,128,256):
    a=cell(sc,'over_ctx'); b=cell(sc,'over_ctx_spill'); ks=[k for k in a if k in b]
    dr=np.array([b[k][0]-a[k][0] for k in ks]); dfp=np.array([b[k][2]-a[k][2] for k in ks])
    print(f'  s{sc}: dRecall {dr.mean():+.3f} (도움 {int((dr>0.02).sum())}·해 {int((dr<-0.02).sum())}·~동 {int((np.abs(dr)<=0.02).sum())}) | dSynFP {dfp.mean():+.3f} (오름 {int((dfp>0.005).sum())}/{len(ks)})')

print('\n=== (c) over vs screen per-scene (washout이 장면별로도?) ===')
for sc in (64,128,256):
    a=cell(sc,'over_ctx'); b=cell(sc,'screen_ctx'); ks=[k for k in a if k in b]
    print(f'  s{sc}: over>screen {int(sum(a[k][0]>b[k][0] for k in ks))}/{len(ks)} 장면 (평균 {np.mean([a[k][0] for k in ks]):.3f} vs {np.mean([b[k][0] for k in ks]):.3f})')

print('\n=== (a-보강) over_ctx@256 하위5/상위5 장면 ===')
srt=sorted(cell(256,'over_ctx').items(), key=lambda kv: kv[1][0])
print('  최저5:', [(k[:14], round(v[0],2)) for k,v in srt[:5]])
print('  최고5:', [(k[:14], round(v[0],2)) for k,v in srt[-5:]])

print('\n=== (d) VFX orientation별 over_ctx recall (height-scale가 세로긴/수평에 불리?·⑥) ===')
try:
    vman={r['scene_id']:r for r in csv.DictReader(open(f'{DR}/firecrop_src/vfx_bank/manifest.csv'))}
    for sc in (64,256):
        byori=defaultdict(list)
        for k,v in cell(sc,'over_ctx').items(): byori[vman.get(k,{}).get('orientation','?')].append(v[0])
        print(f'  s{sc}: ' + ' · '.join(f'{o} n{len(vs)} {np.mean(vs):.3f}' for o,vs in sorted(byori.items())))
except Exception as e:
    print('  (orientation 조인 실패:', str(e)[:50], ')')


# ========== CELL 34: miss 육안 + 소스품질 지표 — 전멸4 vs 우수4 (무엇이 소스를 약하게 하나) ==========
# 목적(리뷰): "큐레이션이 레버"는 아직 관찰. 처방이 되려면 컬 기준이 필요 → 전멸/우수 매트 나란히 + 후보지표(알파커버·파편수·종횡비).
#   composite로 blind(any conf 0) vs 위치오류(any 높은데 TP 0) 구분.
# ⚠️ Phase B 주의: base가 놓치는 어려운 소스 = 학습 정보량 큼 → (A)프록시용 컬이 (B)학습엔 역효과일 수. 순환("못잡는걸 빼면 잘잡음") 경계.
import os, csv, subprocess, sys, json, numpy as np
from scipy import ndimage
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'
m=YOLO(f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt')
vman={r['scene_id']:r for r in csv.DictReader(open(f'{VFXB}/manifest.csv'))}
rows=json.load(open(f'{DR}/synth_sweep/ablation_rows.json'))
rec256={r[1]:r[4] for r in rows if r[0]=='VFX' and r[2]==256 and r[3]=='over_ctx'}
tpc256={r[1]:r[5] for r in rows if r[0]=='VFX' and r[2]==256 and r[3]=='over_ctx'}  # 18bg 평균 tp conf: ~0=진짜블라인드 · 0.1~0.24=약함(임계미달)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
def ropen(p):
    try: return Image.open(p).convert('RGB')
    except: return None
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def comp_over(bg, flame, scale, anchor, box):
    bgn=np.asarray(bg).astype(np.float32); H,Wd=bgn.shape[:2]; fr=Image.fromarray(flame)
    tw=max(1,int(fr.width*scale/fr.height)); flr=np.asarray(fr.resize((tw,scale))).astype(np.float32); fh,fw=flr.shape[:2]
    px=int((box[0]+box[2])//2-fw//2); py=int(box[3]-int(anchor*(fh-1)))
    x0,y0=max(0,px),max(0,py); xe,ye=min(Wd,px+fw),min(H,py+fh); fx0,fy0=x0-px,y0-py; rw,rh=xe-x0,ye-y0
    out=bgn.copy(); A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; a=reg[...,3:4]/255.
        out[y0:y0+rh,x0:x0+rw]=out[y0:y0+rh,x0:x0+rw]*(1-a)+reg[...,:3]*a; A[y0:y0+rh,x0:x0+rw]=reg[...,3]/255.
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)), gt
def dets(pil):
    r=m.predict(pil,conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
    return [] if r.boxes is None else list(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))
def quality(fl):                      # 소스 품질 후보지표(리뷰 3가설): 알파커버·연결조각수·코어휘도
    a=fl[...,3]/255.; tot=int((a>0.3).sum()); cov=float((a>0.3).mean())          # 가설1: 알파/bbox 면적비(파편=낮음)
    lbl,_=ndimage.label(a>0.3); sizes=np.bincount(lbl.ravel())[1:]
    frags=int((sizes>=0.02*tot).sum()) if tot>0 and len(sizes) else 0            # 가설2: 연결성분 수(흩어짐)
    m5=a>0.5; lum=0.299*fl[...,0]+0.587*fl[...,1]+0.114*fl[...,2]
    corelum=float(lum[m5].mean()) if m5.any() else 0.0                           # 가설3(부분): 소스 코어휘도(배경대비델타는 composite-특이·후속)
    return cov, frags, corelum
DEAD=['10141290','13915730','15502132','3514521']; TOP=['15884546','13915751','20025238','16212846']
fig,ax=plt.subplots(8,3,figsize=(12,26)); ri=0
for gname,ids in [('DEAD',DEAD),('TOP',TOP)]:
    for sid in ids:
        r=vman.get(sid);
        if r is None: print(f'{gname} {sid} 매니페스트에 없음(다른 매트일 수)'); ri+=1; continue
        fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
        cov,frags,corelum=quality(fl); H,W=fl.shape[:2]; asp=round(W/H,2)
        ax[ri,0].imshow(fl); ax[ri,0].axis('off'); ax[ri,0].set_title(f'[{gname}] {sid} {W}x{H} asp{asp}\ncov{cov:.2f} frags{frags} lum{corelum:.0f} rec{rec256.get(sid,-1):.2f} tp{tpc256.get(sid,-1):.2f}',fontsize=8)
        ax[ri,1].imshow(fl[...,3],cmap='gray'); ax[ri,1].axis('off'); ax[ri,1].set_title('alpha',fontsize=8)
        n,bg,box=bgs[0]; comp,gt=comp_over(bg,fl,256,anc,box); ds=dets(comp); vis=comp.copy(); d=ImageDraw.Draw(vis)
        if gt: d.rectangle(list(gt),outline=(0,255,0),width=4)
        for xy,cf in ds:
            if cf>=0.10: d.rectangle([int(t) for t in xy],outline=(255,0,0),width=2)
        tp=max([cf for xy,cf in ds if gt and iou(xy,gt)>=0.5]+[0.0]); anyt=max([cf for _,cf in ds]+[0.0])
        ax[ri,2].imshow(vis); ax[ri,2].axis('off'); ax[ri,2].set_title(f'{n[:6]}@256 TP{tp:.2f}/any{anyt:.2f}',fontsize=8)
        print(f'{gname:5}{sid:10} {W}x{H} asp{asp} cov{cov:.3f} frags{frags} corelum{corelum:.0f} rec256={rec256.get(sid,-1):.2f} tp256={tpc256.get(sid,-1):.3f}')
        ri+=1
fig.suptitle('소스품질: DEAD(전멸4) vs TOP(우수4) | matte·alpha·composite@256 | cov=알파커버·frags=유의미조각·asp=종횡비',fontsize=10)
plt.tight_layout(); plt.savefig(f'{DR}/synth_sweep/miss_quality.png',dpi=80,bbox_inches='tight'); plt.show()
print('\n※ blind vs 약함(리뷰): tp256 ~0 = 진짜 블라인드(컬 후보) · 0.1~0.24 = 약한신호(임계미달=threshold 대상, 컬 아님).')
print('※ 컬 기준: DEAD가 TOP 대비 cov낮음/frags많음/lum낮음/극단asp 이면 후보. 차이 없으면 큐레이션 처방 근거 약함(=관찰로 남김).')
print('⚠️ Phase B: 못잡는 어려운 소스=학습 정보량 큼 → (A)용 컬이 (B)엔 역효과일 수. 순환 경계.')
# (CELL 35 폐기·36/36b는 메시지 이력 → 커밋 시 통합. 아래 37은 GT 임계 스윕.)


# ========== CELL 37: GT alpha 임계 스윕 — rec@.5 by GT{0.1,0.3,0.5} (대형 저recall=라벨정의?) ==========
# 리뷰: @256 gap 0.372 = 낮은 recall의 절반이 GT(alpha>0.1) 헐렁 탓 추정. alpha 0.3/0.5로 GT 조이면 rec@.5 회복?
#   검출은 GT무관 → 한 번 추론에서 GT 3임계 동시 집계(재추론 불요). over>screen 순위 유지 확인 = 결론 견고성.
#   GT>.1=원래(CELL32) · GT>.5 ≈ D-Fire 본체박싱. 조밀소스 회복 작고 성긴소스(VFX256) 크게 회복 예상.
import os, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
import numpy as np
from PIL import Image
from collections import defaultdict
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'
W=f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; BG_ROOT=f'{DR}/realneg_frames/synth'
VFXB=f'{FSRC}/vfx_bank'; NISTB=f'{FSRC}/nist_bank'
SCALES=[64,128,256]; SEEDS=3; CONF=0.25; GTS=[0.1,0.3,0.5]
def screen(a,b): return 255.0-(255.0-a)*(255.0-b)/255.0
def ropen(p):
    for q in (p, unicodedata.normalize('NFC',p), unicodedata.normalize('NFD',p)):
        try: return Image.open(q).convert('RGB')
        except: pass
    return None
def core_lum(rgba):
    a=rgba[...,3]/255.; lum=0.299*rgba[...,0]+0.587*rgba[...,1]+0.114*rgba[...,2]; mm=a>0.5
    return float(np.percentile(lum[mm],98)) if mm.any() else 200.0
def composite(bg_pil, flame, point, scale_px, anchor_frac, blend='over', spill=False, hard=False):
    bg=np.asarray(bg_pil).astype(np.float32); H,Wd=bg.shape[:2]
    fr=Image.fromarray(flame); tw=max(1,int(fr.width*scale_px/fr.height))
    flr=np.asarray(fr.resize((tw,scale_px))).astype(np.float32); fh,fw=flr.shape[:2]
    ax=fw//2; ay=int(anchor_frac*(fh-1)); px=int(point[0]-ax); py=int(point[1]-ay)
    cx,cy=px+fw//2,py+fh//2; out=bg.copy()
    if spill:
        yy,xx=np.mgrid[0:H,0:Wd]; d2=(xx-cx)**2+(yy-cy)**2; r0=fh*0.3
        inten=(core_lum(flame)/255.)*(r0*r0/(d2+r0*r0))*0.7
        out=np.clip(out+np.dstack([inten*255,inten*140,inten*40]).astype(np.float32),0,255)
    x0c,y0c=max(0,px),max(0,py); xe=min(Wd,px+fw); ye=min(H,py+fh)
    fx0,fy0=x0c-px,y0c-py; rw,rh=xe-x0c,ye-y0c; A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; rgb=reg[...,:3]; fa=reg[...,3:4]/255.
        al=np.ones_like(fa) if hard else fa
        dst=out[y0c:y0c+rh,x0c:x0c+rw]; bl=screen(dst,rgb) if blend=='screen' else rgb
        out[y0c:y0c+rh,x0c:x0c+rw]=dst*(1-al)+bl*al; A[y0c:y0c+rh,x0c:x0c+rw]=fa[...,0]
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)), A
def bbox_at(A,thr):
    ys,xs=np.where(A>thr)
    return (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def dets_of(mm,pil):
    r=mm.predict(pil,conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    return list(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))
def load_bank(bd,filt=None):
    out=[]
    for r in csv.DictReader(open(f'{bd}/manifest.csv')):
        if filt and not filt(r): continue
        out.append((r['source'],r['scene_id'],f'{bd}/{r["matte"]}',float(r['anchor_frac']),int(r['h'])))
    return out
vfx=load_bank(VFXB); bsd={}
for s in vfx: bsd.setdefault(s[1],s)
vfx=list(bsd.values()); nist=load_bank(NISTB, filt=lambda r:r.get('equip_flag')!='Y'); sources=vfx+nist
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
print(f'소스 {len(sources)} · 배경 {len(bgs)}'); m=YOLO(W)
randpts={}
for n,im,box in bgs:
    Wd,Hd=im.size; pl=[]
    for sd in range(SEEDS):
        rr=np.random.default_rng(1000*sd+sum(map(ord,n))); pl.append((int(rr.uniform(0.2,0.8)*Wd),int(rr.uniform(0.4,0.7)*Hd)))
    randpts[n]=pl
LEVELS=[('over_ctx','over',False,'context',False),('screen_ctx','screen',False,'context',False),('over_rand','over',False,'random',False)]
rows=[]
for si,(src,scene,mpath,anchor,mh) in enumerate(sources):
    flame=np.asarray(Image.open(mpath).convert('RGBA'))
    for scale in SCALES:
        if mh<scale: continue
        for lname,blend,spill,pos,hard in LEVELS:
            hits={g:[] for g in GTS}
            for n,im,box in bgs:
                pts=[((box[0]+box[2])//2,box[3])] if pos=='context' else randpts[n]
                for point in pts:
                    comp,A=composite(im,flame,point,scale,anchor,blend,spill,hard)
                    if not (A>0.1).any(): continue
                    ds=[xy for xy,cf in dets_of(m,comp) if cf>=CONF]
                    for g in GTS:
                        gt=bbox_at(A,g); best=max([iou(xy,gt) for xy in ds]+[0.0]) if gt else 0.0
                        hits[g].append(1 if best>=0.5 else 0)
            if hits[0.1]: rows.append((src,scene,scale,lname,*[float(np.mean(hits[g])) for g in GTS]))
    if (si+1)%5==0: print(f'  ...{si+1}/{len(sources)}')
agg=defaultdict(list)
for r in rows: agg[(r[0],r[2],r[3])].append(r[4:])
print(f'\n{"src":7}{"scale":>6}  {"level":12}{"GT>.1":>7}{"GT>.3":>7}{"GT>.5":>7}{"scenes":>7}')
for k in sorted(agg):
    v=np.array(agg[k]); print(f'{k[0]:7}{k[1]:>6}  {k[2]:12}{v[:,0].mean():>7.3f}{v[:,1].mean():>7.3f}{v[:,2].mean():>7.3f}{len(v):>7}')
print('\n※ GT 조일수록(.1→.5) rec@.5 회복 = 대형 저recall이 GT아티팩트. GT>.1이 CELL32 재현되는지 sanity. over>screen 순위 전 GT서 유지 확인.')


# ========== CELL 36: 검출 vs 위치 재집계 — rec@IoU0.5(위치) vs rec@IoU0.1(검출) + cov↔Δ 상관 ==========
# 리뷰 point3: 전 결론(over>screen·scale·spill)이 IoU0.5 기반. IoU0.1(검출됐나)로 재집계해 순위 유지 확인.
#   + cov↔Δ(=rec@.1−rec@.5) 상관: 성긴(low cov) 소스일수록 Δ 커야 "성김→헐렁GT→위치오류" 가설 지지.
# any(배경FP 섞임) 대신 IoU≥0.1(불꽃 겹친 검출=bg FP 배제). GT는 alpha>0.1 그대로(임계는 별도 변수).
import os, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
import numpy as np
from PIL import Image
from collections import defaultdict
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'
W=f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; BG_ROOT=f'{DR}/realneg_frames/synth'
VFXB=f'{FSRC}/vfx_bank'; NISTB=f'{FSRC}/nist_bank'; OUT=f'{DR}/synth_sweep'
SCALES=[64,128,256]; SEEDS=3; CONF=0.25
def screen(a,b): return 255.0-(255.0-a)*(255.0-b)/255.0
def ropen(p):
    for q in (p, unicodedata.normalize('NFC',p), unicodedata.normalize('NFD',p)):
        try: return Image.open(q).convert('RGB')
        except: pass
    return None
def core_lum(rgba):
    a=rgba[...,3]/255.; lum=0.299*rgba[...,0]+0.587*rgba[...,1]+0.114*rgba[...,2]; mm=a>0.5
    return float(np.percentile(lum[mm],98)) if mm.any() else 200.0
def composite(bg_pil, flame, point, scale_px, anchor_frac, blend='over', spill=False, hard=False):
    bg=np.asarray(bg_pil).astype(np.float32); H,Wd=bg.shape[:2]
    fr=Image.fromarray(flame); tw=max(1,int(fr.width*scale_px/fr.height))
    flr=np.asarray(fr.resize((tw,scale_px))).astype(np.float32); fh,fw=flr.shape[:2]
    ax=fw//2; ay=int(anchor_frac*(fh-1)); px=int(point[0]-ax); py=int(point[1]-ay)
    cx,cy=px+fw//2,py+fh//2; out=bg.copy()
    if spill:
        yy,xx=np.mgrid[0:H,0:Wd]; d2=(xx-cx)**2+(yy-cy)**2; r0=fh*0.3
        inten=(core_lum(flame)/255.)*(r0*r0/(d2+r0*r0))*0.7
        out=np.clip(out+np.dstack([inten*255,inten*140,inten*40]).astype(np.float32),0,255)
    x0c,y0c=max(0,px),max(0,py); xe=min(Wd,px+fw); ye=min(H,py+fh)
    fx0,fy0=x0c-px,y0c-py; rw,rh=xe-x0c,ye-y0c; A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; rgb=reg[...,:3]; fa=reg[...,3:4]/255.
        al=np.ones_like(fa) if hard else fa
        dst=out[y0c:y0c+rh,x0c:x0c+rw]; bl=screen(dst,rgb) if blend=='screen' else rgb
        out[y0c:y0c+rh,x0c:x0c+rw]=dst*(1-al)+bl*al; A[y0c:y0c+rh,x0c:x0c+rw]=fa[...,0]
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)), gt
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def dets_of(mm,pil):
    r=mm.predict(pil,conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    return list(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))
def load_bank(bd,filt=None):
    out=[]
    for r in csv.DictReader(open(f'{bd}/manifest.csv')):
        if filt and not filt(r): continue
        out.append((r['source'],r['scene_id'],f'{bd}/{r["matte"]}',float(r['anchor_frac']),int(r['h'])))
    return out
vfx=load_bank(VFXB); bs={}
for s in vfx: bs.setdefault(s[1],s)
vfx=list(bs.values()); nist=load_bank(NISTB, filt=lambda r:r.get('equip_flag')!='Y'); sources=vfx+nist
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
print(f'소스 {len(sources)} · 배경 {len(bgs)}'); m=YOLO(W)
randpts={}
for n,im,box in bgs:
    Wd,Hd=im.size; pl=[]
    for sd in range(SEEDS):
        rr=np.random.default_rng(1000*sd+sum(map(ord,n))); pl.append((int(rr.uniform(0.2,0.8)*Wd),int(rr.uniform(0.4,0.7)*Hd)))
    randpts[n]=pl
LEVELS=[('0a_hard_ctx','over',False,'context',True),('over_rand','over',False,'random',False),
        ('over_ctx','over',False,'context',False),('screen_ctx','screen',False,'context',False),
        ('over_ctx_spill','over',True,'context',False)]
rows=[]; covmap={}
for si,(src,scene,mpath,anchor,mh) in enumerate(sources):
    flame=np.asarray(Image.open(mpath).convert('RGBA')); covmap[scene]=float((flame[...,3]/255.>0.3).mean())
    for scale in SCALES:
        if mh<scale: continue
        for lname,blend,spill,pos,hard in LEVELS:
            h50=[];h10=[]
            for n,im,box in bgs:
                pts=[((box[0]+box[2])//2,box[3])] if pos=='context' else randpts[n]
                for point in pts:
                    comp,gt=composite(im,flame,point,scale,anchor,blend,spill,hard)
                    if gt is None: continue
                    ds=[(xy,cf) for xy,cf in dets_of(m,comp) if cf>=CONF]
                    best=max([iou(xy,gt) for xy,cf in ds]+[0.0])
                    h50.append(1 if best>=0.5 else 0); h10.append(1 if best>=0.1 else 0)
            if h50: rows.append((src,scene,scale,lname,float(np.mean(h50)),float(np.mean(h10))))
    if (si+1)%5==0: print(f'  ...{si+1}/{len(sources)}')
agg=defaultdict(list)
for src,scene,scale,lname,r50,r10 in rows: agg[(src,scale,lname)].append((r50,r10))
print(f'\n{"src":7}{"scale":>6}  {"level":16}{"rec@.5":>8}{"rec@.1":>8}{"gap":>7}{"scenes":>7}')
for k in sorted(agg):
    v=agg[k]; r50=np.mean([x[0] for x in v]); r10=np.mean([x[1] for x in v])
    print(f'{k[0]:7}{k[1]:>6}  {k[2]:16}{r50:>8.3f}{r10:>8.3f}{r10-r50:>7.3f}{len(v):>7}')
print('\n=== cov ↔ Δ(rec@.1−rec@.5) 상관 (VFX over_ctx·성김이 위치오류 원인인가) ===')
for scL in (64,128,256):
    pts=[(covmap[sc], r10-r50) for (s,sc,scl,ln,r50,r10) in rows if s=='VFX' and ln=='over_ctx' and scl==scL]
    if len(pts)>=3:
        cs=np.array([p[0] for p in pts]); ds=np.array([p[1] for p in pts]); cc=np.corrcoef(cs,ds)[0,1] if ds.std()>0 else float('nan')
        print(f'  s{scL}: cov-Δ corr={cc:+.2f} (음수=성길수록 Δ↑=가설지지) · Δ mean {ds.mean():.3f} n={len(pts)}')
json.dump(rows, open(f'{OUT}/ablation_iou.json','w'))
print('\n※ rec@.5=위치(원래·CELL32 재현) · rec@.1=검출(느슨) · gap=위치오류로 깎인 양.')
print('  판정: over>screen·scale 순위가 rec@.1서도 유지 → 검출효과(결론 견고). gap 크고 cov-Δ 음상관 → 위치오류=성김/GT아티팩트 확증.')
# ※ 파일 셀 순서 흐트러짐(34→37→36→38) — 커밋 시 32→33→34→36→37→38로 정리 예정. 각 셀 자립형이라 실행엔 무관.


# ========== CELL 38: miss 육안 — @64 검출실패 3분류 + @256 국소화실패 성격 ==========
# @64(진짜 검출병목): (1)배경묻힘(대비=bgL) (2)너무작음(해상도=fh) (3)형태소실(크롭 육안). + miss/hit bgL 대비.
# @256(base 느슨 국소화·2/3): 검출박스가 GT 대비 작나(코어만)/크나(주변까지)/옮겨졌나 — det/GT면적·중심이동.
import os, csv, subprocess, sys, json, numpy as np
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'
m=YOLO(f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt')
seen=set(); vscenes=[]
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')):
    if r['scene_id'] not in seen: seen.add(r['scene_id']); vscenes.append(r)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
def ropen(p):
    try: return Image.open(p).convert('RGB')
    except: return None
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def comp_over(bg, flame, scale, anchor, box):
    bgn=np.asarray(bg).astype(np.float32); H,Wd=bgn.shape[:2]; fr=Image.fromarray(flame)
    tw=max(1,int(fr.width*scale/fr.height)); flr=np.asarray(fr.resize((tw,scale))).astype(np.float32); fh,fw=flr.shape[:2]
    px=int((box[0]+box[2])//2-fw//2); py=int(box[3]-int(anchor*(fh-1)))
    x0,y0=max(0,px),max(0,py); xe,ye=min(Wd,px+fw),min(H,py+fh); fx0,fy0=x0-px,y0-py; rw,rh=xe-x0,ye-y0
    out=bgn.copy(); A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; a=reg[...,3:4]/255.
        out[y0:y0+rh,x0:x0+rw]=out[y0:y0+rh,x0:x0+rw]*(1-a)+reg[...,:3]*a; A[y0:y0+rh,x0:x0+rw]=reg[...,3]/255.
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8)), gt
def dets(pil):
    r=m.predict(pil,conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
    return [] if r.boxes is None else list(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))
def crop(pil, gt, zoom=2.5):
    W,H=pil.size; cx,cy=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2; bw,bh=gt[2]-gt[0],gt[3]-gt[1]
    hw,hh=max(bw*zoom,60)/2,max(bh*zoom,60)/2
    x0,y0=int(max(0,cx-hw)),int(max(0,cy-hh)); x1,y1=int(min(W,cx+hw)),int(min(H,cy+hh))
    return pil.crop((x0,y0,x1,y1)),(x0,y0)
def bglum(bg, gt):
    a=np.asarray(bg).astype(np.float32); lum=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    return float(lum[gt[1]:gt[3], gt[0]:gt[2]].mean()) if gt[3]>gt[1] and gt[2]>gt[0] else 0.0
CONF=0.25
# --- @64: miss/hit bgL 대비 + miss 15 몽타주 ---
m64=[]; bglM=[]; bglH=[]
for r in vscenes:
    fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
    for n,bg,box in bgs[:6]:
        comp,gt=comp_over(bg,fl,64,anc,box)
        if gt is None: continue
        ds=[(xy,cf) for xy,cf in dets(comp) if cf>=CONF]; best=max([iou(xy,gt) for xy,cf in ds]+[0.0]); bl=bglum(bg,gt)
        if best<0.5:
            bglM.append(bl)
            if len(m64)<15: m64.append((r['scene_id'],comp,gt,ds,bl))
        else: bglH.append(bl)
print(f'@64 대비체크: miss bgL {np.mean(bglM):.0f}(n{len(bglM)}) vs hit bgL {np.mean(bglH) if bglH else 0:.0f}(n{len(bglH)}) — miss가 높으면 밝은배경 묻힘 신호')
fig,ax=plt.subplots(3,5,figsize=(18,11)); ax=ax.ravel()
for i,(sid,comp,gt,ds,bl) in enumerate(m64[:15]):
    sub,(ox,oy)=crop(comp,gt); d=ImageDraw.Draw(sub)
    d.rectangle([gt[0]-ox,gt[1]-oy,gt[2]-ox,gt[3]-oy],outline=(0,255,0),width=2); top=0.0
    for xy,cf in ds:
        if cf>=0.10: d.rectangle([xy[0]-ox,xy[1]-oy,xy[2]-ox,xy[3]-oy],outline=(255,0,0),width=1); top=max(top,cf)
    ax[i].imshow(sub); ax[i].axis('off'); ax[i].set_title(f'{sid[:8]} fh{gt[3]-gt[1]} bgL{bl:.0f} det{top:.2f}',fontsize=8)
for j in range(len(m64[:15]),15): ax[j].axis('off')
fig.suptitle('@64 detection-miss (VFX over_ctx) | green=GT red=det>=0.1 | fh=flameHeight bgL=bgLuma -- classify: buried(contrast)/tiny(res)/shapeless',fontsize=10)
plt.tight_layout(); plt.savefig(f'{DR}/synth_sweep/miss64.png',dpi=85,bbox_inches='tight'); plt.show()
# --- @256: 국소화실패(검출됐으나 IoU 0.1~0.5) 8 ---
l256=[]
for r in vscenes:
    fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
    for n,bg,box in bgs[:4]:
        comp,gt=comp_over(bg,fl,256,anc,box)
        if gt is None: continue
        ds=[(xy,cf) for xy,cf in dets(comp) if cf>=CONF]
        if not ds: continue
        xy,cf=max(ds,key=lambda z:z[1]); ii=iou(xy,gt)
        if 0.1<=ii<0.5 and len(l256)<8: l256.append((r['scene_id'],comp,gt,xy,cf,ii))
    if len(l256)>=8: break
fig,ax=plt.subplots(2,4,figsize=(18,9)); ax=ax.ravel()
for i,(sid,comp,gt,xy,cf,ii) in enumerate(l256[:8]):
    sub,(ox,oy)=crop(comp,gt,zoom=1.8); d=ImageDraw.Draw(sub)
    d.rectangle([gt[0]-ox,gt[1]-oy,gt[2]-ox,gt[3]-oy],outline=(0,255,0),width=3)
    d.rectangle([xy[0]-ox,xy[1]-oy,xy[2]-ox,xy[3]-oy],outline=(255,0,0),width=2)
    gta=(gt[2]-gt[0])*(gt[3]-gt[1]); da=(xy[2]-xy[0])*(xy[3]-xy[1])
    sh=(((gt[0]+gt[2])/2-(xy[0]+xy[2])/2)**2+((gt[1]+gt[3])/2-(xy[1]+xy[3])/2)**2)**.5/max(1,gt[3]-gt[1])
    ax[i].imshow(sub); ax[i].axis('off'); ax[i].set_title(f'{sid[:8]} IoU{ii:.2f} det/GT{da/gta:.2f} sh{sh:.2f}',fontsize=8)
for j in range(len(l256[:8]),8): ax[j].axis('off')
fig.suptitle('@256 localization-fail | green=GT red=det | det/GT<1=core-only small, >1=wide · sh=centerShift/GTh',fontsize=10)
plt.tight_layout(); plt.savefig(f'{DR}/synth_sweep/loc256.png',dpi=85,bbox_inches='tight'); plt.show()
print('저장: miss64.png · loc256.png')
print('※ @64 3분류: bgL 높고 불꽃 흐림=묻힘(대비) · fh작고 텍스처뭉갬=해상도 · 불꽃형태 소실=형태.')
print('※ @256: det/GT<1=코어만 작게잡음 · >1=주변까지 넓게 · sh 큼=옮겨짐 → "느슨한 국소화" 성격.')


# ========== CELL 39: D-Fire 박싱 관행 확인 — under-box 3-way 귀속(GT 불일치 / base 약점 / 소스분포) ==========
# under-box(합성서 base가 밝은코어만·det/GT 0.2-0.3)가 (i)우리 GT off (ii)base 약점 (iii)소스분포 불일치 중 무엇?
# ★같은 luma 정의((R>B+30&R>90)|lum>210)를 D-Fire 박스 + 우리 VFX 매트 양쪽에 적용해 bright-ratio 직접비교(cov[alpha]는 정의 달라 폐기). + D-Fire 박스 육안 몽타주(convention).
# D-Fire=/content(리셋 시 재다운로드·api_key·CELL1~3 돌렸으면 재사용). VFX=Drive.
import os, glob, csv, subprocess, sys, numpy as np
for pkg in ('roboflow','PIL','yaml'):
    mod={'PIL':'PIL','yaml':'yaml'}.get(pkg,pkg)
    try: __import__(mod)
    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q',{'PIL':'Pillow','yaml':'pyyaml'}.get(pkg,pkg)],check=True)
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
def flame_mask(rgb):   # ★양쪽 동일 정의(NIST 매트 추출과 같은 임계)
    R,G,B=rgb[...,0].astype(float),rgb[...,1].astype(float),rgb[...,2].astype(float); lum=0.299*R+0.587*G+0.114*B
    return ((R>B+30)&(R>90))|(lum>210)
# --- (1) 우리 VFX bright-ratio (같은 luma 정의·alpha>0.1 bbox 내) ---
VFXB='/content/drive/MyDrive/firecrop_src/vfx_bank'; vbr=[]; seen=set()
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')):
    if r['scene_id'] in seen: continue
    seen.add(r['scene_id']); fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); ys,xs=np.where(fl[...,3]>25)
    if len(xs): vbr.append(float(flame_mask(fl[ys.min():ys.max()+1,xs.min():xs.max()+1,:3]).mean()))
vbr=np.array(vbr)
# --- (2) D-Fire 확보 ---
def find_split():
    if os.path.isdir('/content/D-Fire-1/train/images'):   # 이미 다운된 raw 재사용(재다운 방지)
        import yaml; nm=yaml.safe_load(open('/content/D-Fire-1/data.yaml')).get('names',['fire','smoke'])
        return '/content/D-Fire-1/train', {i:n for i,n in enumerate(nm)}
    for c in ('/content/dfire_fireonly/train','/content/dfire_ptrain/train'):
        if os.path.isdir(c+'/images'): return c, {0:'fire'}
    return None, None
sp, names = find_split()
if sp is None:
    from roboflow import Roboflow
    rf = Roboflow(api_key="너의_로보플로우_키")   # ← CELL 2와 같은 키 (D-Fire /content에 없을 때만·~10분)
    ds = rf.workspace("kyungho-moon").project("d-fire-aqheb-6iyqy").version(1).download("yolov11")
    root = ds.location; sp = f'{root}/train'
    import yaml; nm = yaml.safe_load(open(f'{root}/data.yaml')).get('names',['fire']); names = {i:n for i,n in enumerate(nm)}
fire_idx = next((i for i,n in names.items() if 'fire' in str(n).lower()), 0)
print(f'D-Fire split={sp} · names={names} · fire_idx={fire_idx}')
# --- (3) D-Fire fire 박스 bright-ratio(같은 luma) + 몽타주 케이스 ---
imgs=sorted(glob.glob(f'{sp}/images/*.jpg')+glob.glob(f'{sp}/images/*.png')+glob.glob(f'{sp}/images/*.jpeg'))
import random; random.seed(0); random.shuffle(imgs)   # ★대표성: 정렬-첫N(한 카메라 연속프레임) 편향 제거
cases=[]; dbr=[]; bhs=[]
for ip in imgs:
    lp=f'{sp}/labels/'+os.path.splitext(os.path.basename(ip))[0]+'.txt'
    if not os.path.exists(lp): continue
    boxes=[[float(x) for x in ln.split()[1:5]] for ln in open(lp) if len(ln.split())>=5 and int(float(ln.split()[0]))==fire_idx]
    if not boxes: continue
    im=np.asarray(Image.open(ip).convert('RGB')); H,W=im.shape[:2]
    for cx,cy,bw,bh in boxes:
        x0,y0=max(0,int((cx-bw/2)*W)),max(0,int((cy-bh/2)*H)); x1,y1=min(W,int((cx+bw/2)*W)),min(H,int((cy+bh/2)*H))
        if x1>x0 and y1>y0: dbr.append(float(flame_mask(im[y0:y1,x0:x1]).mean())); bhs.append(y1-y0)
    if len(cases)<30: cases.append((im,boxes))
    if len(dbr)>=400: break
dbr=np.array(dbr); bhs=np.array(bhs)
def q(a): return f'{a.mean():.3f} (Q1/med/Q3 {np.percentile(a,25):.2f}/{np.percentile(a,50):.2f}/{np.percentile(a,75):.2f})'
print(f'\n★같은 luma 정의 bright-ratio 직접비교:')
print(f'  D-Fire fire 박스 (n{len(dbr)}): {q(dbr)}')
print(f'  우리 VFX 매트     (n{len(vbr)}): {q(vbr)}')
print(f'D-Fire 박스높이(px·416): mean {bhs.mean():.0f} · med {np.percentile(bhs,50):.0f} · <64비율 {(bhs<64).mean():.0%}')
# --- (4) 몽타주(D-Fire 박스·육안 정본) ---
fig,ax=plt.subplots(5,6,figsize=(20,17)); ax=ax.ravel()
for i,(im,boxes) in enumerate(cases[:30]):
    H,W=im.shape[:2]; vis=Image.fromarray(im.copy()); d=ImageDraw.Draw(vis)
    for cx,cy,bw,bh in boxes:
        d.rectangle([int((cx-bw/2)*W),int((cy-bh/2)*H),int((cx+bw/2)*W),int((cy+bh/2)*H)],outline=(0,255,0),width=2)
    ax[i].imshow(vis); ax[i].axis('off')
for j in range(len(cases[:30]),30): ax[j].axis('off')
fig.suptitle('D-Fire fire boxes (green=human GT) -- hug bright core(tight) vs wrap full/dim(loose)?',fontsize=12)
plt.tight_layout(); plt.savefig('/content/dfire_boxing.png',dpi=80,bbox_inches='tight'); plt.show()
print('\n※ 3-way 귀속 (수치=정도·육안=의도·둘 다):')
print('  A) 몽타주 D-Fire 박스가 밝은코어 hug(tight) → 우리 GT(alpha0.1 wispy) off·base 정상 = GT 불일치 → Phase B: GT 맞춤')
print('  B) D-Fire 박스 loose(전체·dim) + bright-ratio ≈ VFX → base 전체학습인데 under-box = base 약점 → Phase B: base 재학습')
print('  C) D-Fire 박스 loose + bright-ratio ≫ VFX(D-Fire 조밀) → base 조밀불만 학습·성긴 우리불엔 코어만 = 소스분포 불일치 → Phase B: 성긴 불꽃 학습투입')
print('  ⚠️분모 성격 약간 다름(VFX 매트=black filler·D-Fire 박스=real bg filler) → 정도 비교용·육안 병행.')


# ========== CELL 40: base det 박스 내 bright-ratio 직접측정 — under-box=A+C·B기각 확정 ==========
# 간접추론(det/GT≈VFX bright-ratio) → 직접: base det 박스 안이 밝은불로 꽉 찼나(=D-Fire tight 0.72) vs GT는 wispy(0.25).
# 같은 합성본(scale256·over_ctx)서 [det 박스 bright-ratio] vs [GT 박스 bright-ratio]. det≈0.72면 base가 D-Fire대로 tight 박싱=B기각.
import os, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'
W=f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'; CONF=0.25
def ropen(p):
    for qq in (p, unicodedata.normalize('NFC',p), unicodedata.normalize('NFD',p)):
        try: return Image.open(qq).convert('RGB')
        except: pass
    return None
def flame_mask(rgb):
    R,G,B=rgb[...,0].astype(float),rgb[...,1].astype(float),rgb[...,2].astype(float); lum=0.299*R+0.587*G+0.114*B
    return ((R>B+30)&(R>90))|(lum>210)
def bratio(img, box):
    x0,y0,x1,y1=[int(v) for v in box]; x0,y0=max(0,x0),max(0,y0); x1,y1=min(img.shape[1],x1),min(img.shape[0],y1)
    return float(flame_mask(img[y0:y1,x0:x1]).mean()) if (x1>x0 and y1>y0) else None
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def composite(bg, flame, point, scale, anchor):
    bgn=np.asarray(bg).astype(np.float32); H,Wd=bgn.shape[:2]; fr=Image.fromarray(flame)
    tw=max(1,int(fr.width*scale/fr.height)); flr=np.asarray(fr.resize((tw,scale))).astype(np.float32); fh,fw=flr.shape[:2]
    px=int(point[0]-fw//2); py=int(point[1]-int(anchor*(fh-1)))
    x0,y0=max(0,px),max(0,py); xe,ye=min(Wd,px+fw),min(H,py+fh); fx0,fy0=x0-px,y0-py; rw,rh=xe-x0,ye-y0
    out=bgn.copy(); A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; a=reg[...,3:4]/255.
        out[y0:y0+rh,x0:x0+rw]=out[y0:y0+rh,x0:x0+rw]*(1-a)+reg[...,:3]*a; A[y0:y0+rh,x0:x0+rw]=reg[...,3]/255.
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return np.clip(out,0,255).astype(np.uint8), gt
m=YOLO(W); vman={}
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')): vman.setdefault(r['scene_id'],r)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
det_br=[]; gt_br=[]; mont=[]
for sid,r in vman.items():
    fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
    for n,bg,box in bgs:
        comp,gt=composite(bg,fl,((box[0]+box[2])//2,box[3]),256,anc)
        if gt is None: continue
        res=m.predict(Image.fromarray(comp),conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
        if res.boxes is None or len(res.boxes)==0: continue
        xy=res.boxes.xyxy.cpu().numpy(); cf=res.boxes.conf.cpu().numpy()
        cand=[(xy[i],cf[i]) for i in range(len(cf)) if cf[i]>=CONF and iou(xy[i],gt)>=0.1]   # 불꽃 검출(bg FP 배제)
        if not cand: continue
        dbox,_=max(cand,key=lambda z:z[1]); db=bratio(comp,dbox); gb=bratio(comp,gt)
        if db is not None and gb is not None:
            det_br.append(db); gt_br.append(gb)
            if len(mont)<8: mont.append((comp,gt,dbox,db,gb))
det_br=np.array(det_br); gt_br=np.array(gt_br)
def qq(a): return f'{a.mean():.3f} (med {np.percentile(a,50):.2f}, n{len(a)})'
print('★같은 luma 정의 bright-ratio 직접비교 (같은 합성본·scale256·over_ctx):')
print(f'  base det 박스 내 : {qq(det_br)}   ← D-Fire tight(0.72) 같으면 base가 밝은불 tight 박싱=B기각 확정')
print(f'  우리 GT 박스 내  : {qq(gt_br)}    ← wispy envelope')
print(f'  참고: D-Fire 박스 0.645(med0.72) · VFX 매트 0.246')
fig,ax=plt.subplots(2,4,figsize=(18,9)); ax=ax.ravel()
for i,(comp,gt,dbox,db,gb) in enumerate(mont[:8]):
    pd=25; x0,y0=max(0,gt[0]-pd),max(0,gt[1]-pd); x1,y1=min(comp.shape[1],gt[2]+pd),min(comp.shape[0],gt[3]+pd)
    sub=Image.fromarray(comp[y0:y1,x0:x1].copy()); d=ImageDraw.Draw(sub)
    d.rectangle([gt[0]-x0,gt[1]-y0,gt[2]-x0,gt[3]-y0],outline=(0,255,0),width=3)
    d.rectangle([int(dbox[0])-x0,int(dbox[1])-y0,int(dbox[2])-x0,int(dbox[3])-y0],outline=(255,0,0),width=2)
    ax[i].imshow(sub); ax[i].axis('off'); ax[i].set_title(f'det br{db:.2f} / GT br{gb:.2f}',fontsize=9)
for j in range(len(mont[:8]),8): ax[j].axis('off')
fig.suptitle('green=GT(wispy) red=base det -- det box mostly bright flame? (br=bright-ratio)',fontsize=11)
plt.tight_layout(); plt.savefig(f'{DR}/synth_sweep/det_bratio.png',dpi=85,bbox_inches='tight'); plt.show()
print('\n※ 판정: det≈0.72 & GT≈0.25 → base는 D-Fire대로 밝은불 tight 박싱(정확)·GT가 wispy(loose)=under-box는 A+C·base 약점 아님(확정).')
print('  det도 낮으면(≈GT) → base가 wispy 전체 박싱 → 재해석 필요.')
# ★주의: 위 CELL 40(luma bright-ratio)은 밝은 주방 bg에 오염(GT box matte0.246→composite0.546)=결과 무효. 정본=아래 CELL 40b(alpha·bg오염0). CELL 39도 편향(정렬-첫N 산불1시퀀스)=정본은 39c(대표 random 1박스/img).


# ========== CELL 39c: @64 상대크기 교란 확인 + D-Fire 대표 재샘플(정본) ==========
import os, glob, csv, json, random, numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'
IMGSZ=640   # ablation predict() imgsz 미지정 = ultralytics 기본
def flame_mask(rgb):
    R,G,B=rgb[...,0].astype(float),rgb[...,1].astype(float),rgb[...,2].astype(float); lum=0.299*R+0.587*G+0.114*B
    return ((R>B+30)&(R>90))|(lum>210)
# (0) ★@64 상대크기: 우리 bg 해상도 → imgsz640 실효크기 vs D-Fire (실측: 1920px → scale64 실효 21px·D-Fire 64px→98px·4.6x)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgsz=[]
for n in list(PLACE):
    try: bgsz.append(Image.open(f'{BG_ROOT}/{BG_MAN[n]}').size)
    except: pass
bgmax=float(np.median([max(w,h) for w,h in bgsz])) if bgsz else 0
print(f'우리 배경 long-side 중앙값: {bgmax:.0f}px (샘플 {bgsz[:3]}, n{len(bgsz)})')
for S in (64,128,256):
    eo=S*IMGSZ/bgmax if bgmax else 0; ed=S*IMGSZ/416
    print(f'  scale {S}px → 우리 실효 {eo:.0f}px | D-Fire {S}px→{ed:.0f}px | 배율차 {ed/eo:.1f}x' if eo else '')
# (1) VFX bright-ratio (같은 luma·matte alpha bbox)
vbr=[]; seen=set()
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')):
    if r['scene_id'] in seen: continue
    seen.add(r['scene_id']); fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); ys,xs=np.where(fl[...,3]>25)
    if len(xs): vbr.append(float(flame_mask(fl[ys.min():ys.max()+1,xs.min():xs.max()+1,:3]).mean()))
vbr=np.array(vbr)
# (2) D-Fire 대표 (1박스/이미지·random shuffle=시퀀스 편향 완화)
sp='/content/D-Fire-1/train'
import yaml; nm=yaml.safe_load(open('/content/D-Fire-1/data.yaml')).get('names',['fire','smoke'])
fire_idx=next((i for i,n in enumerate(nm) if 'fire' in str(n).lower()),0)
imgs=glob.glob(f'{sp}/images/*.jpg')+glob.glob(f'{sp}/images/*.png'); random.seed(0); random.shuffle(imgs)
cases=[]; recs=[]
for ip in imgs:
    lp=f'{sp}/labels/'+os.path.splitext(os.path.basename(ip))[0]+'.txt'
    if not os.path.exists(lp): continue
    fb=[[float(x) for x in ln.split()[1:5]] for ln in open(lp) if len(ln.split())>=5 and int(float(ln.split()[0]))==fire_idx]
    if not fb: continue
    im=np.asarray(Image.open(ip).convert('RGB')); H,W=im.shape[:2]
    cx,cy,bw,bh=max(fb,key=lambda b:b[2]*b[3])   # 이미지당 1박스(최대)
    x0,y0=max(0,int((cx-bw/2)*W)),max(0,int((cy-bh/2)*H)); x1,y1=min(W,int((cx+bw/2)*W)),min(H,int((cy+bh/2)*H))
    if x1>x0 and y1>y0: recs.append((float(flame_mask(im[y0:y1,x0:x1]).mean()), y1-y0))
    if len(cases)<30: cases.append((im,fb))
    if len(recs)>=300: break
recs=np.array(recs); dbr=recs[:,0]; bhs=recs[:,1]
def qf(a): return f'{a.mean():.3f} (Q1/med/Q3 {np.percentile(a,25):.2f}/{np.percentile(a,50):.2f}/{np.percentile(a,75):.2f})'
print(f'\n★D-Fire bright-ratio (1박스/img·n{len(dbr)}): {qf(dbr)}  |  VFX (n{len(vbr)}): {qf(vbr)}   (실측 D-Fire med0.72 tight vs VFX 0.246 wispy)')
print(f'D-Fire 박스높이(px·416): med {np.percentile(bhs,50):.0f} · <64 {(bhs<64).mean():.0%} · <32 {(bhs<32).mean():.0%}')
for lab,msk in [('작은박스<64',bhs<64),('큰박스>=64',bhs>=64)]:
    if msk.sum(): print(f'  {lab} (n{int(msk.sum())}): bright-ratio {dbr[msk].mean():.3f}')
fig,ax=plt.subplots(5,6,figsize=(20,17)); ax=ax.ravel()
for i,(im,fb) in enumerate(cases[:30]):
    H,W=im.shape[:2]; vis=Image.fromarray(im.copy()); d=ImageDraw.Draw(vis)
    for cx,cy,bw,bh in fb: d.rectangle([int((cx-bw/2)*W),int((cy-bh/2)*H),int((cx+bw/2)*W),int((cy+bh/2)*H)],outline=(0,255,0),width=2)
    ax[i].imshow(vis); ax[i].axis('off')
for j in range(len(cases[:30]),30): ax[j].axis('off')
fig.suptitle('D-Fire RANDOM (green=GT) -- composition? tight/loose?',fontsize=12)
plt.tight_layout(); plt.savefig('/content/dfire_rand.png',dpi=80,bbox_inches='tight'); plt.show()
print('\n※ 결과: 우리 실효크기 ≪ D-Fire → @64=순수 크기. D-Fire 대표 bright-ratio 0.72(tight) ≫ VFX 0.246 → 우리 GT/소스가 off. 구성 다양(산불+건물+차량·근접주방 아님).')


# ========== CELL 40b: det vs GT 박스 alpha-밀도 (bg 오염 없는 클린 측정·정본) ==========
# CELL40 luma는 밝은 주방 bg 오염. → 불꽃 alpha로: det 박스 평균alpha vs GT 박스. det>GT면 base가 조밀부 박싱=A+C·B기각.
# 실측: det 평균alpha 0.401·고alpha0.397 > GT 0.256·0.250 (+0.145·n391) → under-box=A(GT정의)+C(밀도)·B기각 확정.
import os, csv, subprocess, sys, json, unicodedata
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'],check=True)
from ultralytics import YOLO
import numpy as np
from PIL import Image
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; FSRC=f'{DR}/firecrop_src'
W=f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'; CONF=0.25
def ropen(p):
    for qq in (p, unicodedata.normalize('NFC',p), unicodedata.normalize('NFD',p)):
        try: return Image.open(qq).convert('RGB')
        except: pass
    return None
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
def composite(bg, flame, point, scale, anchor):
    bgn=np.asarray(bg).astype(np.float32); H,Wd=bgn.shape[:2]; fr=Image.fromarray(flame)
    tw=max(1,int(fr.width*scale/fr.height)); flr=np.asarray(fr.resize((tw,scale))).astype(np.float32); fh,fw=flr.shape[:2]
    px=int(point[0]-fw//2); py=int(point[1]-int(anchor*(fh-1)))
    x0,y0=max(0,px),max(0,py); xe,ye=min(Wd,px+fw),min(H,py+fh); fx0,fy0=x0-px,y0-py; rw,rh=xe-x0,ye-y0
    out=bgn.copy(); A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; a=reg[...,3:4]/255.
        out[y0:y0+rh,x0:x0+rw]=out[y0:y0+rh,x0:x0+rw]*(1-a)+reg[...,:3]*a; A[y0:y0+rh,x0:x0+rw]=reg[...,3]/255.
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return np.clip(out,0,255).astype(np.uint8), gt, A
def box_alpha(A, box):
    x0,y0,x1,y1=[int(v) for v in box]; x0,y0=max(0,x0),max(0,y0); x1,y1=min(A.shape[1],x1),min(A.shape[0],y1)
    if x1<=x0 or y1<=y0: return None,None
    reg=A[y0:y1,x0:x1]; return float(reg.mean()), float((reg>0.5).mean())
m=YOLO(W); vman={}
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')): vman.setdefault(r['scene_id'],r)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
dA=[]; gA=[]; dH=[]; gH=[]
for sid,r in vman.items():
    fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
    for n,bg,box in bgs:
        comp,gt,A=composite(bg,fl,((box[0]+box[2])//2,box[3]),256,anc)
        if gt is None: continue
        res=m.predict(Image.fromarray(comp),conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
        if res.boxes is None or len(res.boxes)==0: continue
        xy=res.boxes.xyxy.cpu().numpy(); cf=res.boxes.conf.cpu().numpy()
        cand=[(xy[i],cf[i]) for i in range(len(cf)) if cf[i]>=CONF and iou(xy[i],gt)>=0.1]
        if not cand: continue
        dbox,_=max(cand,key=lambda z:z[1])
        dma,dhi=box_alpha(A,dbox); gma,ghi=box_alpha(A,gt)
        if None not in (dma,gma): dA.append(dma); gA.append(gma); dH.append(dhi); gH.append(ghi)
dA,gA,dH,gH=map(np.array,(dA,gA,dH,gH))
print('★불꽃 alpha 밀도 (bg 오염 없음·n%d):'%len(dA))
print(f'  base det 박스: 평균alpha {dA.mean():.3f} · 고alpha(>0.5)비율 {dH.mean():.3f}')
print(f'  우리 GT 박스 : 평균alpha {gA.mean():.3f} · 고alpha비율 {gH.mean():.3f}')
print(f'  det−GT: 평균alpha {dA.mean()-gA.mean():+.3f} · 고alpha {dH.mean()-gH.mean():+.3f}')
print('※ det>GT(더 조밀) → base가 조밀코어 박싱(D-Fire 컨벤션 정상)·GT는 저alpha wispy 포함 = under-box A(GT정의)+C(밀도)·B기각.')


# ========== CELL 41: 0-b(생성셋) 특성화 — 생성 vs 컴포지팅 비교의 교란 분리 ==========
# 원 질문: 생성모델(0-b·이미지recall 0.809/박스 0.675)보다 나은 합성법 있나. 우리 best over_ctx@256(rec@.1 0.835/rec@.5 0.464).
# 표면 gap이 GT정의냐 품질이냐 분리: 0-b의 (a)GT bright-ratio(사람박스 tight?) (b)해상도→실효크기 (c)recall 재현.
import os, glob, csv, subprocess, sys, numpy as np
for pkg in ('roboflow','PIL','yaml','ultralytics'):
    mod={'PIL':'PIL','yaml':'yaml'}.get(pkg,pkg)
    try: __import__(mod)
    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q',{'PIL':'Pillow','yaml':'pyyaml'}.get(pkg,pkg)],check=True)
from PIL import Image
from ultralytics import YOLO
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
DR='/content/drive/MyDrive'; W=f'{DR}/dfire_runs/fire_ptrain_b79/weights/best.pt'; IMGSZ=640; CONF=0.25
def flame_mask(rgb):
    R,G,B=rgb[...,0].astype(float),rgb[...,1].astype(float),rgb[...,2].astype(float); lum=0.299*R+0.587*G+0.114*B
    return ((R>B+30)&(R>90))|(lum>210)
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix1-ix0),max(0,iy1-iy0)
    inter=iw*ih; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/ua if ua>0 else 0.0
cand=glob.glob('/content/kitchen-fire-noise-poc-*/train')
if cand and os.path.isdir(cand[0]+'/images'): root=os.path.dirname(cand[0])
else:
    from roboflow import Roboflow
    rf=Roboflow(api_key="너의_로보플로우_키")   # ← CELL 12와 같은 키 (0-b /content에 없을 때만)
    root=rf.workspace("kyungho-moon").project("kitchen-fire-noise-poc").version(1).download("yolov11").location
import yaml; nm=yaml.safe_load(open(f'{root}/data.yaml')).get('names',['fire']); fire_idx=next((i for i,n in enumerate(nm) if 'fire' in str(n).lower()),0)
splits=[s for s in ('train','valid','test') if os.path.isdir(f'{root}/{s}/images')]
print(f'0-b root={root} · names={nm} · splits={splits}')
m=YOLO(W); gbr=[]; bhs=[]; ress=[]; img_hit=0; nimg=0; box_hit=0; nbox=0
for s in splits:
    for ip in sorted(glob.glob(f'{root}/{s}/images/*.jpg')+glob.glob(f'{root}/{s}/images/*.png')):
        lp=f'{root}/{s}/labels/'+os.path.splitext(os.path.basename(ip))[0]+'.txt'
        if not os.path.exists(lp): continue
        fb=[[float(x) for x in ln.split()[1:5]] for ln in open(lp) if len(ln.split())>=5 and int(float(ln.split()[0]))==fire_idx]
        if not fb: continue
        im=np.asarray(Image.open(ip).convert('RGB')); H,Wd=im.shape[:2]; nimg+=1; ress.append(max(H,Wd)); gts=[]
        for cx,cy,bw,bh in fb:
            x0,y0=max(0,int((cx-bw/2)*Wd)),max(0,int((cy-bh/2)*H)); x1,y1=min(Wd,int((cx+bw/2)*Wd)),min(H,int((cy+bh/2)*H))
            if x1>x0 and y1>y0: gts.append((x0,y0,x1,y1)); gbr.append(float(flame_mask(im[y0:y1,x0:x1]).mean())); bhs.append(y1-y0)
        res=m.predict(Image.fromarray(im),conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
        dp=[] if res.boxes is None else [res.boxes.xyxy.cpu().numpy()[i] for i in range(len(res.boxes.conf)) if res.boxes.conf.cpu().numpy()[i]>=CONF]
        if dp: img_hit+=1
        for g in gts:
            nbox+=1
            if any(iou(xy,g)>=0.5 for xy in dp): box_hit+=1
gbr=np.array(gbr); bhs=np.array(bhs); ress=np.array(ress)
def qf(a): return f'{a.mean():.3f} (med {np.percentile(a,50):.2f})'
print(f'\n0-b(생성셋) 이미지 {nimg}·박스 {nbox} · 이미지recall {img_hit/max(1,nimg):.3f}(기록0.809)·박스recall@.5 {box_hit/max(1,nbox):.3f}(기록0.675)')
# ===== 우리 컴포지팅 정합 측정 (VFX@256 over_ctx·18배경·같은 지표) =====
import json, unicodedata
def ropen(p):
    for qz in (p, unicodedata.normalize('NFC',p), unicodedata.normalize('NFD',p)):
        try: return Image.open(qz).convert('RGB')
        except: pass
    return None
def composite(bg, flame, point, scale, anchor):
    bgn=np.asarray(bg).astype(np.float32); H,Wd=bgn.shape[:2]; fr=Image.fromarray(flame)
    tw=max(1,int(fr.width*scale/fr.height)); flr=np.asarray(fr.resize((tw,scale))).astype(np.float32); fh,fw=flr.shape[:2]
    px=int(point[0]-fw//2); py=int(point[1]-int(anchor*(fh-1)))
    x0,y0=max(0,px),max(0,py); xe,ye=min(Wd,px+fw),min(H,py+fh); fx0,fy0=x0-px,y0-py; rw,rh=xe-x0,ye-y0
    out=bgn.copy(); A=np.zeros((H,Wd),np.float32)
    if rw>0 and rh>0:
        reg=flr[fy0:fy0+rh,fx0:fx0+rw]; a=reg[...,3:4]/255.
        out[y0:y0+rh,x0:x0+rw]=out[y0:y0+rh,x0:x0+rw]*(1-a)+reg[...,:3]*a; A[y0:y0+rh,x0:x0+rw]=reg[...,3]/255.
    ys,xs=np.where(A>0.1); gt=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None
    return np.clip(out,0,255).astype(np.uint8), gt
FSRC=f'{DR}/firecrop_src'; BG_ROOT=f'{DR}/realneg_frames/synth'; VFXB=f'{FSRC}/vfx_bank'
vman={}
for r in csv.DictReader(open(f'{VFXB}/manifest.csv')): vman.setdefault(r['scene_id'],r)
BG_MAN=json.load(open(f'{FSRC}/manifest.json')); PLACE=json.load(open(f'{FSRC}/placement.json'))
bgs=[(n,ropen(f'{BG_ROOT}/{BG_MAN[n]}'),PLACE[n]) for n in sorted(PLACE)]; bgs=[b for b in bgs if b[1]]
o_gbr=[]; o_res=[]; o_ih=0; o_ni=0; o_hit=0
for sid,r in vman.items():
    fl=np.asarray(Image.open(f'{VFXB}/{r["matte"]}').convert('RGBA')); anc=float(r['anchor_frac'])
    for n,bg,box in bgs:
        comp,gt=composite(bg,fl,((box[0]+box[2])//2,box[3]),256,anc)
        if gt is None: continue
        o_ni+=1; o_res.append(max(comp.shape[:2])); o_gbr.append(float(flame_mask(comp[gt[1]:gt[3],gt[0]:gt[2]]).mean()))
        res=m.predict(Image.fromarray(comp),conf=0.001,iou=0.6,max_det=300,verbose=False)[0]
        dp=[] if res.boxes is None else [res.boxes.xyxy.cpu().numpy()[i] for i in range(len(res.boxes.conf)) if res.boxes.conf.cpu().numpy()[i]>=CONF]
        if dp: o_ih+=1
        if any(iou(xy,gt)>=0.5 for xy in dp): o_hit+=1
o_gbr=np.array(o_gbr); o_res=np.array(o_res)
# ===== 정합 비교 =====
print('\n'+'='*58)
print(f'{"지표":26}{"0-b 생성":>11}{"우리 컴포지팅":>15}')
print(f'{"이미지 recall@0.25":26}{img_hit/max(1,nimg):>11.3f}{o_ih/max(1,o_ni):>15.3f}  <- apples')
print(f'{"박스 recall@IoU0.5":26}{box_hit/max(1,nbox):>11.3f}{o_hit/max(1,o_ni):>15.3f}  <- GT정의 다름(0-b사람/우리alpha0.1)')
print(f'{"GT bright-ratio(composite)":26}{gbr.mean():>11.3f}{o_gbr.mean():>15.3f}  <- 같은 basis(둘다 bg포함)')
print(f'{"해상도 long med":26}{np.percentile(ress,50):>11.0f}{np.percentile(o_res,50):>15.0f}')
print(f'{"실효 불꽃 med(px)":26}{np.percentile(bhs,50)*IMGSZ/np.percentile(ress,50):>11.0f}{85:>15}')
print('='*58)
print('※ 이미지recall 대등 → "생성>컴포지팅" 착시. 박스 gap=GT정의(bright-ratio로). 실효크기 차=검출 교란.')
print('⚠️ unpaired=두 테스트셋(다른 배경·불꽃). 진짜 paired(같은 배경 생성 vs 컴포지팅)는 미실시.')


# ========== CELL 45 (발표용·정직 before/after): 같은 프리즈 base — 불 없는 장면(검출0) vs 합성 불꽃(검출N) ==========
# [목적] 팀 발표용 정직한 대안 이미지. 팀원이 원한 "기성 base 0검출 → 파인튜닝 N검출" 형식은 정직하게 못 만듦:
#   ① 파인튜닝 모델 없음((B) = 실 급식실 화재 데이터 부재로 봉쇄)  ② 프리즈 base가 합성을 이미 잘 잡음(recall 0.809~0.994) → "기성 0검출"이 거짓(만들면 조작).
# [정직한 재프레이밍] 바뀐 변수 = 모델(기성 vs 파인튜닝)이 아니라 **장면(불 없음 vs 합성 조리유불)**. 모델은 두 컷 다 동일한 프리즈 base(파인튜닝 0).
#   → 우리 (A) 프록시의 시각화: "파인튜닝 없이도 base가 합성 유류불을 인식"(필요조건). 실전 검출 성능 증명 아님((B)).
#   → "0검출 → N검출" 시각 효과는 유지하되, 축이 '모델'이 아니라 '장면'이라 데이터 조작이 아님.
# [설계] CELL 28 페어(맨배경 vs 불꽃추가)를 그대로(같은 검증된 함수·클린 NIST 조리유 KEEP6) + 출력만 발표용 몽타주(CELL 38 스타일 box+conf).
# [⚠️공유] 배경 = realneg_frames/synth = 학교 CCTV(외부공유 제한·HANDOFF). **내부 발표용.** 외부 공유 시 배경 블러/교체 필요(불꽃 NIST=퍼블릭도메인은 무관).
# [스코프] scale 0.25 = 충분크기(recall~0.994) 체제만 — 소형(조기)불 크기한계는 별건(13번/CELL27 0.994→0.694). 프리즈 프록시(실전 아님).
import os, glob, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib, matplotlib.pyplot as plt, matplotlib.patches as patches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'   # 조리유 불꽃(퍼블릭도메인·공유가능)
BG  = '/content/drive/MyDrive/realneg_frames/synth'                 # 학교 CCTV(외부공유 제한)
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 0; FS = 0.25; CONF = 0.25; IOU_ON = 0.3; N_SCAN = 200; PAIRS = 6
KEEP = ['1574198232-Evt3', '1574198232-EvtP', '1574199884-Evt3', '1574199884-EvtP',
        '1508954077-EvtP', '1508958465-EvtP']

def _set_ko_font():   # 한글 폰트(실패시 T()가 영문 라벨로 폴백 — 렌더 tofu 방지)
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts():
        if any(k in f.lower() for k in ('nanum', 'malgun', 'notosanscjk', 'notosanskr', 'notosans-cjk')):
            try:
                fm.fontManager.addfont(f); matplotlib.rc('font', family=fm.FontProperties(fname=f).get_name())
                matplotlib.rcParams['axes.unicode_minus'] = False; return True
            except Exception: pass
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'koreanize-matplotlib'], check=True)
        import koreanize_matplotlib; return True
    except Exception: return False
KO = _set_ko_font()
def T(ko, en): return ko if KO else en

# --- 헬퍼 (CELL 28과 동일·검증됨) ---
def extract_flame(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8))

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        if not any(k in os.path.basename(p) for k in KEEP): continue
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl = extract_flame(p)
        if fl is None or max(fl.size) < 60: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def boxes(m, pil):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return np.zeros((0, 4)), np.zeros(0)
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

flames = load_flames(); assert flames, '클린 불꽃 0 (KEEP/SRC 확인)'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); assert bgs, '배경 0 (BG 확인)'
m = YOLO(W); rng = np.random.default_rng(SEED)
sel = [bgs[i] for i in rng.choice(len(bgs), size=min(N_SCAN, len(bgs)), replace=False)]
print(f'뱅크 {len(flames)}종 · 스캔 배경 {len(sel)}장 · conf{CONF} · scale{FS} · font_ko={KO}')

# 스캔: 발표 페어 수집(맨배경 클린 & 합성 검출) + 정직 집계(전수 — FP/recall 숨기지 않음)
pairs = []; n_bare_clean = 0; n_bare_fp = 0; hits = 0; confs = []
for bp in sel:
    bg = Image.open(bp).convert('RGB'); Wd, Hd = bg.size
    xyb, cfb = boxes(m, bg)                                     # (1) 맨 배경 = "불 없음"(before)
    bare_clean = not bool((cfb >= CONF).any())
    n_bare_clean += int(bare_clean); n_bare_fp += int(not bare_clean)
    nm, fl = flames[int(rng.integers(len(flames)))]
    th = max(1, int(Hd*FS)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
    px = int(Wd*rng.uniform(0.15, 0.85) - tw/2); py = int(Hd*rng.uniform(0.35, 0.75) - th/2)
    comp, gt = paste(bg, fl_r, px, py)                         # (2) 합성 불꽃 = "불 있음"(after)
    xyc, cfc = boxes(m, comp)
    match = [(xyc[i], cfc[i]) for i in range(len(cfc)) if cfc[i] >= CONF and gt and iou(xyc[i], gt) >= IOU_ON]
    on = len(match) > 0; hits += int(on)
    if on: confs.append(max(cf for _, cf in match))
    if bare_clean and on and len(pairs) < PAIRS:
        pairs.append((bp, bg, comp, gt, match, nm))

n = len(sel)
print(f'\n=== 정직 집계 (전수 n={n}) ===')
print(f'  맨배경 base 침묵(검출0)  : {n_bare_clean}/{n} = {n_bare_clean/n:.1%}   ← "before" = 불 없으면 base 조용(정상)')
print(f'  맨배경 헛불(FP≥{CONF})    : {n_bare_fp}/{n} = {n_bare_fp/n:.1%}   ← FP는 배경 색혼동(별건·CELL28/§견고성)')
print(f'  합성 불꽃 검출(recall)   : {hits}/{n} = {hits/n:.1%}   ← "after" = 파인튜닝 없이도 base가 인식')
if confs: print(f'  검출 conf 평균(hit)      : {np.mean(confs):.2f} (n{len(confs)})')
print(f'  → 발표 페어 {len(pairs)}쌍 수집(맨배경 클린 & 합성 검출)')

# 발표용 렌더: (1) 개별 페어 PNG(슬라이드 삽입용·고DPI·정직 캡션 내장) + (2) 오버뷰 몽타주  (초록=합성불 GT · 빨강=base 검출)
def _draw_after(a, comp, gt, match):
    a.imshow(comp); a.axis('off')
    a.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1], fill=False, edgecolor='lime', lw=1.5))
    for xy, cf in match:
        a.add_patch(patches.Rectangle((xy[0], xy[1]), xy[2]-xy[0], xy[3]-xy[1], fill=False, edgecolor='red', lw=2))
    return max(cf for _, cf in match)

K = len(pairs); saved = []
CAP = T('동일 프리즈 검출기(파인튜닝 0) · 바뀐 것은 장면(불 없음→합성 조리유불)이지 모델이 아님',
        'Same frozen detector (0 fine-tuning) · the SCENE changed (no fire -> synthetic oil fire), not the model')
# (1) 개별 페어 — 슬라이드에 하나씩 얹기 좋음(정직 캡션 내장 → 이미지가 맥락과 분리돼도 오독 방지)
for i, (bp, bg, comp, gt, match, nm) in enumerate(pairs, 1):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].imshow(bg); ax[0].axis('off'); ax[0].set_title(T('불 없음 · base 검출 0', 'no fire · base 0 det'), fontsize=13)
    best = _draw_after(ax[1], comp, gt, match)
    ax[1].set_title(T(f'합성 조리유불 · base 검출 conf {best:.2f}', f'synthetic oil fire · base det conf {best:.2f}'), fontsize=13)
    fig.text(0.5, 0.03, CAP, ha='center', fontsize=9, style='italic')
    p = f'{OUT}/honest_pair_{i}.png'; plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); saved.append(p)
# (2) 오버뷰 몽타주 — 어느 페어가 좋은지 [번호]로 한눈에 → 대응 honest_pair_[번호].png 를 슬라이드에
if K:
    fig, ax = plt.subplots(K, 2, figsize=(11, K*3.3)); ax = np.array(ax).reshape(K, 2)
    for i, (bp, bg, comp, gt, match, nm) in enumerate(pairs):
        ax[i, 0].imshow(bg); ax[i, 0].axis('off')
        ax[i, 0].set_title(T(f'[{i+1}] 불 없음 · 검출 0', f'[{i+1}] no fire · 0 det'), fontsize=11)
        best = _draw_after(ax[i, 1], comp, gt, match)
        ax[i, 1].set_title(T(f'[{i+1}] 합성 조리유불 · 검출 conf {best:.2f}', f'[{i+1}] synthetic oil fire · det conf {best:.2f}'), fontsize=11)
    sup = T('같은 프리즈 base(파인튜닝 0) — 장면만 바뀜: 불 없음(검출0) → 합성 조리유불(검출N)\n(A) 프록시: 파인튜닝 없이도 base가 합성 불꽃 인식 · 초록=합성불 위치 빨강=base 검출 · 실전 성능 아님((B) 봉쇄)',
            'Same frozen base (0 fine-tuning) - only the SCENE changes: no fire (0 det) -> synthetic oil fire (N det)\n(A) proxy: base recognizes synthetic flame without fine-tuning · green=GT red=base det · not real-world perf ((B) blocked)')
    fig.suptitle(sup, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(f'{OUT}/honest_beforeafter.png', dpi=130, bbox_inches='tight'); plt.show()
    print('\n저장:')
    print(f'  오버뷰 몽타주 : {OUT}/honest_beforeafter.png  (어느 페어 좋은지 [번호]로 고르기)')
    for p in saved: print(f'  슬라이드용    : {p}')
    print('  → 발표엔 개별 honest_pair_[번호].png 를 슬라이드에 삽입(정직 캡션 내장). 몽타주는 오버뷰용.')
else:
    print('  (페어 0 — N_SCAN↑ 또는 CONF/FS 조정)')

print('\n※ 정직 캡션(발표 이미지 밑에 붙일 문구):')
print('   "동일한 프리즈 검출기(파인튜닝 0). 왼쪽=불 없음→검출 0, 오른쪽=합성 조리유불→검출.')
print('    바뀐 것은 모델이 아니라 장면이다. 파인튜닝 없이도 base가 합성 불꽃을 인식함을 보인다(필요조건·프록시).')
print('    실전 검출 성능은 별개이며 실 급식실 화재 데이터 부재로 미검증((B))."')
print('※ 배경=학교 CCTV → 내부 발표용. 외부 공유 시 배경 블러/교체(불꽃 NIST=퍼블릭도메인은 공유 무관).')
print('※ 더 사실적 접지(조리면 위)를 원하면: placement.json 18배경 + CELL 38 comp_over(anchor 접지)로 교체 가능(별도 요청).')


# ========== CELL 46 (발표 13번용·크기 사다리): 실사 배경 + 불꽃 크기↓ → 검출 박스 사라지는 지점 ==========
# [목적] 섹션 13 "부러지는 지점"을 실사+박스로. 같은 급식실 장면·같은 불꽃을 크기만 줄이며 base 검출 → 큰 불 검출·작을수록 약해지다 놓침.
# [설계] 열=장면(고정)·행=크기. 불꽃 base(하단)를 장면마다 한 점에 고정 → 크기가 유일 변수(한 변수). 패널=검출 conf 또는 "놓침".
#   집계 recall은 여기서 재측정 안 함(§13 확정치 0.994/0.694와 경쟁 숫자 회피) — 이 그림은 *예시 시각화*, 집계는 §13 차트가 담당.
# [공유] 배경=학교CCTV(내부용). 불꽃=NIST 조리유 KEEP6(퍼블릭도메인). 프리즈 프록시(실전 아님). CELL 45와 동일 검증 함수.
import os, glob, subprocess, sys, hashlib
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib, matplotlib.pyplot as plt, matplotlib.patches as patches

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W   = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
SRC = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG  = '/content/drive/MyDrive/realneg_frames/synth'
OUT = '/content/drive/MyDrive/synth_sweep'; os.makedirs(OUT, exist_ok=True)
SEED = 1; CONF = 0.25; IOU_ON = 0.3; SCALES_L = [0.40, 0.25, 0.11]; N_SHOW = 5
KEEP = ['1574198232-Evt3', '1574198232-EvtP', '1574199884-Evt3', '1574199884-EvtP',
        '1508954077-EvtP', '1508958465-EvtP']

def _set_ko_font():   # 한글 폰트(실패시 T()가 영문 폴백)
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts():
        if any(k in f.lower() for k in ('nanum', 'malgun', 'notosanscjk', 'notosanskr', 'notosans-cjk')):
            try:
                fm.fontManager.addfont(f); matplotlib.rc('font', family=fm.FontProperties(fname=f).get_name())
                matplotlib.rcParams['axes.unicode_minus'] = False; return True
            except Exception: pass
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'koreanize-matplotlib'], check=True)
        import koreanize_matplotlib; return True
    except Exception: return False
KO = _set_ko_font()
def T(ko, en): return ko if KO else en

# --- 헬퍼 (CELL 45/28과 동일·검증됨) ---
def extract_flame(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]; lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, _ = ndimage.label(mask); c = np.bincount(lbl.ravel()); c[0] = 0
    m = ndimage.binary_dilation(lbl == c.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0 = max(0, xs.min()-pad); y0 = max(0, ys.min()-pad)
    x1 = min(im.shape[1]-1, xs.max()+pad); y1 = min(im.shape[0]-1, ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160., 0, 1)*mm*255]).astype(np.uint8))

def load_flames():
    out, seen = [], set()
    for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
        if not any(k in os.path.basename(p) for k in KEEP): continue
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if md5 in seen: continue
        fl = extract_flame(p)
        if fl is None or max(fl.size) < 60: continue
        seen.add(md5); out.append((os.path.basename(p).split('__')[0], fl))
    return out

def paste(bg_img, fl_rgba, px, py):
    bg = np.asarray(bg_img.convert('RGB')).astype(np.float32); H, W_ = bg.shape[:2]
    fl = np.asarray(fl_rgba).astype(np.float32); fh, fw = fl.shape[:2]
    x0c, y0c = max(0, px), max(0, py); x1 = min(W_, px+fw); y1 = min(H, py+fh)
    fx0, fy0 = x0c-px, y0c-py; rw, rh = x1-x0c, y1-y0c
    out = bg.copy(); A = np.zeros((H, W_), np.float32)
    if rw > 0 and rh > 0:
        reg = fl[fy0:fy0+rh, fx0:fx0+rw]; a = reg[..., 3:4]/255.
        out[y0c:y0c+rh, x0c:x0c+rw] = out[y0c:y0c+rh, x0c:x0c+rw]*(1-a) + reg[..., :3]*a
        A[y0c:y0c+rh, x0c:x0c+rw] = reg[..., 3]/255.
    ys, xs = np.where(A > 0.1)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), box

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def boxes(m, pil):
    r = m.predict(pil, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return np.zeros((0, 4)), np.zeros(0)
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

def comp_scale(bg, fl, fs, cx, cy):   # 불꽃 base(하단)를 (cx,cy)에 고정 → 크기만 변수
    Wd, Hd = bg.size
    th = max(1, int(Hd*fs)); tw = max(1, int(fl.width*th/fl.height)); fl_r = fl.resize((tw, th))
    px = int(Wd*cx - tw/2); py = int(Hd*cy - th)
    return paste(bg, fl_r, px, py)

flames = load_flames(); assert flames, '클린 불꽃 0 (KEEP/SRC 확인)'
bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True)); assert bgs, '배경 0 (BG 확인)'
m = YOLO(W); rng = np.random.default_rng(SEED)
sel = [bgs[i] for i in rng.choice(len(bgs), size=min(N_SHOW, len(bgs)), replace=False)]
scenes = []   # 장면 고정(배경·불꽃·배치점) — 크기만 바꿈
for bp in sel:
    bg = Image.open(bp).convert('RGB'); nm, fl = flames[int(rng.integers(len(flames)))]
    cx = float(rng.uniform(0.30, 0.70)); cy = float(rng.uniform(0.62, 0.82))
    scenes.append((bg, fl, cx, cy, nm))
print(f'뱅크 {len(flames)}종 · 장면 {len(scenes)} · 크기 {SCALES_L} · conf{CONF} · font_ko={KO}')

R, Cn = len(SCALES_L), len(scenes)
fig, ax = plt.subplots(R, Cn, figsize=(3.2*Cn, 3.0*R)); ax = np.array(ax).reshape(R, Cn)
for ri, fs in enumerate(SCALES_L):
    ndet = 0
    for ci, (bg, fl, cx, cy, nm) in enumerate(scenes):
        comp, gt = comp_scale(bg, fl, fs, cx, cy); a = ax[ri, ci]; a.imshow(comp)
        det = None
        if gt is not None:
            a.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1], fill=False, edgecolor='lime', lw=1.0))
            xy, cf = boxes(m, comp)
            cand = [(xy[i], cf[i]) for i in range(len(cf)) if cf[i] >= CONF and iou(xy[i], gt) >= IOU_ON]
            if cand:
                xyb, cfb = max(cand, key=lambda z: z[1]); det = float(cfb)
                a.add_patch(patches.Rectangle((xyb[0], xyb[1]), xyb[2]-xyb[0], xyb[3]-xyb[1], fill=False, edgecolor='red', lw=2))
        ndet += int(det is not None)
        a.set_title(T(f'검출 {det:.2f}', f'det {det:.2f}') if det is not None else T('놓침', 'miss'),
                    fontsize=10, color=('black' if det is not None else 'crimson'))
        if ci == 0:
            a.set_xticks([]); a.set_yticks([]); a.set_ylabel(T(f'크기 {fs:.2f}', f'scale {fs:.2f}'), fontsize=13)
            for s in a.spines.values(): s.set_visible(False)
        else:
            a.axis('off')
    print(f'  크기 {fs:.2f}: 이 그림 {Cn}장 중 {ndet} 검출')
sup = T('같은 급식실 장면·같은 불꽃, 크기만 축소 — 큰 불 검출 · 작을수록 약해지다 놓침 (크기가 유일 변수)\n초록=합성불 위치 · 빨강=base 검출 · 집계 recall: 크기≥0.25 → 0.994 · 0.11 → 0.694 (§13, 이 그림은 예시)',
        'Same scene & flame, only size shrinks — large detected, smaller weakens then missed (size = only variable)\ngreen=GT · red=base det · aggregate recall: scale>=0.25 -> 0.994 · 0.11 -> 0.694 (§13; this figure is illustrative)')
fig.suptitle(sup, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f'{OUT}/scale_ladder.png', dpi=130, bbox_inches='tight'); plt.show()
print(f'\n저장: {OUT}/scale_ladder.png  (섹션 13용·실사+박스)')
print('※ 예시 시각화(집계 아님) — 크기 0.11 열에 검출/놓침이 섞이면 그게 recall 0.694(불확실해지는 지점)의 육안 표현.')
print('※ 배경=학교 CCTV → 내부 발표용.')
