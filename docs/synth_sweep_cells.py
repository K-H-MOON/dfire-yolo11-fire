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
nist = load_bank(NISTB, filt=lambda r: r.get('equip_flag') != 'Y' and int(r['h']) >= 256)   # 깨끗 peak만
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
