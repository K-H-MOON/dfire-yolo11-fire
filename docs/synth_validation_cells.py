# 합성 데이터 검증 셀 (파이프라인 2번) — base=ptrain_b79(D-Fire 학습). CELL 1(Drive) 이후 실행.
# CELL 12=합성 양성 recall/precision · CELL 13=합성 음성 헛불률(B). D-Fire baseline은 dfire_baseline_cells.py.

# ========== CELL 12: 합성 데이터 검증 — base(ptrain_b79)를 라벨된 합성셋에 평가 ==========
# 합성 = Roboflow kyungho-moon/kitchen-fire-noise-poc v1 (351장·fire box). base는 D-Fire만 학습
# → 합성은 완전 fresh held-out(누수 0). recall/precision 높으면 = 합성이 실제-학습 검출기에 불처럼 보임(리얼).
# 전제: CELL 1(Drive) — best.pt Drive에 있음. api_key 필요.
import subprocess, sys, os, glob, shutil
for pkg in ('roboflow','ultralytics','yaml'):
    mod='yaml' if pkg=='yaml' else pkg
    try: __import__(mod)
    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','pyyaml' if pkg=='yaml' else pkg], check=True)
from roboflow import Roboflow
from ultralytics import YOLO
import numpy as np, yaml

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'Drive 마운트 필요(CELL 1) — best.pt 없음: '+W

rf=Roboflow(api_key="너의_로보플로우_키")                    # ← 네 진짜 키
proj=rf.workspace("kyungho-moon").project("kitchen-fire-noise-poc")
ds=proj.version(1).download("yolov11")
root=ds.location
print('합성셋:', root)

cfg=yaml.safe_load(open(f'{root}/data.yaml'))
names=cfg.get('names')
names={int(k):v for k,v in names.items()} if isinstance(names,dict) else {i:n for i,n in enumerate(names)}
fire_ids={i for i,n in names.items() if 'fire' in str(n).lower() or str(n) in ('불','flame')}
print('클래스:', names, '· fire ids:', fire_ids)
assert fire_ids, 'fire 클래스 못 찾음 — names 확인'

# 전 split(train/valid/test) 병합 + fire만 class 0 정규화 → base엔 전부 처음 보는 held-out
OUT='/content/synth_eval'
if os.path.isdir(OUT): shutil.rmtree(OUT)
os.makedirs(f'{OUT}/images'); os.makedirs(f'{OUT}/labels')
n_img=n_box=0
for split in ('train','valid','val','test'):
    idir=f'{root}/{split}/images'
    if not os.path.isdir(idir): continue
    for ip in glob.glob(f'{idir}/*'):
        nm=os.path.basename(ip); stem=os.path.splitext(nm)[0]
        dst=f'{OUT}/images/{nm}'
        if os.path.exists(dst): continue
        try: os.symlink(os.path.realpath(ip),dst)
        except Exception: shutil.copy(ip,dst)
        lines=[]
        lp=f'{root}/{split}/labels/{stem}.txt'
        if os.path.exists(lp):
            for ln in open(lp):
                t=ln.split()
                if len(t)>=5:
                    try: c=int(float(t[0]))
                    except ValueError: continue
                    if c in fire_ids: lines.append('0 '+' '.join(t[1:5]))
        with open(f'{OUT}/labels/{stem}.txt','w') as f:
            if lines: f.write('\n'.join(lines)+'\n')
        n_img+=1; n_box+=len(lines)
open(f'{OUT}/data.yaml','w').write(f"path: {OUT}\ntrain: images\nval: images\ntest: images\nnc: 1\nnames: ['fire']\n")
print(f'평가셋 병합: {n_img}장 · fire박스 {n_box}')

# 박스 단위 평가 (D-Fire와 동일 기준)
def c1(a):
    a=np.asarray(a,float); return a[0] if a.ndim==2 else a
m=YOLO(W)
b=m.val(data=f'{OUT}/data.yaml', split='test', iou=0.5, conf=0.001, plots=False, verbose=False).box
R=c1(b.r_curve); P=c1(b.p_curve); F1=c1(b.f1_curve); i=int(np.argmax(F1))
idx=np.where(R>=0.80)[0]; p80=(float(P[int(idx[-1])]) if len(idx) else float('nan'))
print('\n=== 합성 검증 (base ptrain_b79 · 박스단위 iou=0.5) ===')
print(f'  mAP50 {b.map50:.3f} · recall천장 {R.max():.3f} · P@F1 {P[i]:.3f} · R@F1 {R[i]:.3f} · P@R0.8 {p80:.3f} · mAP50-95 {b.map:.3f}')

# 이미지 단위 recall (발화율 · 박스 위치 무관)
imgs=glob.glob(f'{OUT}/images/*'); mc=[]
for ip in imgs:
    r=m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    mc.append(float(r.boxes.conf.max()) if (r.boxes is not None and len(r.boxes)) else 0.0)
mc=np.array(mc)
print('  이미지단위 recall(발화율):', ' · '.join(f'conf≥{C}: {(mc>=C).mean():.3f}' for C in (0.05,0.25,0.50)))
print(f'\n※ 비교 기준 = D-Fire 정직 baseline recall천장 0.894·mAP50 0.660.')
print('  합성 recall천장/mAP이 이에 근접·이상 = 합성이 실제-학습 base에 불처럼 보임(리얼). 낮음 = 합성 off-distribution.')
print('  (이 351장 다 양성 → recall+박스precision. 빈 급식실 오탐은 음성셋 있어야 별도.)')

# ========== CELL 13: 합성 음성셋 헛불률(B) 측정 — base(ptrain_b79)를 불 없는 급식실 합성에 ==========
# 정답=불없음 이미지에 모델 돌려 발화하면 = 헛불(FP). conf별 헛불률 + '무엇에 찍었나' 몽타주.
# 전제: CELL 1(Drive) — best.pt. 음성 28장을 Colab에 올려두고 NEG_DIR 지정.
import subprocess, sys, os, glob
try: import ultralytics
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'Drive 마운트 필요(CELL 1) — best.pt 없음: '+W
NEG_DIR='/content/drive/MyDrive/synth_neg'          # ← 업로드한 음성 폴더 경로
assert os.path.isdir(NEG_DIR), '음성 폴더 없음 — 업로드 후 경로 지정: '+NEG_DIR

EXTS=('*.jpg','*.jpeg','*.png','*.webp')
imgs=[]
for e in EXTS: imgs+=glob.glob(os.path.join(NEG_DIR,e))+glob.glob(os.path.join(NEG_DIR,e.upper()))
imgs=sorted(set(imgs))
print(f'음성 이미지 {len(imgs)}장')
assert imgs, 'NEG_DIR에 이미지 없음'

m=YOLO(W)
res=[]   # (path, topconf, xyxy, confs)
for ip in imgs:
    r=m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        cf=r.boxes.conf.cpu().numpy(); xy=r.boxes.xyxy.cpu().numpy()
        res.append((ip, float(cf.max()), xy, cf))
    else:
        res.append((ip, 0.0, np.zeros((0,4)), np.zeros(0)))
tc=np.array([r[1] for r in res]); N=len(res)

print('\n=== 헛불률(B) · 정답=불없음 이미지에서 발화 비율 ===')
print(f'{"conf":>6}{"헛불률":>10}{"헛불장수":>12}   (참고: D-Fire 야외음성)')
ref={0.05:0.081,0.25:0.021,0.50:0.008}
for C in (0.05,0.25,0.50):
    fp=tc>=C; print(f'{C:>6.2f}{fp.mean():>10.3f}{str(int(fp.sum()))+"/"+str(N):>12}   D-Fire {ref[C]:.3f}')
print(f'\n※ N={N} 작음 → 거친 추정. D-Fire 야외음성 헛불률과 대조.')

# 헛불 케이스 몽타주 (conf≥0.25 발화 + 무엇에 찍었나)
C0=0.25
fires=[r for r in res if r[1]>=C0]
print(f'\nconf≥{C0} 헛불 {len(fires)}장 — 모델이 무엇을 불로 오인했나:')
if fires:
    K=len(fires); cols=min(5,K); rows=(K+cols-1)//cols
    fig,ax=plt.subplots(rows,cols,figsize=(3.4*cols,3.4*rows)); ax=np.atleast_2d(ax)
    for i in range(rows*cols):
        a=ax[i//cols,i%cols]; a.axis('off')
        if i>=K: continue
        ip,tcf,xy,cf=fires[i]
        im=Image.open(ip).convert('RGB'); a.imshow(im)
        for (x1,y1,x2,y2),c in zip(xy,cf):
            if c>=C0: a.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=2))
        a.set_title(f'top conf {tcf:.2f} · {os.path.basename(ip)[:14]}',fontsize=7)
    plt.tight_layout(); plt.show()
    print('※ 빨강=모델이 불이라 찍은 곳. 수증기·반사·주황조명 등에 찍혔으면 헛불 원인 확인.')
else:
    print('  conf≥0.25에서 헛불 0장 — 합성 음성에 오경보 안 냄(좋음).')
