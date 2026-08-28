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

# ========== CELL 14: 실제 무화재 급식실 헛불률(B) — base(ptrain_b79)를 oilfire_realtest 실음성에 ==========
# oilfire_realtest_share.zip(Drive) 안 nofire_kitchen(실 조리 무화재)+nofire_presrc(대조)에 base 돌려 헛불률.
# 28장 합성 예비치를 실제 데이터로 대체. 전제: CELL 1(Drive) — best.pt·zip 모두 Drive.
import subprocess, sys, os, glob, zipfile
try: import ultralytics
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'Drive 마운트 필요(CELL 1) — best.pt 없음: '+W
EXTS=('.jpg','.jpeg','.png','.webp')

# 1) zip 찾기 — 경로 하드코딩 안 함(MyDrive 재귀 검색). 경로 알면 ZIP_OVERRIDE에 직접.
ZIP_OVERRIDE=''
if ZIP_OVERRIDE and os.path.exists(ZIP_OVERRIDE):
    zpath=ZIP_OVERRIDE
else:
    print('MyDrive에서 oilfire_realtest_share.zip 검색 중...(수십초 가능)')
    hits=glob.glob('/content/drive/MyDrive/**/oilfire_realtest_share.zip', recursive=True)
    assert hits, 'oilfire_realtest_share.zip 못 찾음 — ZIP_OVERRIDE에 경로 지정하거나 파일 위치 확인'
    zpath=hits[0]
print('zip:', zpath)

# 2) zip 내부 구조 확인(추출 전) — 추측 안 하고 실제로 봄
zf=zipfile.ZipFile(zpath)
names=zf.namelist()
tops=sorted(set(n.split('/')[0] for n in names if n.strip('/')))
print(f'zip 항목 {len(names)}개 · 최상위 항목:', tops[:12])
def zimgs(sub): return [n for n in names if f'{sub}/' in n and n.lower().endswith(EXTS)]
print(f'  zip 내 nofire_kitchen 이미지 {len(zimgs("nofire_kitchen"))} · nofire_presrc 이미지 {len(zimgs("nofire_presrc"))}')

# 3) 추출
EXTRACT='/content/oilfire_real'
if os.path.isdir(EXTRACT) and any(os.scandir(EXTRACT)):
    print('이미 추출됨 →', EXTRACT)
else:
    zf.extractall(EXTRACT); print('추출 완료 →', EXTRACT)
zf.close()

# 4) 이미지 폴더 탐색 — 이름 매칭 + 전체 나열로 검증(없으면 멈춤, 추측 금지)
def count_imgs(d): return sorted(os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(EXTS))
targets={}; alldirs=[]
for root,dirs,files in os.walk(EXTRACT):
    ims=[f for f in files if f.lower().endswith(EXTS)]
    if ims:
        alldirs.append((root,len(ims)))
        b=os.path.basename(root).lower()
        if b in ('nofire_kitchen','nofire_presrc'): targets[b]=root
print('\n추출된 이미지 폴더(개수順):')
for d,n in sorted(alldirs, key=lambda x:-x[1])[:15]:
    print(f'  {n:5d}  {d.replace(EXTRACT,"").lstrip("/")}')
assert 'nofire_kitchen' in targets, f'nofire_kitchen 폴더 못 찾음 — 위 목록서 실제 폴더명 확인 후 targets 로직 수정. 발견된 target={list(targets)}'

# 5) base 헛불률 측정
DREF={0.05:0.081,0.25:0.021,0.50:0.008}   # D-Fire 야외음성(참고)
SREF={0.05:0.071,0.25:0.036,0.50:0.000}   # 합성28 예비(참고)
m=YOLO(W)
def fp_eval(name, d):
    imgs=count_imgs(d); res=[]
    for ip in imgs:
        r=m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            res.append((ip, float(r.boxes.conf.max().cpu()), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        else:
            res.append((ip, 0.0, np.zeros((0,4)), np.zeros(0)))
    tc=np.array([x[1] for x in res]); N=len(res)
    print(f'\n[{name}] 실제 무화재 {N}장 · 헛불률:')
    for C in (0.05,0.25,0.50):
        fp=tc>=C
        print(f'   conf>={C:.2f}: {fp.mean():.3f} ({int(fp.sum())}/{N})   D-Fire야외 {DREF[C]:.3f} · 합성28 {SREF[C]:.3f}')
    return res

results={}
for nm in ('nofire_kitchen','nofire_presrc'):
    if nm in targets: results[nm]=fp_eval(nm, targets[nm])

# 6) 급식실 실제 헛불 몽타주(conf≥0.25) — 무엇을 불로 오인했나
C0=0.25
res=results.get('nofire_kitchen',[])
fires=sorted([r for r in res if r[1]>=C0], key=lambda x:-x[1])
print(f'\nnofire_kitchen conf>={C0} 헛불 {len(fires)}장 — 실제로 무엇을 불로 오인했나:')
if fires:
    K=min(24,len(fires)); order=fires[:K]
    cols=min(4,K); rows=(K+cols-1)//cols
    fig,ax=plt.subplots(rows,cols,figsize=(4*cols,3.4*rows)); ax=np.atleast_2d(ax)
    for i in range(rows*cols):
        a=ax[i//cols,i%cols]; a.axis('off')
        if i>=K: continue
        ip,tcf,xy,cf=order[i]; im=Image.open(ip).convert('RGB'); a.imshow(im)
        for (x1,y1,x2,y2),c in zip(xy,cf):
            if c>=C0: a.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=2))
        a.set_title(f'top {tcf:.2f} · {os.path.basename(ip)[:16]}',fontsize=7)
    plt.tight_layout(); plt.show()
    print('※ 빨강=모델이 불이라 찍은 곳. 실제 수증기·스테인리스 반사·조명에 찍혔으면 헛불 원인 확정.')
else:
    print('  conf>=0.25 헛불 0장.')
print('\n※ 실제 음성 기준 = 합성 28장 예비치 대체. N 커서 신뢰도↑. presrc는 대조(비-급식실).')

# ========== CELL 15: nofire_kitchen 헛불 재검증 — 전체+박스 & 박스확대 + 크기% + PNG저장 ==========
# 원 몽타주는 저장 안 됐어 재생성. 결정론(같은 모델·이미지·conf) → 같은 7건. 판독 쉽게 확대·크기% 추가.
import subprocess, sys, os
try: import ultralytics
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'Drive 마운트 필요(CELL 1) — best.pt 없음: '+W
hits=[r for r,_,_ in os.walk('/content/oilfire_real') if os.path.basename(r).lower()=='nofire_kitchen']
assert hits, 'nofire_kitchen 폴더 없음 — CELL 14로 추출 먼저'
KDIR=hits[0]; print('nofire_kitchen:', KDIR)
EXTS=('.jpg','.jpeg','.png','.webp')
imgs=sorted(os.path.join(KDIR,f) for f in os.listdir(KDIR) if f.lower().endswith(EXTS))
print('이미지', len(imgs))

m=YOLO(W); C0=0.25
fa=[]   # (path, xyxy[keep], conf[keep], topconf)
for ip in imgs:
    r=m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: continue
    xy=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
    keep=cf>=C0
    if keep.any(): fa.append((ip, xy[keep], cf[keep], float(cf.max())))
fa.sort(key=lambda x:-x[3])

print(f'\nconf>={C0} 헛불 {len(fa)}장 (박스 크기로 표시등[작음] vs 음식[큼] 구분):')
for ip,xy,cf,tc in fa:
    b=xy[int(cf.argmax())]; w=b[2]-b[0]; h=b[3]-b[1]
    W0,H0=Image.open(ip).size
    print(f'  {os.path.basename(ip):24} top {tc:.2f} · top박스 {int(w)}x{int(h)}px = 이미지의 {100*w*h/(W0*H0):.1f}%')

if fa:
    K=len(fa)
    fig,ax=plt.subplots(K,2,figsize=(11,4.3*K)); ax=np.atleast_2d(ax)
    for i,(ip,xy,cf,tc) in enumerate(fa):
        im=Image.open(ip).convert('RGB'); W0,H0=im.size
        ax[i,0].imshow(im)
        for (x1,y1,x2,y2),c in zip(xy,cf):
            ax[i,0].add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=2.5))
            ax[i,0].text(x1,max(0,y1-4),f'{c:.2f}',color='red',fontsize=10,weight='bold')
        ax[i,0].set_title(f'{os.path.basename(ip)} · 전체',fontsize=9); ax[i,0].axis('off')
        b=xy[int(cf.argmax())]; pad=max(30,(b[2]-b[0])*0.8,(b[3]-b[1])*0.8)
        cx1=max(0,int(b[0]-pad)); cy1=max(0,int(b[1]-pad)); cx2=min(W0,int(b[2]+pad)); cy2=min(H0,int(b[3]+pad))
        ax[i,1].imshow(im.crop((cx1,cy1,cx2,cy2)))
        ax[i,1].add_patch(patches.Rectangle((b[0]-cx1,b[1]-cy1),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='red',linewidth=2.5))
        ax[i,1].set_title(f'top {tc:.2f} 박스 확대 — 안에 뭐가 있나?',fontsize=9); ax[i,1].axis('off')
    plt.tight_layout()
    OUT='/content/drive/MyDrive/kitchen_fp_verify.png'
    plt.savefig(OUT, dpi=110, bbox_inches='tight'); plt.show()
    print(f'\n저장 → {OUT}  (Drive/파일탐색기서 열어 확대·다운로드 가능)')
    print('※ 오른쪽 확대크롭 안을 봐: 붉은 LED 표시등? 주황 음식? 수증기? 반사? — 네가 직접 판정')
    print('  원본 이미지 직접 보려면: 파일탐색기 →', KDIR, '→ 위 파일명 더블클릭')
else:
    print('conf>=0.25 헛불 0장.')
