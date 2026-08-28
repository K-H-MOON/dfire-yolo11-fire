# D-Fire YOLO11 baseline — Colab 전체 셀 (순서대로 실행). 새 노트북 A100 기준.
# 검증됨: 문법(ast) + split 공정성(cap1/ptrain val·test 동일·누수0). 2026-08-28 세션.

# ==================== CELL 1: Drive 마운트 ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== CELL 2: D-Fire 다운로드 + dedup(cl 생성) — api_key 채우기 · ~10분 ====================
# ② D-Fire 다운로드 + dedup 클러스터(cl) 생성 — DO_HASH 필수
HAM = 6; DO_HASH = True
import subprocess, sys
for pkg in ('roboflow','imagehash','PIL'):
    try: __import__('PIL' if pkg=='PIL' else pkg)
    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','Pillow' if pkg=='PIL' else pkg], check=True)
from roboflow import Roboflow
rf = Roboflow(api_key="너의_로보플로우_키")                 # ← 네 진짜 키
project = rf.workspace("kyungho-moon").project("d-fire-aqheb-6iyqy")
dataset = project.version(1).download("yolov11")
root = dataset.location
print('D-Fire 위치:', root)

import os, glob
import numpy as np
from PIL import Image
import imagehash
EXTS=('.jpg','.jpeg','.png','.webp','.bmp')
def splits_of(rt):
    s={}
    for n in ('train','valid','val','test'):
        if os.path.isdir(os.path.join(rt,n,'images')): s[n]=os.path.join(rt,n)
    if not s and os.path.isdir(os.path.join(rt,'images')): s['(root)']=rt
    return s
def pairs_of(d):
    im=[]
    for e in EXTS: im+=glob.glob(os.path.join(d,'images','*'+e))+glob.glob(os.path.join(d,'images','*'+e.upper()))
    im=sorted(set(im)); out=[]
    for p in im:
        lp=os.path.join(d,'labels',os.path.splitext(os.path.basename(p))[0]+'.txt')
        out.append((p, lp if os.path.exists(lp) else None))
    return out
def boxes_of(lp):
    if not lp: return []
    o=[]
    for ln in open(lp,encoding='utf-8',errors='ignore'):
        t=ln.split()
        if len(t)<5: continue
        try: c=int(float(t[0])); cx,cy,w,h=(float(x) for x in t[1:5])
        except ValueError: continue
        o.append((c,cx,cy,w,h))
    return o
sp=splits_of(root); assert sp, f'split 못 찾음: {root}'
allrec=[]
for s,d in sp.items():
    pr=pairs_of(d)
    for p,lp in pr: allrec.append((p, boxes_of(lp), s))
    print(f'[{s}] 이미지 {len(pr)}')
assert DO_HASH
print('dHash(21k라 ~10분)...')
N=len(allrec); Hs=np.empty(N,dtype=np.uint64)
for i,(p,_,_) in enumerate(allrec):
    try:
        hh=imagehash.dhash(Image.open(p).convert('RGB')); v=0
        for b in hh.hash.flatten(): v=(v<<1)|int(b)
        Hs[i]=np.uint64(v)
    except Exception: Hs[i]=np.uint64(0)
    if (i+1)%3000==0: print(f'  {i+1}/{N}')
POP=np.array([bin(i).count('1') for i in range(1<<16)],dtype=np.uint8)
def ham(h):
    x=Hs^h
    return (POP[np.asarray(x&np.uint64(0xFFFF),dtype=np.uint32)]+POP[np.asarray((x>>np.uint64(16))&np.uint64(0xFFFF),dtype=np.uint32)]+POP[np.asarray((x>>np.uint64(32))&np.uint64(0xFFFF),dtype=np.uint32)]+POP[np.asarray((x>>np.uint64(48))&np.uint64(0xFFFF),dtype=np.uint32)])
par=list(range(N))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for i in range(N):
    dd=ham(Hs[i])
    for j in np.where(dd<=HAM)[0]:
        if j>i:
            ra,rb=find(i),find(int(j))
            if ra!=rb: par[max(ra,rb)]=min(ra,rb)
cl={}
for i in range(N): cl.setdefault(find(i),[]).append(i)
print(f'장면(클러스터) {len(cl)} / 전체 {N} · 최대 {max(len(v) for v in cl.values())}')
print('allrec·cl 준비 완료.')

# ==================== CELL 3: cleanbuild -> dfire_fireonly (cap1) ====================
# ③ cleanbuild → dfire_fireonly (cap1: 전 split CAP=1)
assert 'allrec' in dir() and 'cl' in dir(), '② 먼저'
import os, random, shutil, re
from collections import defaultdict
CAP=1; FIRE_CLS=0; RATIOS=(0.8,0.1,0.1); SEED=0
OUTDIR='/content/dfire_fireonly'
rng=random.Random(SEED)
clusters=list(cl.values())
reps_per=[sorted(m)[:CAP] for m in clusters]
def src_of(imgp):
    base=os.path.basename(imgp); stem0=base.split('_jpg')[0] if '_jpg' in base else os.path.splitext(base)[0]
    m=re.match(r'[A-Za-z]+', stem0); return m.group(0) if m else 'OTHER'
def stratum(members):
    imgp,boxes,_=allrec[members[0]]; return (src_of(imgp), any(b[0]==FIRE_CLS for b in boxes))
strata=defaultdict(list)
for k,members in enumerate(reps_per): strata[stratum(members)].append(k)
split_of={}
for key, ks in sorted(strata.items(), key=lambda kv:str(kv[0])):
    rng.shuffle(ks); n_img=sum(len(reps_per[k]) for k in ks); cap_tr=n_img*RATIOS[0]; cap_va=n_img*RATIOS[1]; c_tr=c_va=0
    for k in ks:
        n=len(reps_per[k])
        if c_tr+n<=cap_tr or c_tr==0: sp='train'; c_tr+=n
        elif c_va+n<=cap_va or c_va==0: sp='valid'; c_va+=n
        else: sp='test'
        split_of[k]=sp
assert 'dfire_fireonly' in os.path.basename(OUTDIR)
if os.path.isdir(OUTDIR): shutil.rmtree(OUTDIR)
for s in ('train','valid','test'): os.makedirs(f'{OUTDIR}/{s}/images',exist_ok=True); os.makedirs(f'{OUTDIR}/{s}/labels',exist_ok=True)
stat={s:{'pos':0,'neg':0} for s in ('train','valid','test')}; seen=set(); bb=ba=0
for k,members in enumerate(reps_per):
    s=split_of[k]
    for i in members:
        imgp,boxes,_=allrec[i]; fireb=[b for b in boxes if b[0]==FIRE_CLS]; bb+=len(fireb)
        nm=os.path.basename(imgp); stem=os.path.splitext(nm)[0]; dst=f'{OUTDIR}/{s}/images/{nm}'
        if not os.path.exists(dst):
            try: os.symlink(os.path.realpath(imgp),dst)
            except Exception: shutil.copy(imgp,dst)
        with open(f'{OUTDIR}/{s}/labels/{stem}.txt','w') as f:
            for (c,cx,cy,w,h) in fireb: f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'); ba+=1
        seen.add(nm); stat[s]['pos' if fireb else 'neg']+=1
open(f'{OUTDIR}/data.yaml','w').write("path: "+OUTDIR+"\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n")
assert bb==ba and sum(stat[s]['pos']+stat[s]['neg'] for s in stat)==len(seen)
print('[cap1] 총', len(seen), '· train 양성', stat['train']['pos'], '· test', str(stat['test']['pos'])+'/'+str(stat['test']['pos']+stat['test']['neg']))
print('가드 통과 OK ·', OUTDIR)

# ==================== CELL 4: per-split -> dfire_ptrain (train CAP=3) ====================
# ④ per-split → dfire_ptrain (train CAP=3, val/test=1 · 분할은 ③과 동일)
assert 'allrec' in dir() and 'cl' in dir(), '② 먼저'
import os, random, shutil, re
from collections import defaultdict
CAP_TRAIN=3; FIRE_CLS=0; RATIOS=(0.8,0.1,0.1); SEED=0
OUTDIR='/content/dfire_ptrain'
rng=random.Random(SEED)
clusters=list(cl.values())
def src_of(imgp):
    base=os.path.basename(imgp); stem0=base.split('_jpg')[0] if '_jpg' in base else os.path.splitext(base)[0]
    m=re.match(r'[A-Za-z]+', stem0); return m.group(0) if m else 'OTHER'
def stratum(members):
    imgp,boxes,_=allrec[members[0]]; return (src_of(imgp), any(b[0]==FIRE_CLS for b in boxes))
reps1=[sorted(m)[:1] for m in clusters]
strata=defaultdict(list)
for k,members in enumerate(reps1): strata[stratum(members)].append(k)
split_of={}
for key,ks in sorted(strata.items(), key=lambda kv:str(kv[0])):
    rng.shuffle(ks); n_img=sum(len(reps1[k]) for k in ks); cap_tr=n_img*RATIOS[0]; cap_va=n_img*RATIOS[1]; c_tr=c_va=0
    for k in ks:
        n=len(reps1[k])
        if c_tr+n<=cap_tr or c_tr==0: sp='train'; c_tr+=n
        elif c_va+n<=cap_va or c_va==0: sp='valid'; c_va+=n
        else: sp='test'
        split_of[k]=sp
assert 'dfire' in os.path.basename(OUTDIR)
if os.path.isdir(OUTDIR): shutil.rmtree(OUTDIR)
for s in ('train','valid','test'): os.makedirs(f'{OUTDIR}/{s}/images',exist_ok=True); os.makedirs(f'{OUTDIR}/{s}/labels',exist_ok=True)
stat={s:{'pos':0,'neg':0} for s in ('train','valid','test')}; seen=set(); bb=ba=0
for k,members in enumerate(clusters):
    s=split_of[k]; cap=CAP_TRAIN if s=='train' else 1
    for i in sorted(members)[:cap]:
        imgp,boxes,_=allrec[i]; fireb=[b for b in boxes if b[0]==FIRE_CLS]; bb+=len(fireb)
        nm=os.path.basename(imgp); stem=os.path.splitext(nm)[0]; dst=f'{OUTDIR}/{s}/images/{nm}'
        if not os.path.exists(dst):
            try: os.symlink(os.path.realpath(imgp),dst)
            except Exception: shutil.copy(imgp,dst)
        with open(f'{OUTDIR}/{s}/labels/{stem}.txt','w') as f:
            for (c,cx,cy,w,h) in fireb: f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'); ba+=1
        seen.add(nm); stat[s]['pos' if fireb else 'neg']+=1
open(f'{OUTDIR}/data.yaml','w').write("path: "+OUTDIR+"\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n")
assert bb==ba and sum(stat[s]['pos']+stat[s]['neg'] for s in stat)==len(seen)
print('[ptrain] 총', len(seen), '· train 양성', stat['train']['pos'], '· test', str(stat['test']['pos'])+'/'+str(stat['test']['pos']+stat['test']['neg']))
print('  (③ cap1 과 test 수 동일해야 정상)')
print('가드 통과 OK ·', OUTDIR)

# ==================== CELL 5: 학습(batch=79)+오버핏+공정비교 ====================
import subprocess, sys
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
# ⑤ cap1·ptrain 동일설정(batch=79) 학습 + 오버핏 확인 + 동일 cap1 test 공정 비교
from ultralytics import YOLO
import numpy as np, torch, gc, os
import pandas as pd, matplotlib.pyplot as plt
RUNS='/content/drive/MyDrive/dfire_runs'
DATA1='/content/dfire_fireonly/data.yaml'      # cap1 (test 기준)
DATA_PT='/content/dfire_ptrain/data.yaml'       # train CAP3
COMMON=dict(epochs=100, imgsz=640, patience=25, batch=79, cache='disk',
            seed=0, deterministic=True, project=RUNS, exist_ok=True, plots=True)
RUNSPEC=[('cap1_b79', DATA1), ('ptrain_b79', DATA_PT)]

# --- 1) 둘 다 동일 설정 학습(데이터만 다름) ---
for name, data in RUNSPEC:
    YOLO('yolo11s.pt').train(data=data, name='fire_'+name, **COMMON)
    gc.collect(); torch.cuda.empty_cache()

def c1(a):
    a=np.asarray(a, float); return a[0] if a.ndim==2 else a

def col(df, sub):
    for c in df.columns:
        if sub in c: return df[c]
    return None

# --- 2) 오버핏 곡선(손실 + val mAP + best epoch) ---
for name, data in RUNSPEC:
    run=f'{RUNS}/fire_{name}'
    df=pd.read_csv(f'{run}/results.csv'); df.columns=[c.strip() for c in df.columns]
    ep=col(df,'epoch'); m50=col(df,'mAP50(B)'); m5095=col(df,'mAP50-95(B)')
    best_ep=int(ep.iloc[int(m5095.idxmax())])
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for nm in ('train/box_loss','val/box_loss'):
        y=col(df,nm)
        if y is not None: ax[0].plot(ep,y,label=nm)
    ax[0].set_xlabel('epoch'); ax[0].set_title(name+' loss (val 오르면 오버핏)'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep,m50,label='val mAP50'); ax[1].plot(ep,m5095,label='val mAP50-95')
    ax[1].axvline(best_ep,color='r',ls=':',label=f'best {best_ep}')
    ax[1].set_xlabel('epoch'); ax[1].set_title(name+' val mAP'); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
    tag='조기종료 잘됨' if best_ep < len(df)-24 else ('마지막=과소학습 가능' if best_ep>=len(df)-1 else '중간')
    print(f'{name}: 총 {len(df)}ep · best {best_ep} → {tag}')

# --- 3) train/val/test 지표 + recall천장 + 오버핏 gap (둘 다 동일 cap1 test) ---
def probe(w, own_data, test_data):
    m=YOLO(w)
    bt =m.val(data=test_data, split='test',  iou=0.5, plots=False, verbose=False).box
    bv =m.val(data=own_data,  split='val',   iou=0.5, plots=False, verbose=False).box
    btr=m.val(data=own_data,  split='train', iou=0.5, plots=False, verbose=False).box
    R=c1(bt.r_curve); P=c1(bt.p_curve); F1=c1(bt.f1_curve); i=int(np.argmax(F1))
    idx=np.where(R>=0.80)[0]; p80=(P[int(idx[-1])] if len(idx) else float('nan'))
    return dict(test50=float(bt.map50), test=float(bt.map), val=float(bv.map), train=float(btr.map),
                gap=float(btr.map-bt.map), maxR=float(R.max()), f1R=float(R[i]), f1P=float(P[i]), p80=float(p80))

print('\n=== batch=79 · 동일설정 · 데이터만 CAP1 vs CAP3-train · test=cap1 공정 비교 ===')
hdr=('model','test_mAP50','recall천장','F1_R','F1_P','P@R.8','train_m','val_m','test_m','gap')
print(('{:12}{:>10}{:>10}{:>7}{:>7}{:>7}{:>8}{:>7}{:>7}{:>7}').format(*hdr))
for name, data in RUNSPEC:
    r=probe(f'{RUNS}/fire_{name}/weights/best.pt', data, DATA1)
    print(('{:12}{:>10.3f}{:>10.3f}{:>7.3f}{:>7.3f}{:>7.3f}{:>8.3f}{:>7.3f}{:>7.3f}{:>7.3f}').format(
        name, r['test50'], r['maxR'], r['f1R'], r['f1P'], r['p80'], r['train'], r['val'], r['test'], r['gap']))
print('\n※ gap = train − test (mAP50-95, 오버핏 크기). 둘 다 batch79·cache=disk·seed0·같은세션 · test=cap1.')
print('  판정: ptrain_b79 이 recall천장↑·test_mAP50↑ 이면서 gap 이 cap1_b79 대비 안 커졌으면 → CAP↑ 진짜 이득.')
print('        차이 작거나 gap 만 커졌으면 → 이득 없음/착시. (1-seed 노이즈 감안)')


# ==================== CELL 6: 임베딩 기반 누수 감사 (leaky-split) ====================
# 미제 ②: pHash(dHash Ham≤6)가 놓친 cross-split near-dup 정량.
# 데이터만·detection 모델 불필요. CELL 3(cap1 빌드) 이후 실행.
# 각 test 이미지의 train 최근접 코사인 → 임계값별 누수율 + 상위 쌍 몽타주 육안확인.
# ★cap1 기준 감사: ptrain train은 같은 클러스터의 해시-중복 rep만 더한 것 →
#   새 cross-cluster 누수 없음 → cap1 감사가 ptrain에도 유효.
import subprocess, sys, os, glob
try: import open_clip
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','open_clip_torch'], check=True)
    import open_clip
import torch, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT='/content/dfire_fireonly'
assert os.path.isdir(ROOT), 'CELL 3(cap1 빌드) 먼저'
EXTS=('*.jpg','*.jpeg','*.png','*.webp','*.bmp')
def imgs_of(split):
    ps=[]
    for e in EXTS:
        ps+=glob.glob(os.path.join(ROOT,split,'images',e))
        ps+=glob.glob(os.path.join(ROOT,split,'images',e.upper()))
    return sorted(set(ps))
def has_fire(imgp, split):
    lp=os.path.join(ROOT,split,'labels',os.path.splitext(os.path.basename(imgp))[0]+'.txt')
    return os.path.exists(lp) and os.path.getsize(lp)>0

train_ps=imgs_of('train'); test_ps=imgs_of('test')
te_fire=np.array([has_fire(p,'test') for p in test_ps])
print(f'train {len(train_ps)} · test {len(test_ps)} (양성 {int(te_fire.sum())} · 음성 {int((~te_fire).sum())})')

dev='cuda' if torch.cuda.is_available() else 'cpu'
model,_,preproc=open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model=model.to(dev).eval()

@torch.no_grad()
def embed(paths, bs=256):
    out=[]
    for i in range(0,len(paths),bs):
        batch=[]
        for p in paths[i:i+bs]:
            try: batch.append(preproc(Image.open(p).convert('RGB')))
            except Exception: batch.append(torch.zeros(3,224,224))
        f=model.encode_image(torch.stack(batch).to(dev))
        f=torch.nn.functional.normalize(f, dim=-1)
        out.append(f.cpu().float())
        if (i//bs)%8==0: print(f'  embed {min(i+bs,len(paths))}/{len(paths)}')
    return torch.cat(out)

Etr=embed(train_ps); Ete=embed(test_ps)
sims=Ete @ Etr.T                      # [n_test, n_train] cosine (CPU)
maxsim, argmax = sims.max(dim=1)
maxsim=maxsim.numpy(); argmax=argmax.numpy()

print('\n=== test 이미지의 train 최근접 cosine — 임계값별 누수 후보 ===')
print(f'{"thr":>6}{"전체":>10}{"양성(불)":>12}{"음성":>10}')
for t in (0.80,0.90,0.95,0.98,0.99,0.995):
    m=maxsim>=t; na=int(m.sum()); npos=int((m&te_fire).sum()); nneg=int((m&~te_fire).sum())
    print(f'{t:>6.3f}{na:>4d} ({100*na/len(maxsim):4.1f}%){npos:>5d} ({100*npos/max(1,int(te_fire.sum())):4.1f}%){nneg:>5d}')
print(f'중앙값 {np.median(maxsim):.3f} · 평균 {maxsim.mean():.3f} · 최대 {maxsim.max():.3f}')

# 양성 test 중 최근접 유사도 상위 K 쌍 몽타주 (진짜 dup인지 육안)
K=24
pos_idx=np.where(te_fire)[0]
order=pos_idx[np.argsort(-maxsim[pos_idx])[:K]]
rows=(K+3)//4
fig,ax=plt.subplots(rows,8,figsize=(20,2.6*rows))
ax=np.atleast_2d(ax)
for r,ti in enumerate(order):
    tr=int(argmax[ti])
    a=ax[r//4,(r%4)*2]; b=ax[r//4,(r%4)*2+1]
    try: a.imshow(Image.open(test_ps[ti]).convert('RGB'))
    except Exception: pass
    a.set_title(f'test#{ti}',fontsize=7); a.axis('off')
    try: b.imshow(Image.open(train_ps[tr]).convert('RGB'))
    except Exception: pass
    b.set_title(f'train sim={maxsim[ti]:.3f}',fontsize=7); b.axis('off')
for j in range(len(order),rows*4):
    ax[j//4,(j%4)*2].axis('off'); ax[j//4,(j%4)*2+1].axis('off')
plt.tight_layout(); plt.show()
print('※ 왼=test, 오=최근접 train. 같은 장면=누수 / 다른 장면인데 유사=정상(도메인 유사).')
print('  임계값은 이 몽타주로 "진짜 dup 시작 지점"을 눈으로 정함. 그 이하 카운트가 누수 규모.')

# ========== CELL 7 (엄밀판): 각 모델을 '자기 train' 기준 누수 제거 후 재평가 ==========
# 옵션1 · 재학습 없음. cap1은 cap1-train, ptrain은 ptrain-train 기준으로 각각 누수 산출(가정 없음).
# 전제: CELL 3(dfire_fireonly)·CELL 4(dfire_ptrain) 둘 다 빌드돼 있어야 함.
#   ※ 이번 세션에 CELL 4 안 돌렸으면 CELL 4 먼저 실행(allrec·cl 메모리에 있음).
import subprocess, sys, os, glob, shutil
try: import open_clip
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','open_clip_torch'], check=True); import open_clip
import torch, numpy as np
try: import ultralytics
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO

CAP1='/content/dfire_fireonly'; PTR='/content/dfire_ptrain'
assert os.path.isdir(CAP1), 'CELL 3 먼저 (dfire_fireonly 없음)'
assert os.path.isdir(PTR),  'CELL 4 먼저 (dfire_ptrain 없음) — allrec·cl 메모리에 있으니 CELL 4 실행'
RUNS='/content/drive/MyDrive/dfire_runs'
# (모델, 가중치, 그 모델의 실제 train 이미지 폴더)
MODELS=[('cap1_b79',   f'{RUNS}/fire_cap1_b79/weights/best.pt',   f'{CAP1}/train/images'),
        ('ptrain_b79', f'{RUNS}/fire_ptrain_b79/weights/best.pt', f'{PTR}/train/images')]
# 공유 test = dfire_fireonly/test (cap1·ptrain 분할 동일, test 1068 동일)
TEST_DIR=f'{CAP1}/test'
THRS=[('full',2.0),('t995',0.995),('t990',0.990),('t980',0.980)]
EXTS=('*.jpg','*.jpeg','*.png','*.webp','*.bmp')

def imgs_of(d):
    ps=[]
    for e in EXTS: ps+=glob.glob(os.path.join(d,e))+glob.glob(os.path.join(d,e.upper()))
    return sorted(set(ps))
def has_fire(imgp, labels_dir):
    lp=os.path.join(labels_dir, os.path.splitext(os.path.basename(imgp))[0]+'.txt')
    return os.path.exists(lp) and os.path.getsize(lp)>0

test_ps=imgs_of(f'{TEST_DIR}/images')
te_fire=np.array([has_fire(p, f'{TEST_DIR}/labels') for p in test_ps])
print(f'공유 test {len(test_ps)}장 (양성 {int(te_fire.sum())} · 음성 {int((~te_fire).sum())})')

dev='cuda' if torch.cuda.is_available() else 'cpu'
model_clip,_,preproc=open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model_clip=model_clip.to(dev).eval()
from PIL import Image
@torch.no_grad()
def embed(paths, bs=256):
    out=[]
    for i in range(0,len(paths),bs):
        batch=[]
        for p in paths[i:i+bs]:
            try: batch.append(preproc(Image.open(p).convert('RGB')))
            except Exception: batch.append(torch.zeros(3,224,224))
        f=model_clip.encode_image(torch.stack(batch).to(dev))
        f=torch.nn.functional.normalize(f, dim=-1)
        out.append(f.cpu().float())
    return torch.cat(out)

Ete=embed(test_ps)   # 공유 test 임베딩 (1회)

def build_clean_test(keep_mask, tag):
    out=f'/content/dfire_clean_{tag}'
    assert 'dfire_clean' in os.path.basename(out)
    if os.path.isdir(out): shutil.rmtree(out)
    os.makedirs(f'{out}/test/images'); os.makedirs(f'{out}/test/labels')
    for p,k in zip(test_ps, keep_mask):
        if not k: continue
        nm=os.path.basename(p); stem=os.path.splitext(nm)[0]
        try: os.symlink(os.path.realpath(p), f'{out}/test/images/{nm}')
        except Exception: shutil.copy(p, f'{out}/test/images/{nm}')
        lp=f'{TEST_DIR}/labels/{stem}.txt'; dst=f'{out}/test/labels/{stem}.txt'
        if os.path.exists(lp): shutil.copy(lp,dst)
        else: open(dst,'w').close()
    open(f'{out}/data.yaml','w').write(
        f"path: {out}\ntrain: test/images\nval: test/images\ntest: test/images\nnc: 1\nnames: ['fire']\n")
    return out

def c1(a):
    a=np.asarray(a,float); return a[0] if a.ndim==2 else a
def probe(w, yaml):
    b=YOLO(w).val(data=yaml, split='test', iou=0.5, plots=False, verbose=False).box
    R=c1(b.r_curve); P=c1(b.p_curve); F1=c1(b.f1_curve); i=int(np.argmax(F1))
    idx=np.where(R>=0.80)[0]; p80=(P[int(idx[-1])] if len(idx) else float('nan'))
    return dict(map50=float(b.map50), maxR=float(R.max()), f1R=float(R[i]),
                f1P=float(P[i]), p80=float(p80), mapc=float(b.map))

print('\n=== 정직한 baseline: 각 모델을 자기 train 기준 누수 제거 재평가 ===')
hdr=('model','test','test_mAP50','recall천장','F1_R','F1_P','P@R.8','mAP50-95','n_test')
print(('{:12}{:>7}{:>12}{:>11}{:>7}{:>7}{:>7}{:>10}{:>8}').format(*hdr))
for mname,w,train_imgs in MODELS:
    Etr=embed(imgs_of(train_imgs))                 # 이 모델의 실제 train
    maxsim=(Ete @ Etr.T).max(dim=1).values.numpy() # 각 test의 자기-train 최근접
    for tag,thr in THRS:
        keep = maxsim < thr
        r=probe(w, f'{build_clean_test(keep,mname+"_"+tag)}/data.yaml')
        print(('{:12}{:>7}{:>12.3f}{:>11.3f}{:>7.3f}{:>7.3f}{:>7.3f}{:>10.3f}{:>8d}').format(
            mname, tag, r['map50'], r['maxR'], r['f1R'], r['f1P'], r['p80'], r['mapc'], int(keep.sum())))
    npos990=int(((maxsim>=0.990)&te_fire).sum()); npos980=int(((maxsim>=0.980)&te_fire).sum())
    print(f'   └ {mname} 누수(자기 train): 양성 test ≥0.990 {npos990}장 · ≥0.980 {npos980}장 / {int(te_fire.sum())}\n')

print('※ full 행이 CELL5 재현(cap1 0.658 / ptrain 0.686)이어야 정상. t990=육안확인 dup 제거=1차 정직 baseline.')

# ========== CELL 8: 0.980–0.990 구간 육안확인 (정직 baseline 점추정 확정용) ==========
# ptrain_b79(채택 모델) 기준. 이 구간이 진짜 dup이면 정직치는 t980(0.660)쪽, 아니면 t990(0.675)쪽.
import subprocess, sys, os, glob, numpy as np, torch
try: import open_clip
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','open_clip_torch'], check=True); import open_clip
from PIL import Image
import matplotlib.pyplot as plt
CAP1='/content/dfire_fireonly'; PTR='/content/dfire_ptrain'
EXTS=('*.jpg','*.jpeg','*.png','*.webp','*.bmp')
def imgs_of(d):
    ps=[]
    for e in EXTS: ps+=glob.glob(os.path.join(d,e))+glob.glob(os.path.join(d,e.upper()))
    return sorted(set(ps))
test_ps=imgs_of(f'{CAP1}/test/images')
def has_fire(p):
    lp=f'{CAP1}/test/labels/'+os.path.splitext(os.path.basename(p))[0]+'.txt'
    return os.path.exists(lp) and os.path.getsize(lp)>0
te_fire=np.array([has_fire(p) for p in test_ps])
train_ps=imgs_of(f'{PTR}/train/images')

dev='cuda' if torch.cuda.is_available() else 'cpu'
m,_,preproc=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai'); m=m.to(dev).eval()
@torch.no_grad()
def embed(paths,bs=256):
    out=[]
    for i in range(0,len(paths),bs):
        bb=[]
        for p in paths[i:i+bs]:
            try: bb.append(preproc(Image.open(p).convert('RGB')))
            except Exception: bb.append(torch.zeros(3,224,224))
        f=m.encode_image(torch.stack(bb).to(dev)); f=torch.nn.functional.normalize(f,dim=-1)
        out.append(f.cpu().float())
    return torch.cat(out)

Ete=embed(test_ps); Etr=embed(train_ps)
sims=Ete@Etr.T; mx=sims.max(dim=1); maxsim=mx.values.numpy(); argmax=mx.indices.numpy()
band=np.where((maxsim>=0.980)&(maxsim<0.990)&te_fire)[0]
band=band[np.argsort(-maxsim[band])]
print(f'ptrain-train 기준 0.980–0.990 양성 test: {len(band)}장')

K=min(24,len(band)); order=band[:K]; rows=max(1,(K+3)//4)
fig,ax=plt.subplots(rows,8,figsize=(20,2.6*rows)); ax=np.atleast_2d(ax)
for r,ti in enumerate(order):
    tr=int(argmax[ti]); a=ax[r//4,(r%4)*2]; b=ax[r//4,(r%4)*2+1]
    try: a.imshow(Image.open(test_ps[ti]).convert('RGB'))
    except Exception: pass
    a.set_title(f'test#{ti}',fontsize=7); a.axis('off')
    try: b.imshow(Image.open(train_ps[tr]).convert('RGB'))
    except Exception: pass
    b.set_title(f'train sim={maxsim[ti]:.3f}',fontsize=7); b.axis('off')
for j in range(K,rows*4):
    ax[j//4,(j%4)*2].axis('off'); ax[j//4,(j%4)*2+1].axis('off')
plt.tight_layout(); plt.show()
print('※ 같은 장면 많으면 정직 baseline은 t980(0.660)쪽 · 다른 장면이면 t990(0.675)쪽.')
