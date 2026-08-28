# ################################################################
# 재빌드 + 배분 실측 — 셀 1~5 를 순서대로 각각 별도 Colab 셀에 붙여넣기
# CELL 5(재학습) · CELL 7 불필요. 셀2에 api_key 채울 것.
# ################################################################

# ============ 셀 1 : Drive 마운트 ============
from google.colab import drive
drive.mount('/content/drive')


# ============ 셀 2 : 다운로드 + dedup (api_key 필요 · ~10분) ============
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


# ============ 셀 3 : cap1 빌드 + 게이트 ============
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
for _s in ('train','valid','test'):
    _p,_n=stat[_s]['pos'],stat[_s]['neg']
    print(f"   {_s:6} 총 {_p+_n:6d} · 양성 {_p:5d} · 음성 {_n:5d} · {100*(_p+_n)/len(seen):5.1f}%")
print('가드 통과 OK ·', OUTDIR)

# ---- [재집계 삽입] 하드 게이트: 안 맞으면 여기서 멈춤 ----
_gA=(len(seen), stat['train']['pos'], stat['test']['pos'], stat['test']['pos']+stat['test']['neg'])
assert _gA==(10624,3205,404,1068), f'✗ cap1 게이트 불일치: {_gA} != (10624, 3205, 404, 1068) — allrec/cl 이 이전과 다름. 멈춤.'
print('✓ cap1 게이트 통과')

# ============ 셀 4 : ptrain 빌드 + 게이트 ============
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
_nc={_s:sum(1 for _k,_v in split_of.items() if _v==_s) for _s in ('train','valid','test')}
for _s in ('train','valid','test'):
    _p,_n=stat[_s]['pos'],stat[_s]['neg']
    print(f"   {_s:6} 총 {_p+_n:6d} · 양성 {_p:5d} · 음성 {_n:5d} · 덩어리 {_nc[_s]:5d} · 덩어리당 {(_p+_n)/max(_nc[_s],1):4.2f}장")
print('  (③ cap1 과 test 수 동일해야 정상)')
print('가드 통과 OK ·', OUTDIR)

# ---- [재집계 삽입] 하드 게이트 ----
_gB=(stat['train']['pos'], stat['test']['pos'], stat['test']['pos']+stat['test']['neg'])
assert _gB==(3433,404,1068), f'✗ ptrain 게이트 불일치: {_gB} != (3433, 404, 1068). 멈춤.'
print('✓ ptrain 게이트 통과 — 셀 C 진행 가능')

# ============ 셀 5 : 배분 실측 + 층화 확인 ============
# ===== 배분 실측 — 재빌드 없음 · /content 빌드 결과를 세기만 함 (수 초) =====
import os, glob, re, collections
EXTS=('*.jpg','*.jpeg','*.png','*.webp','*.bmp')
def imgs_of(d):
    ps=[]
    for e in EXTS: ps+=glob.glob(os.path.join(d,e))+glob.glob(os.path.join(d,e.upper()))
    return sorted(set(ps))
def count(root):
    out={}
    for s in ('train','valid','test'):
        ps=imgs_of(f'{root}/{s}/images'); pos=0
        for p in ps:
            lp=f'{root}/{s}/labels/'+os.path.splitext(os.path.basename(p))[0]+'.txt'
            if os.path.exists(lp) and os.path.getsize(lp)>0: pos+=1
        out[s]=(len(ps),pos)
    return out

for name,root in (('cap1','/content/dfire_fireonly'),('ptrain','/content/dfire_ptrain')):
    if not os.path.isdir(root):
        print(f'[{name}] 없음: {root} (셀 A/B 로 재빌드 필요)'); continue
    r=count(root); tot=sum(v[0] for v in r.values()); tp=sum(v[1] for v in r.values())
    print(f'[{name}] 합계 {tot} · 양성 {tp} · 음성 {tot-tp}')
    for s in ('train','valid','test'):
        n,p=r[s]
        print(f'   {s:6} 총 {n:6d} · 양성 {p:5d} · 음성 {n-p:5d} · {100*n/tot:5.1f}%')

# ===== 층화 확인: 출처 접두사가 정말 AoF/WEB 둘뿐인가 =====
ROOT='/content/dfire_fireonly'
if os.path.isdir(ROOT):
    c=collections.Counter(); cp=collections.Counter()
    for s in ('train','valid','test'):
        for p in imgs_of(f'{ROOT}/{s}/images'):
            b=os.path.basename(p)
            stem0=b.split('_jpg')[0] if '_jpg' in b else os.path.splitext(b)[0]
            m=re.match(r'[A-Za-z]+', stem0); k=m.group(0) if m else 'OTHER'
            c[k]+=1
            lp=f'{ROOT}/{s}/labels/'+os.path.splitext(b)[0]+'.txt'
            if os.path.exists(lp) and os.path.getsize(lp)>0: cp[k]+=1
    print('\n출처 접두사별 (전체 / 양성):')
    for k,v in c.most_common(): print(f'   {k:12} {v:6d} / {cp[k]:6d}')
    print(f'   → 접두사 {len(c)}종 · 층 {2*len(c)}개')
