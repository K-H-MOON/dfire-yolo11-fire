# ===== 접두사별 누수 분포 + 누수 제거 후 출처별 재평가 =====
# 질문: 방송 프레임 누수가 WEB 에 몰려 있나? 그렇다면 WEB 0.676 도 부풀려진 값인가?
# 전제: 셀1(Drive 마운트) + 셀3(dfire_fireonly) + 셀4(dfire_ptrain). 재학습 없음.
import subprocess, sys, os, glob, re, shutil
try: import open_clip
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','open_clip_torch'], check=True); import open_clip
import torch, numpy as np
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO
from PIL import Image

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
TEST='/content/dfire_fireonly/test'
TRAIN='/content/dfire_ptrain/train/images'
assert os.path.exists(W), 'Drive 마운트/가중치 없음: '+W
assert os.path.isdir(TEST), '셀3(dfire_fireonly) 먼저'
assert os.path.isdir(TRAIN), '셀4(dfire_ptrain) 먼저'
EXTS=('*.jpg','*.jpeg','*.png','*.webp','*.bmp')

def imgs_of(d):
    ps=[]
    for e in EXTS: ps+=glob.glob(os.path.join(d,e))+glob.glob(os.path.join(d,e.upper()))
    return sorted(set(ps))
def src_of(p):
    b=os.path.basename(p); s0=b.split('_jpg')[0] if '_jpg' in b else os.path.splitext(b)[0]
    m=re.match(r'[A-Za-z]+', s0); return m.group(0) if m else 'OTHER'
def is_pos(p):
    lp=f'{TEST}/labels/'+os.path.splitext(os.path.basename(p))[0]+'.txt'
    return os.path.exists(lp) and os.path.getsize(lp)>0

ps=imgs_of(f'{TEST}/images'); tr=imgs_of(TRAIN)
src=np.array([src_of(p) for p in ps]); pos=np.array([is_pos(p) for p in ps])
SRCS=sorted(set(src.tolist()))
print(f'test {len(ps)}장 · train {len(tr)}장 · 출처 {SRCS}')

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
        out.append(torch.nn.functional.normalize(f,dim=-1).cpu().float())
    return torch.cat(out)

print('임베딩 중...')
Ete=embed(ps); Etr=embed(tr)
maxsim=(Ete @ Etr.T).max(dim=1).values.numpy()

# ---------- 1) 접두사별 누수 분포 ----------
print('\n=== 1) 접두사별 누수 (ptrain train 기준 최근접 유사도) ===')
print(f'{"source":14}{"n":>6}{"pos":>6}{">=0.980":>9}{">=0.990":>9}{"pos>=.980":>11}{"pos>=.990":>11}{"중앙값":>9}')
for s in ['ALL']+SRCS:
    m=np.ones(len(ps),bool) if s=='ALL' else (src==s)
    v=maxsim[m]; vp=maxsim[m&pos]
    print(f'{s:14}{int(m.sum()):>6}{int((m&pos).sum()):>6}'
          f'{int((v>=0.980).sum()):>9}{int((v>=0.990).sum()):>9}'
          f'{int((vp>=0.980).sum()):>11}{int((vp>=0.990).sum()):>11}{np.median(v):>9.3f}')
print(f'※ 교차확인 — ALL 의 pos>=0.990 이 25 면 HANDOFF(ptrain 25장)와 일치.')

# ---------- 2) 누수 제거 후 출처별 재평가 ----------
def build(mask, tag):
    out=f'/content/dfire_leak_{tag}'
    assert 'dfire_leak' in os.path.basename(out)
    if os.path.isdir(out): shutil.rmtree(out)
    os.makedirs(f'{out}/test/images'); os.makedirs(f'{out}/test/labels')
    for p,k in zip(ps, mask):
        if not k: continue
        nm=os.path.basename(p); stem=os.path.splitext(nm)[0]
        try: os.symlink(os.path.realpath(p), f'{out}/test/images/{nm}')
        except Exception: shutil.copy(p, f'{out}/test/images/{nm}')
        lp=f'{TEST}/labels/{stem}.txt'; dst=f'{out}/test/labels/{stem}.txt'
        if os.path.exists(lp): shutil.copy(lp,dst)
        else: open(dst,'w').close()
    open(f'{out}/data.yaml','w').write(
        f"path: {out}\ntrain: test/images\nval: test/images\ntest: test/images\nnc: 1\nnames: ['fire']\n")
    return out+'/data.yaml'
def c1(a):
    a=np.asarray(a,float); return a[0] if a.ndim==2 else a
def probe(yaml):
    b=YOLO(W).val(data=yaml, split='test', iou=0.5, plots=False, verbose=False).box
    R=c1(b.r_curve); F1=c1(b.f1_curve); i=int(np.argmax(F1))
    return float(b.map50), float(R.max()), float(b.map)

print('\n=== 2) 출처별 · 누수 제거(t980) 전후 ===')
print(f'{"source":14}{"n_full":>8}{"mAP50":>8}{"|":>3}{"n_t980":>8}{"mAP50":>8}{"maxR":>8}{"delta":>9}')
clean = maxsim < 0.980
for s in ['ALL']+SRCS:
    m=np.ones(len(ps),bool) if s=='ALL' else (src==s)
    f50,_,_ = probe(build(m, s+'_full'))
    mc = m & clean
    if mc.sum() < 5:
        print(f'{s:14}{int(m.sum()):>8}{f50:>8.3f}{"|":>3}{int(mc.sum()):>8}   (남은 장수 부족)')
        continue
    c50,cmr,_ = probe(build(mc, s+'_t980'))
    print(f'{s:14}{int(m.sum()):>8}{f50:>8.3f}{"|":>3}{int(mc.sum()):>8}{c50:>8.3f}{cmr:>8.3f}{c50-f50:>+9.3f}')

print('\n※ delta 가 WEB 에서만 크게 음수면 = 누수가 WEB 에 몰려 WEB 을 부풀렸던 것 → 출처 간 실제 격차는 더 큼.')
print('※ 소수 출처는 남는 장수가 적어 delta 를 방향으로만 읽을 것.')
