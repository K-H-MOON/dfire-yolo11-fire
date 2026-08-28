# ===== (A) 출처별 성능 — 재학습 없음 · ptrain_b79 · cap1 test 1068장 =====
# 전제: 셀1(Drive 마운트) + 셀3(dfire_fireonly 빌드). 셀4 불필요.
import subprocess, sys, os, glob, re, shutil, math
try: import ultralytics
except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np

W='/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
TEST='/content/dfire_fireonly/test'
assert os.path.exists(W), 'Drive 마운트/가중치 없음: '+W
assert os.path.isdir(TEST), '셀3(dfire_fireonly) 먼저'
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

ps=imgs_of(f'{TEST}/images')
src=np.array([src_of(p) for p in ps]); pos=np.array([is_pos(p) for p in ps])
SRCS=sorted(set(src.tolist()))
print(f'test {len(ps)}장 · 출처 {SRCS}')
for s in SRCS:
    m=src==s
    print(f'   {s:14} {int(m.sum()):5d}장 · 양성 {int((m&pos).sum()):4d} · 음성 {int((m&~pos).sum()):4d}')

# ---------- 1) 출처별 박스 지표 ----------
def build(mask, tag):
    out=f'/content/dfire_src_{tag}'
    assert 'dfire_src' in os.path.basename(out)
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
    R=c1(b.r_curve); P=c1(b.p_curve); F1=c1(b.f1_curve); i=int(np.argmax(F1))
    return float(b.map50), float(R.max()), float(R[i]), float(P[i]), float(b.map)

print('\n=== 1) 출처별 박스 지표 (iou 0.5) ===')
print(f'{"source":14}{"n":>6}{"pos":>6}{"mAP50":>9}{"maxR":>8}{"R@F1":>8}{"P@F1":>8}{"mAP5095":>10}')
for s in ['ALL']+SRCS:
    m=np.ones(len(ps),bool) if s=='ALL' else (src==s)
    a,b_,c,d,e = probe(build(m, s))
    print(f'{s:14}{int(m.sum()):>6}{int((m&pos).sum()):>6}{a:>9.3f}{b_:>8.3f}{c:>8.3f}{d:>8.3f}{e:>10.3f}')

# ---------- 2) 이미지 단위 + 부트스트랩 구간 ----------
mdl=YOLO(W); maxconf=np.zeros(len(ps)); B=64
for i in range(0,len(ps),B):
    for j,r in enumerate(mdl.predict(ps[i:i+B], conf=0.01, verbose=False)):
        cf=r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.array([])
        maxconf[i+j]=float(cf.max()) if cf.size else 0.0

def wilson(k, n, z=1.96):
    """비율의 95% 신뢰구간. 부트스트랩과 달리 11/11 같은 경우에도 구간이 붕괴하지 않음."""
    if n==0: return float('nan'), float('nan'), float('nan')
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0.0,c-h), min(1.0,c+h)

for T in (0.05,0.25):
    print(f'\n=== 2) 이미지 단위 · conf {T} · Wilson 95% 구간 ===')
    for s_ in ['ALL']+SRCS:
        m=np.ones(len(ps),bool) if s_=='ALL' else (src==s_)
        hp=(maxconf[m&pos]>=T); hn=(maxconf[m&~pos]>=T)
        r,rlo,rhi=wilson(int(hp.sum()), len(hp))
        f,flo,fhi=wilson(int(hn.sum()), len(hn))
        print(f'{s_:14} pos {len(hp):4d} recall {r:.3f} [{rlo:.3f},{rhi:.3f}] (폭 {rhi-rlo:.3f})   '
              f'neg {len(hn):4d} FP {f:.3f} [{flo:.3f},{fhi:.3f}]')
print('\n※ 구간이 넓으면 = 그 출처는 이 split 으로 판정 불가(N 부족). 점추정만 읽지 말 것.')
print('※ full test(누수 미제거) 기준 — 절대값보다 출처 간 대비를 볼 것.')
