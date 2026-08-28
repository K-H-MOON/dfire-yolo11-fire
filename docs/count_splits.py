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
