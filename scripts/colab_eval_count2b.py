import os, glob, json, unicodedata, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files, drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'
STM  = f'{SRC}/steam'
ROUT = f'{SRC}/runs_s1'
EOUT = f'{SRC}/eval_s1'

SPLITS = [1, 2, 3, 4, 5]
BASE   = 0.10
CMIN   = 0.01
N_E    = 12
N_D    = 6
CW, CH = 960, 540
D_DIRS = ['steam_near', 'steam_far', 'steam_in']

drive.mount('/content/drive')

print('=' * 78)
print('정탐 위치 세기 2판 **고침판** — 상자가 0개로 나오던 결함을 고침')
print('=' * 78)
print('  2판 시트에서 E 60칸 중 아홉 칸이 `상자 0개` 로 나왔음.')
print('  라벨의 conf 는 0.10 이 넘는데 그림에는 상자가 없었음. 둘이 어긋남')
print()
print('  짐작한 원인   perframe_conf.json 은 **출처별로 묶어** 추론했고,')
print('               2판 시트는 **여러 출처를 섞어** 16장씩 묶어 추론했음.')
print('               ultralytics 는 묶음 안의 크기가 다르면 letterbox 를')
print('               정사각으로 바꾸므로 같은 장이라도 결과가 달라짐')
print()
print('  고친 것       묶지 않고 **한 장씩** 추론함. 그러면 출처별 묶음과 같은 기하가 됨')
print('  안 고친 것    세는 규칙(노란 표) · 읽는 기준(0.5 / 0.8) · 뽑는 규칙')
print()
print('  아직 아무도 2판을 세지 않았으므로 지금 고치는 것은 사후 조정이 아님')
print('=' * 78)


def norm(s):
    return unicodedata.normalize('NFC', s)


pf = f'{EOUT}/perframe_conf.json'
if not os.path.exists(pf):
    raise SystemExit(f'{pf} 가 없음 — colab_eval_look.py 를 먼저 돌릴 것')
PF = json.load(open(pf))

DMAP = {}
for d in D_DIRS:
    for p in glob.glob(f'{STM}/{d}/*.jpg'):
        DMAP[norm(os.path.basename(p))] = p

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO
print(f'\nultralytics {ultralytics.__version__} · CUDA {torch.cuda.is_available()}')


def even_pick(hits, n):
    r'''출처별로 고르게 나누되, 모자란 출처의 몫은 남은 출처에 다시 고르게 나눔.
       같은 출처 안에서는 시간이 가장 벌어지게 고름. 무작위 없음'''
    grp = {}
    for h in hits:
        grp.setdefault(h[0], []).append(h)
    keys = sorted(grp, key=lambda k: (-len(grp[k]), k))
    if not keys:
        return []
    quota = {k: 0 for k in keys}
    left = min(n, len(hits))
    pool = [k for k in keys if len(grp[k]) > quota[k]]
    while left > 0 and pool:
        share = left // len(pool)
        if share == 0:
            for k in pool[:left]:
                quota[k] += 1
            left = 0
            break
        for k in pool:
            add = min(share, len(grp[k]) - quota[k])
            quota[k] += add
            left -= add
        pool = [k for k in pool if len(grp[k]) > quota[k]]
    out = []
    for k in keys:
        v = sorted(grp[k], key=lambda h: h[1])
        q = quota[k]
        if q <= 0:
            continue
        idx = (np.linspace(0, len(v) - 1, q).round().astype(int) if q > 1
               else [len(v) // 2])
        out += [v[i] for i in sorted(set(int(i) for i in idx))]
    return out


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


def infer(model, path):
    r'''한 장씩 추론. 묶음 안의 크기가 섞이는 일이 없으므로 출처별 묶음과 같은 기하임'''
    r = model([path], conf=CMIN, verbose=False)[0]
    cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
    bx = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
    mk = cf >= BASE
    return bx[mk], cf[mk]


def infer_mixed(model, paths, B=16):
    r'''2판이 한 방식 그대로 — 여러 출처를 섞어 묶어 추론. 원인 시험용'''
    out = []
    for i in range(0, len(paths), B):
        for r in model(paths[i:i + B], conf=CMIN, verbose=False):
            cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
            out.append(float(cf.max()) if len(cf) else 0.0)
    return out


F = korean_font(19)
INDEX = {}
CHK = []
TESTED = False

for s in SPLITS:
    key = str(s)
    w = f'{ROUT}/s{s}/best.pt'
    if key not in PF or not os.path.exists(w):
        print(f'\n분할 {s} — 자료나 가중치가 없어 건너뜀')
        continue

    ehit = [tuple(r) for r in PF[key]['E'] if r[2] >= BASE]
    dhit = [tuple(r) for r in PF[key]['D'] if r[2] >= BASE]
    ep, dp = even_pick(ehit, N_E), even_pick(dhit, N_D)
    ec = {}
    for h in ep:
        ec[h[0]] = ec.get(h[0], 0) + 1
    dc = {}
    for h in dp:
        dc[h[0]] = dc.get(h[0], 0) + 1
    print(f'\n분할 {s}  E {len(ehit)}장 -> {len(ep)}칸  ' +
          ' · '.join(f'{k} {v}' for k, v in sorted(ec.items())))
    print(f'         D {len(dhit)}장 -> {len(dp)}칸  ' +
          ' · '.join(f'{k} {v}' for k, v in sorted(dc.items())))

    pick = [('E',) + h for h in ep] + [('D',) + h for h in dp]
    paths = []
    for kind, src, name, c in pick:
        p = f'{EXT}/smoke/{name}' if kind == 'E' else DMAP.get(norm(name))
        paths.append(p if p and os.path.exists(p) else None)

    m = YOLO(w)

    if not TESTED:
        TESTED = True
        good = [p for p in paths if p]
        mixed = infer_mixed(m, good)
        print('\n  [원인 시험] 같은 장을 두 방식으로 잼 — 분할 1 의 열여덟 칸')
        print(f'  {"칸":6}{"출처":10}{"기록된 conf":>12}{"한 장씩":>10}{"섞어 묶음":>11}   어긋남')
        print('  ' + '-' * 62)
        nd = 0
        gi = 0
        for i, (kind, src, name, c) in enumerate(pick):
            if paths[i] is None:
                continue
            _, cf1 = infer(m, paths[i])
            v1 = float(cf1.max()) if len(cf1) else 0.0
            v2 = mixed[gi];  gi += 1
            bad = abs(v1 - v2) > 0.02
            nd += int(bad)
            print(f'  t{s}-{i+1:02d}  {src:10}{c:12.3f}{v1:10.3f}{v2:11.3f}   '
                  + ('**어긋남**' if bad else ''))
        print(f'  -> 열여덟 칸 중 {nd}칸이 어긋남')
        print('     `기록된 conf` 와 `한 장씩` 이 같고 `섞어 묶음` 만 다르면')
        print('     짐작한 원인이 맞는 것임')

    cells = []
    for i, (kind, src, name, c) in enumerate(pick):
        tag = f'u{s}-{i+1:02d}'
        if paths[i] is None:
            cells.append((tag, kind, src, name, c, 0.0, 0, None))
            continue
        im = Image.open(paths[i]).convert('RGB')
        W, H = im.size
        r = min(CW / W, CH / H)
        lw = max(2, int(round(4 / r)))
        rad = max(6, int(round(11 / r)))
        d = ImageDraw.Draw(im)
        bx, cf = infer(m, paths[i])
        top = int(np.argmax(cf)) if len(cf) else -1
        arat = 0.0
        for j, ((x0, y0, x1, y1), cc) in enumerate(zip(bx, cf)):
            main = (j == top)
            d.rectangle([x0, y0, x1, y1],
                        outline=(255, 40, 40) if main else (255, 150, 150),
                        width=lw if main else max(2, lw // 2))
            if main:
                arat = float((x1 - x0) * (y1 - y0) / (W * H))
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                          outline=(255, 235, 0), width=lw)
                d.line([cx - rad * 1.8, cy, cx + rad * 1.8, cy], fill=(255, 235, 0),
                       width=max(2, lw // 2))
                d.line([cx, cy - rad * 1.8, cx, cy + rad * 1.8], fill=(255, 235, 0),
                       width=max(2, lw // 2))
        drawn = float(cf.max()) if len(cf) else 0.0
        CHK.append((tag, float(c), drawn, int(len(cf))))
        cells.append((tag, kind, src, name, c, arat, len(cf), im))
        INDEX.setdefault(s, []).append({'tag': tag, 'kind': kind, 'src': src,
                                        'file': name, 'conf': float(c),
                                        'drawn_conf': drawn,
                                        'area_ratio': arat, 'n_box': int(len(cf))})

    rows = (len(cells) + 1) // 2
    sh = Image.new('RGB', (CW * 2, (CH + 34) * rows + 8), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for i, (tag, kind, src, name, c, arat, nb, im) in enumerate(cells):
        gx, gy = i % 2, i // 2
        x, y = gx * CW, gy * (CH + 34)
        dr.text((x + 8, y + 6),
                f'{tag}  [{kind}]  {src} · {name} · conf {c:.3f} · '
                f'상자 {nb}개 · 최고상자 넓이비 {arat:.3f}',
                fill=(120, 220, 255) if kind == 'E' else (255, 170, 120), font=F)
        if im is None:
            continue
        r = min(CW / im.width, CH / im.height)
        im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                        Image.LANCZOS)
        sh.paste(im2, (x + (CW - im2.width) // 2, y + 34 + (CH - im2.height) // 2))
    p = f'{EOUT}/_count2b_s{s}.jpg'
    sh.save(p, quality=88)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(cells)}칸)')
    files.download(p)

json.dump(INDEX, open(f'{EOUT}/count2b_index.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/count2b_index.json')

print('\n' + '=' * 78)
print('검산 — 라벨의 conf 와 실제로 그린 최고 상자의 conf 가 같은가')
print('=' * 78)
tiny = [t for t in CHK if 0.001 < abs(t[1] - t[2]) <= 0.02]
big  = [t for t in CHK if abs(t[1] - t[2]) > 0.02]
zero = [t for t in CHK if t[3] == 0]
print(f'  칸 {len(CHK)}개 · 0.001 넘게 어긋난 칸 {len(tiny)}개 · '
      f'0.02 넘게 어긋난 칸 {len(big)}개 · 상자 0개인 칸 {len(zero)}개')
print('  0.02 이하의 어긋남은 cuDNN 이 묶음 크기에 따라 다른 셈법을 고르는 데서 옴.')
print('  기하가 바뀐 것과는 다른 일이므로 이 정도는 봐 줌')
for t in tiny[:10]:
    print(f'    [작음] {t[0]}  라벨 {t[1]:.3f} · 그린 것 {t[2]:.3f} · 상자 {t[3]}개')
if big:
    for t in big[:20]:
        print(f'    **{t[0]}  라벨 {t[1]:.3f} · 그린 것 {t[2]:.3f} · 상자 {t[3]}개**')
    print('  **0.02 를 넘게 어긋난 칸이 있음. 이 시트로 세지 말고 알려 주십시오**')
else:
    print('  0.02 를 넘게 어긋난 칸은 없음. 그림의 상자가 평가에서 센 그 상자임')
if zero:
    print(f'  상자 0개인 칸 — ' + ' · '.join(t[0] for t in zero))
    print('  이 칸은 `못 셈` 으로 적고 분모에서 뺄 것')

if INDEX:
    ar = [c['area_ratio'] for v in INDEX.values() for c in v if c['kind'] == 'E']
    if ar:
        ar = np.array(ar)
        print(f'\n[참고] E 칸의 최고 상자 넓이비')
        print(f'  중앙 {np.median(ar):.3f} · 25% {np.percentile(ar,25):.3f} · '
              f'75% {np.percentile(ar,75):.3f} · 최대 {ar.max():.3f}')
        print(f'  화면의 25% 를 넘는 상자 {(ar > 0.25).sum()}/{len(ar)}칸 · '
              f'50% 를 넘는 상자 {(ar > 0.5).sum()}/{len(ar)}칸')

print('\n' + '=' * 78)
print('세는 법 — 2판과 같음')
print('=' * 78)
print('  빨간 굵은 상자가 최고 신뢰도 상자이고, 그 중심에 노란 표가 있음')
print('  연한 빨강은 나머지 상자임')
print()
print('  **노란 표가 연기(또는 김) 위에 있는가** 만 보면 됨')
print('  [E] 는 연기 위인 칸의 번호를, [D] 는 김 위인지 다른 것인지를 적어 주면 됨')
print('  애매하면 애매하다고 적을 것')
print()
print('이 확인이 못 보는 것')
print('  분할마다 18칸 · 90칸임. 전수가 아님')
print('  정탐이 0장인 출처는 아예 안 나옴')
print('  1판과 2판을 먼저 본 기억이 남아 세 세기가 서로 독립적이지 않음')
print('  가로 960 으로 줄인 그림임. 옅은 김은 1판에서도 사람마다 갈렸음')
