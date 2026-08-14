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
print('정탐 위치 세기 **2판** — 1판에서 찾은 결함 넷을 고침')
print('=' * 78)
print('  **판정은 이 2판으로 함. 1판은 대조군임.** 세기 전에 못 박았음')
print()
print('  고친 것 넷')
print('    선 굵기   1판은 원본에 그린 뒤 축소해 선이 절반이 됐음(s2-01 이 안 보였음)')
print('              -> 축소 뒤 4px 가 되도록 원본에서 굵게 그림')
print('    세는 규칙  1판 `상자가 연기 위` 는 화면 절반을 덮는 상자에서 자동 참이 됐음')
print('              -> **최고 신뢰도 상자의 중심**이 연기 위인가. 중심에 표를 찍음')
print('              -> 상자 넓이비도 함께 적음')
print('    뽑기      1판은 정탐에서 무작위라 잘 울리는 출처에 쏠렸음(분할 5 는 12칸 중 8칸이 07)')
print('              -> **출처별로 고르게** 나눔')
print('    연속 프레임 1판은 12칸이 사실상 서너 장면이었음')
print('              -> 같은 출처 안에서는 **시간이 가장 벌어지게** 고름')
print()
print('  **읽는 기준은 안 바꿈** — 0.5 미만 / 0.5~0.8 / 0.8 이상. 답에 관한 선이므로')
print('  뽑기에 무작위가 없음(고르게 나누고 고르게 벌림). 시드가 필요 없음')
print()
print('  **못 가르는 것** — 자(선·규칙)와 표본(뽑기)을 동시에 바꿨으므로,')
print('  1판과 값이 달라도 **원인을 둘로 못 가름**. 가르려면 시트가 셋이어야 함')
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
    r'''출처별로 고르게 나누고, 같은 출처 안에서는 시간이 가장 벌어지게 고름.
       무작위 없음 — 같은 자료면 같은 결과가 나옴'''
    grp = {}
    for h in hits:
        grp.setdefault(h[0], []).append(h)
    keys = sorted(grp, key=lambda k: (-len(grp[k]), k))
    if not keys:
        return []
    quota = {k: n // len(keys) for k in keys}
    for i in range(n - sum(quota.values())):
        quota[keys[i % len(keys)]] += 1
    out = []
    for k in keys:
        v = sorted(grp[k], key=lambda h: h[1])
        q = min(quota[k], len(v))
        if q <= 0:
            continue
        idx = (np.linspace(0, len(v) - 1, q).round().astype(int) if q > 1
               else [len(v) // 2])
        out += [v[i] for i in sorted(set(int(i) for i in idx))]
    short = n - len(out)
    if short > 0:
        rest = [h for k in keys for h in sorted(grp[k], key=lambda h: h[1])
                if h not in out]
        out += rest[:short]
    return out[:n]


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


F = korean_font(19)
INDEX = {}

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
    print(f'\n분할 {s}  E 정탐 {len(ehit)}장 -> {len(ep)}칸  ' +
          ' · '.join(f'{k} {v}' for k, v in sorted(ec.items())))
    print(f'         D 오탐 {len(dhit)}장 -> {len(dp)}칸  ' +
          ' · '.join(f'{k} {v}' for k, v in sorted(dc.items())))

    pick = [('E',) + h for h in ep] + [('D',) + h for h in dp]
    paths = []
    for kind, src, name, c in pick:
        p = f'{EXT}/smoke/{name}' if kind == 'E' else DMAP.get(norm(name))
        paths.append(p if p and os.path.exists(p) else None)

    m = YOLO(w)
    good = [(i, p) for i, p in enumerate(paths) if p]
    drawn = {}
    B = 16
    idxs = [i for i, _ in good];  ps = [p for _, p in good]
    for i in range(0, len(ps), B):
        for j, r in zip(idxs[i:i + B], m(ps[i:i + B], conf=CMIN, verbose=False)):
            cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
            bx = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
            mk = cf >= BASE
            drawn[j] = (bx[mk], cf[mk])

    cells = []
    for i, (kind, src, name, c) in enumerate(pick):
        tag = f't{s}-{i+1:02d}'
        if paths[i] is None:
            cells.append((tag, kind, src, name, c, 0.0, 0, None));  continue
        im = Image.open(paths[i]).convert('RGB')
        W, H = im.size
        r = min(CW / W, CH / H)
        lw = max(2, int(round(4 / r)))
        rad = max(6, int(round(11 / r)))
        d = ImageDraw.Draw(im)
        bx, cf = drawn.get(i, (np.zeros((0, 4)), np.zeros(0)))
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
        cells.append((tag, kind, src, name, c, arat, len(cf), im))
        INDEX.setdefault(s, []).append({'tag': tag, 'kind': kind, 'src': src,
                                        'file': name, 'conf': float(c),
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
    p = f'{EOUT}/_count2_s{s}.jpg'
    sh.save(p, quality=88)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(cells)}칸)')
    files.download(p)

json.dump(INDEX, open(f'{EOUT}/count2_index.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/count2_index.json')

if INDEX:
    ar = [c['area_ratio'] for v in INDEX.values() for c in v if c['kind'] == 'E']
    if ar:
        ar = np.array(ar)
        print(f'\n[참고] E 칸의 **최고 상자 넓이비** — 사람이 세기 전에 숫자로 먼저 봄')
        print(f'  중앙 {np.median(ar):.3f} · 25% {np.percentile(ar,25):.3f} · '
              f'75% {np.percentile(ar,75):.3f} · 최대 {ar.max():.3f}')
        print(f'  화면의 25% 를 넘는 상자 {(ar > 0.25).sum()}/{len(ar)}칸 · '
              f'50% 를 넘는 상자 {(ar > 0.5).sum()}/{len(ar)}칸')
        print(f'  이 값이 크면 1판의 `상자가 연기 위` 규칙이 왜 무의미했는지가 드러남')

print('\n' + '=' * 78)
print('세는 법 — 1판과 규칙이 다름')
print('=' * 78)
print('  빨간 굵은 상자가 **최고 신뢰도 상자**이고, 그 중심에 **노란 표**가 있음')
print('  연한 빨강은 나머지 상자임')
print()
print('  **노란 표가 연기(또는 김) 위에 있는가** 만 보면 됨')
print('  [E] 는 연기 위인 칸의 번호를, [D] 는 김 위인지 다른 것인지를 적어 주면 됨')
print('  애매하면 애매하다고 적을 것')
print()
print('이 확인이 못 보는 것')
print('  분할마다 18칸 · 90칸임. 전수가 아님')
print('  정탐이 0장인 출처는 아예 안 나옴 (분할 5 의 p2 · 05)')
print('  자와 표본을 함께 바꿨으므로 1판과 달라져도 원인을 못 가름')
print('  1판을 먼저 센 기억이 남아 두 세기가 완전히 독립적이지 않음')
