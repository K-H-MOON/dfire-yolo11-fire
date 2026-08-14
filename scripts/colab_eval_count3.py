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
print('정탐 위치 세기 3판 — **노란 표만** 찍음. 빨간 상자를 아예 안 그림')
print('=' * 78)
print('  왜 다시 만드는가')
print('    2판 고침판에서 두 사람이 서로 다른 규칙으로 셌음.')
print('    한쪽은 `상자 안에 연기가 들어와 있는가`(1판 규칙), 다른 쪽은')
print('    `노란 표가 연기 위인가`(2판 규칙) 로 셌음. E 일치가 0.542 였음')
print('    그림에 빨간 상자를 그려 두고 표만 보라고 한 것이 원인이었음')
print()
print('  이번 판이 바꾼 것 — **그림뿐임. 규칙은 안 바꿈**')
print('    빨간 상자를 안 그림. 최고 신뢰도 상자의 중심에 노란 표만 찍음')
print('    라벨에서 conf · 상자 수 · 넓이비를 뺌. 그 값들은 화면과 json 에만 남김')
print('    뽑는 규칙 · 세는 규칙(노란 표) · 읽는 기준(0.5 / 0.8) 은 그대로임')
print()
print('  칸 이름   v1-01 ~ v5-18. **v_N-MM 은 u_N-MM 과 같은 장임**')
print('  못 가르는 것  사용자가 이미 제 목록을 봤으므로 이 세기는 앞선 둘만큼')
print('               독립적이지 않음. 기록에 적을 것')
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
    r'''한 장씩 추론. 묶음 안의 크기가 섞이는 일이 없으므로 평가 본편과 같은 기하임'''
    r = model([path], conf=CMIN, verbose=False)[0]
    cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
    bx = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
    mk = cf >= BASE
    return bx[mk], cf[mk]


F = korean_font(22)
INDEX = {}
CHK = []

for s in SPLITS:
    key = str(s)
    w = f'{ROUT}/s{s}/best.pt'
    if key not in PF or not os.path.exists(w):
        print(f'\n분할 {s} — 자료나 가중치가 없어 건너뜀')
        continue

    ehit = [tuple(r) for r in PF[key]['E'] if r[2] >= BASE]
    dhit = [tuple(r) for r in PF[key]['D'] if r[2] >= BASE]
    ep, dp = even_pick(ehit, N_E), even_pick(dhit, N_D)
    pick = [('E',) + h for h in ep] + [('D',) + h for h in dp]

    paths = []
    for kind, src, name, c in pick:
        p = f'{EXT}/smoke/{name}' if kind == 'E' else DMAP.get(norm(name))
        paths.append(p if p and os.path.exists(p) else None)

    m = YOLO(w)
    print(f'\n분할 {s}')
    print(f'  {"칸":8}{"갈래":5}{"출처":12}{"conf":>8}{"상자":>6}{"넓이비":>9}  파일')

    cells = []
    for i, (kind, src, name, c) in enumerate(pick):
        tag = f'v{s}-{i+1:02d}'
        if paths[i] is None:
            cells.append((tag, kind, None))
            print(f'  {tag:8}{kind:5}{src:12}{"파일 없음":>8}')
            continue
        im = Image.open(paths[i]).convert('RGB')
        W, H = im.size
        r = min(CW / W, CH / H)
        lw = max(2, int(round(4 / r)))
        rad = max(6, int(round(13 / r)))
        d = ImageDraw.Draw(im)
        bx, cf = infer(m, paths[i])
        arat, drawn = 0.0, 0.0
        if len(cf):
            t = int(np.argmax(cf))
            x0, y0, x1, y1 = bx[t]
            arat = float((x1 - x0) * (y1 - y0) / (W * H))
            drawn = float(cf[t])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      outline=(0, 0, 0), width=lw + 2)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      outline=(255, 235, 0), width=lw)
            for a, b, e, f2 in [(-rad * 2.0, 0, -rad * 0.6, 0), (rad * 0.6, 0, rad * 2.0, 0),
                                (0, -rad * 2.0, 0, -rad * 0.6), (0, rad * 0.6, 0, rad * 2.0)]:
                d.line([cx + a, cy + b, cx + e, cy + f2], fill=(0, 0, 0), width=lw + 2)
                d.line([cx + a, cy + b, cx + e, cy + f2], fill=(255, 235, 0), width=lw)
        CHK.append((tag, float(c), drawn, int(len(cf))))
        cells.append((tag, kind, im))
        INDEX.setdefault(s, []).append({'tag': tag, 'kind': kind, 'src': src,
                                        'file': name, 'conf': float(c),
                                        'drawn_conf': drawn, 'area_ratio': arat,
                                        'n_box': int(len(cf))})
        print(f'  {tag:8}{kind:5}{src:12}{c:8.3f}{len(cf):6d}{arat:9.3f}  {name}')

    rows = (len(cells) + 1) // 2
    sh = Image.new('RGB', (CW * 2, (CH + 34) * rows + 8), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for i, (tag, kind, im) in enumerate(cells):
        gx, gy = i % 2, i // 2
        x, y = gx * CW, gy * (CH + 34)
        dr.text((x + 8, y + 5), f'{tag}   [{kind}]',
                fill=(120, 220, 255) if kind == 'E' else (255, 170, 120), font=F)
        if im is None:
            continue
        r = min(CW / im.width, CH / im.height)
        im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                        Image.LANCZOS)
        sh.paste(im2, (x + (CW - im2.width) // 2, y + 34 + (CH - im2.height) // 2))
    p = f'{EOUT}/_count3_s{s}.jpg'
    sh.save(p, quality=88)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(cells)}칸)')
    files.download(p)

json.dump(INDEX, open(f'{EOUT}/count3_index.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/count3_index.json')

print('\n' + '=' * 78)
print('검산 하나 — 표를 찍은 상자가 평가가 센 그 상자인가')
print('=' * 78)
big  = [t for t in CHK if abs(t[1] - t[2]) > 0.02]
zero = [t for t in CHK if t[3] == 0]
print(f'  칸 {len(CHK)}개 · 0.02 넘게 어긋난 칸 {len(big)}개 · 상자 0개인 칸 {len(zero)}개')
for t in big[:20]:
    print(f'    **{t[0]}  기록 {t[1]:.3f} · 그린 것 {t[2]:.3f} · 상자 {t[3]}개** -> 못 셈')
if zero:
    print('    상자 0개 — ' + ' · '.join(t[0] for t in zero) + ' -> 못 셈')
if not big and not zero:
    print('  전부 일치. 모든 칸이 셀 수 있는 칸임')

print('\n' + '=' * 78)
print('검산 둘 — 2판 고침판과 같은 장을 골랐는가')
print('=' * 78)
q = f'{EOUT}/count2b_index.json'
if os.path.exists(q):
    Q = json.load(open(q))
    bad = 0
    for s in INDEX:
        a = [c['file'] for c in INDEX[s]]
        b = [c['file'] for c in Q.get(str(s), [])]
        if a != b:
            bad += 1
            print(f'  분할 {s} — **다름**')
    print(f'  {"다섯 분할 모두 같은 장임. v_N-MM 은 u_N-MM 과 같은 칸" if not bad else "**다른 장이 섞임. 견주지 말 것**"}')
else:
    print('  count2b_index.json 이 없어 못 견줌')

print('\n' + '=' * 78)
print('세는 법')
print('=' * 78)
print('  그림에는 **노란 표 하나만** 있음. 빨간 상자는 없음')
print()
print('  **그 노란 표가 연기(또는 김) 위에 있는가** 만 보면 됨')
print('  [E] 는 연기 위인 칸 번호를, [D] 는 김 위인지 다른 것인지를 적어 주면 됨')
print('  표 둘레가 아니라 **표가 찍힌 그 자리**를 봄')
print('  애매하면 `애매` 로 적을 것. 억지로 가르지 않는 편이 나음')
print()
print('이 확인이 못 보는 것')
print('  분할마다 18칸 · 90칸임. 전수가 아님')
print('  정탐이 0장인 출처는 아예 안 나옴')
print('  가로 960 으로 줄인 그림임. 옅은 김은 앞선 세기에서도 사람마다 갈렸음')
print('  1판·2판을 먼저 셌고 상대 목록도 봤으므로 세 세기가 서로 독립적이지 않음')
