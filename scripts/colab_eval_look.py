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
MINSRC = 10
BATCH  = 16

ID = {1: 's2', 2: 'ft', 3: 'j01', 4: 'm3', 5: 'p2', 6: '07', 7: 'kt', 8: '12',
      9: 'kfire03', 10: 'q1', 11: 'j04', 12: '05', 13: 'j12', 14: 'kfire02', 15: '04'}
EVAL = {1: [2, 3, 5, 10, 13], 2: [1, 2, 12, 14, 15], 3: [3, 4, 6, 9, 10],
        4: [2, 4, 5, 7, 12], 5: [5, 6, 10, 11, 12]}
D_DIRS = ['steam_near', 'steam_far', 'steam_in']

drive.mount('/content/drive')
os.makedirs(EOUT, exist_ok=True)

print('=' * 78)
print('눈으로 확인 — 1층이 요구한 마지막 줄')
print('=' * 78)
print('  원문 `장 단위의 맹점은 그대로 남음 — 어디를 봤는지 모름. 뒤쪽 국솥에')
print('        반응해도 정탐이 됨. **정탐이 냄비 위가 아닌 곳에서 나오는지는')
print('        1회차 결과에서 눈으로 확인함.**`')
print()
print('  **고를 규칙을 그림을 보기 전에 못 박음**')
print('    E 정탐 최고    분할마다 conf 가장 높은 장 하나')
print('    E 겨우 잡음    conf 0.10 이상 중 가장 낮은 장 하나')
print('    D 오탐 최고    분할마다 conf 가장 높은 장 하나')
print('    배경 오탐 최고  로봇고(쉐이크) 중 가장 높은 장 하나')
print('  그림에는 conf 0.10 이상 상자를 전부 그리고 값을 적음. 원본 배율')
print('=' * 78)


def norm(s):
    return unicodedata.normalize('NFC', s)


def dhash(path, size=8):
    g = np.asarray(Image.open(path).convert('L').resize((size + 1, size), Image.LANCZOS),
                   np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def dedup(paths):
    keep, hs = [], []
    for p in sorted(paths):
        h = dhash(p)
        if all(int(np.unpackbits(h ^ g).sum()) > 0 for g in hs):
            keep.append(p);  hs.append(h)
    return keep


r'''자료 — 평가 2판과 같은 규칙'''
pool = {}
for p in glob.glob(f'{EXT}/smoke/*.jpg'):
    b = norm(os.path.basename(p))
    for k in ID.values():
        if b.startswith(norm(k) + '_'):
            pool.setdefault(k, []).append(p)
            break
ESRC = {k: dedup(pool.get(k, [])) for k in ID.values()}
ndd = sum(len(v) for v in ESRC.values())

dpaths = []
for d in D_DIRS:
    dpaths += glob.glob(f'{STM}/{d}/*.jpg')
dsite = {}
for p in dpaths:
    dsite.setdefault(norm(os.path.basename(p)).split('_')[0], []).append(p)
DSRC = {k: dedup(v) for k, v in dsite.items()}
dtot = sum(len(v) for v in DSRC.values())

bgall = glob.glob(f'{STM}/bg/*.jpg')
BR = dedup([p for p in bgall
            if norm(os.path.basename(p)).startswith(norm('로봇고') + '_')
            and norm('쉐이크') in norm(os.path.basename(p))])
print(f'\n[검산] E 서로 다름 {ndd} (666) · D {dtot} (2,044) · 로봇고(쉐이크) {len(BR)} (29)')
ok = (ndd, dtot, len(BR)) == (666, 2044, 29)
print(f'  {"일치" if ok else "**불일치 — 아래 그림을 2판과 나란히 읽지 말 것**"}')

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO
print(f'ultralytics {ultralytics.__version__} · CUDA {torch.cuda.is_available()}')


def scan(model, paths):
    r'''장마다 (최대 신뢰도, 상자 목록). 상자는 conf 0.10 이상만 남김'''
    out = []
    for i in range(0, len(paths), BATCH):
        for p, r in zip(paths[i:i + BATCH], model(paths[i:i + BATCH], conf=CMIN,
                                                  verbose=False)):
            c = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
            b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
            m = c >= BASE
            out.append((p, float(c.max()) if len(c) else 0.0,
                        b[m].tolist(), c[m].tolist()))
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


F = korean_font(26)
FB = korean_font(30)

PICK = {'E_top': [], 'E_edge': [], 'D_top': [], 'BG_top': []}
STORE = {}

for s in SPLITS:
    w = f'{ROUT}/s{s}/best.pt'
    if not os.path.exists(w):
        print(f'\n분할 {s} — 가중치가 없어 건너뜀')
        continue
    print(f'\n분할 {s} 훑는 중...')
    m = YOLO(w)
    keys = [ID[i] for i in EVAL[s] if len(ESRC[ID[i]]) >= MINSRC]
    erec, drec = [], []
    for k in keys:
        erec += [(k,) + t for t in scan(m, ESRC[k])]
    for k in sorted(DSRC):
        drec += [(k,) + t for t in scan(m, DSRC[k])]
    brec = [('로봇고',) + t for t in scan(m, BR)]

    ehit = [r for r in erec if r[2] >= BASE]
    dhit = [r for r in drec if r[2] >= BASE]
    bhit = [r for r in brec if r[2] >= BASE]
    print(f'  E 정탐 {len(ehit)}/{len(erec)} · D 오탐 {len(dhit)}/{len(drec)} · '
          f'로봇고 오탐 {len(bhit)}/{len(brec)}')

    if ehit:
        PICK['E_top'].append((s, max(ehit, key=lambda r: r[2])))
        PICK['E_edge'].append((s, min(ehit, key=lambda r: r[2])))
    if dhit:
        PICK['D_top'].append((s, max(dhit, key=lambda r: r[2])))
    if bhit:
        PICK['BG_top'].append((s, max(bhit, key=lambda r: r[2])))

    STORE[s] = {'E': [[r[0], os.path.basename(r[1]), r[2]] for r in erec],
                'D': [[r[0], os.path.basename(r[1]), r[2]] for r in drec],
                'BG': [[r[0], os.path.basename(r[1]), r[2]] for r in brec]}

json.dump(STORE, open(f'{EOUT}/perframe_conf.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/perframe_conf.json  (장마다의 신뢰도. 다음부터 재추론 안 해도 됨)')


def sheet(items, tag, note):
    if not items:
        print(f'  {tag} — 뽑힌 장이 없음')
        return
    rows = []
    for s, r in items:
        src, path, mx, boxes, confs = r
        im = Image.open(path).convert('RGB')
        d = ImageDraw.Draw(im)
        for (x0, y0, x1, y1), c in zip(boxes, confs):
            d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=4)
            d.text((x0 + 6, max(y0 - 32, 2)), f'{c:.3f}', fill=(255, 40, 40), font=F)
        lab = (f'분할 {s} · {src} · {os.path.basename(path)} · '
               f'최대 conf {mx:.3f} · 상자 {len(boxes)}개')
        rows.append((lab, np.asarray(im)))
    CW = max(a.shape[1] for _, a in rows)
    H = sum(a.shape[0] + 40 for _, a in rows) + 8
    sh = Image.new('RGB', (CW, H), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    y = 0
    for lab, a in rows:
        dr.text((8, y + 6), lab, fill=(255, 220, 0), font=FB)
        sh.paste(Image.fromarray(a), (0, y + 40))
        y += a.shape[0] + 40
    p = f'{EOUT}/_det_{tag}.jpg'
    sh.save(p, quality=88)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(rows)}줄 · 원본 배율)')
    print(f'     {note}')
    files.download(p)


print('\n' + '=' * 78)
print('그림')
print('=' * 78)
sheet(PICK['E_top'], 'E_top', 'E 정탐 중 가장 확신한 장. **상자가 연기 위에 있는가**')
sheet(PICK['E_edge'], 'E_edge', 'E 정탐 중 문턱에 겨우 걸친 장. 무엇을 보고 잡았는가')
sheet(PICK['D_top'], 'D_top', 'D 오탐 중 가장 확신한 장. **무엇에 울렸는가**')
sheet(PICK['BG_top'], 'BG_top', '김도 없는 로봇고(쉐이크)에서 가장 확신한 오탐')

print('\n' + '=' * 78)
print('이 확인이 못 보는 것')
print('=' * 78)
print('  분할마다 한 장씩 · 네 갈래라 **최대 20장**임. 전수가 아님')
print('  `최고`와 `문턱`만 봄 — 가운데가 어떤지는 안 봄')
print('  `연기 위에 있는가` 는 사람이 판정하는 것이고 숫자로 안 잼')
