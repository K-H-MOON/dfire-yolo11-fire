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
SEED   = 1
N_E    = 12
N_D    = 6
CW, CH = 960, 540
D_DIRS = ['steam_near', 'steam_far', 'steam_in']

drive.mount('/content/drive')
rng = np.random.default_rng(SEED)

print('=' * 78)
print('정탐 위치를 세어 봄 — 1층 `눈으로 확인함` 을 표본을 늘려 다시')
print('=' * 78)
print('  앞서 본 스무 장은 **최고와 문턱** 이라 양극단 편향일 수 있음. 무작위로 다시 뽑음')
print()
print('  **뽑는 규칙 — 세기 전에 못 박음**')
print(f'    분할마다 E 정탐(conf {BASE} 이상) 중 무작위 {N_E}장 · D 오탐 중 무작위 {N_D}장')
print(f'    시드 {SEED}')
print()
print('  **읽는 기준 — 세기 전에 못 박음. 0.5 와 0.8 은 임의임**')
print('    E 정탐 중 `상자가 연기 위` 비율이')
print('      0.5 미만   장 단위 채점이 이 모델에서 뜻을 잃었다고 적음')
print('                판별비를 `연기와 김을 가르는 능력` 으로 읽지 않음')
print('      0.5~0.8    판별비를 **상한으로만** 읽음')
print('      0.8 이상   앞 스무 장이 편향이었던 것으로 보고 판별비를 그대로 읽음')
print()
print('  **축소에 대하여** — 시트는 가로 960 으로 줄임. 이번은 `상자가 어디에 있는가`')
print('  라는 **위치** 판정이라 축소가 됨. 기록 87 에서 문제였던 것은 `옅은 연기가')
print('  보이는가` 라는 **세기** 판정이었음. 원본이 필요하면 파일 이름이 시트에 있음')
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


F = korean_font(20)

INDEX = {}
for s in SPLITS:
    key = str(s)
    if key not in PF:
        print(f'\n분할 {s} — perframe_conf.json 에 없어 건너뜀')
        continue
    w = f'{ROUT}/s{s}/best.pt'
    if not os.path.exists(w):
        print(f'\n분할 {s} — 가중치가 없어 건너뜀')
        continue

    ehit = [r for r in PF[key]['E'] if r[2] >= BASE]
    dhit = [r for r in PF[key]['D'] if r[2] >= BASE]
    ei = rng.permutation(len(ehit))[:N_E]
    di = rng.permutation(len(dhit))[:N_D]
    pick = ([('E',) + tuple(ehit[i]) for i in sorted(ei)]
            + [('D',) + tuple(dhit[i]) for i in sorted(di)])
    print(f'\n분할 {s}  E 정탐 {len(ehit)}장에서 {len(ei)}장 · '
          f'D 오탐 {len(dhit)}장에서 {len(di)}장 뽑음')

    paths = []
    for kind, src, name, c in pick:
        p = (f'{EXT}/smoke/{name}' if kind == 'E' else DMAP.get(norm(name)))
        if p is None or not os.path.exists(p):
            print(f'  **못 찾음 — {kind} {name}**')
            p = None
        paths.append(p)
    good = [(i, p) for i, p in enumerate(paths) if p]

    m = YOLO(w)
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
        tag = f's{s}-{i+1:02d}'
        if paths[i] is None:
            cells.append((tag, kind, src, name, c, None))
            continue
        im = Image.open(paths[i]).convert('RGB')
        d = ImageDraw.Draw(im)
        bx, cf = drawn.get(i, (np.zeros((0, 4)), np.zeros(0)))
        lw = max(3, int(round(im.width / 480)))
        for (x0, y0, x1, y1), cc in zip(bx, cf):
            d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=lw)
        cells.append((tag, kind, src, name, c, im))
        INDEX.setdefault(s, []).append({'tag': tag, 'kind': kind, 'src': src,
                                        'file': name, 'conf': float(c),
                                        'n_box': int(len(cf))})

    rows = (len(cells) + 1) // 2
    sh = Image.new('RGB', (CW * 2, (CH + 34) * rows + 8), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for i, (tag, kind, src, name, c, im) in enumerate(cells):
        gx, gy = i % 2, i // 2
        x, y = gx * CW, gy * (CH + 34)
        lab = f'{tag}  [{kind}]  {src} · {name} · conf {c:.3f}'
        dr.text((x + 8, y + 6), lab, fill=(120, 220, 255) if kind == 'E'
                else (255, 170, 120), font=F)
        if im is None:
            continue
        r = min(CW / im.width, CH / im.height)
        im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                        Image.LANCZOS)
        sh.paste(im2, (x + (CW - im2.width) // 2, y + 34 + (CH - im2.height) // 2))
    p = f'{EOUT}/_count_s{s}.jpg'
    sh.save(p, quality=86)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(cells)}칸)')
    files.download(p)

json.dump(INDEX, open(f'{EOUT}/count_index.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {EOUT}/count_index.json')

print('\n' + '=' * 78)
print('세는 법')
print('=' * 78)
print('  파란 글씨 [E] 가 정탐 · 주황 글씨 [D] 가 김 오탐임')
print('  각 칸의 빨간 상자가 **연기(또는 김) 위에 있는가** 를 보고,')
print('  **연기 위인 칸의 번호만** 적어 주면 됨 (보기 — s1-02 s1-05 s1-11 ...)')
print('  [D] 는 김 위인지 아니면 사람·기계·창문 같은 다른 것인지로 적어 주면 됨')
print()
print('  애매하면 애매하다고 적을 것. 억지로 가르지 않는 편이 나음')
print()
print('  세고 나서 위에 못 박은 기준(0.5 / 0.8)을 그대로 적용함')
print()
print('이 확인이 못 보는 것')
print(f'  분할마다 {N_E + N_D}칸 · 다섯 분할이라 **{(N_E + N_D) * 5}칸**임. 전수가 아님')
print('  `연기 위인가` 는 사람이 판정하는 것이고 숫자로 안 잼')
print('  상자가 연기를 걸치기만 해도 `위` 로 셀지 사람마다 다를 수 있음')
