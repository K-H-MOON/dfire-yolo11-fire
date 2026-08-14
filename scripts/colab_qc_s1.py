import os, glob, json, shutil, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import files, drive

SRC   = '/content/drive/MyDrive/smoke_frames'
MAT   = f'{SRC}/matte'
BGDIR = f'{SRC}/steam/bg'
WORK  = '/content/ds_s1'
DOUT  = f'{SRC}/ds_s1'
QOUT  = f'{SRC}/qc_s1'

SPLITS = [1, 2, 3, 4, 5]
THR    = 0.06
SEED   = 1
GAIN   = 12

drive.mount('/content/drive')
os.makedirs(QOUT, exist_ok=True)
rng = np.random.default_rng(SEED)

print('=' * 78)
print('본 합성 QC — 학습을 돌리기 전에 숫자와 그림을 함께 봄')
print('=' * 78)
print('  이 회차에서 그림이 숫자를 두 번 뒤집었음 (m3 배율 0.073\\~0.174 · 07 피복률 0.671)')
print('  볼 것 셋 — 여유가 가장 작은 장에 연기가 보이는가 · 벽/바닥/사람 위에 뜬 그림이')
print('  어떻게 보이는가 · 라벨 상자가 얼마나 헐거운가')
print('=' * 78)


r'''작업본이 없으면 Drive 압축본을 풀어서 씀'''
for n in SPLITS:
    root = f'{WORK}/split{n}'
    if not os.path.exists(f'{root}/manifest.json'):
        z = f'{DOUT}/split{n}.zip'
        if not os.path.exists(z):
            raise SystemExit(f'{z} 가 없음 — 본 합성을 먼저 돌릴 것')
        os.makedirs(root, exist_ok=True)
        shutil.unpack_archive(z, root)
        print(f'  분할 {n} 압축본을 품')

MAN = {n: json.load(open(f'{WORK}/split{n}/manifest.json')) for n in SPLITS}
W = H = None


def bgpath(name):
    p = f'{BGDIR}/{name}'
    if os.path.exists(p):
        return p
    tgt = unicodedata.normalize('NFC', name)
    for q in glob.glob(f'{BGDIR}/*.jpg'):
        if unicodedata.normalize('NFC', os.path.basename(q)) == tgt:
            return q
    raise SystemExit(f'배경을 못 찾음 — {name}')


ALPHA = {}


def alpha8(key, file):
    if (key, file) not in ALPHA:
        a = np.asarray(Image.open(f'{MAT}/{key}/{file}'))[..., 3].copy()
        a[a < int(round(THR * 255))] = 0
        ALPHA[(key, file)] = a
    return ALPHA[(key, file)]


def sub_alpha(r):
    r'''그 장에 실제로 얹힌 알파에서 라벨 상자 안만 잘라 돌려줌'''
    a8 = alpha8(r['key'], r['file'])
    pw = max(int(round(a8.shape[1] * r['s'])), 4)
    ph = max(int(round(a8.shape[0] * r['s'])), 4)
    al = np.asarray(Image.fromarray(a8).resize((pw, ph), Image.LANCZOS), np.float32) / 255.0
    al[al < THR] = 0
    x0, y0, x1, y1 = r['box']
    xx, yy = r['pos']
    return al[y0 - yy:y1 - yy, x0 - xx:x1 - xx]


r'''1. 숫자 QC — 2,380장 전수'''
print('\n' + '=' * 78)
print('[1] 숫자 QC — 전수')
print('=' * 78)
STAT = {}
for n in SPLITS:
    rows = MAN[n]['rows']
    img0 = np.asarray(Image.open(f'{WORK}/split{n}/images/train/{rows[0]["name"]}.jpg'))
    if W is None:
        H, W = img0.shape[:2]
    fill, arat, shortv = [], [], []
    for r in rows:
        s = sub_alpha(r)
        bw, bh = r['box'][2] - r['box'][0], r['box'][3] - r['box'][1]
        fill.append(float(s.sum() / max(bw * bh, 1)))
        arat.append(bw * bh / (W * H))
        shortv.append(min(bw, bh))
    m = np.array([r['chg'] / r['floor'] for r in rows])
    cx = np.array([(r['box'][0] + r['box'][2]) / 2 / W for r in rows])
    cy = np.array([(r['box'][1] + r['box'][3]) / 2 / H for r in rows])
    STAT[n] = {'margin': m, 'fill': np.array(fill), 'arat': np.array(arat),
               'short': np.array(shortv), 'cx': cx, 'cy': cy,
               's': np.array([r['s'] for r in rows])}

print(f'  {"분할":>4}{"여유 최소":>10}{"여유 중앙":>10}{"채움 중앙":>10}'
      f'{"상자 넓이비 중앙":>18}{"짧은 변 최소":>13}')
print('  ' + '-' * 66)
for n in SPLITS:
    d = STAT[n]
    print(f'  {n:>4}{d["margin"].min():>10.2f}{np.median(d["margin"]):>10.2f}'
          f'{np.median(d["fill"]):>10.3f}{np.median(d["arat"]):>18.3f}'
          f'{int(d["short"].min()):>13}')
print('  ' + '-' * 66)
print(f'  2층이 적어 둔 채움 0.049 와 견줄 것. 짧은 변 하한은 24화소(격자 한 칸)')

print(f'\n  자리가 고르게 흩어졌는가 — 상자 중심')
print(f'  {"분할":>4}{"가로 10%":>10}{"가로 50%":>10}{"가로 90%":>10}'
      f'{"세로 10%":>10}{"세로 50%":>10}{"세로 90%":>10}')
print('  ' + '-' * 64)
for n in SPLITS:
    d = STAT[n]
    q = lambda v, p: np.percentile(v, p)
    print(f'  {n:>4}{q(d["cx"], 10):>10.2f}{q(d["cx"], 50):>10.2f}{q(d["cx"], 90):>10.2f}'
          f'{q(d["cy"], 10):>10.2f}{q(d["cy"], 50):>10.2f}{q(d["cy"], 90):>10.2f}')
print('  ' + '-' * 64)
print('  중심이 0.5 근처로 몰리는 것은 정상임 — 조각이 화면에 다 들어가야 하므로')
print('  큰 조각일수록 놓을 수 있는 자리가 가운데로 좁아짐')

print(f'\n  [검산] 라벨 txt 가 manifest 와 맞는가 (분할마다 20장 표본)')
bad = 0
for n in SPLITS:
    rows = MAN[n]['rows']
    for r in [rows[i] for i in np.linspace(0, len(rows) - 1, 20).round().astype(int)]:
        t = open(f'{WORK}/split{n}/labels/train/{r["name"]}.txt').read().split()
        x0, y0, x1, y1 = r['box']
        want = [0, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
        got = [int(t[0])] + [float(v) for v in t[1:]]
        if len(t) != 5 or any(abs(a - b) > 2e-6 for a, b in zip(want, got)):
            bad += 1
print(f'      어긋난 장 {"없음" if bad == 0 else f"**{bad}장 — 못 씀**"}')

print(f'\n  [검산] 한 장에 라벨 한 줄인가 (전수)')
nline = 0
for n in SPLITS:
    for p in glob.glob(f'{WORK}/split{n}/labels/train/*.txt'):
        if sum(1 for _ in open(p) if _.strip()) != 1:
            nline += 1
print(f'      두 줄 이상이거나 빈 장 {"없음" if nline == 0 else f"**{nline}장**"}')


r'''2. 그림 QC'''
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


F = korean_font(30)


def make_sheet(picks, tag, note):
    r'''줄마다 배경 · 합성(라벨 상자) · 차이x12. 원본 배율, 축소 안 함'''
    rowsimg = []
    for n, r in picks:
        comp = np.asarray(Image.open(f'{WORK}/split{n}/images/train/{r["name"]}.jpg'),
                          np.float32)
        bg = np.asarray(Image.open(bgpath(r['bg'])).convert('RGB'), np.float32)
        dif = np.clip(np.abs(comp - bg).mean(2) * GAIN, 0, 255).astype(np.uint8)
        box = Image.fromarray(comp.astype(np.uint8))
        d = ImageDraw.Draw(box)
        x0, y0, x1, y1 = r['box']
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 40, 40), width=4)
        s = sub_alpha(r)
        fill = float(s.sum() / max((x1 - x0) * (y1 - y0), 1))
        lab = (f'분할 {n} · {r["name"]} · {r["key"]} {r["file"]} · 배율 {r["s"]:.3f} · '
               f'변화 {r["chg"]:.2f} / 바닥 {r["floor"]:.2f} = 여유 {r["chg"]/r["floor"]:.2f} · '
               f'상자 {x1-x0}x{y1-y0} · 채움 {fill:.3f}')
        rowsimg.append((lab, [bg.astype(np.uint8), np.asarray(box),
                              np.dstack([dif] * 3)]))
    CW = rowsimg[0][1][0].shape[1]
    RH = rowsimg[0][1][0].shape[0]
    sh = Image.new('RGB', (CW * 3, (RH + 48) * len(rowsimg) + 8), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    y = 0
    for lab, arrs in rowsimg:
        dr.text((8, y + 9), lab + '    (배경 · 합성+라벨상자 · 차이x12)',
                fill=(255, 220, 0), font=F)
        for j, a in enumerate(arrs):
            sh.paste(Image.fromarray(a), (CW * j, y + 48))
        y += RH + 48
    p = f'{QOUT}/_qc_{tag}.jpg'
    sh.save(p, quality=88)
    print(f'  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB · {len(rowsimg)}줄 · 원본 배율)')
    print(f'     {note}')
    files.download(p)


print('\n' + '=' * 78)
print('[2] 그림 QC — 원본 배율, 축소하지 않음')
print('=' * 78)

worst, rand, best = [], [], []
for n in SPLITS:
    rows = MAN[n]['rows']
    o = sorted(rows, key=lambda r: r['chg'] / r['floor'])
    worst.append((n, o[0]))
    best.append((n, o[-1]))
    rand.append((n, rows[int(rng.integers(0, len(rows)))]))

make_sheet(worst, 'worst', '분할마다 **여유가 가장 작은 장**. 연기가 보이는지 볼 것')
make_sheet(rand, 'rand', '분할마다 **무작위 한 장**(시드 1). 자리와 상자가 어떤지 볼 것')
make_sheet(best, 'best', '분할마다 **여유가 가장 큰 장**. 위 둘과 견줄 기준')

print('\n' + '=' * 78)
print('이 QC 가 못 보는 것')
print('=' * 78)
print('  전수 2,380장 중 그림으로 본 것은 **15장**임')
print('  차이 칸에는 저장 JPEG q95 오차 0.18 계조도 섞여 있음 (x12 하면 2.1)')
print('  `연기처럼 보이는가` 는 사람이 판정하는 것이고 숫자로 안 잼')
