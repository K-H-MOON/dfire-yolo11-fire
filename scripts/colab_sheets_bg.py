# ===== 배경 오탐 후보 훑기 시트 =====
#
# 논현중 · 로봇고_bg · 숭곡중_bg 는 `김 없음` 으로 적혀 있으나 확인 전임.
# 김이 섞인 채로 배경 음성에 넣으면 **김이 있는 프레임을 김이 없다고 채점**하게 됨.
#
# **이미 뽑아 둔 프레임으로 시트를 만듦.** 영상을 다시 읽지 않으므로 2~3분이면 끝남
# (`colab_extract_steam.py` 가 만든 `steam/bg/` 를 그대로 씀).
#
# ---------------------------------------------------------------------------
# 왜 잘라내지 않는가 — 재 보고 바꾼 것
#
# 인화여중에서는 같은 칸 크기(440px)에 **솥만 잘라낸 쪽이 전체 화면보다 훨씬 잘
# 보였음.** 잘라낸 440px 이 전체 화면 900px 보다 나았음. 그래서 처음에는 여기도
# 잘라내려 했음.
#
# 그런데 논현중 프레임(60·200·381초)을 실제로 보니 **튀김솥이 이미 화면을 거의
# 채우고 있음.** 잘라내 봐야 1.37배밖에 안 커지고, 대신 화각이 바뀌는 구간
# (428초 무렵 조리사·바구니 쪽)에서 내용을 잘라낼 위험이 생김.
# **이득이 작고 위험이 있어 잘라내지 않기로 함.**
#
# 인화여중은 솥이 화면 구석에 작게 있었고 논현중은 가운데 크게 있음.
# **잘라내기가 이득인지는 자료마다 다름.**
#
# ---------------------------------------------------------------------------
# 이 시트로 하는 일 — 1단계
#
#   김일 수도 있어 보이는 자리를 **짚기만** 하면 됨. 확정하지 않아도 됨.
#   짚어 주신 자리는 원본 해상도로 확대해 2단계에서 가름.
#
# 인화여중 68.0초가 그렇게 확정됐음 — 축소된 시트에서 사용자가 짚었고
# 원본(1920×1080)으로 확대해 김을 확인했음. **옅은 김은 시트에서 안 보일 수 있으므로
# `확실하지 않지만 뭔가 있는 것 같다` 도 짚어 주시는 편이 나음.**
#
# 짚으실 때 적어 주실 것 — **시트 번호와 칸에 적힌 초.**

import os, glob, math, shutil, unicodedata
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

BG  = '/content/drive/MyDrive/smoke_frames/steam/bg'
OUT = '/content/sheets_bg'

# 시트 하나에 48칸(6×8). 칸 가로 460px → 시트 2,760px
COLS, ROWS, CW, HDR = 6, 8, 460, 44
PER = COLS * ROWS

SITES = ['논현중', '로봇고_bg', '숭곡중_bg']


def norm(s):
    return unicodedata.normalize('NFC', s)


def sec_of(name):
    """파일명 끝의 `_00428.00.jpg` 에서 초를 꺼냄."""
    return float(os.path.splitext(name)[0].rsplit('_', 1)[-1])


drive.mount('/content/drive')
allf = sorted(glob.glob(f'{BG}/*.jpg'))
if not allf:
    raise SystemExit(f'{BG} 에 프레임이 없음. colab_extract_steam.py 를 먼저 돌릴 것.')

shutil.rmtree(OUT, ignore_errors=True);  os.makedirs(OUT)

try:
    F  = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    FH = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
except Exception:
    F = FH = ImageFont.load_default()

total = 0
for site in SITES:
    ps = sorted([p for p in allf if norm(os.path.basename(p)).startswith(norm(site) + '_')],
                key=lambda p: (os.path.basename(p), sec_of(os.path.basename(p))))
    if not ps:
        print(f'{site:<10}[없음]');  continue

    n_sheet = math.ceil(len(ps) / PER)
    w0, h0 = Image.open(ps[0]).size
    ch = round(CW * h0 / w0)

    for s in range(0, len(ps), PER):
        chunk = ps[s:s + PER]
        rows = math.ceil(len(chunk) / COLS)
        sh = Image.new('RGB', (COLS * CW, rows * ch + HDR), (16, 16, 16))
        d = ImageDraw.Draw(sh)
        idx = s // PER + 1
        d.text((8, 8), f'{site}  ·  시트 {idx}/{n_sheet}  ·  '
                       f'{sec_of(os.path.basename(chunk[0])):.0f}s ~ '
                       f'{sec_of(os.path.basename(chunk[-1])):.0f}s  ·  '
                       f'김일 수도 있어 보이는 칸의 초를 적어 주십시오',
               fill=(255, 220, 0), font=FH)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert('RGB').resize((CW, ch), Image.LANCZOS)
            x, y = (j % COLS) * CW, (j // COLS) * ch + HDR
            sh.paste(im, (x, y))
            d.rectangle([x, y, x + CW - 1, y + ch - 1], outline=(90, 90, 90))
            d.rectangle([x + 2, y + 2, x + 118, y + 34], fill=(0, 0, 0))
            d.text((x + 6, y + 4), f'{sec_of(os.path.basename(p)):.0f}s',
                   fill=(255, 220, 0), font=F)
        sh.save(f'{OUT}/{site}_{idx:02d}.jpg', quality=84)

    total += n_sheet
    print(f'{site:<10}{len(ps):>4}장  시트 {n_sheet:>2}장')

shutil.make_archive('/content/sheets_bg', 'zip', OUT)
mb = os.path.getsize('/content/sheets_bg.zip') / 1e6
print(f'\n시트 {total}장  ·  sheets_bg.zip  {mb:.1f}MB')
print('짚으실 때 — 시트 번호와 칸의 초를 적어 주십시오. 확정하지 않아도 됩니다.')
files.download('/content/sheets_bg.zip')
