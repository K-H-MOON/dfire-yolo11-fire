# ===== 발연 구간 프레임 추출 =====
# 드라이브 smoke_frames 폴더의 영상에서, 사람이 직접 확인한 시간대만 뽑음.
#
# 왜 이 방식인가 — 영상 전체에 프레임을 고르게 흩뿌리면 10분 영상에서 15초에 한 장이 됨.
# 발화 직전 흰 연기는 짧게는 2초짜리라 그 구간이 통째로 빠짐. 영상을 재생할 수 있는
# 사람이 구간을 짚고 그 구간만 촘촘히 뽑는 쪽이 정확함.
#
# 프레임은 드라이브에 저장하고, 확인용 시트만 내려받음.
# Colab 새 노트북에 이 셀 하나. GPU 불필요.

import os, glob, json, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

SRC = '/content/drive/MyDrive/smoke_frames'
OUT = f'{SRC}/extracted'          # 프레임 저장 위치 (드라이브)

# 태그별 초당 장수
#   발연        학습 대상. 촘촘히
#   발연_재가열  한 번 불이 붙었다 꺼진 뒤의 기름 연기. 그을음이 섞였을 수 있어 분리
#   발연_조각    2초 이하. 학습에만 쓰고 평가에는 넣지 않음. 자료원 수에도 안 셈
#   발화        학습 클래스 아님. 대조·예시용이므로 낮게
FPS = {'발연': 2.0, '발연_재가열': 2.0, '발연_조각': 2.0, '발화': 0.5}

# ---------------------------------------------------------------------------
# 시간대 — 2026-08-09 사람이 영상을 재생하며 직접 확인한 값
#
# 제외한 영상과 사유
#   02 Cooking fire demo with NH State Fire Marshal    연기 출처 불명.
#      0:31-0:48 연기 앞에 0:10-0:17 발화가 있고, 그 사이 0:17-0:31 이 인터뷰로
#      대체되어 소화 여부를 확인할 수 없음. 발화 전 기름 연기인지 잔여 연기인지 불명
#   03 Cooking Fire Safety                             발연 2초. 게다가 0:00 시작이라
#      연기가 생겨나는 과정이 화면에 없음
#   10 Kitchen Grease Fire Safety                      발연 1초
#   11 Putting out kitchen grease fires                의미 있는 발연 구간 없음
# ---------------------------------------------------------------------------
TIMELINE = {
    # key            영상 파일명에 들어 있는 고유 문자열
    'turkey': dict(match='deep frying a turkey', ranges=[
        ('발연', '0:32', '0:35'), ('발연', '0:40', '0:42'),
        ('발연', '0:54', '0:57'), ('발연', '1:41', '1:45'),
        ('발화', '0:42', '0:46'), ('발화', '0:58', '1:01'),
    ]),
    # 빠른 배속. 재생 6초가 실제로는 더 길 수 있음 — 프레임 보고 확인할 것
    'prevention': dict(match='Fire Prevention - Cooking', ranges=[
        ('발연', '0:25', '0:31'),
        ('발화', '0:32', '0:33'),
    ]),
    # 슬로우모션. 재생 7초가 실제로는 2초 안팎 — 장수는 나오나 실제 시간은 짧음
    'letschat': dict(match='Cooking Fire Prevention and Response', ranges=[
        ('발연', '2:33', '2:40'),
        ('발화', '2:41', '2:45'),
    ]),
    # 발연이 발화로 끊김 없이 이어짐 (0:16, 1:26)
    'spread': dict(match='how quickly grease fires', ranges=[
        ('발연', '0:13', '0:16'), ('발연', '1:20', '1:26'),
        ('발화', '0:16', '0:21'), ('발화', '0:36', '0:45'), ('발화', '1:26', '1:28'),
    ]),
    # 0:44-1:19 이 35초 연속. 중간에 화각이 바뀌므로 나중에 같은 화각끼리 묶을 것.
    # 1:44-1:52 는 뚜껑을 덮어 진화 — 물이 닿지 않았으므로 이후 연기는 김이 아님
    'greasekitchen': dict(match='Grease Fires in the Kitchen', ranges=[
        ('발연', '0:38', '0:40'), ('발연', '0:44', '1:19'),
        ('발연_재가열', '1:52', '1:54'), ('발연_재가열', '2:02', '2:03'),
        ('발화', '1:20', '1:30'), ('발화', '1:42', '1:44'), ('발화', '2:03', '2:15'),
    ]),
    # 화면에 자막이 섞임. 띠 안에 있으면 잘라내고, 연기를 덮으면 해당 프레임 제외
    # 발화는 사람 판단으로 부적합하여 뽑지 않음
    'avoid': dict(match='How to avoid kitchen grease fires', ranges=[
        ('발연', '0:22', '0:26'), ('발연', '1:16', '1:24'),
    ]),
    # 2초뿐. 자료원으로 세지 않음. 발연→발화 연결은 확인됨(1:07)
    'fire411': dict(match='Kitchen Fire 411', ranges=[
        ('발연_조각', '1:05', '1:07'),
    ]),
    # 1:25-2:36 이 71초 연속. 현재 가장 긴 구간
    'deepfry': dict(match='Deep-Frying', ranges=[
        ('발연', '1:25', '2:36'),
        ('발화', '2:38', '2:50'),
    ]),
}


def sec(t):
    p = [float(x) for x in str(t).split(':')]
    while len(p) < 3:
        p.insert(0, 0.0)
    return p[0] * 3600 + p[1] * 60 + p[2]


drive.mount('/content/drive')
vids = sorted(glob.glob(f'{SRC}/*.mp4') + glob.glob(f'{SRC}/*.MP4') +
              glob.glob(f'{SRC}/*.mov') + glob.glob(f'{SRC}/*.MOV'))
assert vids, f'{SRC} 에서 영상을 찾지 못했습니다'

# 매칭 — 하나의 key 가 정확히 영상 하나에 붙는지 확인
paths = {}
for k, cfg in TIMELINE.items():
    hit = [v for v in vids if cfg['match'].lower() in os.path.basename(v).lower()]
    assert len(hit) == 1, f'{k}: "{cfg["match"]}" 에 맞는 영상이 {len(hit)}개 (1개여야 함)'
    paths[k] = hit[0]
    print(f'{k:<15} {os.path.basename(hit[0])[:64]}')
print()

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
shutil.rmtree('/content/smoke_sheets', ignore_errors=True)
os.makedirs('/content/smoke_sheets', exist_ok=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
except Exception:
    F = ImageFont.load_default()

COLS, ROWS, CW = 6, 8, 400
manifest, tally = [], {}

for k in TIMELINE:
    src = paths[k]
    bytag = {}
    for tag, a, b in TIMELINE[k]['ranges']:
        bytag.setdefault(tag, []).append((sec(a), sec(b), a, b))

    for tag, rngs in bytag.items():
        d = f'{OUT}/{k}/{tag}'
        os.makedirs(d, exist_ok=True)
        n = 0
        picked = []
        for (t0, t1, a, b) in rngs:
            tmp = '/content/_seg'
            shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
            subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0}', '-i', src,
                            '-t', f'{t1 - t0}', '-vf', f'fps={FPS[tag]}',
                            '-q:v', '2', f'{tmp}/%04d.jpg'], check=False)
            for j, p in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
                n += 1
                t = t0 + j / FPS[tag]
                name = f'{k}_{tag}_{n:03d}.jpg'
                shutil.move(p, f'{d}/{name}')
                picked.append((name, f'{d}/{name}', t))
                manifest.append({'video': k, 'tag': tag, 'file': name,
                                 'sec': round(t, 2), 'range': f'{a}-{b}'})
            shutil.rmtree(tmp, ignore_errors=True)

        if not picked:
            print(f'{k:<15} {tag:<12} 0장 — 추출 실패');  continue
        tally[(k, tag)] = len(picked)

        w0, h0 = Image.open(picked[0][1]).size
        ch = round(CW * h0 / w0)
        per = COLS * ROWS
        for s in range(0, len(picked), per):
            chunk = picked[s:s + per]
            rows = (len(chunk) + COLS - 1) // COLS
            sheet = Image.new('RGB', (COLS * CW, rows * ch), (20, 20, 20))
            dr = ImageDraw.Draw(sheet)
            for j, (nm, pp, t) in enumerate(chunk):
                im = Image.open(pp).convert('RGB').resize((CW, ch))
                x, y = (j % COLS) * CW, (j // COLS) * ch
                sheet.paste(im, (x, y))
                dr.rectangle([x, y, x + 150, y + 34], fill=(0, 0, 0))
                dr.text((x + 5, y + 3), f'{nm.split("_")[-1][:3]} {t:.1f}s',
                        fill=(255, 220, 0), font=F)
                dr.rectangle([x, y, x + CW - 1, y + ch - 1], outline=(80, 80, 80))
            sheet.save(f'/content/smoke_sheets/{k}__{tag}__{s // per + 1:02d}.jpg',
                       quality=86)
        print(f'{k:<15} {tag:<12} {len(picked):>4}장')

json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), ensure_ascii=False, indent=1)
shutil.copy(f'{OUT}/manifest.json', '/content/smoke_sheets/manifest.json')

print('\n' + '=' * 46)
for tag in ['발연', '발연_재가열', '발연_조각', '발화']:
    tot = sum(v for (k, t), v in tally.items() if t == tag)
    srcn = len({k for (k, t) in tally if t == tag})
    if tot:
        print(f'{tag:<12} {tot:>5}장   출처 {srcn}개')
print(f'{"합계":<12} {sum(tally.values()):>5}장')

shutil.make_archive('/content/smoke_sheets', 'zip', '/content/smoke_sheets')
print(f'\n프레임 -> 드라이브 {OUT}')
print(f'시트   -> smoke_sheets.zip  '
      f'{os.path.getsize("/content/smoke_sheets.zip") / 1e6:.1f}MB')
files.download('/content/smoke_sheets.zip')
