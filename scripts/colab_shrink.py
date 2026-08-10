# ===== 큰 영상을 첨부 가능한 크기로 줄이기 =====
# 원본은 드라이브에 그대로 두고, 화면을 확인할 수 있을 정도로만 줄인 사본을 만듦.
# 프레임을 실제로 뽑는 작업은 나중에 드라이브의 원본으로 하므로 화질 손실이 남지 않음.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 1~3분.

import os, glob, subprocess, unicodedata
from google.colab import drive, files

SRC    = '/content/drive/MyDrive/smoke_frames'
PICK   = '発生するまで'   # 파일명에 들어 있는 고유 문자열
TARGET = 22              # 목표 크기 (MB)


def norm(s):
    """일본어 파일명 대조용.

    `で` 는 한 글자로 저장할 수도 있고 `て` + 탁점 두 글자로 저장할 수도 있음.
    눈에는 같아 보이나 문자열로는 다름. 드라이브가 내려주는 이름이 후자인 경우가 있어
    양쪽을 같은 방식(NFC)으로 맞춘 뒤 비교함.
    """
    return unicodedata.normalize('NFC', s)


drive.mount('/content/drive')
allf = glob.glob(f'{SRC}/*')
hit = [p for p in allf if norm(PICK) in norm(os.path.basename(p))]
assert len(hit) == 1, f'"{PICK}" 에 맞는 파일이 {len(hit)}개 (1개여야 함)\n' + \
                      '\n'.join(os.path.basename(p) for p in allf)
src = hit[0]
print(f'원본  {os.path.basename(src)}  {os.path.getsize(src)/1e6:.1f}MB\n')

# 화질을 조금씩 낮춰 가며 목표 크기 아래로 내려갈 때까지 반복함.
# 소리는 버림 — 판단에 쓰지 않고 용량만 차지함.
out = '/content/small.mp4'
for h, crf in [(720, 28), (720, 32), (540, 32), (540, 36), (432, 36), (360, 38)]:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src,
                    '-vf', f'scale=-2:{h}', '-c:v', 'libx264', '-crf', str(crf),
                    '-preset', 'veryfast', '-an', out], check=False)
    mb = os.path.getsize(out) / 1e6
    print(f'  세로 {h}px · crf {crf}  ->  {mb:.1f}MB')
    if mb <= TARGET:
        print(f'\n목표 이하. 이 사본을 씀. (세로 {h}px)')
        break
else:
    print('\n목표까지 못 내렸음. 마지막 사본을 그대로 씀.')

files.download(out)
