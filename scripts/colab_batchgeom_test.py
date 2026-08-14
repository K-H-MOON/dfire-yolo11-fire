import os, glob, json, unicodedata, subprocess, sys
import numpy as np
from PIL import Image
from google.colab import drive

SRC  = '/content/drive/MyDrive/smoke_frames'
EXT  = f'{SRC}/extracted'
ROUT = f'{SRC}/runs_s1'
EOUT = f'{SRC}/eval_s1'
BASE, CMIN = 0.10, 0.01
TARGET = ['p2_00042.5.jpg', 'p2_00045.0.jpg', 'q1_00047.5.jpg']

drive.mount('/content/drive')
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
from ultralytics import YOLO
print(f'ultralytics {ultralytics.__version__}')

PF = json.load(open(f'{EOUT}/perframe_conf.json'))
rec = {r[1]: (r[0], r[2]) for r in PF['1']['E']}
hit = [r for r in PF['1']['E'] if r[2] >= BASE]
path = lambda n: f'{EXT}/smoke/{n}'
size = lambda n: Image.open(path(n)).size

bysrc = {}
for s, n, c in hit:
    bysrc.setdefault(s, []).append(n)
bysize = {}
for s, v in bysrc.items():
    bysize.setdefault(size(v[0]), []).append(s)
print('\n분할 1 의 E 출처와 그림 크기')
for k, v in sorted(bysize.items()):
    print(f'  {k}   {" · ".join(v)}')

m = YOLO(f'{ROUT}/s1/best.pt')


def mx(paths):
    out = []
    for r in m(paths, conf=CMIN, verbose=False):
        c = r.boxes.conf
        out.append(float(c.max()) if len(c) else 0.0)
    return out


print(f'\n{"파일":22}{"기록":>8}{"가 한장":>9}{"나 같은출처":>12}'
      f'{"다 크기섞음":>12}{"라 크기같음":>12}')
print('-' * 76)
for t in TARGET:
    if t not in rec:
        print(f'{t:22}  기록에 없음')
        continue
    src, c0 = rec[t]
    sz = size(t)
    a = mx([path(t)])[0]
    same = [n for n in bysrc[src] if n != t][:15]
    b = mx([path(t)] + [path(n) for n in same])[0]
    other = [n for s2, v in bysrc.items() if s2 != src for n in v if size(n) != sz]
    c = mx([path(t)] + [path(n) for n in other[:15]])[0] if other else float('nan')
    eq = [n for s2, v in bysrc.items() if s2 != src for n in v if size(n) == sz]
    d = mx([path(t)] + [path(n) for n in eq[:15]])[0] if eq else float('nan')
    print(f'{t:22}{c0:8.3f}{a:9.3f}{b:12.3f}{c:12.3f}{d:12.3f}')

print('\n읽는 법')
print('  가 · 나 가 기록과 같고 다 만 다르면  -> 원인은 **묶음 안의 크기가 섞인 것**')
print('  라 도 기록과 같으면                  -> 묶는 것 자체는 죄가 없음')
print('  가 부터 이미 기록과 다르면           -> 원인이 다른 데 있음. 다시 생각해야 함')
print('  라 가 nan 이면 크기가 같은 다른 출처가 없어 그 칸은 못 잰 것임')

print('\nargs.rect =', getattr(m.predictor.args, 'rect', '그런 값이 없음'))
print('  이 값이 True 라야 auto 가 켜짐. False 면 늘 정사각이라 갈릴 수가 없음')
