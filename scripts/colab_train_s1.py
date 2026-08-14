import os, json, time, shutil, glob, subprocess, sys
from google.colab import drive

SRC   = '/content/drive/MyDrive/smoke_frames'
DOUT  = f'{SRC}/ds_s1'
WORK  = '/content/ds_s1'
RUNS  = '/content/runs'
ROUT  = f'{SRC}/runs_s1'

SPLITS = [1, 3, 4, 5]
SKIP_DONE = True
MODEL = 'yolov8s.pt'
EPOCHS, IMGSZ, BATCH = 60, 640, 16
SEED = 1

drive.mount('/content/drive')
os.makedirs(ROUT, exist_ok=True)

print('=' * 78)
print('학습 — 화재 저장소 6회차와 같은 자리에서 고름')
print('=' * 78)
print(f'  모델     {MODEL}    에폭 {EPOCHS} · imgsz {IMGSZ} · 배치 {BATCH}')
print(f'  data     val: images/train    (그쪽 synthesize_smoke.py 278-280줄과 같음)')
print(f'  가중치    best.pt             (그쪽 round6_smoke.ipynb 셀 [6] 과 같음)')
print(f'  분할     {SPLITS}   (Drive 에 이미 있으면 건너뜀: {SKIP_DONE})')
print('=' * 78)
print('  **한계** — val 이 train 과 같으므로 best.pt 는 가장 잘 외운 체크포인트를 고를 수')
print('  있음. 조기 종료의 뜻이 없고 여기 나오는 mAP 는 학습 자료 위의 값이라 성능이')
print('  아님. **판정은 1층 평가군으로만 함**')
print('  **증강** — 2층이 안 정한 칸임. ultralytics 기본값을 그대로 씀. 그쪽도 인자를')
print('  안 넘겨 기본값이었으므로 `그쪽과 같게 둔 것`이고 우리 자료를 보고 고른 값이 아님')
print('  실제로 쓰인 증강 값은 아래에 전부 찍고 args.yaml 로 남김')
print('=' * 78)



r"""환경 — 한 번만"""
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO

print(f'\nultralytics {ultralytics.__version__} · torch {torch.__version__}')
print(f'CUDA {torch.cuda.is_available()} · '
      f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음"}')
if not torch.cuda.is_available():
    print('  **GPU 가 없음. 런타임 유형을 GPU 로 바꿀 것**')


TIMES = {}
for SPLIT in SPLITS:
    out = f'{ROUT}/s{SPLIT}'
    if SKIP_DONE and os.path.exists(f'{out}/best.pt'):
        print(f'\n분할 {SPLIT} — Drive 에 best.pt 가 이미 있어 건너뜀')
        continue
    print('\n' + '=' * 78)
    print(f'분할 {SPLIT}')
    print('=' * 78)
    root = f'{WORK}/split{SPLIT}'
    if not os.path.exists(f'{root}/data.yaml'):
        z = f'{DOUT}/split{SPLIT}.zip'
        if not os.path.exists(z):
            raise SystemExit(f'{z} 가 없음 — 본 합성을 먼저 돌릴 것')
        os.makedirs(root, exist_ok=True)
        shutil.unpack_archive(z, root)
        print(f'\n분할 {SPLIT} 압축본을 품')

    open(f'{root}/data.yaml', 'w').write(
        f'path: {os.path.abspath(root)}\ntrain: images/train\nval: images/train\n'
        f'nc: 1\nnames: [\'smoke\']\n')

    nimg = len(glob.glob(f'{root}/images/train/*.jpg'))
    nlab = len(glob.glob(f'{root}/labels/train/*.txt'))
    man = json.load(open(f'{root}/manifest.json'))
    print(f'\n이미지 {nimg}장 · 라벨 {nlab}장 · manifest {len(man["rows"])}줄')
    ok = nimg == nlab == len(man['rows']) == 476
    print(f'  [검산] 셋이 다 476 인가 — {"통과" if ok else "**실패 — 멈춤**"}')
    assert ok, '장수가 안 맞음'
    print(f'  {open(f"{root}/data.yaml").read().strip()}')
    name = f's{SPLIT}'
    shutil.rmtree(f'{RUNS}/{name}', ignore_errors=True)
    t0 = time.time()
    m = YOLO(MODEL)
    m.train(data=f'{root}/data.yaml', epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            project=RUNS, name=name, seed=SEED, exist_ok=True, verbose=True)
    dt = time.time() - t0
    print(f'\n분할 {SPLIT} 학습 시간 **{dt/60:.1f}분** ({dt:.0f}초)')

    rd = f'{RUNS}/{name}'
    best = f'{rd}/weights/best.pt'
    last = f'{rd}/weights/last.pt'
    print(f'\n  [검산] best.pt {"있음" if os.path.exists(best) else "**없음**"} · '
          f'last.pt {"있음" if os.path.exists(last) else "**없음**"}')

    args = f'{rd}/args.yaml'
    if os.path.exists(args):
        print('\n실제로 쓰인 증강 값 — 2층이 안 정한 칸이므로 전부 적어 둠')
        keep = ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear',
                'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'copy_paste',
                'erasing', 'close_mosaic', 'auto_augment', 'augment', 'rect', 'seed',
                'optimizer', 'lr0', 'lrf', 'warmup_epochs', 'patience', 'cos_lr')
        for line in open(args):
            k = line.split(':')[0].strip()
            if k in keep:
                print('    ' + line.rstrip())

    csv = f'{rd}/results.csv'
    if os.path.exists(csv):
        rows = [l.strip().split(',') for l in open(csv) if l.strip()]
        head = [h.strip() for h in rows[0]]
        print(f'\n마지막 에폭 (학습 자료 위의 값이므로 성능이 아님)')
        for k, v in zip(head, rows[-1]):
            if 'metrics' in k or k.strip() == 'epoch':
                print(f'    {k.strip():<28}{v.strip()}')

    out = f'{ROUT}/{name}'
    os.makedirs(out, exist_ok=True)
    for f in ('weights/best.pt', 'weights/last.pt', 'results.csv', 'args.yaml'):
        p = f'{rd}/{f}'
        if os.path.exists(p):
            shutil.copy(p, f'{out}/{os.path.basename(f)}')
    json.dump({'split': SPLIT, 'model': MODEL, 'epochs': EPOCHS, 'imgsz': IMGSZ,
               'batch': BATCH, 'seed': SEED, 'seconds': dt,
               'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'n_images': nimg},
              open(f'{out}/train_meta.json', 'w'), ensure_ascii=False, default=float)
    print(f'\n-> {out} 에 best.pt · last.pt · results.csv · args.yaml · train_meta.json 을 넣음')

    TIMES[SPLIT] = dt

print('\n' + '=' * 78)
print('요약')
print('=' * 78)
for k, v in TIMES.items():
    print(f'  분할 {k}  {v/60:>6.1f}분')
if TIMES:
    print(f'  합 {sum(TIMES.values())/3600:.2f}시간')
print(f'  가중치는 {ROUT}/s*/best.pt 에 있음')
print('\n다음 — 평가 스크립트. 1층 보고 규칙 여섯 줄을 전부 채울 것 —')
print('  E · D · 판별비 · conf 네 점 · 다섯 분할 각각 · 출처별 ·')
print('  배경 오탐군 두 출처 · 논현중 포함/제외 · 추론 시간')
