# 실음성/배경 재빌드 — 조리 영상 QC 셀. base=ptrain_b79(D-Fire). 상위: docs/HANDOFF_DFIRE_YOLO11.md · 기준: docs/PREREGISTER_DFIRE_QC.md
# CELL 16=사용성 프로브(디코딩·길이·fps·해상도·프레임수·추정수율·플래그+콘택트시트). 육안(장면·불)은 사용자 풀해상 병렬 검수=정본.
# ★Colab 콜드 재연결 대비: 완전 자립(마운트·설치 가드·폴더 탐색). 이전 셀 변수 의존 없음.

# ========== CELL 16: 조리 영상 QC 프로브 (사용성 감사) ==========
import os, sys, glob, json, subprocess, unicodedata, re
from collections import defaultdict
NFC = lambda s: unicodedata.normalize('NFC', s)   # ★드라이브 파일명이 NFD(분해형)라 매칭 전 조합형 정규화 필수

# 1) Drive 마운트 (콜드 재연결 시 /content 초기화 → 재마운트 필요)
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
assert os.path.exists('/content/drive/MyDrive'), 'Drive 마운트 실패'

# 2) ffprobe/ffmpeg 가드 (Colab 기본 탑재이나 재연결 안전)
def _has(cmd):
    return subprocess.run(['which', cmd], capture_output=True, text=True).returncode == 0
if not (_has('ffprobe') and _has('ffmpeg')):
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], check=False)
assert _has('ffprobe') and _has('ffmpeg'), 'ffmpeg/ffprobe 없음 — 설치 실패'

# 3) 영상 폴더 탐색 (경로 하드코딩 안 함 · hankookro 공유는 '내 드라이브에 바로가기 추가' 필요)
FOLDER_OVERRIDE = ''            # 경로 알면 직접 지정
FOLDER_NAME = '조리 데이터 영상'
if FOLDER_OVERRIDE and os.path.isdir(FOLDER_OVERRIDE):
    VDIR = FOLDER_OVERRIDE
else:
    hits = [d for d in glob.glob(f'/content/drive/MyDrive/**/{FOLDER_NAME}', recursive=True) if os.path.isdir(d)]
    assert hits, (f"'{FOLDER_NAME}' 폴더 못 찾음. hankookro 공유라면 Drive에서 우클릭 '내 드라이브에 바로가기 추가' 후 재실행, "
                  f"또는 FOLDER_OVERRIDE에 경로 지정.")
    if len(hits) > 1: print('폴더 후보 여러 개:', hits, '→ 첫 번째 사용. 다르면 FOLDER_OVERRIDE 지정.')
    VDIR = hits[0]
print('영상 폴더:', VDIR)

# 4) 영상 수집
EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm')
vids = sorted(p for p in glob.glob(f'{VDIR}/**/*', recursive=True)
              if os.path.splitext(p)[1].lower() in EXTS)
print(f'영상 파일 {len(vids)}개  (사전등록 인벤토리 = 28 · 다르면 폴더/스크롤 재확인)')
assert vids, 'VDIR에 영상 없음'

# 5) 학교/유형 파싱 (규약: 학교_유형(단계).ext · '개원중cctv'→개원중)
def parse_school(path):
    base = NFC(os.path.splitext(os.path.basename(path))[0])
    head = re.split(r'[_\s]', base, 1)[0]     # 밑줄 또는 공백 구분(영동중은 공백)
    return head.replace('cctv', '').strip()
def parse_dish(path):
    n = NFC(os.path.basename(path))
    for k in ('튀김', '볶음', '국탕'):
        if k in n: return k
    return '기타'

# 6) 지표: ffprobe → 실패/불완전 시 cv2 폴백(개원중 .avi 등 ffprobe 빈값 대응)
def _fps(s):
    for key in ('avg_frame_rate', 'r_frame_rate'):   # '0/0' 은 건너뛰고 유효값 채택
        v = s.get(key) or ''
        try:
            a, b = v.split('/'); f = float(a) / float(b) if float(b) else 0.0
            if f > 0: return f
        except Exception:
            pass
    return 0.0

def probe(path):
    # 1차 ffprobe
    try:
        r = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
                            '-show_format', '-show_streams', path],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and r.stdout.strip():
            j = json.loads(r.stdout)
            vs = [s for s in j.get('streams', []) if s.get('codec_type') == 'video']
            if vs:
                s = vs[0]
                dur = float(j.get('format', {}).get('duration') or s.get('duration') or 0)
                fps = _fps(s)
                w, h = int(s.get('width', 0)), int(s.get('height', 0))
                try: nb = int(s.get('nb_frames'))
                except (TypeError, ValueError): nb = 0
                if not nb and dur and fps: nb = int(dur * fps)
                if not dur and nb and fps: dur = nb / fps
                if dur and (w or h):
                    return {'ok': True, 'dur': dur, 'fps': fps, 'w': w, 'h': h, 'nb': nb, 'codec': s.get('codec_name', '?'), 'src': 'ffprobe'}
    except Exception:
        pass
    # 2차 cv2 폴백
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            dur = nb / fps if (nb and fps) else 0.0
            if (dur or nb) and (w or h):
                return {'ok': True, 'dur': dur, 'fps': fps, 'w': w, 'h': h, 'nb': nb, 'codec': 'cv2', 'src': 'cv2'}
    except Exception as e:
        return {'ok': False, 'err': 'cv2:' + str(e)[:40]}
    return {'ok': False, 'err': 'ffprobe+cv2 fail'}

# 7) 잠정 임계 (★프로브 분포 보고 확정 — 자동 탈락 아님, 사람 판정용 플래그)
STRIDE_S = 2.0        # 배경 샘플 간격(초) — 근접중복 억제
MIN_USABLE_S = 8.0    # 이보다 짧으면 SHORT
LOW_RES_MIN = 240     # 짧은 변이 이 미만이면 LOWRES

rows = []
for p in vids:
    info = probe(p)
    sz = os.path.getsize(p) / 1e6
    sch, dish = parse_school(p), parse_dish(p)
    flags = []
    if not info['ok']:
        flags.append('DECODE_FAIL')
        rows.append((sch, dish, os.path.basename(p), sz, info, 0, flags)); continue
    est = int(info['dur'] // STRIDE_S) if info['dur'] else 0
    if info['dur'] < MIN_USABLE_S: flags.append('SHORT')
    if min(info['w'], info['h']) and min(info['w'], info['h']) < LOW_RES_MIN: flags.append('LOWRES')
    if est <= 2: flags.append('YIELD<=2')
    rows.append((sch, dish, os.path.basename(p), sz, info, est, flags))

# 8) 영상별 표
print('\n=== 영상별 기술지표 (사용성 감사) ===')
print(f'{"학교":<9}{"유형":<5}{"길이s":>7}{"fps":>6}{"해상도":>11}{"프레임":>8}{"MB":>7}{"수율":>6}  플래그')
for sch, dish, nm, sz, info, est, flags in rows:
    if info['ok']:
        res = f'{info["w"]}x{info["h"]}'
        print(f'{sch:<9}{dish:<5}{info["dur"]:>7.1f}{info["fps"]:>6.1f}{res:>11}{info["nb"]:>8}{sz:>7.1f}{est:>6}  {",".join(flags)}')
    else:
        print(f'{sch:<9}{dish:<5}{"-":>7}{"-":>6}{"-":>11}{"-":>8}{sz:>7.1f}{est:>6}  {",".join(flags)} ({info.get("err","")})')

# 9) 학교·유형 집계 (부적합 후보 제외 시 남는 규모 가늠 — 자동 제외는 안 함)
def agg(rows, keep=None):
    sset, dcnt, nv, te = set(), defaultdict(int), 0, 0
    for sch, dish, nm, sz, info, est, flags in rows:
        if keep is not None and not keep(info, est, flags): continue
        sset.add(sch); dcnt[dish] += 1; nv += 1; te += est
    return sset, dcnt, nv, te

aS, aD, aN, aE = agg(rows)
print(f'\n[전체]        영상 {aN} · 학교 {len(aS)} · 유형 {dict(aD)} · 추정수율합 {aE}')
okkeep = lambda info, est, flags: info['ok'] and not flags
oS, oD, oN, oE = agg(rows, okkeep)
print(f'[플래그無 통과] 영상 {oN} · 학교 {len(oS)} · 유형 {dict(oD)} · 추정수율합 {oE}')
print('  → 플래그는 자동 탈락 아님. 사용자 풀해상 육안(장면·불)으로 적합/부분적합/부적합 최종 판정.')
risk = sorted(aS - oS)
if risk: print('  플래그로 통과셋서 빠질 위험 학교:', risk)
for dish in ('튀김', '볶음', '국탕'):
    schools = sorted({s for s, d, *_ in rows if d == dish})
    print(f'  [{dish}] 학교: {schools}')

# 10) 썸네일 콘택트시트 (영상당 중간 1프레임 → 그리드 PNG를 Drive 저장; 사용자·나 교차확인)
OUT_THUMB = '/content/drive/MyDrive/qc_contact_sheet.png'
try:
    import numpy as np, matplotlib.pyplot as plt
    from PIL import Image
    try:   # 한글 폰트(콘택트시트 라벨 깨짐 방지) — best effort
        import matplotlib.font_manager as fm
        cand = [f for f in fm.findSystemFonts() if 'Nanum' in f]
        if not cand:
            subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'], check=False)
            cand = [f for f in fm.findSystemFonts() if 'Nanum' in f]
        picked = None
        for fp in cand:
            try:
                fm.fontManager.addfont(fp)          # ★등록해야 matplotlib이 family 인식
                picked = picked or fm.FontProperties(fname=fp).get_name()
            except Exception:
                pass
        if picked:
            plt.rcParams['font.family'] = picked
            plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    tmp = '/content/qc_thumbs'; os.makedirs(tmp, exist_ok=True)
    thumbs = []
    for i, (sch, dish, nm, sz, info, est, flags) in enumerate(rows):
        cand = [v for v in vids if os.path.basename(v) == nm]
        p = cand[0] if cand else os.path.join(VDIR, nm)
        op = os.path.join(tmp, f'{i:02d}.jpg')
        t = (info['dur'] / 2) if (info['ok'] and info['dur']) else 0
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(t), '-i', p,
                        '-frames:v', '1', '-vf', 'scale=320:-1', op], check=False)
        thumbs.append((op, f'{sch}·{dish}·{",".join(flags) or "ok"}', nm))
    K = len(thumbs); cols = 4; rows_n = (K + cols - 1) // cols
    fig, ax = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n))
    ax = np.array(ax).reshape(rows_n, cols)
    for i in range(rows_n * cols):
        a = ax[i // cols, i % cols]; a.axis('off')
        if i >= K: continue
        op, cap, nm = thumbs[i]
        if os.path.exists(op) and os.path.getsize(op) > 0:
            a.imshow(Image.open(op).convert('RGB'))
        else:
            a.text(0.5, 0.5, 'NO FRAME', ha='center', va='center', fontsize=9)
        a.set_title(f'{cap}\n{nm[:24]}', fontsize=7)
    plt.tight_layout(); plt.savefig(OUT_THUMB, dpi=100, bbox_inches='tight'); plt.show()
    print(f'\n콘택트시트 저장 → {OUT_THUMB}  (풀해상 판정은 원본 재생이 정본)')
except Exception as e:
    print('썸네일 생성 스킵:', str(e)[:100])

print('\n※ 다음: 이 표+콘택트시트 + 사용자 육안(장면·불·timestamp) → 적합/부분/부적합 확정 →')
print('  검증된 [영상·학교·유형]로 분할 사전등록(PREREGISTER_DFIRE_QC.md) → 프레임 추출 → base FP 재측정.')


# ========== CELL 17: 확정 분할대로 프레임 추출 (eval/synth 배경·무화재) ==========
# 사전등록(PREREGISTER_DFIRE_QC.md §분할 확정) 그대로. STRIDE 샘플·학교별 cap·가시불꽃 프레임은 spot-check로 검증.
# ★자립(마운트·ffmpeg 가드·폴더 탐색) · env안전(rmtree 없음·스킵방식). CELL 16의 NFC/parse 함수 필요.
import os, sys, glob, subprocess, shutil, unicodedata, re
from collections import defaultdict
NFC = lambda s: unicodedata.normalize('NFC', s)
if 'parse_school' not in dir():
    def parse_school(path):
        head = re.split(r'[_\s]', NFC(os.path.splitext(os.path.basename(path))[0]), maxsplit=1)[0]
        return head.replace('cctv', '').strip()

# 0) 마운트·ffmpeg·폴더 (콜드 재연결 자립)
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], check=False)
FOLDER_OVERRIDE = ''
if FOLDER_OVERRIDE and os.path.isdir(FOLDER_OVERRIDE):
    VDIR = FOLDER_OVERRIDE
else:
    hits = [d for d in glob.glob('/content/drive/MyDrive/**/조리 데이터 영상', recursive=True) if os.path.isdir(d)]
    assert hits, "'조리 데이터 영상' 폴더 못 찾음 — 바로가기 추가 또는 FOLDER_OVERRIDE 지정."
    VDIR = hits[0]
EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm')
vids = sorted(p for p in glob.glob(f'{VDIR}/**/*', recursive=True) if os.path.splitext(p)[1].lower() in EXTS)
print('영상 폴더:', VDIR, '·', len(vids), '개')

# 1) 확정 분할 (사전등록 §분할 확정) — 학교 단위, 교집합 ∅
SPLIT = {
    '개원중': 'eval', '부산체고': 'eval', '논현중': 'eval', '인화여중': 'eval', '영동중': 'eval',
    '금정초': 'synth', '남일고': 'synth', '로봇고': 'synth', '원촌중': 'synth', '진선여고': 'synth',
    '울산현대차': 'synth', '내곡중': 'synth', '숭곡중': 'synth',
}
EXCLUDE_SUB = ('오토틸팅',)          # 부적합 확정(육안): 화질 불량·비조리
STRIDE_S = 2.0                        # 샘플 간격(초)
CAP_PER_SCHOOL = 150                  # 학교별 상한(균형·큰영상 지배 방지) — 잠정

LOCAL = '/content/realneg_extract'    # 임시(안전)
OUT = '/content/drive/MyDrive/realneg_frames'
os.makedirs(LOCAL, exist_ok=True)

# 2) 영상별 stride 추출 → 학교별 풀
school_frames = defaultdict(list)     # school -> [(group, vstem, framepath)]
skipped = []
for p in vids:
    nm = os.path.basename(p)
    if any(x in NFC(nm) for x in EXCLUDE_SUB):
        skipped.append(('제외영상', nm)); continue
    sch = parse_school(p); grp = SPLIT.get(sch)
    if grp is None:
        skipped.append(('분할외학교:' + sch, nm)); continue
    vstem = re.sub(r'[^0-9A-Za-z가-힣]+', '_', NFC(os.path.splitext(nm)[0]))[:40]
    tmpd = f'{LOCAL}/{grp}__{sch}__{vstem}'
    os.makedirs(tmpd, exist_ok=True)
    if not any(f.endswith('.jpg') for f in os.listdir(tmpd)):    # 이미 추출됐으면 스킵(재실행 안전)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', p,
                        '-vf', f'fps={1.0/STRIDE_S:.6f}', '-qscale:v', '2',
                        f'{tmpd}/f_%05d.jpg'], check=False)
    for fr in sorted(glob.glob(f'{tmpd}/*.jpg')):
        school_frames[sch].append((grp, vstem, fr))

# 3) 학교별 cap(균등 서브샘플) → Drive 저장
import numpy as np
manifest = []
for sch, items in school_frames.items():
    grp = items[0][0]
    if len(items) > CAP_PER_SCHOOL:
        idx = sorted(set(np.linspace(0, len(items) - 1, CAP_PER_SCHOOL).round().astype(int)))
        items = [items[i] for i in idx]
    dst = f'{OUT}/{grp}/{sch}'; os.makedirs(dst, exist_ok=True)
    for k, (g, vstem, fr) in enumerate(items):
        shutil.copy(fr, f'{dst}/{vstem}_{k:04d}.jpg')
    manifest.append((grp, sch, len(items)))

# 4) 요약 + manifest 저장
print('\n=== 추출 결과 (학교별) ===')
gtot = defaultdict(int)
for grp in ('eval', 'synth'):
    print(f'[{grp}]')
    for g, sch, n in sorted([m for m in manifest if m[0] == grp], key=lambda x: -x[2]):
        print(f'   {sch:<9} {n:5d}'); gtot[grp] += n
print(f'\n합계 · eval {gtot["eval"]} · synth {gtot["synth"]} · 총 {sum(gtot.values())}')
with open(f'{OUT}/manifest.csv', 'w', encoding='utf-8') as f:
    f.write('group,school,frames\n')
    for grp, sch, n in manifest: f.write(f'{grp},{sch},{n}\n')
print('manifest →', f'{OUT}/manifest.csv')
if skipped:
    print('스킵:', skipped)

# 5) 가시 불꽃 spot-check 몽타주 (그룹별 균등 샘플 — 화구/버너 불꽃 섞였나 육안 검증)
try:
    import matplotlib.pyplot as plt
    from PIL import Image
    for grp in ('eval', 'synth'):
        allf = sorted(glob.glob(f'{OUT}/{grp}/**/*.jpg', recursive=True))
        if not allf: continue
        K = min(24, len(allf))
        pick = [allf[i] for i in np.linspace(0, len(allf) - 1, K).round().astype(int)]
        cols = 6; rows = (K + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows)); ax = np.array(ax).reshape(rows, cols)
        for i in range(rows * cols):
            a = ax[i // cols, i % cols]; a.axis('off')
            if i < K: a.imshow(Image.open(pick[i]).convert('RGB'))
        fig.suptitle(f'{grp} spot-check (n={len(allf)}) — 불꽃 섞였나 확인', fontsize=11)
        op = f'{OUT}/spotcheck_{grp}.png'
        plt.tight_layout(); plt.savefig(op, dpi=90, bbox_inches='tight'); plt.show()
        print(f'spot-check → {op}')
except Exception as e:
    print('spot-check 스킵:', str(e)[:80])

print('\n※ 다음: spot-check서 불꽃 없음 확인 →')
print('  eval 프레임으로 base(ptrain_b79) FP 재측정(CELL 14 방식) · synth 프레임은 (A) 합성 배경으로 확보.')


# ========== CELL 18: eval 프레임에 base(ptrain_b79) FP 재측정 (누수 없는 급식실 무화재) ==========
# 구 nofire_kitchen(3.9%@0.25) 대체. eval 5개교 무화재 프레임에 base → 전체+학교별 헛불률 + 오탐 몽타주.
import os, glob, subprocess, sys
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'best.pt(ptrain_b79) 없음 — Drive 확인: ' + W
EVAL_DIR = '/content/drive/MyDrive/realneg_frames/eval'
assert os.path.isdir(EVAL_DIR), 'eval 프레임 없음 — CELL 17 먼저: ' + EVAL_DIR

m = YOLO(W); CONFS = (0.05, 0.25, 0.50)
schools = sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(f'{EVAL_DIR}/{d}'))
rows = []; per = defaultdict(list)
for sch in schools:
    for ip in sorted(glob.glob(f'{EVAL_DIR}/{sch}/*.jpg')):
        r = m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
        has = (r.boxes is not None and len(r.boxes))
        tc = float(r.boxes.conf.max().cpu()) if has else 0.0
        xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
        cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
        rows.append((sch, ip, tc, xy, cf)); per[sch].append(tc)

tc_all = np.array([x[2] for x in rows]); N = len(rows)
print(f'=== eval 급식실 무화재 헛불률(B) · N={N} · 구 nofire_kitchen 3.9%@0.25 대체 ===')
print(f'{"":12}' + ''.join(f'conf{c:>6}' for c in CONFS))
print(f'{"[전체]":12}' + ''.join(f'{(tc_all>=c).mean():>10.3f}' for c in CONFS))
for sch in schools:
    t = np.array(per[sch])
    print(f'{sch:12}' + ''.join(f'{(t>=c).mean():>10.3f}' for c in CONFS) + f'   (n={len(t)})')

C0 = 0.25
fires = sorted([x for x in rows if x[2] >= C0], key=lambda z: -z[2])
print(f'\nconf>={C0} 헛불 {len(fires)}장 (전체 {N}장 중 {len(fires)/N:.1%}) — 무엇에 오탐?')
try:
    import matplotlib.pyplot as plt, matplotlib.patches as patches
    from PIL import Image
    K = min(24, len(fires))
    if K:
        cols = 4; rr = (K + cols - 1) // cols
        fig, ax = plt.subplots(rr, cols, figsize=(4 * cols, 3.2 * rr)); ax = np.array(ax).reshape(rr, cols)
        for i in range(rr * cols):
            a = ax[i // cols, i % cols]; a.axis('off')
            if i >= K: continue
            sch, ip, tc, xy, cf = fires[i]; im = Image.open(ip).convert('RGB'); a.imshow(im)
            for (x1, y1, x2, y2), c in zip(xy, cf):
                if c >= C0: a.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            a.set_title(f'{sch} top{tc:.2f}', fontsize=8)
        op = '/content/drive/MyDrive/eval_fp_montage.png'
        plt.tight_layout(); plt.savefig(op, dpi=95, bbox_inches='tight'); plt.show()
        print('몽타주 →', op, '(풀해상 판정이 정본)')
    else:
        print('  conf>=0.25 헛불 0장.')
except Exception as e:
    print('몽타주 스킵:', str(e)[:80])


# ========== (A) 합성 파이프라인 v0 — CELL 20~22 ==========
# 소스 결정: NIST Stovetop 옥수수유 화재 스냅샷(퍼블릭도메인 추정·발표 전 약관확인). MP4는 HRR오버레이 타임랩스라 폐기, 스냅샷 JPG만.
# (CELL 19=오일화재 로컬 인벤토리는 공개데이터셋 경로로 대체돼 미수록.)

# ========== CELL 20: NIST Stovetop 옥수수유 화재 스냅샷 다운로드 ==========
import os, urllib.request, urllib.parse, glob
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
os.makedirs(OUT, exist_ok=True)
BASE = 'https://nist-el-nfrlhrr.s3.amazonaws.com/HRR/ASSET_FILES/Corn Oil/video'
imgs = {'1574198232-Evt1.jpg':'fuel_pour', '1574198232-Evt2.jpg':'heating_on',
        '1574198232-Evt3.jpg':'ignition_FIRE', '1574198232-EvtP.jpg':'peak_FIRE',
        '1574198232-Evt4.jpg':'fire_out'}
for fn, tag in imgs.items():
    url = urllib.parse.quote(f'{BASE}/{fn}', safe=':/')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, timeout=60).read()
        if data[:2] == b'\xff\xd8':
            open(f'{OUT}/{tag}__{fn}', 'wb').write(data); print('OK  ', tag, f'{len(data)//1024}KB')
        else:
            print('NOT-JPEG', tag)
    except Exception as e:
        print('FAIL', tag, str(e)[:60])
print('저장 →', OUT)


# ========== CELL 21: NIST 오일 불꽃 크롭(마스킹) + synth 배경 합성 v0 + 미리보기 ==========
import os, glob, numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as patches

SRC  = '/content/drive/MyDrive/firecrop_src/nist_stovetop_cornoil'
BG   = '/content/drive/MyDrive/realneg_frames/synth'
OUTI = '/content/drive/MyDrive/synth_composite_v0/images'
OUTL = '/content/drive/MyDrive/synth_composite_v0/labels'
os.makedirs(OUTI, exist_ok=True); os.makedirs(OUTL, exist_ok=True)

def extract_flame(path):     # 어두운 배경 → 주황∪백열 마스크 + 최대 연결성분(잡픽셀 제거) + 소프트 알파
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = im[...,0], im[...,1], im[...,2]
    lum = 0.299*R + 0.587*G + 0.114*B
    mask = ((R > B + 30) & (R > 90)) | (lum > 210)
    if mask.sum() < 50: return None
    lbl, n = ndimage.label(mask)
    counts = np.bincount(lbl.ravel()); counts[0] = 0
    m = ndimage.binary_dilation(lbl == counts.argmax(), iterations=3)
    ys, xs = np.where(m); pad = 10
    x0=max(0,xs.min()-pad); y0=max(0,ys.min()-pad)
    x1=min(im.shape[1]-1,xs.max()+pad); y1=min(im.shape[0]-1,ys.max()+pad)
    crop = im[y0:y1, x0:x1]; mm = m[y0:y1, x0:x1].astype(np.float32); l = lum[y0:y1, x0:x1]
    return Image.fromarray(np.dstack([crop, np.clip(l/160.0, 0, 1) * mm * 255]).astype(np.uint8))

flames = {}
for p in sorted(glob.glob(f'{SRC}/*FIRE*.jpg')):
    f = extract_flame(p)
    if f: flames[os.path.basename(p).split('__')[0]] = f
print('추출 불꽃:', list(flames), [f.size for f in flames.values()])
assert flames, '불꽃 추출 실패'

bgs = sorted(glob.glob(f'{BG}/**/*.jpg', recursive=True))
sel = [bgs[i] for i in np.linspace(0, len(bgs)-1, 12).round().astype(int)]
keys = list(flames); comps = []
for i, bp in enumerate(sel):
    bg = Image.open(bp).convert('RGB'); W, H = bg.size
    fl = flames[keys[i % len(keys)]]
    th = int(H*0.35); tw = max(1, int(fl.width*th/fl.height))
    fl_r = fl.resize((tw, max(1, th)))
    px = int(W*0.5 - tw/2); py = int(H*0.55 - th/2)     # v0: 위치 대충(frozen-base recall 무관)
    canvas = bg.convert('RGBA'); canvas.alpha_composite(fl_r, (px, py))
    canvas.convert('RGB').save(f'{OUTI}/comp_{i:03d}.jpg', quality=92)
    a = np.asarray(fl_r)[...,3]; ys, xs = np.where(a > 10)
    bx0,by0,bx1,by1 = px+xs.min(), py+ys.min(), px+xs.max(), py+ys.max()
    open(f'{OUTL}/comp_{i:03d}.txt','w').write(
        f'0 {(bx0+bx1)/2/W:.6f} {(by0+by1)/2/H:.6f} {(bx1-bx0)/W:.6f} {(by1-by0)/H:.6f}\n')
    comps.append((f'{OUTI}/comp_{i:03d}.jpg', (bx0,by0,bx1,by1)))

K=len(comps); cols=4; rows=(K+cols-1)//cols
fig,ax=plt.subplots(rows,cols,figsize=(4*cols,3*rows)); ax=np.array(ax).reshape(rows,cols)
for i in range(rows*cols):
    a=ax[i//cols,i%cols]; a.axis('off')
    if i>=K: continue
    p,(x0,y0,x1,y1)=comps[i]; a.imshow(Image.open(p))
    a.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=2))
plt.tight_layout(); plt.savefig('/content/drive/MyDrive/synth_composite_v0/preview.png',dpi=90,bbox_inches='tight'); plt.show()
print('합성', K, '장 저장 →', OUTI)


# ========== CELL 22: 합성 v0에 base(ptrain_b79) recall — base가 합성 불을 잡나 ==========
import os, glob, subprocess, sys
try: import ultralytics
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
from ultralytics import YOLO
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image

if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
W = '/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt'
assert os.path.exists(W), 'best.pt 없음: ' + W
IMG = '/content/drive/MyDrive/synth_composite_v0/images'
imgs = sorted(glob.glob(f'{IMG}/*.jpg')); assert imgs, '합성 이미지 없음 — CELL 21 먼저'

m = YOLO(W); CONFS = (0.05, 0.25, 0.50); det = []
for ip in imgs:
    r = m.predict(ip, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
    has = (r.boxes is not None and len(r.boxes))
    tc = float(r.boxes.conf.max().cpu()) if has else 0.0
    xy = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0,4))
    cf = r.boxes.conf.cpu().numpy() if has else np.zeros(0)
    det.append((ip, tc, xy, cf))
tc = np.array([d[1] for d in det]); N = len(det)
print(f'=== 합성 v0 · base recall · N={N} ===')
for c in CONFS: print(f'  conf>={c:.2f}: recall {(tc>=c).mean():.3f}  ({int((tc>=c).sum())}/{N})')
print('  top conf:', np.round(sorted(tc, reverse=True), 2))
# ※ recall 1.0 = 프록시 천장(실 불꽃은 base가 무조건 검출) → 판별력 없음. 다음=ablation(스케일·열화 sweep).

C0 = 0.25; K = len(det); cols = 4; rows = (K + cols - 1) // cols
fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 3*rows)); ax = np.array(ax).reshape(rows, cols)
for i in range(rows*cols):
    a = ax[i//cols, i%cols]; a.axis('off')
    if i >= K: continue
    ip, t, xy, cf = det[i]; a.imshow(Image.open(ip))
    for (x1,y1,x2,y2), c in zip(xy, cf):
        if c >= C0: a.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=2))
    a.set_title(f'top {t:.2f}', fontsize=9)
plt.tight_layout(); plt.savefig('/content/drive/MyDrive/synth_composite_v0/base_detect.png',dpi=90,bbox_inches='tight'); plt.show()
print('검출 시각화 저장.')
