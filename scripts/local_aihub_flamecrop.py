#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================
#  AI-Hub 71751 불꽃 크롭 소스 추출 · 파이프라인 step4 (합성 불꽃 소스 서치)
# =====================================================================================
#  ★★★ 데이터 취급 (라이선스 주체=사용자 승인 · 2026-08-29) ★★★
#  - 원 라이선스=국외반출 금지이나, 사용자가 이 프로젝트 한정 승인:
#    "불꽃 셋 추출 목적 · 비배포 · 비상업 · 결과물 출처표기" 조건으로 Colab/Drive 사용 OK.
#  - ★단 데이터셋 원본/파생의 제3자 공유·재배포는 여전히 금지(라이선스 명시).
#  - 출처표기: 결과물/모델에 "AI-Hub 71751 화재 발생 예측 영상" 명기.
#  - Claude/AI는 AI-Hub 이미지를 context로 읽지 않음(제3자 전송 최소화) — 육안 검수=사용자.
#
#  환경(2026-08-28 점검): Python 3.10.11 · Pillow 12.3.0 · GPU 없음(CPU).
#    - audit 모드: 표준 라이브러리만(zipfile/json) — 설치 불필요.
#    - crop  모드: Pillow 필요(이미 설치됨).
#  실행 예:
#    py -3.10 scripts/local_aihub_flamecrop.py audit
#    py -3.10 scripts/local_aihub_flamecrop.py audit --split both --dump-schema 3
#    py -3.10 scripts/local_aihub_flamecrop.py crop  --device ct --inout in
#    py -3.10 scripts/local_aihub_flamecrop.py crop  --device ct --inout in --fallback-place ENB
#
#  출처: AI-Hub 71751 (화재 발생 예측 영상)  https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71751
#  상위 문맥: docs/HANDOFF_DFIRE_YOLO11.md (§AI-Hub) · 기준: docs/PREREGISTER_DFIRE_QC.md
#
#  파이프라인 위치: step4 = "어떻게 합성해야 base(ptrain_b79)가 잘 인식하나".
#    이 스크립트는 그 재료(=실 실내/조리 불꽃 크롭)를 만드는 소스 추출기.
#    합성·평가는 별도(로컬). 목표는 (A) 합성 개선이지 (B) 전이/실데이터 학습이 아님.
# =====================================================================================

import os, sys, io, re, json, zipfile, hashlib, argparse, unicodedata
from collections import Counter, defaultdict

# Windows 콘솔(cp949)에서 한글 print 시 UnicodeEncodeError 방지 — env 설정 불필요하게.
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding='utf-8')
    except Exception:
        pass

NFC = lambda s: unicodedata.normalize('NFC', s or '')


# ------------------------------------------------------------------------------------
# 대용량 Zip64 이미지 zip(VS.zip 등)이 "zipfiles that span multiple disks" 로 거부되는 경우.
#   원인: 엔트리 6만5천 초과 → Zip64 강제. 패키저가 Zip64 EOCD 로케이터의 disk 필드를
#         0 이 아닌 값으로 잘못 기록(단일 파일 false-span). Python zipfile 은 그걸 보고 거부.
#   대응: multi-disk raise 만 무력화한 _EndRecData64 판본으로 read-only 재시도.
#         진짜 분할본(part 여러 개)이면 이후 중앙디렉토리 읽기서 다른 에러 → 상위서 안내.
#   (읽기 전용 · 원본 zip 미변경.)
# ------------------------------------------------------------------------------------
def _open_zip_tolerant(path):
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        if 'span multiple disks' not in str(e):
            raise
    import struct as _struct
    z = zipfile

    def _patched(fpin, offset, endrec):
        try:
            fpin.seek(offset - z.sizeEndCentDir64Locator, 2)
        except OSError:
            return endrec
        data = fpin.read(z.sizeEndCentDir64Locator)
        if len(data) != z.sizeEndCentDir64Locator:
            return endrec
        sig, diskno, reloff, disks = _struct.unpack(z.structEndArchive64Locator, data)
        if sig != z.stringEndArchive64Locator:
            return endrec
        # ★ multi-disk raise 무력화 (단일 파일로 가정) — 여기만 원본과 다름.
        fpin.seek(offset - z.sizeEndCentDir64Locator - z.sizeEndCentDir64, 2)
        data = fpin.read(z.sizeEndCentDir64)
        if len(data) != z.sizeEndCentDir64:
            return endrec
        vals = _struct.unpack(z.structEndArchive64, data)
        sig = vals[0]
        if sig != z.stringEndArchive64:
            return endrec
        dircount, dircount2, dirsize, diroffset = vals[6], vals[7], vals[8], vals[9]
        endrec[z._ECD_SIGNATURE] = sig
        endrec[z._ECD_DISK_NUMBER] = 0
        endrec[z._ECD_DISK_START] = 0
        endrec[z._ECD_ENTRIES_THIS_DISK] = dircount
        endrec[z._ECD_ENTRIES_TOTAL] = dircount2
        endrec[z._ECD_SIZE] = dirsize
        endrec[z._ECD_OFFSET] = diroffset
        return endrec

    orig = zipfile._EndRecData64
    zipfile._EndRecData64 = _patched
    try:
        return zipfile.ZipFile(path)   # 진짜 분할본이면 여기서 CD 읽기 실패(다른 에러)
    finally:
        zipfile._EndRecData64 = orig

# ------------------------------------------------------------------------------------
# 기본 경로 (핸드오프 기록값 — 폴더명 일부 불명 '...' 이라 자동탐색 폴백)
#   루트(=1.데이터) 아래: {Validation,Training}/{01.원천데이터(이미지 zip), 02.라벨링데이터(json zip)}
#   Validation: VS.zip(1.27GB 이미지) · VL.zip(151MB json 243,529 · 불꽃 76,753)
#   Training  : TS.zip(이미지) · TL.zip(json)
# ------------------------------------------------------------------------------------
DEFAULT_DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')

# 논리 필드 → JSON 후보 키 (AI-Hub 스키마 변형 대비 · 핸드오프 기록 우선)
ATTR_KEYS = {
    'class':      ['class', 'category', 'label', 'obj_class', '객체', '분류'],
    'inout':      ['inout', 'in_out', 'indoor_outdoor', 'in/out', '실내외'],
    'place':      ['place', 'location', 'scene_place', '장소'],
    'device':     ['device', 'apparatus', 'equipment', '기기', '장치'],
    'fire_reason':['fire_reason', 'reason', '발화원인', '원인'],
    'fire_level': ['fire_level', 'level', 'fire_scale', 'scale', '화재강도', '강도'],
    'fps':        ['fps', 'frame_rate'],
    'scene':      ['scene', 'sceneid', 'scene_id', 'scene_no'],
    'clipname':   ['clipname', 'clip', 'clip_name', 'video'],
}


# ====================================================================================
#  경로 탐색
# ====================================================================================
def find_data_root(root_arg):
    """--root(=1.데이터) 우선. 없으면 Downloads 아래 089.화재* / 1.데이터 자동탐색."""
    if root_arg:
        if os.path.isdir(root_arg):
            return root_arg
        sys.exit(f'[ERR] --root 경로 없음: {root_arg}')
    cands = []
    for base in (DEFAULT_DOWNLOADS, os.path.expanduser('~')):
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            # 너무 깊이 안 들어가게 컷 (Downloads 하위 4단계까지)
            if dp[len(base):].count(os.sep) > 5:
                dns[:] = []
                continue
            bn = os.path.basename(dp)
            if bn == '1.데이터' and _looks_like_root(dp):
                cands.append(dp)
    cands = sorted(set(cands))
    if not cands:
        sys.exit("[ERR] AI-Hub '1.데이터' 루트 자동탐색 실패. --root 로 지정하세요 "
                 "(예: --root \"C:\\Users\\jhmoo\\Downloads\\089.화재...\\3.개방데이터\\1.데이터\").")
    if len(cands) > 1:
        print('[warn] 루트 후보 여러 개 → 첫 번째 사용. 다르면 --root 지정:')
        for c in cands:
            print('   ', c)
    return cands[0]


def _looks_like_root(d):
    subs = set(os.listdir(d)) if os.path.isdir(d) else set()
    return any(s in subs for s in ('Validation', 'Training', 'validation', 'training'))


def split_dir(root, split):
    """split 폴더 경로(대소문자 관대)."""
    for cand in (split, split.capitalize(), split.lower(), split.upper()):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p
    return None


def _has_data_subdirs(d):
    """Validation/Training 없이 01.원천데이터/02.라벨링데이터가 바로 있는 샘플 구조 감지."""
    if not os.path.isdir(d):
        return False
    for s in os.listdir(d):
        n = NFC(s)
        if s.startswith('01') or s.startswith('02') or '원천' in n or '라벨' in n:
            return True
    return False


def enumerate_splits(root, arg_split):
    """(label, path) 목록. Validation/Training 있으면 그것들, 없으면(샘플) root 자체를 단일 split."""
    want = ['Validation', 'Training'] if arg_split == 'both' else [arg_split.capitalize()]
    out = [(name, split_dir(root, name)) for name in want]
    out = [(n, p) for n, p in out if p is not None]
    if not out and _has_data_subdirs(root):
        out = [('(sample)', root)]      # 샘플: root 자체가 데이터 폴더(01/02 직속·Validation 없음)
    return out


def _find_zip_or_dir(split_path, kind):
    """kind='label'(02.라벨링데이터) | 'image'(01.원천데이터).
    반환: ('dir', 경로) 느슨한 json/jpg 폴더 · ('zip', [zip경로...]) · None.
    폴더명이 다를 수 있어 이름 키워드 + 내용으로 판별."""
    key = '02' if kind == 'label' else '01'
    kw = '라벨' if kind == 'label' else ('원천' if kind == 'image' else '')
    subdirs = []
    for name in os.listdir(split_path):
        full = os.path.join(split_path, name)
        if os.path.isdir(full) and (name.startswith(key) or kw in NFC(name)):
            subdirs.append(full)
    # 후보 서브폴더 없으면 split_path 자체도 뒤진다
    search_roots = subdirs or [split_path]
    ext = '.json' if kind == 'label' else ('.jpg', '.jpeg', '.png')
    # 1) 느슨한 파일 우선(사용자가 이미 압축 해제한 경우)
    for r in search_roots:
        for dp, dns, fns in os.walk(r):
            if any(f.lower().endswith(ext) for f in fns):
                return ('dir', r)
    # 2) zip
    zips = []
    for r in search_roots:
        for dp, dns, fns in os.walk(r):
            for f in fns:
                if f.lower().endswith('.zip'):
                    zips.append(os.path.join(dp, f))
    if zips:
        return ('zip', sorted(set(zips)))
    return None


# ====================================================================================
#  라벨(JSON) 이터레이터 — 느슨한 파일 or zip 둘 다 지원
# ====================================================================================
def iter_label_records(src, limit=0):
    """yield (member_name, raw_bytes). src = ('dir', path) | ('zip', [paths])."""
    n = 0
    kind, val = src
    if kind == 'dir':
        for dp, dns, fns in os.walk(val):
            for f in sorted(fns):
                if f.lower().endswith('.json'):
                    with open(os.path.join(dp, f), 'rb') as fh:
                        yield f, fh.read()
                    n += 1
                    if limit and n >= limit:
                        return
    else:  # zip
        for zp in val:
            try:
                zf = _open_zip_tolerant(zp)
            except Exception as e:
                print(f'[warn] zip 열기 실패 {zp}: {e}')
                continue
            with zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith('.json'):
                        continue
                    yield os.path.basename(info.filename), zf.read(info)
                    n += 1
                    if limit and n >= limit:
                        return


# ====================================================================================
#  이미지 소스 — 파일명(basename) → 바이트  (crop 모드 전용)
# ====================================================================================
class ImageSource:
    def __init__(self, src):
        self.kind, self.val = src
        self._index = {}      # nfc-lower basename -> (zippath, member) or fullpath
        self._zips = {}
        self._build()

    def _build(self):
        if self.kind == 'dir':
            for dp, dns, fns in os.walk(self.val):
                for f in fns:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self._index[NFC(f).lower()] = os.path.join(dp, f)
        else:
            for zp in self.val:
                try:
                    zf = _open_zip_tolerant(zp)
                except Exception as e:
                    print(f'[warn] 이미지 zip 열기 실패 {zp}: {e}')
                    print('       → 진짜 분할 아카이브일 수 있음(VS.z01 등 동반 파일 확인). '
                          '7-Zip 으로 VS.zip 우클릭→"여기에 압축 풀기" 후, '
                          '--root 는 그대로 두고 재실행(스크립트가 풀린 jpg 폴더 자동 인식).')
                    continue
                self._zips[zp] = zf
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self._index[NFC(os.path.basename(info.filename)).lower()] = (zp, info.filename)

    def __len__(self):
        return len(self._index)

    def get_bytes(self, filename):
        key = NFC(os.path.basename(filename)).lower()
        hit = self._index.get(key)
        if hit is None:
            # .json 이름을 넘긴 경우 대비: 확장자 치환 재시도
            stem = os.path.splitext(key)[0]
            for ext in ('.jpg', '.jpeg', '.png'):
                hit = self._index.get(stem + ext)
                if hit:
                    break
        if hit is None:
            return None
        if isinstance(hit, tuple):
            zp, member = hit
            return self._zips[zp].read(member)
        with open(hit, 'rb') as fh:
            return fh.read()

    def close(self):
        for zf in self._zips.values():
            try:
                zf.close()
            except Exception:
                pass


# ====================================================================================
#  파싱 헬퍼
# ====================================================================================
def parse_filename(name):
    """'sceneID_FL_place_frame.json' → dict. 규약 기록값 기반 · 실패해도 부분 반환.
       파일명은 JSON 키가 달라도 신뢰 가능한 백업(scene/class/place/frame)."""
    stem = os.path.splitext(NFC(os.path.basename(name)))[0]
    parts = stem.split('_')
    out = {'stem': stem}
    if len(parts) >= 1: out['scene_fn'] = parts[0]
    if len(parts) >= 2: out['class_fn'] = parts[1]
    if len(parts) >= 3: out['place_fn'] = parts[2]
    if len(parts) >= 4:
        m = re.search(r'\d+', parts[-1])
        out['frame_fn'] = int(m.group()) if m else None
    # ★클립 단위 키 = 프레임 번호(마지막 토큰)만 뗀 것 = 'sceneID_FL_place'.
    #   (JSON 'scene' 속성은 프레임마다 고유 → 근접중복 억제엔 무용. 파일명 규약이 클립 단위.)
    out['clip_fn'] = '_'.join(parts[:-1]) if len(parts) >= 2 else stem
    return out


def _scalarize(v):
    if isinstance(v, (str, int, float, bool)):
        return v
    return None


def flatten_attrs(attrs):
    """attributes 가 dict 또는 [{...}] 리스트 둘 다 대응 → {logical: value}."""
    raw = {}
    def absorb(d):
        if isinstance(d, dict):
            for k, v in d.items():
                sv = _scalarize(v)
                if sv is not None:
                    raw.setdefault(NFC(str(k)).lower(), sv)
                # {"code":"device","value":"ct"} 형태
                if isinstance(v, (dict, list)):
                    absorb(v)
            # code/value 쌍
            if 'code' in d and 'value' in d:
                cv = _scalarize(d.get('value'))
                if cv is not None:
                    raw.setdefault(NFC(str(d['code'])).lower(), cv)
        elif isinstance(d, list):
            for x in d:
                absorb(x)
    absorb(attrs)
    out = {}
    for logical, cands in ATTR_KEYS.items():
        for c in cands:
            if c.lower() in raw:
                out[logical] = raw[c.lower()]
                break
    return out, raw


def get_record_meta(name, obj):
    """JSON 1건 → 정규화 메타(dict). 파일명 백업과 JSON 속성 병합."""
    fn = parse_filename(name)
    img = obj.get('image') or obj.get('images') or {}
    if isinstance(img, list):
        img = img[0] if img else {}
    attrs = obj.get('attributes') or obj.get('attribute') or obj.get('meta') or {}
    logical, raw = flatten_attrs(attrs)
    anns = obj.get('annotations') or obj.get('annotation') or obj.get('objects') or []
    if isinstance(anns, dict):
        anns = [anns]
    meta = {
        'name': NFC(os.path.basename(name)),
        'img_w': _to_int(img.get('width') or img.get('img_width')),
        'img_h': _to_int(img.get('height') or img.get('img_height')),
        'img_file': NFC(img.get('filename') or img.get('file_name') or img.get('name') or ''),
        'class': logical.get('class') or fn.get('class_fn'),
        'inout': logical.get('inout'),
        'place': logical.get('place') or fn.get('place_fn'),
        'device': logical.get('device'),
        'fire_reason': logical.get('fire_reason'),
        'fire_level': logical.get('fire_level'),
        'scene': logical.get('scene') or fn.get('scene_fn'),
        'clipname': logical.get('clipname'),
        # 근접중복 그룹 키: 파일명 클립키(문서화된 클립단위) 우선 → clipname → scene 폴백.
        'clip': fn.get('clip_fn') or logical.get('clipname') or logical.get('scene') or fn.get('scene_fn'),
        'frame': fn.get('frame_fn'),
        'anns': anns,
        '_raw_attr_keys': sorted(raw.keys()),
    }
    if not meta['img_file']:
        meta['img_file'] = os.path.splitext(meta['name'])[0] + '.jpg'
    return meta


def _to_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def norm(v):
    return NFC(str(v)).strip() if v is not None else ''


# ====================================================================================
#  bbox 해석 (xywh vs xyxy 자동판별 · area 로 교차검증)
# ====================================================================================
def resolve_bbox(bbox, area, iw, ih, fmt='auto'):
    """bbox[4] → (x1,y1,x2,y2) 정수, 이미지 경계 clamp. 실패시 None."""
    if not bbox or len(bbox) < 4:
        return None
    try:
        b = [float(x) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None

    def as_xywh(b):
        x, y, w, h = b
        return x, y, x + w, y + h
    def as_xyxy(b):
        return b[0], b[1], b[2], b[3]

    if fmt == 'xywh':
        x1, y1, x2, y2 = as_xywh(b)
    elif fmt == 'xyxy':
        x1, y1, x2, y2 = as_xyxy(b)
    else:  # auto — area 매칭 우선, 없으면 경계 타당성
        cand = {}
        xw = as_xywh(b); cand['xywh'] = xw
        xy = as_xyxy(b); cand['xyxy'] = xy
        def valid(t):
            a, c, e, g = t
            return e > a and g > c
        def area_err(t):
            if not area:
                return None
            a, c, e, g = t
            aa = (e - a) * (g - c)
            return abs(aa - float(area)) / max(float(area), 1.0)
        scored = []
        for kk, t in cand.items():
            if not valid(t):
                continue
            er = area_err(t)
            scored.append((er if er is not None else 9e9, kk, t))
        if not scored:
            # 둘 다 뒤집힘 → xywh 로 강제
            x1, y1, x2, y2 = xw
        else:
            scored.sort(key=lambda z: z[0])
            x1, y1, x2, y2 = scored[0][2]

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(iw, int(round(x2))); y2 = min(ih, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ====================================================================================
#  MODE: audit  — device/속성 분포 확인 (JSON only · 설치 불필요)
# ====================================================================================
def run_audit(args):
    root = find_data_root(args.root)
    print(f'[root] {root}')
    splits = enumerate_splits(root, args.split)
    if not splits:
        sys.exit('[ERR] Validation/Training 도, 01원천·02라벨 폴더도 못 찾음: ' + root)

    grand = defaultdict(Counter)         # field -> Counter(value)
    cross_dev_inout = Counter()          # (device, inout)
    cross_place_inout = Counter()        # (place, inout)
    cross_dev_class = Counter()          # (device, class)
    cat_ids = Counter()                  # annotations categories_id
    target_hits = Counter()              # 관심 교집합 카운트
    per_scene_target = defaultdict(int)
    n_total = 0
    n_flame_boxes = 0
    schema_dumped = 0
    attr_key_seen = Counter()

    for split, sp in splits:
        src = _find_zip_or_dir(sp, 'label')
        if src is None:
            print(f'[skip] {split} 라벨(json) 소스 없음')
            continue
        print(f'[scan] {split} 라벨 소스: {src[0]} '
              + (f'({len(src[1])} zip)' if src[0] == 'zip' else f'({src[1]})'))
        for name, raw in iter_label_records(src, limit=args.limit):
            n_total += 1
            try:
                obj = json.loads(raw)
            except Exception:
                grand['_parse_error']['bad_json'] += 1
                continue
            meta = get_record_meta(name, obj)
            for k in meta['_raw_attr_keys']:
                attr_key_seen[k] += 1

            cls = norm(meta['class']); ino = norm(meta['inout'])
            plc = norm(meta['place']); dev = norm(meta['device'])
            grand['class'][cls or '(none)'] += 1
            grand['inout'][ino or '(none)'] += 1
            grand['place'][plc or '(none)'] += 1
            grand['device'][dev or '(none)'] += 1
            grand['fire_level'][norm(meta['fire_level']) or '(none)'] += 1
            grand['fire_reason'][norm(meta['fire_reason']) or '(none)'] += 1
            cross_dev_inout[(dev or '(none)', ino or '(none)')] += 1
            cross_place_inout[(plc or '(none)', ino or '(none)')] += 1
            cross_dev_class[(dev or '(none)', cls or '(none)')] += 1

            for a in meta['anns']:
                if isinstance(a, dict):
                    cid = a.get('categories_id', a.get('category_id', a.get('categories', a.get('category'))))
                    cat_ids[norm(cid) or '(none)'] += 1

            # 관심 교집합 (필터는 소문자 관대비교)
            is_flame = (cls.upper() == norm(args.klass).upper()) if cls else False
            is_in = (ino.lower() == norm(args.inout).lower()) if ino else False
            is_ct = (dev.lower() == norm(args.device).lower()) if dev else False
            is_enb = (plc.upper() == norm(args.place).upper()) if plc else False
            if is_flame:
                n_flame_boxes += len(meta['anns'])
            if is_flame and is_in and is_ct:
                target_hits['FL∧in∧device=%s' % args.device] += 1
                per_scene_target[meta['clip']] += 1
            if is_flame and is_in and is_enb:
                target_hits['FL∧in∧place=%s' % args.place] += 1
            if is_flame and is_in:
                target_hits['FL∧in (device 무관)'] += 1

            if args.dump_schema and schema_dumped < args.dump_schema:
                schema_dumped += 1
                print(f'\n--- schema sample #{schema_dumped}: {meta["name"]} ---')
                _print_schema(obj)
                print(f'   parsed: class={cls} inout={ino} place={plc} device={dev} '
                      f'fire_level={norm(meta["fire_level"])} scene={norm(meta["scene"])} '
                      f'#anns={len(meta["anns"])}')

            if n_total % 20000 == 0:
                print(f'   ...{n_total} scanned')

    # ---- 리포트 ----
    print('\n' + '=' * 78)
    print(f'AI-Hub 71751 속성 감사 · JSON {n_total}건'
          + (f' (limit {args.limit})' if args.limit else ''))
    print('=' * 78)
    _print_counter('inout (실내/외)', grand['inout'])
    _print_counter('class (객체)', grand['class'])
    _print_counter('device (기기) ← ★ct=조리기구? 이 분포로 확인', grand['device'], top=40)
    _print_counter('place (장소)', grand['place'], top=40)
    _print_counter('fire_level (강도)', grand['fire_level'])
    _print_counter('fire_reason (원인)', grand['fire_reason'], top=20)
    _print_counter('annotations categories_id (불꽃 카테고리 id 확인용)', cat_ids, top=20)

    print('\n[cross] device × inout (상위 30)')
    for (d, i), c in cross_dev_inout.most_common(30):
        print(f'   device={d:<10} inout={i:<6} : {c}')
    print('\n[cross] place × inout (상위 30)')
    for (p, i), c in cross_place_inout.most_common(30):
        print(f'   place={p:<10} inout={i:<6} : {c}')
    print('\n[cross] device × class (상위 30)')
    for (d, cl), c in cross_dev_class.most_common(30):
        print(f'   device={d:<10} class={cl:<6} : {c}')

    print('\n[target] 합성 불꽃 소스 후보 교집합 카운트')
    for k, c in target_hits.most_common():
        print(f'   {k:<28} : {c}')
    if per_scene_target:
        vals = sorted(per_scene_target.values(), reverse=True)
        print(f'   → device={args.device} 타깃 클립(=12초 영상·파일명 sceneID_FL_place) 수: '
              f'{len(per_scene_target)}  (클립당 프레임 최다 {vals[0]} · 중앙값 {vals[len(vals)//2]})')
        print(f'     → crop 시 per-scene-cap 로 클립당 프레임 제한(근접중복 억제).')

    print('\n[raw attr keys] JSON attributes 에서 실제 관측된 키 (스키마 검증)')
    for k, c in attr_key_seen.most_common(30):
        print(f'   {k:<24} : {c}')

    print('\n※ 판정: 위 device 분포에서 ct 의 정체(조리기구 여부)를 확인.')
    print('  ct 크롭이 부족하면 crop 모드에 --fallback-place ENB (음식점 실내) 추가.')
    print('  다음: py -3.10 scripts/local_aihub_flamecrop.py crop --device ct --inout in '
          '[--fallback-place ENB] [--flame-cat-id <id>]')


def _print_counter(title, counter, top=30):
    tot = sum(counter.values())
    print(f'\n[{title}] (합 {tot})')
    for v, c in counter.most_common(top):
        print(f'   {str(v):<14} {c:>10}  {c/max(tot,1)*100:5.1f}%')
    if len(counter) > top:
        print(f'   ... (+{len(counter)-top} more)')


def _print_schema(obj, prefix='', depth=0, maxdepth=3):
    """JSON 구조(키·타입·스칼라 예시)만 출력 — 배열 전체는 안 펼침."""
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                n = len(v)
                print(f'   {prefix}{k}: {type(v).__name__}[{n}]')
                if isinstance(v, list) and v:
                    _print_schema(v[0], prefix + '  ', depth + 1, maxdepth)
                elif isinstance(v, dict):
                    _print_schema(v, prefix + '  ', depth + 1, maxdepth)
            else:
                sv = str(v)
                if len(sv) > 40:
                    sv = sv[:40] + '…'
                print(f'   {prefix}{k}: {sv}')
    elif isinstance(obj, list) and obj:
        _print_schema(obj[0], prefix, depth + 1, maxdepth)


# ====================================================================================
#  MODE: crop  — 필터 매칭 불꽃 bbox 크롭 추출 (로컬 파일 생성만)
# ====================================================================================
def run_crop(args):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit('[ERR] Pillow 필요: py -3.10 -m pip install Pillow  (audit 모드는 불필요)')

    root = find_data_root(args.root)
    print(f'[root] {root}')
    out_dir = os.path.abspath(args.out)
    crop_dir = os.path.join(out_dir, 'crops')
    os.makedirs(crop_dir, exist_ok=True)
    print(f'[out] {out_dir}   (불꽃추출·비배포·출처표기 조건 · 데이터셋 재배포 금지)')

    splits = enumerate_splits(root, args.split)
    if not splits:
        sys.exit('[ERR] Validation/Training 도, 01원천·02라벨 폴더도 못 찾음: ' + root)
    flame_cat_ids = set(norm(x) for x in args.flame_cat_id.split(',')) if args.flame_cat_id else None

    # 1) 라벨 스캔 → 매칭 레코드 수집 (이미지 아직 안 열음)
    #    매칭 = class==FL ∧ inout==in ∧ (device==ct  또는  --fallback-place 이면 place==ENB)
    matched = []   # dict(meta 축약 + bbox 목록은 crop 시 재해석)
    per_scene = defaultdict(list)
    n_scanned = 0
    for split, sp in splits:
        src = _find_zip_or_dir(sp, 'label')
        if src is None:
            print(f'[skip] {split} 라벨 소스 없음'); continue
        print(f'[scan] {split} 라벨: {src[0]}')
        for name, raw in iter_label_records(src, limit=args.limit):
            n_scanned += 1
            if n_scanned % 20000 == 0:
                print(f'   ...{n_scanned} scanned, {len(matched)} matched')
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            meta = get_record_meta(name, obj)
            cls = norm(meta['class']); ino = norm(meta['inout'])
            dev = norm(meta['device']); plc = norm(meta['place'])
            if cls.upper() != norm(args.klass).upper():
                continue
            if args.inout and ino and ino.lower() != norm(args.inout).lower():
                # inout 값이 있는데 in 이 아니면 스킵. (값 없으면 통과 — 결측 관대)
                continue
            dev_ok = (dev.lower() == norm(args.device).lower()) if dev else False
            place_ok = (args.fallback_place and plc.upper() == norm(args.fallback_place).upper())
            if not (dev_ok or place_ok):
                continue
            meta['_split'] = split
            meta['_match'] = 'device' if dev_ok else 'place'
            matched.append(meta)
            per_scene[meta['clip']].append(meta)   # 클립(=12초) 단위 그룹 = 근접중복 억제

    print(f'\n[match] class={args.klass} ∧ inout={args.inout} ∧ '
          f'(device={args.device}{" | place="+args.fallback_place if args.fallback_place else ""}) '
          f'→ {len(matched)} 프레임 / {len(per_scene)} 클립')
    if not matched:
        print('  매칭 0. audit 로 device/place 코드 재확인하거나 --fallback-place ENB 추가.')
        return

    # 2) 클립당 균등 서브샘플(근접중복 억제) → 전체 cap
    picked = []
    for clip, recs in per_scene.items():
        recs = sorted(recs, key=lambda m: (m['frame'] if m['frame'] is not None else 0))
        if len(recs) > args.per_scene_cap:
            step = (len(recs) - 1) / (args.per_scene_cap - 1) if args.per_scene_cap > 1 else 1
            idx = sorted(set(int(round(i * step)) for i in range(args.per_scene_cap)))
            recs = [recs[i] for i in idx]
        picked.extend(recs)
    picked.sort(key=lambda m: (norm(m['clip']), m['frame'] if m['frame'] is not None else 0))
    if args.total_cap and len(picked) > args.total_cap:
        step = len(picked) / args.total_cap
        picked = [picked[int(i * step)] for i in range(args.total_cap)]
    print(f'[subsample] per-scene-cap {args.per_scene_cap} · total-cap {args.total_cap} '
          f'→ {len(picked)} 프레임 추출 대상')

    # 2b) --filelist-only: 이미지 추출 대신 대상 파일명 목록만(7-Zip 선택추출용) → VS 100GB 다 안 풀어도 됨
    if getattr(args, 'filelist_only', False):
        out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)
        lst = os.path.join(out_dir, 'cook_flame_files.txt')
        with open(lst, 'w', encoding='utf-8') as f:
            for m in picked:
                fn = os.path.basename(m['img_file'] or (os.path.splitext(m['name'])[0] + '.jpg'))
                f.write('*' + fn + '\n')                 # 7-Zip 와일드카드(폴더 무관 매칭)
        print(f'\n[filelist] 조리불꽃 대상 {len(picked)} 파일명 → {lst}')
        print( '  ★7-Zip 선택추출(VS.z01·VS.zip 같은 폴더에 둔 뒤):')
        print(f'    7z x "<...>\\Validation\\01.원천데이터\\VS.zip" -o"{out_dir}\\imgs" -r @"{lst}"')
        print( '    (전체 108GB 대신 이 파일들만 풀림. 풀린 뒤 crop 모드를 그 폴더에 --root 로 실행.)')
        return

    # 3) 이미지 소스 인덱스
    manifest = []
    thumbs = []      # (PIL crop, label) — 몽타주
    n_crop = 0
    n_skip_box = 0
    n_skip_img = 0
    n_read_err = 0
    dup_md5 = set()

    for split, sp in splits:
        recs = [m for m in picked if m['_split'] == split]
        if not recs:
            continue
        isrc = _find_zip_or_dir(sp, 'image')
        if isrc is None:
            print(f'[skip] {split} 이미지 소스 없음 — 크롭 불가'); continue
        print(f'[img] {split} 이미지 인덱싱... ({isrc[0]})')
        IMG = ImageSource(isrc)
        print(f'      {len(IMG)} 이미지 인덱싱됨')
        for m in recs:
            try:
                data = IMG.get_bytes(m['img_file'] or m['name'])
            except Exception as e:
                n_read_err += 1
                if n_read_err == 1:
                    print(f'[warn] 이미지 데이터 읽기 실패(첫 사례) {type(e).__name__}: {str(e)[:70]}')
                continue
            if data is None:
                n_skip_img += 1
                continue
            try:
                im = Image.open(io.BytesIO(data)).convert('RGB')
            except Exception:
                n_read_err += 1
                continue
            iw, ih = im.size
            # img_w/h 결측이면 실제 크기로 채움
            IW = m['img_w'] or iw
            IH = m['img_h'] or ih
            for ai, a in enumerate(m['anns']):
                if not isinstance(a, dict):
                    continue
                if flame_cat_ids is not None:
                    cid = norm(a.get('categories_id', a.get('category_id', a.get('categories', a.get('category')))))
                    if cid not in flame_cat_ids:
                        continue
                bb = a.get('bbox') or a.get('box') or a.get('points')
                area = a.get('area')
                box = resolve_bbox(bb, area, IW, IH, fmt=args.bbox_format)
                if box is None:
                    n_skip_box += 1
                    continue
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) < args.min_box_px:
                    n_skip_box += 1
                    continue
                # 패딩(마스킹 여유) — 이미지 경계 clamp
                px = int(bw * args.pad_frac); py = int(bh * args.pad_frac)
                cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
                cx2 = min(iw, x2 + px); cy2 = min(ih, y2 + py)
                # 좌표 스케일 보정(라벨 좌표가 원본, PIL 크기가 다르면)
                if (IW, IH) != (iw, ih) and IW and IH:
                    sx, sy = iw / IW, ih / IH
                    cx1, cy1, cx2, cy2 = int(cx1*sx), int(cy1*sy), int(cx2*sx), int(cy2*sy)
                crop = im.crop((cx1, cy1, cx2, cy2))
                # 근접중복(동일 크롭 md5) 억제
                h = hashlib.md5(crop.tobytes()).hexdigest()
                if h in dup_md5:
                    continue
                dup_md5.add(h)
                fl = norm(m['fire_level']) or 'NA'
                scn = re.sub(r'[^0-9A-Za-z가-힣]+', '', norm(m['clip']))[:24] or 'clip'
                fr = m['frame'] if m['frame'] is not None else ai
                fname = f"{scn}_{norm(m['place']) or 'NA'}_{m['_match']}_lvl{fl}_f{fr}_{ai}.jpg"
                crop.save(os.path.join(crop_dir, fname), quality=95)
                n_crop += 1
                manifest.append({
                    'file': fname, 'split': split, 'clip': norm(m['clip']),
                    'scene': norm(m['scene']),
                    'place': norm(m['place']), 'device': norm(m['device']),
                    'inout': norm(m['inout']), 'fire_level': fl, 'match': m['_match'],
                    'frame': fr, 'src_img': m['img_file'],
                    'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                    'crop_w': cx2 - cx1, 'crop_h': cy2 - cy1,
                })
                if len(thumbs) < args.montage_n:
                    th = crop.copy(); th.thumbnail((200, 200))
                    thumbs.append((th, f"{scn} lvl{fl}"))
        IMG.close()

    # 4) manifest + 몽타주(로컬 육안용)
    if manifest:
        mani = os.path.join(out_dir, 'crops_manifest.csv')
        cols = ['file', 'split', 'clip', 'scene', 'place', 'device', 'inout', 'fire_level',
                'match', 'frame', 'src_img', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'crop_w', 'crop_h']
        with open(mani, 'w', encoding='utf-8-sig') as f:
            f.write(','.join(cols) + '\n')
            for r in manifest:
                f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f'[manifest] {mani}  ({len(manifest)} 크롭)')

    if thumbs:
        _save_montage(thumbs, os.path.join(out_dir, 'crops_montage.png'))
        print(f'[montage] {os.path.join(out_dir, "crops_montage.png")}  ← 로컬 풀해상 육안 검수')

    print(f'\n[done] 크롭 {n_crop}장 → {crop_dir}')
    print(f'  스킵: 이미지없음 {n_skip_img} · 읽기실패 {n_read_err} · bbox부적합(작음/파싱) {n_skip_box}')
    if n_read_err and n_crop == 0:
        print('\n[진단] 이미지 목록은 읽혔으나 데이터 추출이 전부 실패 = VS.zip 이 '
              '분할(멀티볼륨) 아카이브의 일부이고 실제 이미지가 이 파일 안에 없음.')
        print('  확인: 01.원천데이터 폴더에 VS.z01/VS.z02/… (또는 VS.zip.001 등) 동반 파일이 있는지, '
              '각 크기 합이 수십 GB인지.')
        print('  해결: 모든 조각이 있으면 7-Zip 으로 VS.zip 우클릭→"압축 풀기"(조각 자동결합) → '
              '풀린 jpg 폴더 생기면 같은 명령 재실행(스크립트가 폴더 자동 인식).')
        print('  조각이 VS.zip 하나뿐이면 다운로드가 불완전 → AI-Hub 재다운로드(전 조각) 필요.')
    print('  ★취급: 불꽃추출·비배포·출처표기 조건서 Colab/Drive OK(사용자 승인) · 데이터셋 재배포/공유 금지.')
    print('  다음(로컬): 이 크롭들을 육안 선별 → 마스킹(불꽃 알파) → 합성 → base 추론(로컬 CPU).')


def _save_montage(thumbs, path, cols=8):
    from PIL import Image, ImageDraw
    n = len(thumbs)
    rows = (n + cols - 1) // cols
    cw, ch = 200, 224
    canvas = Image.new('RGB', (cols * cw, rows * ch), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * cw, r * ch
        ox = x + (cw - im.width) // 2
        canvas.paste(im, (ox, y + 4))
        draw.text((x + 4, y + ch - 16), label[:26], fill=(230, 230, 120))
    canvas.save(path)


# ====================================================================================
def run_diag(args):
    """이미지 zip 무결성 진단 — 데이터가 파일 안에 실제로 있는지(오프셋·용량) 판정."""
    root = find_data_root(args.root)
    print(f'[root] {root}')
    splits = enumerate_splits(root, args.split)
    if not splits:
        sys.exit('[ERR] Validation/Training 도, 01원천·02라벨 폴더도 못 찾음: ' + root)
    for split, sp in splits:
        isrc = _find_zip_or_dir(sp, 'image')
        if isrc is None:
            print(f'[skip] {split} 이미지 소스 없음'); continue
        print(f'\n=== {split} 이미지 소스: {isrc[0]} ===')
        if isrc[0] == 'dir':
            n = sum(len([f for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    for _, _, fs in os.walk(isrc[1]))
            print(f'  느슨한 이미지 폴더 {isrc[1]} · jpg {n}개 (이 경우 crop 바로 가능)')
            continue
        for zp in isrc[1]:
            fsize = os.path.getsize(zp)
            print(f'\n[{os.path.basename(zp)}] 파일크기 {fsize/1e9:.3f} GB')
            try:
                zf = _open_zip_tolerant(zp)
            except Exception as e:
                print(f'  OPEN 실패: {type(e).__name__}: {str(e)[:70]}')
                print('  → 조각(VS.z01 등)이 있으면 7-Zip 결합, 없으면 재다운로드.')
                continue
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if not infos:
                print('  엔트리 0'); zf.close(); continue
            comp = sum(i.compress_size for i in infos)
            unco = sum(i.file_size for i in infos)
            maxend = max(i.header_offset + i.compress_size for i in infos)
            maxoff = max(i.header_offset for i in infos)
            print(f'  엔트리 {len(infos)}개 · 압축합 {comp/1e9:.3f} GB · 원본합 {unco/1e9:.3f} GB')
            print(f'  header_offset 최대 {maxoff:,} · (offset+압축) 최대 {maxend:,} · 파일크기 {fsize:,}')
            fits = maxend <= fsize
            print(f'  ▶ 데이터가 파일 안에 다 있나? {"YES (온전)" if fits else "NO ★ 오프셋이 파일 끝을 넘음 = 데이터 없음/불완전/멀티볼륨"}')
            expect = comp / max(len(infos), 1)
            print(f'  참고: 엔트리당 평균 압축 {expect/1024:.1f} KB '
                  f'(1080p JPEG면 보통 100~500KB — 너무 작으면 원본 아님/썸네일/불완전)')
            # 실제 읽기 프로브 — 가장 작은 것 + 오프셋 최대(파일 끝쪽, 잘렸으면 여기서 실패)
            for tag, ent in (('최소', min(infos, key=lambda i: i.compress_size)),
                             ('오프셋최대', max(infos, key=lambda i: i.header_offset))):
                try:
                    b = zf.read(ent)
                    ok = b[:2] == b'\xff\xd8'
                    print(f'  프로브[{tag}] 읽기 OK: {len(b)} bytes · JPEG magic {"YES" if ok else "NO"}')
                except Exception as e:
                    print(f'  프로브[{tag}] 실패: {type(e).__name__}: {str(e)[:55]} → 이 지점 데이터 없음')
            zf.close()
    print('\n※ 판정: "데이터 온전 YES"+"프로브 OK"면 crop 가능(안 되면 다른 버그) · '
          '"NO"/프로브 실패면 VS.zip 불완전 → AI-Hub 원천데이터 재다운로드(전체 용량 확인).')


def build_parser():
    p = argparse.ArgumentParser(
        description='AI-Hub 71751 불꽃 크롭 소스 추출 (불꽃추출·비배포·출처표기 조건서 Colab OK·사용자승인)')
    p.add_argument('--root', default='',
                   help="AI-Hub '1.데이터' 루트(미지정시 Downloads 자동탐색)")
    p.add_argument('--split', default='Validation',
                   choices=['Validation', 'Training', 'both'],
                   help='라벨/이미지 split (기본 Validation — 더 작음/빠름)')
    p.add_argument('--limit', type=int, default=0, help='스캔 JSON 상한(0=전체·빠른 프리뷰용)')
    sub = p.add_subparsers(dest='mode', required=True)

    sub.add_parser('diag', help='이미지 zip 무결성 진단 (데이터 실재 여부·설치 불필요)')

    pa = sub.add_parser('audit', help='device/속성 분포 확인 (JSON only · 설치 불필요)')
    pa.add_argument('--dump-schema', type=int, default=2, help='샘플 JSON 구조 N건 출력(스키마 검증)')
    pa.add_argument('--klass', default='FL', help='불꽃 class 코드(기본 FL)')
    pa.add_argument('--inout', default='in', help='실내 코드(기본 in)')
    pa.add_argument('--device', default='ct', help='조리기구 추정 코드(기본 ct·미검증)')
    pa.add_argument('--place', default='ENB', help='폴백 장소=음식점(기본 ENB)')

    pc = sub.add_parser('crop', help='필터 매칭 불꽃 bbox 크롭 추출 (Pillow)')
    pc.add_argument('--klass', default='FL', help='불꽃 class 코드(기본 FL)')
    pc.add_argument('--inout', default='in', help='실내 필터(기본 in · 값 결측은 관대통과)')
    pc.add_argument('--device', default='ct', help='조리기구 코드(기본 ct)')
    pc.add_argument('--fallback-place', default='', help='ct 부족시 넓힐 장소(예: ENB 음식점)')
    pc.add_argument('--flame-cat-id', default='', help='불꽃 categories_id 제한(콤마 구분·audit로 확인)')
    pc.add_argument('--bbox-format', default='auto', choices=['auto', 'xywh', 'xyxy'])
    pc.add_argument('--min-box-px', type=int, default=40, help='너무 작은 불꽃 제외(px)')
    pc.add_argument('--pad-frac', type=float, default=0.12, help='크롭 패딩(bbox 대비·마스킹 여유)')
    pc.add_argument('--per-scene-cap', type=int, default=8, help='scene당 최대 프레임(근접중복 억제)')
    pc.add_argument('--total-cap', type=int, default=800, help='총 크롭 상한')
    pc.add_argument('--montage-n', type=int, default=64, help='몽타주 썸네일 수(로컬 육안)')
    pc.add_argument('--out', default='aihub_flamecrops',
                    help='출력 폴더(로컬 · 기본 ./aihub_flamecrops)')
    pc.add_argument('--filelist-only', action='store_true',
                    help='이미지 추출 대신 대상 파일명 목록(cook_flame_files.txt·7-Zip 선택추출용)만 출력 — VS 100GB 다 안 풀어도 됨')
    return p


def main():
    args = build_parser().parse_args()
    print('=' * 78)
    print(' AI-Hub 71751 flame-crop (불꽃추출·비배포·출처표기 조건서 Colab OK·사용자승인 2026-08-29 · 데이터셋 재배포 금지)')
    print('=' * 78)
    if args.mode == 'audit':
        run_audit(args)
    elif args.mode == 'crop':
        run_crop(args)
    elif args.mode == 'diag':
        run_diag(args)


if __name__ == '__main__':
    main()
