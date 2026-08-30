#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================
#  VFX 검은배경 불꽃 클립 → 프레임 추출 + luma-key 알파 + QC 몽타주  (Phase 2 파일럿/본수집)
# =====================================================================================
#  왜: VFX 검은배경 소재는 발광강도=알파(측정). SAM2 추정 불필요. HD라 업스케일 문제 없음.
#  luma-key: alpha = clip((luma-lo)/(hi-lo),0,1). lo=회색빛 haze 컷, hi=완전불투명 문턱.
#  독립성: 클립당 1-2프레임(시간축 이격)만 — 인접 프레임은 독립 소스 아님(pseudoreplication).
#  ★파일럿 통과기준(사전): 5클립 중 ≥3에서 "깨끗한 알파"(haze/블록노이즈/워터마크 없음)면 통과.
#
#  준비: CC0(Pexels/Pixabay) 검은배경 불꽃 mp4 를 ASCII 경로 폴더에 저장(예: Downloads\vfx_pilot).
#        ★비ASCII/한글 폴더는 cv2 가 못 읽음 → ASCII 경로 사용.
#  실행: py -3.10 scripts\vfx_extract.py --src "C:\Users\jhmoo\Downloads\vfx_pilot"
#        [옵션] --per-clip 2  --lo 12 --hi 90
#  출력: <src>\extracted\rgba\*.png (알파 매트) · <src>\extracted\vfx_qc.png (몽타주)
#  취급: 소재 라이선스=CC0 우선 · 클립 목록·출처 기록(재현성). 소재 자체 재배포 금지 유의.
# =====================================================================================
import os, sys, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='mp4/mov/webm 클립 폴더(ASCII 경로)')
    ap.add_argument('--per-clip', type=int, default=2, help='클립당 프레임 수(시간축 이격)')
    ap.add_argument('--lo', type=float, default=12.0, help='haze 컷 문턱(이하 투명)')
    ap.add_argument('--hi', type=float, default=90.0, help='불투명 문턱(이상 알파=1)')
    args = ap.parse_args()

    try:
        import cv2, numpy as np
    except ImportError:
        sys.exit('[ERR] opencv/numpy 필요: py -3.10 -m pip install opencv-python numpy')
    from PIL import Image, ImageDraw

    vids = []
    for ext in ('*.mp4', '*.mov', '*.webm', '*.mkv', '*.avi', '*.MP4', '*.MOV'):
        vids += glob.glob(os.path.join(args.src, ext))
    vids = sorted(set(vids))
    if not vids:
        sys.exit(f'[ERR] 영상 없음: {args.src}  (CC0 검은배경 불꽃 mp4 를 여기에 저장)')

    outdir = os.path.join(args.src, 'extracted'); rgba_dir = os.path.join(outdir, 'rgba')
    os.makedirs(rgba_dir, exist_ok=True)

    def luma_key(bgr):
        rgb = bgr[..., ::-1].astype(np.float32)
        luma = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
        a = np.clip((luma - args.lo) / (args.hi - args.lo), 0, 1)
        return np.dstack([rgb, a*255]).astype(np.uint8), luma

    def checker(w, h, s=14):
        a = np.zeros((h, w, 3), np.uint8)
        for y in range(0, h, s):
            for x in range(0, w, s):
                a[y:y+s, x:x+s] = 90 if ((x//s + y//s) % 2 == 0) else 150
        return Image.fromarray(a).convert('RGBA')

    cells = []   # (orig_thumb, alpha_thumb, label, stats)
    print('=' * 70)
    print(f' VFX 추출 · 클립 {len(vids)}개 · 클립당 {args.per_clip}프레임 · luma-key lo{args.lo}/hi{args.hi}')
    print('=' * 70)
    for vi, vp in enumerate(vids):
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            # 일부 코덱은 프레임수 0 보고 → 순차 읽기로 대체
            frames_all = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                frames_all.append(fr)
            n = len(frames_all)
            picks = [int(n * t) for t in ((0.35, 0.65) if args.per_clip == 2 else
                     [(i+1)/(args.per_clip+1) for i in range(args.per_clip)])]
            grabbed = [(k, frames_all[k]) for k in picks if 0 <= k < n]
        else:
            ts = (0.35, 0.65) if args.per_clip == 2 else [(i+1)/(args.per_clip+1) for i in range(args.per_clip)]
            grabbed = []
            for t in ts:
                k = int(n * t); cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, fr = cap.read()
                if ok: grabbed.append((k, fr))
        cap.release()
        base = os.path.splitext(os.path.basename(vp))[0][:20]
        if not grabbed:
            print(f'  [{vi+1}] {base}: ❌ 프레임 읽기 실패(코덱?) — mp4/h264 권장'); continue
        for (k, fr) in grabbed:
            rgba, luma = luma_key(fr)
            h, w = rgba.shape[:2]
            # 통계: 배경 어두움(코너 median), 알파 커버리지, haze(저알파 비율)
            corner = np.concatenate([luma[:20, :20].ravel(), luma[-20:, -20:].ravel()])
            bg_med = float(np.median(corner))
            a = rgba[..., 3] / 255.
            cover = float((a > 0.5).mean())
            haze = float(((a > 0.05) & (a < 0.4)).mean())   # 반투명 잔여(높으면 haze/노이즈 의심)
            fname = f'{base}_f{k}.png'
            Image.fromarray(rgba, 'RGBA').save(os.path.join(rgba_dir, fname))
            # 썸네일
            o = Image.fromarray(fr[..., ::-1]); o.thumbnail((220, 220))
            bg = checker(w, h); bg.alpha_composite(Image.fromarray(rgba, 'RGBA'))
            av = bg.convert('RGB'); av.thumbnail((220, 220))
            flag = 'OK' if (bg_med < 20 and cover > 0.01 and haze < 0.15) else '⚠확인'
            cells.append((o, av, f'{base} f{k}', f'bg{bg_med:.0f} cov{cover*100:.0f}% haze{haze*100:.0f}% {flag}'))
            print(f'  [{vi+1}] {fname}: 배경median {bg_med:.0f}(낮을수록 검정) · 커버{cover*100:.0f}% · haze{haze*100:.0f}% → {flag}')

    if cells:
        cols = 2  # (원본|알파) 쌍 × 2 = 4열
        per = 2; rows = (len(cells) + per - 1) // per
        CW, CH = 232, 262
        canvas = Image.new('RGB', (per*2*CW, rows*CH), (18, 18, 20)); d = ImageDraw.Draw(canvas)
        for i, (o, av, lab, stat) in enumerate(cells):
            r, c = divmod(i, per); x = c*2*CW; y = r*CH
            canvas.paste(o, (x + (CW-o.width)//2, y+8)); canvas.paste(av, (x+CW + (CW-av.width)//2, y+8))
            d.text((x+6, y+CH-30), lab[:26], fill=(230, 230, 120))
            d.text((x+6, y+CH-16), stat, fill=(255, 130, 130) if '⚠' in stat else (140, 235, 140))
        mont = os.path.join(outdir, 'vfx_qc.png'); canvas.save(mont)
        print(f'\n[done] rgba {len(cells)}장 → {rgba_dir}\n[montage] {mont}')
        print('  판정: 각 쌍 [원본|알파(체커)]. 알파에서 불꽃만 남고 배경=격자(투명)면 OK.')
        print('  ⚠확인 = 배경 안검정(haze) / 커버리지0 / 반투명잔여 많음 → 그 클립 제외 후보.')
    else:
        print('\n[done] 추출된 프레임 0 — 코덱 문제일 수 있음(mp4/h264 권장).')


if __name__ == '__main__':
    main()
