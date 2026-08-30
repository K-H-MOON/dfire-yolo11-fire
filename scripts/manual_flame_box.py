#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================================
#  수동 불꽃 박스 크롭 도구 (tkinter) — 자동 세그(색마스크) 실패 대체 · 사람이 정본으로 박스
# =====================================================================================
#  왜: AI-Hub CCTV 실내불꽃은 과노출 코어+갈색 글로우+블러라 색마스크 추출이 깨끗지 않음
#      (v1/v2 QC 실패). → 사람이 불꽃에 사각형을 직접 쳐서 그 박스만 크롭.
#  ★GUI(대화형)라 '사용자 터미널'에서 실행. tkinter=표준라이브러리(설치 불필요), PIL=이미 설치.
#     (opencv 5.0.0 headless 빌드라 cv2 창 불가 → tkinter 사용.)
#  취급: AI-Hub 71751 불꽃추출·비배포·출처표기. 결과=로컬 crops.
#
#  실행:  cd <worktree>;  py -3.10 scripts\manual_flame_box.py
#         [옵션] --all-frames : 클립당 1장 대신 전 프레임(64)
#
#  조작(창):  마우스 드래그=박스 그리기(놓으면 확정, 여러개 가능)
#             [N]다음  [S]건너뜀  [Z]마지막 박스 취소  [Q]저장&종료  (버튼도 있음)
# =====================================================================================
import os, sys, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r"C:\Users\jhmoo\Downloads\aihub_enb\imgs")
    ap.add_argument('--out', default=r"C:\Users\jhmoo\Downloads\aihub_enb\manual_crops")
    ap.add_argument('--all-frames', action='store_true', help='클립당 1장 대신 전 프레임(64)')
    ap.add_argument('--max-w', type=int, default=1400)
    ap.add_argument('--max-h', type=int, default=760)
    args = ap.parse_args()

    try:
        import tkinter as tk
    except Exception:
        sys.exit('[ERR] tkinter 없음(표준설치엔 포함). python.org 재설치 시 tcl/tk 옵션 켜세요.')
    try:
        from PIL import Image, ImageTk
    except ImportError:
        sys.exit('[ERR] Pillow 필요: py -3.10 -m pip install Pillow')

    frames = sorted(glob.glob(os.path.join(args.src, '**', '*.jpg'), recursive=True))
    if not frames:
        sys.exit(f'[ERR] 프레임 없음: {args.src}')
    if not args.all_frames:
        groups = {}
        for f in frames:
            groups.setdefault(os.path.basename(f).split('_')[0], []).append(f)
        frames = sorted(g[len(g)//2] for g in (sorted(v) for v in groups.values()))
    os.makedirs(args.out, exist_ok=True)

    state = {'idx': 0, 'saved': 0, 'boxes': [], 'start': None, 'rect': None,
             'oimg': None, 'scale': 1.0, 'tkimg': None}

    root = tk.Tk()
    root.title('수동 불꽃 박스')
    info = tk.Label(root, text='', anchor='w', justify='left', font=('Consolas', 11))
    info.pack(fill='x')
    canvas = tk.Canvas(root, bg='#111', highlightthickness=0)
    canvas.pack()

    bar = tk.Frame(root); bar.pack(fill='x')
    def mkbtn(t, c):
        b = tk.Button(bar, text=t, command=c, width=12); b.pack(side='left', padx=3, pady=4); return b

    def load():
        f = frames[state['idx']]
        im = Image.open(f).convert('RGB')
        ow, oh = im.size
        sc = min(1.0, args.max_w/ow, args.max_h/oh)
        dim = im.resize((int(ow*sc), int(oh*sc))) if sc < 1 else im.copy()
        state['oimg'] = im; state['scale'] = sc; state['boxes'] = []
        state['tkimg'] = ImageTk.PhotoImage(dim)
        canvas.config(width=dim.width, height=dim.height)
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=state['tkimg'])
        refresh()

    def refresh():
        info.config(text=f"[{state['idx']+1}/{len(frames)}] {os.path.basename(frames[state['idx']])}   "
                         f"박스:{len(state['boxes'])}  누적저장:{state['saved']}   "
                         f"드래그=박스 · N다음 · S건너뜀 · Z취소 · Q저장&종료")

    def on_press(e):
        state['start'] = (e.x, e.y)
        if state['rect']: canvas.delete(state['rect'])
        state['rect'] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='#00e0ff', width=2)

    def on_drag(e):
        if state['start'] and state['rect']:
            canvas.coords(state['rect'], state['start'][0], state['start'][1], e.x, e.y)

    def on_release(e):
        if not state['start']: return
        x0, y0 = state['start']; x1, y1 = e.x, e.y
        state['start'] = None; state['rect'] = None
        bx0, by0, bx1, by1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        if bx1-bx0 < 4 or by1-by0 < 4: return
        state['boxes'].append((bx0, by0, bx1, by1))
        canvas.create_rectangle(bx0, by0, bx1, by1, outline='#33ff66', width=2)
        canvas.create_text(bx0+3, by0+8, anchor='w', text=str(len(state['boxes'])), fill='#33ff66')
        refresh()

    def save_boxes():
        im = state['oimg']; sc = state['scale']
        stem = os.path.splitext(os.path.basename(frames[state['idx']]))[0]
        for i, (bx0, by0, bx1, by1) in enumerate(state['boxes']):
            X0, Y0 = int(bx0/sc), int(by0/sc); X1, Y1 = int(bx1/sc), int(by1/sc)
            crop = im.crop((X0, Y0, X1, Y1))
            crop.save(os.path.join(args.out, f'manual_{stem}_{X0}_{Y0}.jpg'), quality=95)
            state['saved'] += 1

    def go(delta):
        state['idx'] += delta
        if state['idx'] >= len(frames):
            finish(); return
        if state['idx'] < 0: state['idx'] = 0
        load()

    def next_frame(_=None): save_boxes(); go(1)
    def skip(_=None): go(1)
    def undo(_=None):
        if state['boxes']:
            state['boxes'].pop(); load_keep()
    def load_keep():
        # 현재 프레임 다시 그리되 남은 박스 유지
        f = frames[state['idx']]; im = state['oimg']; sc = state['scale']
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=state['tkimg'])
        for i, (bx0, by0, bx1, by1) in enumerate(state['boxes']):
            canvas.create_rectangle(bx0, by0, bx1, by1, outline='#33ff66', width=2)
            canvas.create_text(bx0+3, by0+8, anchor='w', text=str(i+1), fill='#33ff66')
        refresh()
    def finish(_=None):
        try: save_boxes()
        except Exception: pass
        print(f"[done] 수동 크롭 {state['saved']}장 → {args.out}")
        root.destroy()

    canvas.bind('<ButtonPress-1>', on_press)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<ButtonRelease-1>', on_release)
    root.bind('n', next_frame); root.bind('N', next_frame)
    root.bind('s', skip); root.bind('S', skip)
    root.bind('z', undo); root.bind('Z', undo)
    root.bind('q', finish); root.bind('Q', finish)
    root.protocol('WM_DELETE_WINDOW', finish)
    mkbtn('다음(N)', next_frame); mkbtn('건너뜀(S)', skip); mkbtn('취소(Z)', undo); mkbtn('저장&종료(Q)', finish)

    print('=' * 70)
    print(f' 수동 불꽃 박스(tkinter) · 프레임 {len(frames)}장 · 저장→ {args.out}')
    print(' 창에서: 드래그=박스 · N다음 · S건너뜀 · Z취소 · Q저장&종료')
    print('=' * 70)
    load()
    root.mainloop()


if __name__ == '__main__':
    main()
