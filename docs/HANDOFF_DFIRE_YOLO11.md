# HANDOFF — D-Fire YOLO11 화재 baseline

**상태: Phase 0 baseline 확정 (2026-08-28).** 다음 = Phase 1 진단.

## ▶▶▶ 다음 세션 재개 레시피
1. 이 문서 전체 읽기.
2. baseline은 **확정**됨(아래 §결과). 재학습 불필요 — 모델은 Drive에 생존.
3. 다음 단계 = **Phase 1 진단** 또는 미제(§미제) 중 사용자 선택.
4. 전체 Colab 셀 = `docs/dfire_baseline_cells.py` (CELL 1~8, 검증됨). 런타임 끊기면 `/content` 초기화·**Drive 모델 생존** → 재빌드: CELL 1(마운트)→2(다운로드+dedup, **api_key 필요**)→3(cap1)→4(ptrain). 학습(CELL 5)은 Drive 모델 있으면 skip. 누수 감사(6/7/8)는 `/content` 데이터셋만 필요(모델·Drive 불요, CELL 8은 마운트도 불요).
5. **§행동원칙** 준수.

## ★ 결과 (baseline 확정)
- **채택 모델 = `ptrain_b79`** (train만 CAP=3). Drive: `/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt` (참고: `fire_cap1_b79`도 있음).
- **정직한 성능(누수 ≥0.980 제거 = t980)**: **test_mAP50 0.660 · recall천장 0.894 · mAP50-95 0.301** (n_test 982).
  - 누수 포함(full, n_test 1068): mAP50 0.686 · recall천장 0.906.
  - 누수 부풀림은 mAP50 약 **−0.026**(full→t980). 작음.
- **recall천장 0.894** = 양성의 ~11%는 어떤 conf서도 못 잡음 = **base 실제 한계**(누수 착시 아님, 제거해도 유지).
- **ptrain > cap1 견고**: mAP50 +0.028 격차가 full·t990·t980 전 범위서 고정 → 모델 선택 확정.

## 판정 근거 (측정 이력)
- **CELL 5 공정비교**(batch=79·같은세션): ptrain>cap1 전지표 일관·오버핏 gap 안 커짐(0.273→0.248) → ptrain 채택.
- **CELL 6 누수 감사**(open_clip ViT-B/32, test↔train 임베딩 최근접): dHash(Ham≤6)가 놓친 **cross-split near-dup 실재**. 양성 test ≥0.990: cap1 23 / ptrain 25장. 육안 확인 = 같은 방송 프레임(자막 `06:37`·워터마크 `RT`/`BRC`/`TV BARROSO`·동일 차량 등).
- **CELL 7 엄밀 재평가**(각 모델 자기-train 기준, 재학습 없음·test만 정제): 누수 제거 재평가. full/t990/t980 스윕. → 영향 작고 순위 견고(위 §결과).
- **CELL 8 band 육안**(0.980–0.990 양성 28장): 몽타주 24/24 사실상 전부 같은 장면 → 이 구간도 누수 → 정직 점추정 = **t980(0.660)** 확정.
- **오버핏**: val/box_loss는 ep~30 후 재상승(시그니처) 하나 early-stop이 정점 best.pt(cap1 ep67·ptrain ep64) 저장, val mAP는 plateau → 배포관점 문제 아님. 정직 train-test gap ≈0.27(정상 규모).

## 데이터 (고정)
- Roboflow: ws `kyungho-moon` · project `d-fire-aqheb-6iyqy` · **v1** · `download("yolov11")`. 원본 21,522장.
- dedup dHash HAM=6 union-find → **10,624 클러스터**(최대 3,042). (DetectiumFire arXiv:2511.02495 "중복시 절반"·pHash0.15/CNN0.55와 정합 — 검증함.)
- fire-only(smoke 드롭)·층화(src AoF/WEB × has-fire) 그룹 80/10/10 · 구조적 누수 0 · SEED=0 결정론(재빌드시 게이트로 확인).
- **cap1**(전 split CAP1): train 양성 3,205 · test 404/1068. **ptrain**(train만 CAP3): train 양성 3,433 · test 404/1068. **재빌드 게이트: 10624 / 3205·404-1068 / 3433·404-1068.**

## 고정 파라미터
- fire=class 0 · yolo11s · epochs100 · patience25 · imgsz640 · **batch=79** · cache=disk · seed0 · deterministic.
- 누수 임베딩: open_clip `ViT-B-32` pretrained `openai` · 코사인 최근접 · 판정 임계 육안(≥0.980 = dup).

## 미제 (未決)
| 항목 | 상태 | 성격 |
|---|---|---|
| ⑤ cleanlab ObjectLab 라벨 감사 | 미제 | recall천장 0.894의 "못 잡는 11%"가 모델 한계인지 **라벨 오류(라벨 안 된 불)**인지 미분해. 이제 가능(모델 있음). baseline 해석에 직결 |
| multi-seed 재현 | 미제 | 전부 1-seed. ptrain>cap1 방향은 전범위 일관하나 격차 크기 재현 미확인 |
| 0.95–0.98 누수구간 | 미검증 | 여기도 dup이면 정직치 <0.660 가능. 단 median 0.931 의미유사와 섞여 과보정 위험 → 0.660 유지 |
| oilfire_realtest 사용가능? | 미확인 | Phase 1 진단(실내 화재)에 필요 |
| Home-fire (IEEE Access 2025, 6500) | 미검증 | 쓰려면 D-Fire처럼 QC |

## 다음 = Phase 1 진단
base(ptrain_b79)를 3셋에 돌려 recall/precision — (a) D-Fire test (b) **실제 실내/조리 화재** (c) 합성 gen. 읽기: (c)vs(b)=합성 리얼한가 · (b)vs(a)=도메인갭. → Phase 2 분기(합성≈실내&base실내OK=합성검증됨 / base실내약함=Home검증후 fine-tune / 합성≪실내=합성이 문제) → Phase 3(증강목적이면) with/without 실제유용성.
**프록시 경고**: 위 baseline 전부 D-Fire test(야외·웹). 프로젝트(실내·조리) 예측력은 도메인갭 때문에 **가정** — Phase 1서 실측 전까지 미확립.

## ★행동원칙 (지난 세션 신뢰 손상 — 준수)
1. 측정 전 결론 금지. 숫자 없으면 판정 안 함.
2. 사용자 지시 없이 다음 단계 진행 금지. 결과 읽고 선택지만.
3. 주장 전 검증(문법·로직·**환경/설치**). "확실"은 검증 범위 안에서만. (CELL 6/7/8 설치가드 필수 — CELL 5 안 돌린 세션엔 ultralytics·open_clip 없음.)
4. 애매하면 "미확정" + 질문. 추측으로 메우지 말 것. established(측정) vs assumed(프록시) 구분.
5. 논문/데이터셋 주장은 원문 확인 후 인용.
