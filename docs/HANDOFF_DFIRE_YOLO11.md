# HANDOFF — D-Fire YOLO11 화재 baseline

## ▶▶▶▶▶▶▶ 재개 — 새 세션 시작점 (2026-09-01 세션 종료 · 팀 발표자료 전달 국면)
> **한 줄 상태**: (A) 발표 문서 정합 완료 + 팀 공유 md(내부/외부) 생성 + **main 병합(로컬·미푸시)**. 현재 국면 = **팀원 발표자료 전달** — 시각자료 확정됐으나 **(A)/(B) 프레이밍 충돌 미해결**(팀원이 파인튜닝 before/after 이미지 원하나 (B) 미실시).
> **★★ 다음 세션 시작점 = 이 미결**: 팀원(배찬우)이 **"기성 base(파인튜닝 전) 0검출 → 파인튜닝 후 N검출" before/after 이미지**를 원함(사람검출 예시 스타일). = **(B) 전이학습 결과.** **우리는 정직하게 못 만듦**: ①**파인튜닝 모델 없음**((B) = 실 급식실 화재 데이터 부재로 봉쇄) ②**프리즈 base가 합성 이미 잘 잡음**(recall 0.809~0.994)이라 "기성 0검출"이 거짓 → 만들면 데이터 조작. **▶ 다음 할 일 = (1)팀원과 (A)/(B) 정렬**(파인튜닝 안 함·못 함 · 우리 자랑 = "파인튜닝 없이도 base가 합성 인식[프록시]") **(2)정직한 대안 이미지 셀 작성**(base가 합성 composite 검출 box+conf · **빈 배경[불 없음 0] → 합성 불꽃[검출 N] 페어** · CELL 28/38 기반). **셀 미작성.**
> **★★ push 금지 (안전)**: repo `K-H-MOON/kitchen-smoke-poc` = **PUBLIC**(2026-09-01 API 확인 `private:false`). `dfire_journey.md`(내부)에 **이메일 blessmoonkh** + 계정구조 · 문서 전반에 Drive경로·급식실 CCTV 설명 → **push하면 공개 노출.** origin/main = `ac0322e`(미푸시). 백업/협업 필요 시 **repo를 private 전환 후에만** push. [[commit-changes-continuity]]
> **이번 세션 커밋 (전부 main 반영·로컬 cb1c3c8)**: `0da062e`(문서정합 나+다: 4.3 열화 → deck §05 견고성·Q&A Q11·여정 13번) · `4748ebb`(`docs/dfire_journey.md` 팀공유 전체 여정 md) · `5db233f`(재현 부록: Drive경로·Roboflow·계정·라이선스) · `cb1c3c8`(`docs/dfire_journey_external.md` 외부용[계정 제거]·학교CCTV 외부공유금지·HANDOFF vfx_bank 업로드완료).
> **팀 전달물 (2종 md + 아티팩트 3)**: `docs/dfire_journey.md`(내부·이메일 포함) · `docs/dfire_journey_external.md`(외부·계정 제거·"담당자 문의") · 아티팩트 deck `c7a24a9d`·여정 `e8240245`·Q&A `82643b64`(전부 비공개·공유 켜야 열림·in-app 브라우저는 미로그인이라 못 봄).
> **★발표 시각자료 확정(검증완료·savefig grep+이미지 육안)**: **14번** = `loc256`·`miss64`·`det_bratio` PNG 3장(Drive `synth_sweep/`·CELL 38/40b·box recall ablation·실사 composite=학교CCTV배경) + deck **§03**(over/screen)·**§04**(진단 두체제) 캡처 · **12번** = 여정 12번섹션 도해(NIST불꽃+급식실배경=합성·"배관은 작동" 캡션) · **13번** = 여정 13번섹션 도해(크기 0.994→0.694·화질 ≥0.96·"깨끗한 불꽃6종 N=160"). **★제외**: `scale_sweep`/`degrade_sweep`(CELL 24/25 **오염 초기본**·축·지표·값 다름[image recall @conf·px]≠우리 box recall@IoU0.5·0.994는 CELL27 클린뱅크로 별개) · `dryrun_montage`(CELL31·**기각된 발견2[조리면배치]** 담김·블렌드/배치 교란) · `diag_montage`·`vfx_bank_montage`(**존재 안 함**) · `dfire_boxing`/`dfire_rand`(`/content` 휘발·Drive 없음).
> **★프레이밍 원칙 (팀원 커뮤니케이션·핵심)**: (A) = 프리즈 base가 합성 인식[프록시·필요조건] · (B) = 합성/실사로 학습해 실전 성능 오르나[봉쇄]. 팀원이 "sim-to-real 증명"이라 표현하나 **(A)로는 증명 아님**. 발표 = 반드시 **"base가 합성 인식(프록시)"**으로 · "실화재 검출/파인튜닝 성공"으로 쓰면 **과대주장**. 사람검출 before/after(기성0→파인튜닝13)는 (B) 성공담 템플릿 = 우리 단계 아님.
> **행동원칙**: [[no-running-ahead-verify-first]]·[[no-premature-conclusions]]·[[working-style]]. **이 세션 교훈** = 팀원/외부의견 제시 파일명·수치 다수 오류(vfx_bank_montage·diag_montage 존재안함·`realneg_frames/synth`=798장(8교)≠18장[18은 firecrop_src placement.json 선택분]·scale/degrade 오염본·dryrun 기각본·0.994 provenance=CELL27)를 **savefig grep + 이미지 육안 + GitHub API로 매번 정정**. **파일 존재 ≠ 내용이 서사와 일치**. 인용은 셀 코드 확인 후. 커밋은 public repo라 push 금지.

## ▶▶▶▶▶▶ 재개 — (2026-08-31 · (A) 발표 상세 · 아래는 이전 시작점)
> **한 줄 상태**: (A) = **진단-우선 발표로 완결** · **나+다(문서 정합) 완료**(2026-08-31 후속 세션·커밋). 4.3 열화 발견을 deck(§05 신규)·Q&A(Q11)·여정(13번)에 반영 · GT스윕 잔차 0.211 여정 반영 · 여정 13번서 신규축(조명) **미제→닫힘** 명시 · 세 아티팩트 재배포(같은 URL). 프록시 유의미 작업 전부 완료 · 미제 정직 기록. 다음 = 아래 §다음 갈래 (라)/(B) 또는 발표 마무리.
> **산출물 (세 문서 + 소스)**:
> - **요약 발표 도표** = 아티팩트 `c7a24a9d-8013-4811-a293-860926876b14` "낮은 recall의 정체" · 소스 `docs/dfire_diagnosis.html`(커밋). **7-beat**(§05 견고성=4.3 열화 추가) 진단-우선·검증 수치만.
> - **전 과정 eli5 여정** = 아티팩트 `e8240245-4649-4cd0-95e0-0168eb2251e1` "D-Fire 정직한 0.660" · **사용자 라이브 편집중**(repo 미편입 · 수정하려면 Artifact read로 최신본 재확인→병합→재배포. 뷰어는 공유 핀 옮겨야 최신 봄).
> - **예상 질의응답** = 아티팩트 `82643b64-e582-43bb-91de-237eb9d7913b` "예상 질의응답" · **정본 `docs/dfire_qa.md`**(커밋 · HTML 소스는 재생성 가능이라 미커밋).
> - 근거·상세 = 이 HANDOFF ▶▶▶▶▶ 블록(아래) + `SYNTH_METHOD_EVIDENCE.md`(문헌 원장) + 메모리 [[dfire-yolo11-baseline]].
> **파이프라인 정산 (사용자 6단계)**: 1 base ✅(ptrain_b79 0.660) · 2 합성검증 ✅(이미지 recall 0.809 게이트 통과→4) · 3 섞어학습=우회 · 4.1 불꽃셋 ✅ · 4.2 합성법 ✅✅(ablation 완주) · **4.3 열화 ✅ 닫힘**(조도 극단[저조도·과노출]이 지배 killer·순수 노이즈 견고·조기검출엔 조명>압축) · 4.4 GAN/diffusion = ⬜ **선택 프런티어(생성 필요)** · 5·6 프록시판=ablation 흡수(=했음) / 학습전이=**(B) 봉쇄**. → **프록시로 할 수 있는 유의미 작업 완료** · 남은 종착은 새 데이터/생성/학습을 요구(현재 시간·데이터 제약으로 out).
> **미제(정직 기록)**: ①@256 GT 잔차 0.211(3갈래=base tight/소스 wispy/IoU0.5 임계·미제) ②열화 dose-response(강도별) 미측정 — **4.3 자체는 닫힘**, 이 강도축만 잔여 ③0-b 박스 0.675 vs 0.669 미해소(헤드라인 강등·CELL 41 재실행시 확정) ④**(B) 실 급식실 화재 홀드아웃 부재**(조리영상 28개 0건·AI-Hub 조리불 없음)=최대 관문·실 양성 확보가 유일 열쇠.
> **▶ 다음 갈래**: **(나)(다) = ✅완료(2026-08-31 후속)** — 4.3 열화 발견을 **deck §05(신규 견고성)·Q&A Q11·여정 13번**에 반영 · 여정 미제 "스윕 안 돌림"→**잔차 0.211**(14번+미제) · 여정 13번서 신규축(조명) **미제→닫힘** 명시 · deck **6→7-beat**·여정 footer **6→7단** · 세 아티팩트 재배포(같은 URL). **남은 갈래(사용자 선택)**: (가) 발표 이대로 마무리(문서 정합 끝났으니 가능) · (라) 프런티어 4.4 GAN/diffusion(생성 필요) 또는 **(B) 실 홀드아웃 확보**(봉쇄=통제연소 실촬/대안 실양성셋 → 열리면 §▶▶(B) 순서: ①배포목표 먼저부터).
> **재개 레시피**: 런타임 끊기면 `/content` 초기화·Drive 생존 → 각 셀 자립형(self-mount·ultralytics 설치가드). 셀=`docs/synth_sweep_cells.py` + 이 세션 셀들: **CELL 42**(레버 재집계·read-only)·**43**(rec@.1/GT스윕)·**43+**(GT 0.1~0.7)·**44**(열화 aug·numpy결정론). 전부 **sanity 게이트**(VFX none/over_ctx=CELL42 재현) 내장. 원자료 `synth_sweep/ablation_*.json` Drive 생존.
> **행동원칙**: [[no-running-ahead-verify-first]]·[[no-premature-conclusions]]·[[working-style]] 준수. **이 세션 교훈** = 여러 과대주장 → 적대 패널(5렌즈)+원자료 재집계+단일축 분리+sanity 게이트로 **매번 정정**(0.809 vs 0.915·1.9x/7.7x 배수·21/26 체리픽·"스필 null"·잔차 0.211·steam premise·albumentations 대신 numpy결정론 등). 인용은 원장 확인 후.

## ▶▶▶▶▶ (가) 발표 준비 — 숫자 검증 완료·진단-우선 서사 (2026-08-31 후반)
> **방향 확정 = (A) 결과를 "진단-우선 발표"로.** 원 질문 4-beat("D-Fire base→합성 test→개선 방법론→적용시 수치") 판정: 앞 3 beat 부합·마지막 beat는 **"프리즈 검출기의 합성 *인식률*(프록시)"로 한정하면 부합, "검출기 *성능* 개선"으론 미부합((B) 필요)**. 헤드라인=**진단**(개선 크기 아님). 배경=발표 목적이라 (B)(실 홀드아웃) 불요·(A)로 충분. 상세 사고흐름=이 세션.
> **★숫자 전량 검증(read-only 재집계, 재추론 0) — CELL 42(`synth_sweep/ablation_rows.json` 재집계)**: 미검증 버킷 비움. 레버 정본(VFX·장면집계 n=26·rec@IoU0.5·conf0.25·GT alpha>0.1):
> | level | @64 | @128 | @256 | / | over_rand | 0a_hard |
> |---|---|---|---|---|---|---|
> | over_ctx | 0.165 | 0.410 | 0.464 | | @256 0.487 | @256 0.504 |
> | screen_ctx | 0.075 | 0.218 | 0.299 | | @64 0.156 | @64 0.032 |
> | over_ctx_spill | 0.218 | 0.391 | 0.466 | | | @128 0.295 |
> **★진단 3숫자 검증(CELL 43=CELL37+rec@.1+JSON저장·sanity[GT>.1 열이 CELL42 재현]=통과·`synth_sweep/ablation_rec01_gtsweep.json`)**: 두 체제+under-box 한 표(VFX over_ctx):
> | scale | rec@.5 | rec@.1 | GT>.5 | 정체 |
> |---|---|---|---|---|
> | @64 | 0.165 | 0.173 | 0.165(회복0) | 검출병목=실효21px |
> | @256 | 0.464 | 0.835 | 0.581(+0.117) | 위치병목=우리 GT(alpha0.1) 헐거움·base 정상 |
> 스필=소형검출 이득(rec@.1 VFX@64 +0.135·NIST@64 +0.305·대형선 손해).
> **★4 정정(발표 서사 lock)**: ①"21/26"=@128 체리픽(실측 @64 13/26[tie13=바닥효과]·@128 21/26·@256 16/26·**strict 패배 0/1/2=over≥screen 견고**) ②**"스필 0" 철회 → 배치=진짜 null·스필=스케일의존(대형 null·소형 이득)** ③over>조잡(0a_hard)은 @64/@128만(@256 0.504>0.464=박스정합 아티팩트)→견고 레버는 **"over>screen"으로 특정** ④블렌딩 배수 1.55(@256)/1.88(@128)/2.20(@64)·크기는 @128서 포화(@64→128 2.5x·128→256 1.13x). **★"밝은 코어=base 한계" 프레이밍 회귀 금지**(정본=base 정상·tight per 관행 bright-ratio 0.72·우리 GT 헐거움). 배치-null 기전="장면통합 무관"은 해석(랜덤도 밝은면 앉음 caveat).
> **남은 것**: 0-b **비교 수치**(우리 이미지 0.915=프로즈만 미검증 · 0-b 박스 0.675 vs 0.669 불일치)만 미검증=헤드라인서 강등·제외(교차지표 오독[0.809 이미지 vs ablation 박스] 회피). ★단 **0-b 이미지 recall 0.809 자체는 검증됨**(Phase1 GATE 재현) — deck엔 출발점 *맥락*으로만(숫자 없이). 필요시 CELL 41 재실행(api_key·sanity=0.809 재현)로 0.915·박스 확정.
> **★(가) 발표 도표 = 아티팩트 발행됨**: `https://claude.ai/code/artifact/c7a24a9d-8013-4811-a293-860926876b14` (title "낮은 recall의 정체"·검증 수치만·**6-beat**[문제·인식·레버·진단·**정정의연속**·한계]·출발점 0.809[생성셋 이미지 recall·검증됨·격리표기·박스지표와 비교 안 함]·레버표+진단표+2차트[over-vs-screen·두체제]·**정정의 연속 섹션=4 되돌림**[base병목철회·배수7.7×→1.55-2.20×·스필재판정·0a철회, 근거 동반]·배포목표 슬라이드는 제외[안 할 (B) 예고 회피]·기술-에디토리얼 디자인[ember/teal·Noto Serif KR/IBM Plex]·라이트다크). 소스=**`docs/dfire_diagnosis.html`**(repo·커밋됨·scratchpad 원본). **재배포=같은 file_path 또는 url**(url=위 아티팩트). **다음=사용자 조정**(헤드라인·섹션·0-b 맥락/숫자·영문·배포목표 슬라이드). [[no-running-ahead-verify-first]]·[[no-premature-conclusions]] 준수(이 세션 내 여러 과대주장→적대 패널+원자료 대조로 정정).
> **★문헌 근거 (발표용 매핑·원장=`docs/SYNTH_METHOD_EVIDENCE.md` 271줄 vetted)**: 조건별 출처 — over/screen ← **FGL-GAN**(Sensors 2022·불꽃=광원·harmonization 부적합) + **Dwibedi**(ICCV 2017·블렌딩 혼합 +8 AP·**단 비발광 실증=우리 도메인엔 간접**) · 배치 ← **Dvornik**(ECCV 2018·컨텍스트 +5%) ↔ **Ghiasi**(CVPR 2021·랜덤 충분) = **상반→검증 대상** · 스필 ← FGL-GAN(반사광=배경광량·거리 의존 사실성 요소·직접 원리) · 스케일 ← **COCO**(2014·32²/96² 구간)+**SAHI**(2022) — **64/128/256 특정값=우리 판단[근거없음]** · 프리즈 검출기=심판 ← FGL-GAN(conf 품질지표)+**CAS**(NeurIPS 2019·downstream 평가) = **통합 직접 선례 없음=신규 조합**. **★★종속변수 프레임(핵심 방어)**: 선행(Dwibedi +8AP·Dvornik +5%·ODGEN +25.3%)=합성으로 *학습*시킨 효과 · 우리=프리즈 *인식률* → **결과 달라도 모순 아님(다른 질문)=(A)↛(B) 비전이 근거**. [우리 판단·직접선례없음]=64/128/256·L1→L4순서·프리즈심판설계=**신규성**(약점 아님). 원장 정정=BoWFire "80%↓"철회·arXiv:2606.19817(오타)→Borji(2103.09396)·DACBFIAM 미검증. **deck 03 방법레버에 문헌→조건 매핑+DV프레임 반영(발행됨)**. 여정문서(e8240245) step14에도 문헌 매핑 반영·재배포(사용자 라이브 편집중이라 재확인후 병합).
> **★GT임계 확장 스윕(미제 "정답박스 얼마나 조여야" 닫음·CELL 43+ GTS 7개[0.1~0.7]·sanity 통과·재추론)**: **최적점 없음** — VFX over_ctx @256 rec@.5가 [0.1→0.7] 단조 상승(0.464·0.513·0.532·0.541·0.581·0.596·**0.624**)·0.7서도 오름·꺾임 無. **회복 스케일 의존**: @64 flat(0.165→0.171·rec@.1 0.173=검출병목·GT무관) · @128 거의 닫힘(0.410→0.534·rec@.1 0.592·잔차 −0.058) · @256 GT ~43%만(0.464→0.624·rec@.1 0.835·**잔차 −0.211**). **★새 미제=@256 GT로 안 닫히는 잔차 0.211**(원인 미규명·det가 alpha0.7코어보다 tight ↔ 고임계서 wispy불꽃 GT박스 소실[VFX@64spill 0.218→0.197·NIST@64 0.028→0.000 하락 실측] + (c)IoU0.5 임계 자체[잔차=rec@.1−rec@.5 구조]=**3갈래 섞여 실험 하나로 안 갈림·미제로 유지**(닫으려 말 것·발표 결론 불변)). 처방=GT alpha 0.4~0.5로 조이면 회복 대부분(단일 정답임계 없음·threshold≠density 0.401). "recall 최대화"로 팔지 말 것(단조라 게이밍). deck 04·발표=스케일별 회복대조로.
> **★4.3 열화(aug/noise) 유의미성 = ✅닫힘**(실행결과=다음 §4.3 실행·닫힘 항목 · 아래는 당시 계획·근거 기록으로 보존): CELL 25가 부분답 — **충분크기서 견고**(JPEG q8·blur σ5·저해상 0.15× 모두 recall≥0.96)·**소형×JPEG만 붕괴**(scale0.11 q8→0.463). **미측정 신규축 = 센서노이즈(GaussNoise/ISONoise)·밝기(RandomBrightnessContrast)**. 하려면 refined Part1(신규축만·@64/128/256 전부[정보는 소형에]·**시드고정**[albumentations 랜덤·안 하면 sanity 무의미]·재확인축은 "CELL25 replication" 라벨·sanity=no-aug@256 0.464 재현·**스코프=프록시 인식 견고성이지 "aug 학습 도움" 아님[그건 (B)]**). 사전확률=같은 크기의존 패턴이라 큰 반전 낮음·단 미측정("해봐야 앎"). 발표 불요. **★steam FP는 안 함=실데이터가 이미 답**(nofire_kitchen181+realneg684서 수증기 0/7·김 자욱해도 안 찍힘·base fire-only[smoke 드롭]라 기전도 설명·실 트리거=주황/붉은 색혼동). 합성 수증기=더 약한 증거로 닫힌 질문 재개+realism 부담 → 미제로도 불요.
> **★4.3 실행·닫힘(2026-08-31·refined Part1=CELL44·over_ctx×3스케일×열화6·numpy/PIL 결정론·sanity[VFX none=CELL42] 통과·`ablation_aug.json`)**: **예상보다 나은 발견** — 열화 균일 아님. **조도 극단이 지배 killer**(VFX@64 none 0.165→**iso저조도 0.053·bright과노출 0.062**·−62~68%·JPEG 0.085·mblur 0.115·gauss 0.152보다 큼) · **순수 노이즈(gauss) 견고**(전 스케일 −5~8%) · jpeg 중(@64 −48%=**CELL25 재확인**) · mblur 약 · 전부 **소형일수록 심함**(scale 역순). **★귀속(설계 덕 깔끔)**: gauss(노이즈만) 견고 vs bright(과노출·노이즈X) 붕괴 → 붕괴 원인=**노이즈 아니라 조도/노출**(밝은 코어 대비 파괴). **기전(해석·established 연결)**: base=밝은 코어 대비로 검출(bright-ratio 0.72)→저조도는 코어 어둡게·과노출은 배경 희게=대비 파괴·소형(밝은픽셀 적음) 특히 취약. **★실용**: 조기(소형)검출 목표면 **CCTV 조명/노출 관리 > 압축·노이즈**(방향 바뀐 발견). 스코프: 각 열화 단일강도(dose-response 미측정)·NIST n=2 경향(같은 방향 더 강함). → **4.3 부분답→완결.**
> **★발표 예상 질의응답 원문(정본)=`docs/dfire_qa.md`**: 11항목(핵심5[정당성·Q1실제화재·Q3 0.915vs0.809·Q4개선배수·Q6배포오경보] / 참조6[Q2선행·Q5낮은recall·Q7 GT조임·Q8잔차·Q9통계도메인·Q10조건선택])·짧은 A 한 줄+보충·근거 수치. 최상위 방어선="프리즈 인식률(프록시)이지 실검출 아님·실전=(B)·데이터 부재 봉쇄". 발행 아티팩트=**`82643b64-e582-43bb-91de-237eb9d7913b`**("예상 질의응답"·인쇄가능·라이트다크). **HTML 소스는 재생성 가능이라 미커밋**(정본=md·중복 원본 회피).

## ▶▶▶▶ 재개 — (A) 종결. **다음 세션 지금 할 일 = (B) 진입: 아래 §▶▶ (B) 진입 계획 ①(배포 목표)부터**
> 새 세션 시작: ①배포 목표(사용자가 급식실 운영 조건 제공: 어느 크기까지·recall 목표·오경보/일 허용·조기[소형] 우선?) → ②소스 웹확인 → … (전체 순서·주의=§▶▶ (B)). 작업 브랜치=`claude/dfire-yolo11-synthesis-ablation-abf58b`(이 문서·`synth_sweep_cells.py` CELL 32~41 여기). (A) 결론·정정·발표도표는 아래 §ablation 최종결론.
**한 줄 상태**: 불꽃 합성 *방법론* ablation **완주·검출레벨(rec@.1) 재검증·miss 육안까지 완료**(CELL 32~38, Drive 생존). **미규명 전부 해소.** 결론=아래 §ablation 최종결론. 다음 갈래=발표지 배포 / (B) 학습·실데이터 / 조기(소형)검출 프런티어. 발표 도표=아티팩트(claude.ai/code/artifact).

### ▶▶ (B) 진입 계획 (새 세션·(A) 종결 후·리뷰 합의)
**전제**: (A)는 실제 성능을 말하지 않음(합성 프록시). (B) 첫 단계=학습 아니라 **실 홀드아웃 확보**(학습 필요 여부 자체가 미지). **5(생성)·6(재검증)=폐기 아니라 보류** — (B)서 학습데이터 필요해지면 5번이 용도 바꿔 부활(실 web불꽃→급식실 bg 컴포지팅 등).
**★도메인 2축 (배포 진실=급식실 CCTV 실 유류불·직접 확보 난망)**: (A) composite=급식실 bg 실재(✓)+합성 불꽃(✗) · web KitchenFire 홀드아웃=실 불꽃(✓)+bg/카메라 도메인 어긋남(✗·가정/식당·CCTV 아님). **상보적 프록시·어느 것도 배포 아님.** 더 가까운 프록시=실 web불꽃을 급식실 bg에 컴포지팅(둘 다 근접·seam만 합성). 배포 진실=통제연소(급식실 실화재)=프런티어.
**★(B) 첫 관문 = 실 양성(급식실 화재) 부재 — 두 실증 제약(journey doc step15엔 있음·HANDOFF엔 이번 추가)**: ①급식실 조리영상 **28개 전수검수 화재 0건**(정상 — 급식실은 불나면 안 되는 곳이라 자연히 실화재 영상 없음). ②**AI-Hub 71751 조리불 없음(2026-08-30 실증)**: device=촬영기종(카메라)이지 조리기구 아님 · ENB(음식점) 실내불꽃 16장면 전부 일반 실내화재 · 코드북에 조리 원인/장소 라벨 없음 → 조리불 소스로 못 씀. ★따라서 아래 §AI-Hub 상단의 "device ct=조리기구?" 서술은 **이 실증으로 폐기**(구버전). → **실 양성 확보가 (B)의 유일 관문**(대안=통제연소 실촬·다른 실양성셋).
**순서(잠금)**: ①**배포 목표 먼저**(recall/FP 상한·소형 recall인지 전체인지 — 홀드아웃 구성을 결정) → ②소스 웹확인+샘플30 육안(실주방·유류불·CCTV각·라이선스·"이름≠내용"[D-Fire=산불 교훈]) → ③라벨 컨벤션(bright-ratio 0.72·D-Fire 정합) → ④수집·직접박싱(100–150 독립장면·**크기 실효px 기록**·시퀀스 dedup) → ⑤pHash GATE(D-Fire·합성소스 대조·누수) → ⑥B1 baseline(★**크기 층화**·imgsz 640/960/1280·SAHI·conf 스윕 → **3-way 분기**: (a)추론만으로 목표도달=학습 불요 / (b)CCTV각 유류불 놓침=진짜 문제=학습 필요 / (c)놓친 게 요리방송·뉴스 등 out-of-domain=**홀드아웃 재구성 필요**[도메인 2축·miss 육안으로 (b)↔(c) 구분]).
**주의**: NC-ND(DetectiumFire)=평가전용·학습금지(폴더 물리분리·경로 하드코딩 배제). 오버샘플한 소형층 recall≠pooled 배포recall(층별 보고·자연 크기분포 미지라 단일 배포recall 단정 금지). 라벨 QC=bright-ratio 측정(CELL 39/40b 도구). 자산 재사용 금지(NIST/VFX/0-b=합성소스·순환). 기존 negative 18장(급식실 튀김솥)=FP측정+배포 bg축 참조.

### ★ ablation 최종결론 (frozen ptrain_b79 · VFX26+NIST4[peak2+ign2] × 18배경 × scale64/128/256 · 장면집계)
1. **블렌드 over ≫ screen** (established·VFX n=26·전 스케일·전 GT임계·rec@.1 검출레벨서도 유지). screen=밝은면서 불꽃 픽셀을 흰색으로 지움=진짜 검출손실. → **base 블렌딩=over 확정.**
2. **배치 over_ctx ≈ over_rand** (★발견2 강한형 **기각**·established). 조리면 배치가 recall 안 깎음. (약점: 주방 랜덤도 밝은면 앉음 → "grounded-조리면 ≈ random-주방위치"까지만.)
3. **스필: (A)@IoU0.5 드롭**(recall ROI 없음 + synFP 25/26↑) · **단 rec@.1선 소형검출 도움**(VFX@64 +0.135·NIST@64 +0.305·글로우가 검출시키나 박스 번져 IoU↓) → **(B) 소형불 대응 후보**(완전배제 아님).
4. **★scale 지배·두 체제**: **@64=검출병목**(gap≈0·IoU 낮춰도 안 잡힘=진짜 못봄) / **@256=위치병목**(검출 rec@.1 0.835·IoU0.5가 절반 깎음). 128 전이.
5. **⑤ source realism(real oil>movie) 기각**: VFX 저평균=이질성(4전멸+6우수·top 0.94>NIST peak 0.861)+성긴GT지 리얼리즘 아님.
6. **큐레이션 비(非)레버**: dead-4도 검출됨(rec@.1↑)→cov 컬=지표 게이밍(A). (B)선 어려운소스=정보량↑→컬 역효과. (순환 경계.)
7. **성김(cov) 메커니즘 3중 확인**: 소스간(NIST 조밀 gap≈0 vs VFX 0.37)·소스내(cov-Δ corr −0.63@256/−0.52@128/@64 −0.05[gap無])·GT반응(VFX만 회복·NIST 0/−). **r²≈0.40=중간세기·유일원인 아님**(⑥종횡비 잔여).
8. **대형 저recall @IoU0.5 분해**: ~**1/3 GT컨벤션**(alpha>0.1 wispy 헐렁·GT@0.5로 회복 +0.117·성김특이) + ~**2/3 under-box**(base가 밝은 코어만·**det⊂GT·det/GT=IoU=0.18~0.30 실증**). GT@0.5로도 1/3만 회복=det이 alpha0.5 코어보다 작음.
9. **★★통합 = base는 불꽃 "밝은 코어"에 반응**: @64 코어 작아 발화실패 · @256 코어만 박싱(under-box). 두 증상 **한 뿌리**. **합성법 개선불가 = base/Phase B 영역.**
10. **0a_hard "조잡>사실" 철회**: rec@.5 0a승(0.504>0.464)→rec@.1 역전(0.799<0.835)=박스정합 아티팩트. ("realism 무익"은 배치·스필 null로 여전히 성립.)

**★★정정(2026-08-31 후속 — D-Fire 관행 확인 CELL 39/39c/40/40b): ⑧⑨ + "base 병목/약점" + @64 해석 정정. base 약점 아님·병목은 우리쪽·가역.**
- **@64 = 상대크기(순수 소형)**: 우리 배경 long-side 1920px·ablation imgsz640 → scale 64 실효 **21px**(D-Fire 64px→98px·4.6×). 우리 전 스케일(21/43/85px 실효)이 D-Fire 학습분포(49~394px) **하단** → ablation이 실효-작은 구간서만 돎. @64 miss=**크기**(appearance 아님·상대크기 교란 확인). **scale 결론=측정구간 단서·큰 실효크기선 포화 가능.** 소형 대응=imgsz↑/SAHI.
- **under-box = A(GT정의)+C(소스밀도)·B(base약점) 기각**: D-Fire 사람박싱=밝은불 tight(대표 random·1박스/img bright-ratio med **0.72**·편향샘플 0.355는 야간 patchy 산불). 우리 GT(alpha>0.1)=wispy. **CELL40b(bg오염0·alpha밀도): base det 박스 평균alpha 0.401·고alpha 0.397 > 우리 GT 0.256·0.250 (+0.145·n391)** → base가 조밀부(고alpha) 박싱=D-Fire 컨벤션대로 **정상**·우리 GT wispy 포함이 under-box 원인. ⑧ "2/3 under-box=base특성"·⑨ "합성법 개선불가=base영역" **철회.**
- **"base 병목/약점" 철회**: @256 under-box=우리 GT loose+VFX wispy+실효-작은 스케일=**전부 우리쪽·가역**. base는 @256 검출함(rec@.1 0.835)·D-Fire 컨벤션대로 tight 박싱. base 대체로 정상.
- **Phase B 처방(귀속 확정)**: (A) GT를 base 컨벤션에 맞춤(alpha>0.1보다 조임·det 평균alpha 0.40·정확 최적점=IoU-vs-GT임계 스윕 필요) · (C) 밀도 높은 실불 소스 추가(**단 (B)선 wispy도 학습·배제 아님**·실화재도 성김 있음) · 소형=imgsz↑/SAHI. **base 재학습 우선순위 아님.** ★**단 이 처방들의 실제 화재 효과는 미측정((A) 프록시 기준)** — (A)는 실제 성능을 말하지 않음·합성 점수는 실전 대리지표 아님. **★★(B) 첫 단계 = 학습이 아니라 실 주방화재 홀드아웃 확보**(지금은 학습 필요 여부 자체가 미지 — 실 홀드아웃 없이는 판단 불가).
- **D-Fire 구성**: 다양한 대형 실불(산불+건물+차량+도심·뉴스푸티지·**근접 주방 아님·산불 일색도 아님**).
- **방법교훈**: numeric 3회 교란(상대크기·편향샘플[정렬-첫N]·bg오염[밝은 주방서 luma mask 오검]) → 육안+alpha+대표샘플로 매번 정정. [[no-premature-conclusions]] 강화.

**★생성(0-b) vs 컴포지팅 비교 (원 질문 답·CELL 41)**: 검출레벨(이미지 recall·apples) **우리 over@256 0.915 ≥ 0-b 생성 0.809**(실효크기 85 vs 98px·오히려 0-b 큼=크기 아님·flame-only 하한 rec@.1 0.835도 ≥0.809). 박스 recall@IoU0.5는 0-b 0.669 > 우리 0.464이나 **GT정의 교란**(0-b **tight GT** bright-ratio 0.705≈D-Fire[사람/자동라벨 미확인] / 우리 alpha0.1 loose 0.562)·품질 아님. → **"생성이 낫다"=표면 착시(지표+GT)·검출레벨 컴포지팅 ≥ 생성.** ★단 **우리=best config(over@256)·0-b=단일 조건(생성 그대로)** — 우리 평균(전 스케일·조건)은 훨씬 낮음(@64 0.173). **원인 귀속 안 함**(VFX=movie fire·AI-gen 텍스처통계 차이 등 미상·⑤ real>movie 기각됨). ⚠️unpaired·진짜 paired(같은 배경 생성vs컴포지팅) 미실시.

**정밀화(리뷰)**: (a) @64 대비=**판정유보**(n_hit 28 약함)·근거는 육안(어두운 bgL 85/95/111도 det 0.00=밝기 무관 소형불 미발화). (b) under-box **양면성**="base 학습컨벤션 ↔ 우리 GT(alpha>0.1) 정의 불일치"(**D-Fire 박싱관행 미확인**=Phase B 열쇠: base 고칠지 GT 맞출지). (c) 운영=**"대형 검출 확보·조기(소형) 경보 미확보"**(@64 rec@.1 0.173·경보 핵심가치=조기검출).

**한 줄**: 합성 *방법*(over 제외)은 미미한 레버 · over(washout 회피)+충분크기가 큰 축 · **낮은 프록시 recall의 실제 병목 = 우리 GT정의(wispy)+소스밀도+실효-작은 스케일(전부 가역)·base는 대체로 정상**(D-Fire 컨벤션대로 tight 박싱) · 진짜 큰 레버(GAN 생성·학습)+조기(소형)검출은 미탐 프런티어. (A) 프레임 전체 — (B) 실전유용은 별개.

**스코프 한계**: 컴포지팅 knobs만(aug/noise·GAN/diffusion 미측정)·소스 다양성 미조작·(A)프리즈 프록시·NIST n=2(경향)·ign/peak 비독립·D-Fire 박싱관행 미확인·⑥ 종횡비 교락(under-box 잔여에 섞임).

**셀**: `docs/synth_sweep_cells.py` — CELL 32(본실험)·32-nist(NIST뱅크 ign rescue)·33(장면분포)·34(소스품질/miss)·37(GT임계 스윕)·36(검출vs위치 rec@.1+cov-Δ)·38(miss 육안)·**39/39c(D-Fire 박싱관행+@64 상대크기)·40/40b(det 박스 alpha밀도=under-box 귀속)**. 전부 자립형. (셀순서 흐트러짐·자립형이라 무관·CELL35 폐기. 39/40은 D-Fire=/content 재다운로드[api_key] 필요.)

**★방법 교훈(이 실험서 반복)**: Claude가 여러 번 과대판정 → 리뷰가 정정: 스필"해롭다"(→n=2 NIST에 끌림)·큐레이션"레버"(→관찰↔처방 혼동·순환)·"대부분 GT아티팩트"(→rec@.1이 GT+underbox 합침)·under-box"느슨"(→실은 코어만 작게). **rec@.1/GT분해/cov-Δ/육안이 매번 잡음.** [[no-premature-conclusions]]·[[no-running-ahead-verify-first]] 강화.

**상태(2026-08-29→08-31): 1·2 완료 · 3 우회 · 4번(합성 *방법론*) — step4 sweep + ablation 완주 = 종결. 남은 프런티어=(B)실 양성데이터·조기(소형)검출.** (파이프라인 = §프로젝트 파이프라인.)

## ▶▶▶ 2026-08-30 최신 — 불꽃 합성 방법론 ablation 설계 확정 (VFX-pivot)
**목표=여전히 (A)**: frozen YOLO11s(ptrain_b79)가 *합성*을 얼마나 인식하나(recall/precision/confidence). 학습(B) 아님.
- **근거 원장 = [`docs/SYNTH_METHOD_EVIDENCE.md`](SYNTH_METHOD_EVIDENCE.md)** — 각 설계요소의 논문 원문(verbatim)+[직접실증/간접/근거없음] 분류. 보고서 "왜 이 방법" 근거표.
- **소스 전환(AI-Hub→VFX 중심)**: AI-Hub ENB 16 실측=약함(조리불 아님·bbox만·9/16 업스케일·실장면 ~7·recall 0.775<NIST 0.981). → **VFX CC0 검은배경(알파=측정·HD=업스케일0·클립 25-40=독립성↑·screen과 궁합)** 대량 + **NIST 조리유 4(앵커·조리유=대체불가·source 그룹변수)** + **AI-Hub 4장(0970/1169/1170/1187) 보류**.
- **측정=순차 ablation(paired)**: 기준선 **0-a**(하드 사각형·알파X 바닥)·**0-b**(기존 생성셋 NanoBanana ~0.8·**unpaired 절대비교만**·접근성 Phase1 확인)·**0-c**(무화염 배경 FP기준·필수) + 계단 **L1**(over+랜덤)→**L2**(→screen, over *대체*)→**L3**(+컨텍스트배치)→**L4**(+스필). 각 단계 recall+FP 측정.
- **교차/규칙**: SCALE=절대픽셀 불꽃높이 64/128/256(모든 L에 교차·소스별 상한=업스케일금지·결측셀 명시) · SOURCE=VFX/NIST 그룹변수(일반불 vs 조리유). **GT박스=배치박스 고정**(스필로 재추출 금지). **장면단위 집계**(pseudoreplication 회피). pairing: L1↔L2 완전·L2↔L3 위치=변수·L3↔L4 완전. **NIST 셀 n=3=경향만(검정X)**.
- **지표(Phase5)**: recall(img/box)+FP(vs 0-c)+**TP 평균 confidence**(recall 포화 보완·FGL-GAN 선례)+conf sweep PR/AP+IoU분포 + **miss 케이스 육안 20-30**.
- **스필 물리고정**: 코어휘도∝·역제곱(코어반경 clamp)·거리감쇠(FGL-GAN 반사광). 임의값 금지.
- **★정정(SYNTH_METHOD_EVIDENCE §9/Rec 강제)**: "BoWFire FP 80%↓"=근거없음 **철회** · arXiv:2606.19817=미래/오타 **철회**(Borji 2103.09396 대체) · DACBFIAM=미검증 · **L3=가설**(Ghiasi 무작위충분 ↔ Dvornik 컨텍스트우월 양방향) · **L2 screen=간접근거→벤치마크**(TP conf 95%CI 0배제 못하면 over 유지) · **frozen-검출기-심판 통합=직접선례 없음→"표준요소 신규조합"으로 정직 기술**.
- **실행 순서**: Phase1 GATE(pHash≤6+육안: NIST·AIHub4 vs D-Fire · **+0-b Roboflow 접근확인** · D-Fire=로컬없음/Colab Roboflow만) → Phase2 VFX(파일럿 5클립·통과기준 "≥3/5 클린알파"·본수집·QC게이트·GATE-2) → Phase3 뱅크+**배경 마킹(기준명시)·배경수 확정**·NIST 큰화염 2-4장 추가 → Phase4 관통테스트(소스2×배경2) 후 전체.
- **도구**: `scripts/mark_placement.py`(배경 불배치 마킹)·`scripts/manual_flame_box.py`(불꽃 박싱)·`scripts/vfx_extract.py`(VFX 프레임추출+luma-key 알파+QC·영상/이미지 둘다)·`docs/synth_sweep_cells.py` CELL29(AIHub vs NIST recall)·CELL30(추출 원인분리).
- **★Phase1 GATE — 완전 검증 통과(2026-08-30)**:
  - pHash 겹침 0건(NIST14+AIHub16 vs D-Fire 21,522·최소 Hamming 10~17>임계6·6~10 빈구간=무중복 신호) ✅ · 육안 몽타주(`synth_sweep/gate_phash_montage.png`) 최근접도 다른장면 ✅.
  - **positive control 통과(5/5)**: 변형(JPEG40+밝기+리사이즈) D-Fire 5장 자기인식 self Hamming 0~2 = pHash 견고·파이프라인 정상 ✅. **파이프라인이 진짜 near-dup을 H0로 검출함이 실증 → NIST/AIHub 최소 H10은 "도구 무능"이 아니라 진짜 무겹침**(positive control 없었으면 이 구분 불가). (AoF04305 "❌"는 삼각부등식상 AoF07786≡AoF04305[both H0]=D-Fire 내부중복쌍에 동점정렬 걸린 판정기준 아티팩트 — 실질 5/5.)
  - **★D-Fire 내부중복 확인(덤)**: AoF04305≡AoF07786 등 · 알려진 dedup 21,522→10,624와 정합. **GATE엔 무관**(질의=NIST/AIHub). 단 **base가 중복 많은 D-Fire로 학습됨 = 실효 다양성<명목** → **Phase B(학습) 해석 시 고려** 기록.
  - **0-b recall 재현 통과**: 프리즈 base로 gen 351장 추론 → **이미지 recall@0.25 = 0.809 = 기록 0.809 소수셋째까지 일치**(라벨·설정 불변) → 0-b 기준선 유효 ✅. 박스 recall@IoU0.5 = 0.675. **이미지0.809 vs 박스0.675 격차 0.134 = 다불꽃 이미지서 일부만 검출 → Phase4서 두 지표 다 추적(조건별 격차 변화가 신호).**
  - 21,522 vs 논문 21,527: **우리 해싱 스킵 0 확인**(글롭 전체 해시). 5장 차이 **원인 미확인**(Roboflow export or 원본)·영향 미미 — 조용한 드롭 단서로 남김.
  - 사각지대: 크롭-부분영역/동일사건-다른프레임 = pHash 배제 불가 → **출처 논리로 위험 낮음 *판단***(NIST=랩·AIHub=국내CCTV·D-Fire=브라질웹). "확인" 아니라 "판단"으로 기록.
  - **★NIST 장면ID(집계 단위)=화재이벤트 4개**: `1574199884`(alumipan2)·`1574198232`(calphalon)·`1508954077`(massloss13)·`1508958465`(massloss14c). ignition/peak=같은 이벤트. NIST 14프레임 → **집계는 이 4 scene 단위·n≤4·경향만**.
  - **★Phase2 파일럿 통과(2026-08-30)**: CC0 검은배경 불꽃 7영상 → `vfx_extract.py`(luma-key lo12/hi90·클립당2프레임) → **클린 5/7**(육안: 2·3·4·5·6=격자+불꽃만·통과기준 ≥3/5 초과). 탈락=1번(불꽃주변 haze막)·7번(회색배경 bg22/haze90%). **luma-key가 검은배경 불꽃 깨끗분리 실증**(VFX-pivot 근거). 선별기준=순수검정배경+선명불꽃(dim글로우/회색배경 제외).
  - **★Phase2 종결 — VFX 뱅크 확정(2026-08-30)**: CC0 49영상 → md5 dedup(6중복) 43고유 → 1차QC 탈락16(자동⚠4+애매12 불꽃끊김) → 클린27 → 2차QC(뱅크 육안·Claude 판독) 추가정제: 18084807 제외·10141290/9667144/9667220 불량프레임 제거(9667144는 블로운 f267 제거·대각텍스처 f144 유지·5659686 하단=불base라 유지). **최종 49 매트·26 장면·높이 528~1920px(전부≥256=업스케일0)**. 매니페스트=matte→scene_id[Pexels ID]→source=VFX→**orientation(vert40/horiz8/diag1 — 수평/대각은 L3 배치 다르게)**→**anchor_frac(접지선=주불꽃 base y/h·합성 시 배치점에 이 y 정렬=일관 접지)**→**core_lum/whitish_frac/high_bright(L4 스필 소스의존 해석용·VFX가 대체로 밝음=movie-fire)**. **★집계=26 장면 단위**(클립당 2프레임 유사→scene ID로 pseudoreplication 회피·NIST와 동일). `vfx_bank.zip`(34.3MB·crops+manifest) → Drive `firecrop_src/` **업로드 완료**(CELL 32~44 실행이 vfx_bank/manifest.csv를 읽어 정상 산출 = 확인·2026-08-31). 도구=`vfx_extract.py`(md5 dedup·영상/이미지).
  - **★Phase3 배경 마킹 완료(2026-08-30)**: CELL A로 주방배경 18장 샘플 다운 → `mark_placement.py`로 조리면에 불배치 박스 마킹(박스 하단=조리표면) → **placement.json 18장 전부** + manifest.json(bg→Drive rel) → Drive `firecrop_src/` 업로드. (tkinter 창 닫기 먹통이나 증분저장 덕에 데이터 안전.) 소스 확정=VFX26장면+NIST(peak2=256px가능)+AIHub4보류.
  - **★Phase4 관통 테스트 = `synth_sweep_cells.py` CELL 31(DRY-RUN)**: 2소스(VFX1+NIST1)×2배경×L1-L4(scale128). 합성함수(screen 블렌딩·물리스필[코어휘도∝·역제곱]·anchor_frac 접지·GT=불꽃bbox 고정)+지표(tp_conf/fp_conf)+몽타주 육안 검증. **통과시 전체 ablation(0-a/b/c + L1-L4 × scale64/128/256 × VFX/NIST · 장면집계 · 지표 recall/FP/conf/PR-AP/IoU/miss육안)로.**
  - **★CELL 31 dry-run/diagnostic 결과(2026-08-30·2소스×2배경): 파이프라인 정상 + 2발견**:
    - **★★발견 2(진짜 수확·교란없음)**: blend 고정(over) 순수 배치효과서 **over_rand→over_ctx가 bg00 유지(0.483→0.579)·bg01 붕괴(0.240→0.001·0.466→0.000)**. = **물리적으로 옳은 배치(밝은 조리면 위)가 프리즈 base엔 오히려 더 어렵다**(D-Fire=야외/일반화재라 밝은 조리면 저대비 불꽃 미학습=도메인갭). **Dvornik/Ghiasi(둘 다 *학습*효과)에 제3의 답=*평가난이도*↑.** Phase B 근거(base 놓침=학습 정보량 큼). "(A) recall최대화=쉬운샘플 선택" 함정이 실측됨.
    - **screen washout(발견1)**: bg00 over_ctx 0.579 vs screen_ctx 0.025(23×). 밝은 배경서 screen(≈255)이 불꽃 지움·over는 교체라 살림. **단 근거 n=2(bg00만)·spill행은 교란(spill이 blend무관 항상 screen)** → "bg00 한정 확인·본실험서 다배경 확정" (screen 기각 *방향* 맞음).
    - **spill=판정불가**(0.595 vs 0.471·방향갈림·n=2). "중립" 아님.
  - **★본실험 전 코드수정 4**: (1)spill글로우=additive(blend분리) (2)FP=0-c 배경FP집합 먼저→합성서 IoU높은것 제외 (3)랜덤위치 다중시드 (4)**NIST 매트 파일화(nist_bank·VFX와 동형식)** — 즉석추출이면 SOURCE 비교 오염.
  - **설계 확정**: base 블렌딩=**over**(screen=문서화된 기각방법·bg00). 계단 over_rand→over_ctx→over_ctx_spill + over_ctx/screen_ctx 다배경 비교. 18배경·scale64/128/256·장면집계.
  - **NIST 매트 뱅크 완성(2026-08-30)**: `nist_bank` 8매트·4이벤트 → **쓸만한 2장면**(alumipan2·calphalon peak·256px가능). ign(50~57px)=64px에도 업스케일→제외. massloss=금속리그+수평화염→제외. NIST 유효=2장면(조리유 앵커·경향만).
  - **★본실험 셀 = `synth_sweep_cells.py` CELL 32 작성완료(커밋 407ceed)·미실행**: over base + 수정4(additive스필·0-c FP·다중시드·NIST파일뱅크) · 조건0a_hard + 계단 over_rand→over_ctx→over_ctx_spill + screen_ctx · SCALE64/128/256×SOURCE(VFX26/NIST2)×18배경 · 장면집계 · recall/tp_conf/synth_FP. 런타임 ~5-10분.
  - **★★현 위치(2026-08-31) = ablation 완주·종결.** CELL 32~38 실행·검출레벨 재검증·miss 육안 완료. **결론=최상단 ▶▶▶▶ §ablation 최종결론**(over≫screen·scale 두체제·스필정밀화·⑤기각·큐레이션 비레버·성김 3중·대형gap 1/3 GT+2/3 under-box·통합=코어반응·0a철회). NIST ign rescue(32-nist)로 이벤트 2·소형유류 스트레스행 추가.


## ▶▶▶ step4 sweep 결과 요약 (2026-08-29 · 셀=`docs/synth_sweep_cells.py` · 상세=PREREGISTER §step4 sweep)
**목표 재확인**: (A) *합성*을 base가 잘 인식하게 개선(프록시=frozen-base recall on 합성). (B)전이/실데이터 학습 아님.
- **소스 결정**: 실사 NIST 스토브탑 corn oil **6종**만 사용(생성형 다양성은 *학습 단계*로 보류 — 지금 섞으면 "크기·화질" 인과에 "가짜불꽃" 교란). 공개 실사 마스킹가능 조리유류불은 ~6-12장이 천장(NIST FCD)·Kitchen Room Fire는 광각(마스킹난)·heptane은 도메인갭+라이선스 → 다 후순위.
- **★결론(클린 뱅크·N=160·검정력충분)**:
  1. **깨끗·충분크기(scale≥~0.25) 합성 불꽃 → base recall 0.994(≈1.0)**. base 진짜 블라인드 ≈0.6%(1/160·유일 미스=불꽃이 튀김 식재료에 겹침=색혼동).
  2. **크기가 유일한 실질 한계**: scale 0.11(작은불) → recall 0.694. base는 작은/초기 불꽃에 약함(운영 리스크).
  3. **화질은 충분크기서 견고**(JPEG q8·blur σ5·저해상 0.15× 모두 recall≥0.96). **작은 불꽃에서만** JPEG 해로움(scale0.11 q8 → 0.463). blur·downscale은 어디서도 ~평평.
  4. **★추출/큐레이션 품질이 결정적 레버**: 뱅크에 금속·조각 쓰레기 섞이면 0.994→0.369 폭락. 합성 요건 = **깨끗한 추출 + 충분한 크기**.
  5. **합성유발 FP 없음(CELL 28·paired n=300)**: 같은 배경 맨것 vs 불꽃붙임 비교. ΔFP@0.25 +0.030±0.048(CI 0 가로지름). 사용자 풀해상: 불꽃외 빨강 = ①타이머LED·③조리기구·④바닥 빨간스티커(=배경 색혼동, 맨배경에도 있음) + ②불꽃 부분·⑤불꽃+음식 큰박스(=불꽃 자체 IoU<0.3 오분류=측정아티팩트). **무고 영역 새 헛불 유발 없음.** FP=배경 색혼동(데이터/도메인)이지 합성 탓 아님. (맨배경 FP@0.25=8.3%는 synth split 튀김장면 많아 eval 0.7%보다 높음 — 다른 배경, 관찰만.)
- **★정정(중간 오류 철회)**: 세션 중 "plateau ~0.856·~14% 놓침·base 블라인드 ~4%·얇은 실오라기"는 **전부 추출 아티팩트(largest-CC 조각냄 + Evt2/4/5 비불꽃[가열코일·금속자] 유입)**였음. base 한계 아님. v0의 "recall 1.0 프록시 천장"이 (깨끗+충분크기서) 옳았던 것. **다음 세션은 클린 뱅크(KEEP 6종) 수치만 신뢰.**
- **방법 교훈**: 진단 중 이벤트범위+필터+마스크 **3개 동시 변경**→교란(recall 폭락 원인 못 가림). 앞으로 **한 변수씩**. (메모리 no-running-ahead-verify-first 반영.)

**▶ 4번 진행 로그·기준 = `docs/PREREGISTER_DFIRE_QC.md`, 셀 = `docs/realneg_qc_cells.py`(CELL 16~22).** 요약:
- **실배경/실음성 재빌드(누수통제)**: hankookro 조리영상 28개→QC(오토틸팅1 제외)→학교단위 분할 **eval 5교 684장 / synth 8교 798장**(교집합∅). Drive `realneg_frames/{eval,synth}/`.
- **base FP 재측정**(구 nofire_kitchen 3.9% 대체): eval서 **conf0.25 0.7%·0.50 0%**(트리거=붉은버튼·주황테이프 색혼동, 불꽃0). recall(B)는 실양성0이라 미측정.
- **(A) 합성 v0**: 불소스=NIST Stovetop 옥수수유 스냅샷(ignition·peak 2장, Drive `firecrop_src/`) → synth 배경에 copy-paste 합성(`synth_composite_v0/`) → **base recall 1.0@all conf(프록시 천장·판별력 없음)**. 배관 확인까지.
- **★★파이프라인 재확인(2026-08-28 후반·중요): 목표=(A) base가 *합성*을 잘 인식(recall/precision)하게 *합성을 개선*(=step3·4·5·6). (B)전이/실데이터학습 아님** — Claude가 외부보고서 (B)프레이밍으로 두 번 오독, 사용자 정정. 외부보고서 (B)조언(pretrain→ft·실홀드아웃 학습)은 **범위 밖**, 리얼리즘 조언(컴포지팅·오일불)만 유효. **스필/페더=저ROI 접음**(밝은 스테인리스서 screen 무효, 크랭크1.5도 육안 무차). **v2 생성기(CELL 23b·클린불꽃6×스케일0.15~0.40·위치 다양·SEED)=완성**(`synth_composite_v2/`).
- **현재=step4 불꽃 소스 서치.** 확보: **NIST 옥수수유 6장**(Colab-safe·PD·`firecrop_src/nist_stovetop_cornoil/`·품질필터 md5/면적0.30/60px로 10→6). 후보: 소방청(공누리 미검증)·**AI-Hub 71751(=불꽃 *크롭 소스*·학습데이터 아님).**
- **★★AI-Hub 취급(2026-08-29 사용자=라이선스주체 승인으로 변경)**: 원 라이선스=국외반출 금지이나, 사용자가 **이 프로젝트 한정 승인**: "불꽃 셋 추출 목적·비배포·비상업·**결과물 출처표기**" 조건서 **Colab/Drive 사용 OK**. → **AI-Hub 원본/크롭을 Drive 업로드·Colab GPU 처리 가능**(전이 학습도 이제 가능). **★단 데이터셋 원본/파생의 제3자 공유·재배포는 여전히 금지.** 출처표기="AI-Hub 71751 화재 발생 예측 영상". **Claude/AI는 AI-Hub 이미지를 context로 안 읽음**(제3자 전송 최소화)—육안검수=사용자. (이전 "로컬 전용" 하드룰 폐기.)
- **▶다음 스텝(사용자 결정): AI-Hub `inout:in`+`device:ct`(조리기구) 불꽃 크롭 → 부족하면 `place:ENB`(음식점) 실내로 넓힘.** 로컬 크롭 스크립트(PIL+json·ultralytics 불필요)부터 · **device 코드 분포 먼저 확인**(ct=조리기구 추정·미검증). 크롭→마스킹(AI-Hub 배경 다양=NIST보다 어려울 수)→로컬 합성→base 로컬 추론(CPU). 상세=§AI-Hub.
- (참고) AI-Hub는 (A) 루프 *필수 아님*(base-on-합성이라 실데이터 불요) — 불꽃 소스 후보로만. 안 쓰고 NIST/소방청만으로 가도 됨.

## ▶ AI-Hub 71751 (불꽃 크롭 소스 · Colab OK[사용자승인] · 2026-08-29)
**★진행중(2026-08-29): VS.z01(100GB) 다운로드 중.** 스크립트 `scripts/local_aihub_flamecrop.py`(audit/crop/diag/**filelist**·PIL+json) 완성·검증(전체 zip구조 + **샘플 구조[01/02 직속·Validation없음] 둘 다 지원**·enumerate_splits). audit 실측(VL.zip 라벨): FL∧in∧ct=**46,440프레임/129클립**·불꽃 categories_id=1·폴백 place=ENB(5,760).
- **★VS 멀티볼륨 판명**: `Validation/01.원천데이터` = **VS.z01(100GB=이미지 데이터)** + **VS.zip(1GB=목록/카탈로그 조각)**. 이전 "VS.zip diag 불완전(목록 243,529 온전하나 데이터 없음)"의 정체 = **VS.zip[마지막 조각]만 받고 VS.z01[데이터 조각]을 안 받음.** → **VS.z01만 추가로 받으면 완성**(둘을 같은 폴더에 두고 7-Zip 결합).
- **★취급 변경**: 사용자 승인으로 **Colab/Drive OK**(불꽃추출·비배포·출처표기). 로컬 CPU 학습 비현실 → **Colab GPU로 전이학습 가능해짐.**
- **★용량 전략**: 108GB 통짜 업로드 비현실 → **로컬서 조리불꽃만 선택추출**(`crop --filelist-only` 로 `cook_flame_files.txt` 생성 → 7-Zip `-r @목록` 로 수 GB만 풀기) → 그 subset만 Drive 업로드. (단 전체 100GB 내부구조는 다운 완료 후 파악해 subset vs 전체 결정.)
- 샘플(1180장)은 실외 1클립(FWW·device none)뿐이라 조리불꽃 0 — 툴·스키마 실검증용으로만 소용(완본이 정본).

- **경로(로컬)**: `C:\Users\jhmoo\Downloads\089.화재 발생 예측 영상_고도화_...\3.개방데이터\1.데이터\{Validation,Training}\{01.원천데이터,02.라벨링데이터}`. Validation: **VS.zip(1.27GB=원본 이미지)·VL.zip(151MB=JSON 243,529·불꽃 76,753)**. Training(TS/TL.zip)·Other/Sublabel zip도 있음. 압축 상태(미해제).
- **파일명**: `sceneID_FL_place_frame.json`(씬당 360프레임=12초×30fps). place 분포(불꽃): GAH(주택)18720·MS14760·FWW14040·RE9000·OLMF6480·ERBF6120·**ENB(음식점)5760**·VTSP1440.
- **JSON 스키마**: `image{width,height,filename}` · `attributes{class:FL, inout(in/out), place, device(ct=조리기구?), fire_reason, fire_level, fps, scene, clipname}` · `annotations[{bbox:[4], area, categories_id}]`. → 실내·조리·bbox 필터 다 됨(inout·device는 JSON, place는 파일명에도).
- **라이선스**: 모델 상업화 OK(출처표시)·데이터셋 재판매만 별도협의·내국인만 신청·원문상 국외반출 금지. **★2026-08-29 사용자(라이선스주체) 승인으로 이 프로젝트 한정 Colab/Drive 사용 OK(불꽃추출·비배포·비상업·출처표기).** 단 데이터셋 원본/파생 제3자 공유·재배포 금지 유지. 출처 [AI-Hub 71751](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71751).
- **로컬 환경(2026-08-28 점검)**: `python`=3.10.11(권장)·`py`=3.13.9 · **ultralytics/torch 없음** · **GPU 없음(Intel Iris Xe·CPU only)**. → 크롭은 설치 불필요(PIL+json) · base 추론은 `pip install ultralytics`(CPU) 필요 · 학습은 CPU라 비현실(하지만 (A)엔 학습 불요).

## ▶▶▶ 다음 세션 재개 레시피
1. 이 문서 전체 읽기 + `docs/synth_validation_cells.py`(CELL 12~15 = 합성 검증·헛불).
2. **완료**: baseline `ptrain_b79`(정직 mAP50 0.660·recall천장 0.894) · 합성 검증(§합성 데이터 검증: recall0.8 게이트 통과) · 헛불(B) 실데이터(실 급식실 3.9% 낮음·트리거=주황물체/색혼동·수증기아님). 재학습 불필요(모델 Drive 생존).
3. **다음 = 사용자 선택 3갈래**:
   - **(가) 실내 실데이터 후보 웹검색 확정** — 다른 세션 보고서가 준 후보(**AI-Hub 한국 "화재감시"[inout/place음식점/fire_level]=1순위** · Home-fire[가정·CC BY-NC] · Zenodo Indoor Fire Smoke · NIST FCD)는 **전부 미검증(내가 웹확인 안함·arxiv 인용 수상)**. 웹검색으로 실재/접근/실내비중 확정 → 통과시 **(B) 실내 held-out** 세워 "합성품질 vs 도메인갭" 교란분리. ⚠️받으면 D-Fire처럼 dedup/누수/라벨 QC 필수·"실내"≠"급식실 조리".
   - **(나) 4번 진입** — 합성 파이프라인 R&D(생성기 비교·aug/noise 유의미성·GAN/diffusion). 지표=(A)base-on-합성 recall/precision(프록시). 겨냥음성=주황물체·색혼동. 4→5(생성)→6(base 재검증) 루프.
   - **(다) 3번** — 실외(D-Fire) pretrain→실내 소량 fine-tune(대안전략과 일치).
4. **★(A)vs(B)**: (A)=base가 합성 인식하나(리얼리즘 프록시·빠름·2번서 실행·필요조건이나 충분조건 아님) · (B)=합성으로 학습시 실 실내 성능 오르나(실전 유용성·실내 실데이터 필요). 6번 정의=(A). "실전 유용"은 (B)라야 닫힘.
5. 전체 Colab 셀 = `docs/dfire_baseline_cells.py`(CELL 1~11) + `docs/synth_validation_cells.py`(CELL 12 합성검증·13 합성음성헛불·14 실음성헛불·15 헛불재검증몽타주). 보조: `colab_rebuild_full.py`·`count_splits.py`·`colab_src_breakdown.py`·`colab_leak_by_source.py`. 런타임 끊기면 `/content` 초기화·**Drive 모델 생존** → 재빌드 CELL1(마운트)→2(api_key·게이트10624)→3(cap1)→4(ptrain). 합성검증 재개: CELL12(합성양성)·CELL14(실음성=`oilfire_realtest_share.zip` in Drive `fire_frames/`, 정본=nofire_kitchen). 각 셀 open_clip·ultralytics 설치가드 있음.
6. **§행동원칙** 준수. **★특히: 다른 세션의 조사 요약·데이터셋 주장은 관찰데이터(미검증) — 웹검색/원문 확인 전 사실 단정 금지. 육안 판독은 정본 아님(사용자 풀해상 판정이 정본).**

## ★ 결과 (baseline 확정)
- **채택 모델 = `ptrain_b79`** (train만 CAP=3). Drive: `/content/drive/MyDrive/dfire_runs/fire_ptrain_b79/weights/best.pt` (참고: `fire_cap1_b79`도 있음).
- **정직한 성능(누수 ≥0.980 제거 = t980)**: **test_mAP50 0.660 · recall천장 0.894 · mAP50-95 0.301** (n_test 982).
  - 누수 포함(full, n_test 1068): mAP50 0.686 · recall천장 0.906.
  - 누수 부풀림은 mAP50 약 **−0.026**(full→t980). 작음.
- **recall천장 0.894** = 양성의 ~11%는 어떤 conf서도 못 잡음 = **base 실제 한계**(누수 착시 아님, 제거해도 유지).
- **ptrain > cap1 견고**: mAP50 +0.028 격차가 full·t990·t980 전 범위서 고정 → 모델 선택 확정.

### 성능 매트릭스 (CELL 7 측정 · 박스단위 · iou=0.5)
precision은 conf 의존이라 두 지점: **P@F1**(best-F1 운영점 정밀도)·**P@R0.8**(recall 0.8일 때 정밀도).

| 모델 | 정제 | mAP50 | recall천장 | F1_R | P@F1 | P@R0.8 | mAP50-95 | n_test |
|---|---|---|---|---|---|---|---|---|
| **ptrain_b79 ★** | **t980(정직)** | **0.660** | **0.894** | 0.648 | 0.667 | 0.367 | 0.301 | 982 |
| ptrain_b79 | t990 | 0.675 | 0.900 | 0.653 | 0.682 | 0.400 | 0.314 | 1037 |
| ptrain_b79 | full | 0.686 | 0.906 | 0.671 | 0.679 | 0.426 | 0.323 | 1068 |
| cap1_b79 | t980 | 0.632 | 0.875 | 0.617 | 0.657 | 0.318 | 0.290 | 987 |
| cap1_b79 | t990 | 0.647 | 0.884 | 0.630 | 0.669 | 0.347 | 0.301 | 1039 |
| cap1_b79 | full | 0.658 | 0.885 | 0.627 | 0.692 | 0.365 | 0.311 | 1068 |

### 이미지 단위 (CELL 11 · ptrain_b79 · 배포 경보 관점 · full test)
| conf | 이미지recall | 배포놓침 | 음성오경보 |
|---|---|---|---|
| 0.05 | 0.980 | 2.0% | 0.081 |
| 0.10 | 0.968 | 3.2% | 0.054 |
| 0.25 | 0.933 | 6.7% | 0.021 |
| 0.40 | 0.891 | 10.9% | 0.012 |
| 0.50 | 0.807 | 19.3% | 0.008 |

※ 곡선 PNG(PR/P/R/F1/results)는 Drive `dfire_runs/fire_{ptrain_b79,cap1_b79}/`에 있음(val 기준·repo 미포함). 이미지단위는 full test라 양성누수 약간 낙관.

## 판정 근거 (측정 이력)
- **CELL 5 공정비교**(batch=79·같은세션): ptrain>cap1 전지표 일관·오버핏 gap 안 커짐(0.273→0.248) → ptrain 채택.
- **CELL 6 누수 감사**(open_clip ViT-B/32, test↔train 임베딩 최근접): dHash(Ham≤6)가 놓친 **cross-split near-dup 실재**. 양성 test ≥0.990: cap1 23 / ptrain 25장. 육안 확인 = 같은 방송 프레임(자막 `06:37`·워터마크 `RT`/`BRC`/`TV BARROSO`·동일 차량 등).
- **CELL 7 엄밀 재평가**(각 모델 자기-train 기준, 재학습 없음·test만 정제): 누수 제거 재평가. full/t990/t980 스윕. → 영향 작고 순위 견고(위 §결과).
- **CELL 8 band 육안**(0.980–0.990 양성 28장): 몽타주 24/24 사실상 전부 같은 장면 → 이 구간도 누수 → 정직 점추정 = **t980(0.660)** 확정.
- **오버핏**: val/box_loss는 ep~30 후 재상승(시그니처) 하나 early-stop이 정점 best.pt(cap1 ep67·ptrain ep64) 저장, val mAP는 plateau → 배포관점 문제 아님. 정직 train-test gap ≈0.27(정상 규모).

## ⑤ recall천장 분해 (CELL 9/10/11 · 전부 측정)
질문: recall천장 0.894/0.906 = "양성 ~11% 미검출"이 모델 한계인가 라벨 오류인가.
- **CELL 9**(ptrain_b79 conf≈0, GT 1122박스): 천장 미검출 105(9.4%, full test 0.906과 일치) = **완전미검 IoU<0.1: 21(1.9%)** + **근접실패 0.1≤IoU<0.5: 84(7.5%)**.
- **CELL 10 IoU 스윕**: recall천장 IoU0.5=0.906 → 0.4=0.943 → 0.3=0.963 → 0.2=0.972 → **0.1=0.981**. 문턱 늦추면 단조 회복 = **miss 대부분이 박스 위치(IoU 엄격성)지 모델 실명 아님(수치)**. near-miss 84장 겹친예측 최대conf 중앙값 0.303·conf≥0.25가 57% = 모델이 그 불에 자신있는 검출 올림(수치). 단 24%는 conf<0.05(약함).
- **CELL 11 이미지단위(배포 경보)**: conf0.05 recall0.980(놓침2.0%·오경보8.1%) / conf0.25 0.933(6.7%·2.1%) / conf0.50 0.807(19.3%·0.8%).
- **결론(측정)**: 천장의 정체 = **박스 타이트니스**. 진짜 모델 실명 **1.9%**뿐(성격=CELL9 육안: 대부분 극한불[작은·먼·가림]·소수 연기라벨). 배포 실효 놓침 2~7%(운영점 따라). **라벨 오류·모델 한계 어느 쪽도 주범 아님 → recall천장은 탐지실패가 아니라 위치 지표.**
- ⚠️ full test라 양성 누수(~53장 쉬움) 이미지recall 약간 낙관 · 1.9% 성격만 육안.

## 합성 데이터 검증 (파이프라인 2번 · CELL 12)
D-Fire 학습 base(ptrain_b79)를 **한 번도 안 본 라벨된 합성셋**에 평가 → "합성이 실제-학습 검출기에 불처럼 보이나(리얼리즘)".
- 합성셋 = Roboflow `kyungho-moon/kitchen-fire-noise-poc` **v1 · 351장**(급식실 CCTV 유류불 · nano-banana/codex/FLUX 생성) · fire box 라벨 · base엔 완전 held-out(누수 0). 병합 351장·fire박스 354.

| 지표 | 합성셋(351) | D-Fire base(정직 t980) | 차이 |
|---|---|---|---|
| mAP50 | 0.652 | 0.660 | ≈동일 |
| recall천장 | 0.833 | 0.894 | −0.061 |
| P@F1 | 0.786 | 0.667 | +0.119 |
| R@F1 | 0.590 | 0.648 | −0.058 |
| P@R0.8 | 0.187 | 0.367 | −0.180 |
| mAP50-95 | 0.259 | 0.301 | −0.042 |
| Box P/R(운영점) | 0.758 / 0.593 | 0.675 / 0.632 | — |
| 이미지recall@conf0.25 | 0.809 | 0.933 | −0.124 |
| 이미지recall@conf0.05 | 0.880 | 0.980 | −0.100 |

- **읽기**: precision은 합성이 더 높고(발화 시 정확) recall은 낮음(더 놓침) → mAP50 거의 동일(0.652≈0.660). **합성 off-distribution 아님**(그랬으면 0.1~0.2) = **리얼리즘 상당.**
- **게이트(파이프라인 2번, recall 0.8)**: 감지=이미지 recall로 판정 → **0.809(@0.25)·recall천장 0.833 ≥ 0.8 통과** → 파이프라인상 **4번(합성 파이프라인 구축)**으로. (박스 R@F1 0.59는 위치 기준 다른 질문이라 게이트 아님.)
- **⚠️ 교란(미분리)**: recall gap이 **합성 품질 부족 vs 실내 도메인갭**(base 야외학습) 섞임 — 이 테스트만으론 분리 불가. 실제 실내 화재셋 있으면 4번서 분리.
- **precision-B(헛불률) 측정(CELL 13 · 합성 음성 28장 FLUX)**: 헛불률 conf0.05 **0.071**(2/28) · conf0.25 **0.036**(1/28) · conf0.50 **0.000**(0/28). D-Fire 야외음성(0.081/0.021/0.008)과 **N=28서 통계적 구분 불가**(1/28 신뢰구간 ~0.6–18% 넓음) → **대량 헛불 관측 안 됨(예비치).** 박스 precision(A)=0.786은 별개 측정됨.
  - **진단(몽타주)**: 유일한 conf≥0.25 헛불 1건(flux5_111, top conf 0.47) = **냄비 위 수증기(김)를 불로 오인 + 천장 밝은 반사/조명.** → 프로젝트 역사적 병목(조리 수증기·반사 헛불)과 **일치하는 실패모드가 소표본서도 실제 관측.** 4번 겨냥 음성 = **수증기·반사.**
  - ⚠️ N=28 예비 · 합성 음성 ≠ 실제(진짜 수증기/스테인리스 반사는 다를 수, 실제 배포 헛불 더 많을 수) · 1사례라 빈도 정량 불가.
- **★precision-B 실데이터 측정(CELL 14 · 실제 무화재)**: `oilfire_realtest_share.zip`(Drive `fire_frames/`) 안 **nofire_kitchen 181장(실 급식실 조리 무화재, ck03~ck21)** + nofire_presrc 301장(대조)에 base 평가. 코드=`docs/synth_validation_cells.py`.
  - **실제 급식실 헛불률**: conf0.05 0.221(40/181) · **conf0.25 0.039(7/181)** · **conf0.50 0.006(1/181)**. → **운영 conf서 낮음** = base가 실제 급식실 무화재서 대량 헛불 안 냄(N=181로 합성28 예비 뒷받침). 0.05의 22%는 저conf 잡박스(운영점 아님).
  - **★트리거(사용자 풀해상 직접판정 = 정본)**: 몽타주 7건(conf≥0.25) = **주황색 물체 3**(음식 ck21 · 바닥 주황테이프/스티커 ck07·ck20) + **시간 표시등 1**(ck04_0039) + **조리기구 일부분(미상) 3**(ck04_0045·ck03_0033·ck03_0015). **수증기 0/7 — 김 자욱한 장면서도 안 찍힘 → 수증기는 트리거 아님(확정).** 두드러진 패턴 = **주황/붉은 색 혼동**(주황 물체 3 + 붉은 표시등 1). → 4번 겨냥 음성 = **주황색 비화재 물체(조리음식·바닥 테이프/스티커)·조리기구 요소**(수증기 아님).
  - **★내 오류 정정(철회)**: 앞서 "6/7 붉은 LED 표시등"은 **저해상 육안 오독**. 사용자가 풀해상 파일탐색기로 직접 판정 → 표시등은 1건뿐이고 **주황 물체가 3건**. 육안을 정본처럼 단정한 것 철회. 확정 트리거는 위 사용자 판정본.
  - **대조(nofire_presrc) 헛불 높음**(conf0.25 0.203·0.50 0.146). **정체 확정(v2 정본·md5 d0fa13f46c: zip=추출본)**: `nofire_presrc = 같은 오일화재 데모영상(튀김유 天ぷら油火災·chip pan·grease·IH/NIST cooktop)의 '발화 직전' 프레임`(README+manifest accepted). **급식실 아님·불 임박 → 20%는 배포 지표 아님.** nofire_kitchen(정상 급식실 조리)이 배포 음성이고 헛불 낮음(3.9%). (앞서 본 v1 README/manifest는 이 zip에 없어 폐기.)

## 프로젝트 파이프라인 (사용자 정의)
1. Roboflow 공개 화재(야외+실내) 수집 → YOLO11 화재 base. ✅ 완료(=이 문서 baseline).
2. base로 합성(gpt/gemini/nano-banana) 검증(recall/precision). 감지 잘되면(recall≥0.8)→4 / 아니면→3. **✅ 통과(이미지recall 0.809)→4.**
3. (2 실패시) 공개+합성 섞어 학습 · train/val/test 비율 세팅.
4. ★메인 = 합성 파이프라인 구축: 어떻게 합성해야 인식 잘 되나 + data aug/noise 유의미성 + GAN/diffusion. (되고안되고 아닌 파이프라인 구축)

## 데이터 (고정)
- Roboflow: ws `kyungho-moon` · project `d-fire-aqheb-6iyqy` · **v1** · `download("yolov11")`. 원본 21,522장.
- dedup dHash HAM=6 union-find → **10,624 클러스터**(최대 3,042). (DetectiumFire arXiv:2511.02495 "중복시 절반"·pHash0.15/CNN0.55와 정합 — 검증함.)
- fire-only(smoke 드롭)·층화(src × has-fire) 그룹 80/10/10 · 구조적 누수 0 · SEED=0 결정론.
- **★출처 접두사 = 3종(층 6개) — 측정 확정(2026-08-28 · `docs/count_splits.py`).** 이전 기록 `AoF/WEB` 2종(층 4개)은 **오류**. `src_of()`는 파일명 앞 알파벳을 뽑을 뿐 이름을 열거하지 않고 `strata` 키가 어디에도 print 되지 않아, 지금까지 눈에 띌 경로 자체가 없었음. **숫자·결론에는 영향 없음**(층화 코드는 키 개수와 무관하게 동작).

  | 접두사 | 전체 | 양성 | 양성비 |
  |---|---|---|---|
  | WEB | 9,431 (88.8%) | 3,758 | 39.8% |
  | PublicDataset | 669 (6.3%) | 138 | 20.6% |
  | AoF | 524 (4.9%) | 112 | 21.4% |

- **cap1**(전 split CAP1): train 양성 3,205 · test 404/1068. **ptrain**(train만 CAP3): train 양성 3,433 · test 404/1068. **재빌드 게이트: 10624 / 3205·404-1068 / 3433·404-1068.**
- **★split 실측(2026-08-28)** — 빌드 셀이 총계·train양성·test만 찍어 그동안 미기록이던 값.

  | split | cap1 장수 | cap1 양성 | ptrain 장수 | ptrain 양성 |
  |---|---|---|---|---|
  | train | 8,496 | 3,205 | **9,417** | 3,433 |
  | valid | 1,060 | 399 | 1,060 | 399 |
  | test | 1,068 | 404 | 1,068 | 404 |
  | 합계 | 10,624 | 4,008 | **11,545** | 4,236 |

  - 음성: cap1 6,616 · ptrain 7,309 · **test 음성 664**(이미지단위 오경보율의 분모).
  - **CAP3는 상한일 뿐 — 실제 증가는 train 8,496→9,417 = +921장(+10.8%)**, 덩어리당 평균 1.11장(양성만 1.07 · 음성 1.13). 대부분 덩어리가 1장짜리. **즉 데이터 +10.8%로 mAP50 +0.028을 얻고 오버핏 gap은 오히려 감소(0.273→0.248).**
- **★결정론 검증됨(2026-08-28)**: 다른 세션·다른 런타임에서 CELL 1→2→3→4 재빌드 시 게이트 4값 전원 재현. 이어 CELL 7 경로도 `pos≥0.990=25` · `n_t980=982` · `mAP50=0.660` 재현 → **"재빌드시 게이트로 확인"이 미검증 주장에서 실측으로 승격.**

## 출처별 분해 (2026-08-28 · `docs/colab_src_breakdown.py` · `docs/colab_leak_by_source.py`)
질문: test 가 WEB 에 89% 쏠려 있으니 **0.660 이 쉬운 다수 덕에 부풀려진 값 아닌가.** 재학습 없이 test 만 갈라 재평가(ptrain_b79).

| 출처 | n | 양성 | 누수양성(≥.980) | mAP50 full | **mAP50 정직(t980)** | 하락 | mAP50-95 full | P@F1 full | 이미지recall@0.25 (Wilson 95%) |
|---|---|---|---|---|---|---|---|---|---|
| WEB | 945 | 377 | **52** | 0.676 | **0.649** | **−0.027** | 0.315 | 0.671 | 0.960 [0.935, 0.976] |
| PublicDataset | 69 | 15 | **0** | 0.773 | 0.770 | −0.002 | 0.431 | 0.846 | 0.867 [0.621, 0.963] |
| AoF | 54 | 12 | 1 | 0.903 | **0.900** | −0.003 | 0.503 | 0.839 | 0.917 [0.646, 0.985] |
| **전체** | 1068 | 404 | 53 | 0.686 | **0.660** | −0.026 | 0.323 | 0.679 | 0.955 [0.931, 0.972] |

- **판정: 의심 기각 · 방향은 반대.** 다수 출처 WEB 이 셋 중 **가장 어려움**(정직 0.649 vs 0.770 · 0.900). 쏠림이 낙관 쪽으로 작용하지 않았으므로 **0.660 인용 유효.** 단 엄밀히는 0.660 이 WEB 단독 0.649 보다 **+0.011** — 쉬운 소수가 끌어올린 몫.
- **누수는 WEB 에만 실질 영향.** 하락 폭 WEB −0.027 vs 나머지 −0.002·−0.003. 노이즈 아니라 **기전이 설명됨**: 누수 양성이 PublicDataset **0장** · AoF **1장** · WEB **52장**. CELL 8 의 방송 프레임(자막 `06:37` · 워터마크 `RT`/`BRC`/`TV BARROSO`)은 전부 WEB 쪽 이야기였음. → **정직하게 재면 출처 간 격차가 오히려 확대**(0.227 → 0.251).
- **검출은 출처 무관, 위치만 갈림.** 이미지단위 recall 은 세 출처 **Wilson 구간이 전부 겹쳐 구분 불가**인데 박스단위 mAP50 만 크게 갈림 → §⑤ 결론(천장 = 위치 지표)이 출처 단위에서도 재현.
- **⚠️ 소수 출처는 "쟀다"고 하기 어려움**: 양성 12·15 장. 이미지단위 구간 폭 0.34, 박스 mAP 에는 구간을 못 붙임. **0.900·0.770 은 방향으로만 읽을 것.** 확정하려면 재분할+재학습(k-fold) 필요 — 대상 도메인이 실내라 우선순위 낮음.
- 접두사별 누수율(전체 기준): WEB 79/945=8.4% · PublicDataset 4/69=5.8% · AoF 3/54=5.6%. 절대 수의 92%(79/86)가 WEB 이나 이는 대체로 WEB 비중(89%) 때문이고, **양성 누수는 WEB 13.8% vs AoF 8.3% vs PublicDataset 0%** 로 편중이 뚜렷.

## 고정 파라미터
- fire=class 0 · yolo11s · epochs100 · patience25 · imgsz640 · **batch=79** · cache=disk · seed0 · deterministic.
- 누수 임베딩: open_clip `ViT-B-32` pretrained `openai` · 코사인 최근접 · 판정 임계 육안(≥0.980 = dup).

## 미제 (未決)
| 항목 | 상태 | 성격 |
|---|---|---|
| ⑤ recall천장 분해(라벨 감사) | **완료(CELL 9/10/11)** | 위 §⑤: 천장 miss는 박스 위치(IoU 엄격성)·모델 실명 1.9%뿐. cleanlab 자동스캔은 육안상 라벨오류 얇아 우선순위 낮음(미실행) |
| multi-seed 재현 | 미제 | 전부 1-seed. ptrain>cap1 방향은 전범위 일관하나 격차 크기 재현 미확인. (※ 빌드 결정론은 2026-08-28 재빌드로 검증됨 — 학습 seed 재현과는 별개) |
| 소수 출처(AoF·PublicDataset) 성능 | **N 부족** | 갈라 재긴 함(§출처별 분해). test 양성 12·15장이라 이미지단위 구간이 WEB과 전부 겹치고 박스 mAP엔 구간 못 붙임 → 방향 이상 못 읽음. 확정하려면 재분할+재학습(k-fold). 대상이 실내라 우선순위 낮음 |
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
