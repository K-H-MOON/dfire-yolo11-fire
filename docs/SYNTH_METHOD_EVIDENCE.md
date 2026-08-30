<!--
  provenance: 사용자 제공 근거 원장. 원파일 = compass_artifact_wf-fe21fc3d-6018-52cc-a2bc-1f621aa3ebb4_text_markdown.md (2026-08, Downloads).
  용도: docs/synth_sweep_cells.py CELL 31(불꽃 합성 ablation) + 실험 설계도(HANDOFF §설계근거)의 "왜 이 방법을 썼나" 근거표.
  취급: 논문 게재용 아님. 실험 보고서에 설계 근거·출처 표시용. 정직성 최우선(찾지못함/미검증 명시).
  ★이 문서가 강제하는 정정(HANDOFF §설계근거에도 반영):
    - "BoWFire FP 최대 80% 감소" = 정량 근거 미확인 → 철회/정성화(§9).
    - arXiv:2606.19817(FID/KID 한계) = 미래/오타 식별자 → 철회, Borji(2103.09396)로 대체.
    - DACBFIAM(IEEE 9979846) = 미검증.
    - L3(컨텍스트 배치) = 근거 양방향(Ghiasi 무작위도 충분 ↔ Dvornik 컨텍스트 우월) → "가설"로 측정, 자명 가정 금지.
    - L2(screen) = 발광체 직접 실증 없음(간접) → 벤치마크: TP confidence CI가 0 배제 못하면 over 유지.
    - "프리즈 검출기를 사실성 심판으로 쓰는 계단식 ablation" 통합 = 직접 선례 없음 → "표준요소의 신규 조합"으로 정직 기술(약점 아님).
-->

# 주방 불꽃 합성 검출 실험 설계도 — 구성요소별 근거 문헌 원문 발췌

> 본 문서는 사용자의 실험 설계도 각 구성요소에 대해 **근거 문헌의 영어 원문(verbatim)**을 발췌하고 한국어 번역과 출처를 병기한 것이다. 논문 게재용이 아니라 "실험 보고서에 설계 근거를 표시"하기 위한 용도다. **정직성을 최우선**으로 하여, 근거를 찾지 못했거나 검증 실패한 항목은 명시적으로 "찾지 못함/미검증"으로 표기한다.

## TL;DR
- **핵심 근거인 FGL-GAN(Qin et al., Sensors 2022)은 설계의 "물리 모델"(불꽃=광원, halo 전이, 반사광 가변성)과 "검출기 confidence를 품질 지표로 쓰기"를 원문으로 강하게 뒷받침한다. 다만 FGL-GAN은 합성 품질을 FID·ResNet accuracy·YOLOv5 confidence로만 평가했을 뿐, 다운스트림 검출 성능(mAP/recall) 개선은 측정하지 않았다.**
- **"프리즈된 검출기를 계단식 ablation의 사실성 심판으로 쓰는" 정확히 동일한 통합 선행 연구는 찾지 못했다.** 그러나 이를 구성하는 세 축 — ① 검출기 confidence를 합성 품질 지표로(FGL-GAN), ② downstream task 성능으로 생성모델 평가(CAS/Inception/FID), ③ 요소별 기여 분리(ablation) — 은 각각 확립된 근거가 있다. 따라서 이 설계는 "표준 방법론 구성요소의 신규 조합"으로 규정하는 것이 정직하다.
- **개별 방법 계단·인자에는 강한 직접 실증이 존재한다:** blending 혼합(Dwibedi, +8 AP), 컨텍스트 배치(Dvornik, 무작위 대비 평균 +5%; BIB-Copy-Paste +1.19% vs +0.18%), 무작위 배치도 충분(Ghiasi), COCO 스케일 정의(32²/96²), shortcut learning(Geirhos), pseudoreplication(Hurlbert), 데이터 누출(Kapoor & Narayanan). 반면 SCALE 절대값(64/128/256px), 클립 수(25~40), 페어링 시드, SOURCE 2수준 분류 등은 **논문 근거 없는 연구자 판단**이다.

---

## Key Findings

1. **FGL-GAN이 설계의 물리·방법론적 뼈대를 원문으로 제공한다.** "불꽃은 광원이라 색·질감이 배경 영향을 받지 않으므로 전통적 harmonization이 부적합"하고, 문제가 "image blending(halo)"과 "shadow generation(반사광)" 두 단계로 단순화되며, cut-paste는 halo도 반사광도 못 만든다는 서술이 모두 존재한다. **단, downstream 검출 mAP/recall은 측정하지 않았다.**
2. **"검출기를 합성 품질 평가자로" 방법론 선례는 존재하나(FGL-GAN의 YOLOv5 confidence, Ravuri & Vinyals의 CAS), "프리즈 검출기 + 계단식 ablation + 합성법 변주"라는 통합 설계의 직접 선례는 발견되지 않았다.**
3. **blending이 검출 성능에 영향을 준다는 직접 실증(Dwibedi 2017)이 있으며, 여러 blending을 섞으면 단일 모드보다 낫다(no blending 65.9 → all blend + same image 73.7 mAP, +8 AP).** 이것은 L1(over)→L2(blending) 계단의 최강 직접 근거이며, 사용자의 페어링 프로토콜과 동일하다.
4. **컨텍스트 배치 우월성은 실증되지만(Dvornik: 무작위 대비 평균 +5%; BIB-Copy-Paste), 무작위 배치도 강력하다는 반대 실증(Ghiasi)도 있어 L3의 근거는 "양방향"이며 도메인 특수적이다.** 사용자가 flagged한 "BIB-Copy-Paste +0.18% vs +1.1%"는 **실재하며, 정확히는 +0.18%(무작위) vs +1.19%(컨텍스트, BlitzNet, PASCAL VOC 2012)**.
5. **SCALE 인자는 COCO의 32²/96² 정의(Lin et al.)로 개념적 근거가 있으나, 64/128/256px라는 특정 절대값 선택은 임의다.**

---

## Details

아래 각 항목은 **영어 원문(verbatim) + 한국어 번역 + 출처/섹션**으로 제시한다.

### ■ 1. FGL-GAN (최우선 근거)
**출처:** Kui Qin, Xinguo Hou, Zhengjun Yan, Feng Zhou, Leping Bu, "FGL-GAN: Global-Local Mask Generative Adversarial Network for Flame Image Composition," *Sensors* 2022, 22(17):6332. DOI: 10.3390/s22176332. (MDPI 원문 및 PMC9460294에서 확인)

**(a) 불꽃은 광원이라 전통적 harmonization이 부적합**
- 원문 (Section 2.2 Image Compositing): *"The flame itself is a light source and does not need to be harmonized by the background. Therefore, the flame image compositing can be simplified into two steps: image blending and shadow generation."*
- 원문 (Section 1 Introduction, 전통적 방법 비판): *"Traditional methods change the flame's color and texture information, based on the background information. However, the flame is used as the light source, whose color and texture are not affected by the background during the compositing process. Therefore, traditional methods are not suitable for solving the problem of flame image compositing."*
- 번역: "불꽃 자체가 광원이므로 배경에 의해 조화(harmonize)될 필요가 없다. 따라서 불꽃 이미지 합성은 image blending과 shadow generation 두 단계로 단순화될 수 있다." / "전통적 방법은 배경 정보에 근거해 불꽃의 색·질감을 바꾼다. 그러나 불꽃은 광원으로 사용되며 그 색과 질감은 합성 과정에서 배경의 영향을 받지 않는다. 따라서 전통적 방법은 불꽃 이미지 합성 문제에 부적합하다."

**(b) 합성 문제가 image blending + shadow generation 2단계로 단순화**
- 원문 (Section 2.2): *"Of the four steps of flame image compositing, since manual placement of flame position is more reasonable compared with machine placement, manual placement is chosen as the object placement in this paper. ... Image blending is mainly the blurring transition of the boundary between the flame and the background, and the generation of halo around the flame. Shadow generation is mainly the generation of flame reflection."*
- 번역: "불꽃 이미지 합성의 네 단계(object placement, image blending, image harmonization, shadow generation) 중, 기계 배치보다 수동 배치가 더 합리적이므로 본 논문은 object placement로 수동 배치를 택한다. … image blending은 주로 불꽃과 배경 경계의 흐림 전이 및 불꽃 주위 halo 생성이다. shadow generation은 주로 불꽃 반사광(reflection) 생성이다."
- **설계 매핑:** 이 문장이 사용자의 L1/L2(=image blending, halo/경계), L4(=shadow generation, 반사광), L3(=수동 object placement)로 정확히 대응한다.

**(c) 실제 불꽃 주위 halo가 배경으로 부드럽게 전이되며 cut-paste는 이를 못 만든다**
- 원문 (Section 1 Introduction): *"First, there is a halo around the real flame, therefore the flame can transition smoothly into the background and there is no obvious boundary. The cut-paste algorithm cannot paste the halo around the flame in the new background, which renders the composite flame different in appearance from the real flame."*
- 번역: "첫째, 실제 불꽃 주위에는 halo가 있어 불꽃이 배경으로 부드럽게 전이되며 뚜렷한 경계가 없다. cut-paste 알고리즘은 새 배경에 불꽃 주위 halo를 붙일 수 없어 합성된 불꽃이 실제 불꽃과 외형이 달라진다."

**(d) 불꽃이 광원이라 바닥에 반사광을 드리우며, 반사광 면적이 배경 광량·불꽃 색·형태에 따라 변한다 (L4 스필 물리 모델의 근거)**
- 원문 (Section 1, cut-paste 한계): *"Second, the real flame itself is a light source, which will cast reflection on the ground. However, the flame composited by the cut-paste algorithm cannot produce reflection, which makes the composite image unrealistic."*
- 원문 (Section 1, FIS-GAN 비판 — 반사광 가변성): *"the area of flame reflection changes with the color, shape of the flame, and background light intensity. The fixed area flame reflection limits the use of FIS-GAN."*
- 번역: "둘째, 실제 불꽃 자체가 광원이므로 바닥에 반사광을 드리운다. 그러나 cut-paste로 합성된 불꽃은 반사광을 만들 수 없어 합성 이미지가 비현실적이 된다." / "불꽃 반사광의 면적은 불꽃의 색·형태와 배경 광량(light intensity)에 따라 변한다. 고정 면적 반사광은 FIS-GAN의 활용을 제한한다."
- **설계 매핑:** 사용자의 "L4 스필(반사광/글로우)이 배경 광량과 거리에 따라 변한다"는 물리 모델이 이 문장으로 직접 뒷받침된다.

**(e) 평가 지표 수치 (Table 1)**
- 원문 (Section 5, 결과): *"the images composited by FGL-GAN reach 29.75 on the FID, reach 0.9386 on resnet accuracy, reach 0.7534 on yolov5 confidence, reach 0.583 on global user evaluation, and reach 0.636 on local user evaluation. The results of FGL-GAN are all better than the other compared methods."*
- 원문: *"FGL-GAN proposed in this paper has the lowest FID of 29.75, which indicates that the halo and reflection rendered by FGL-GAN are closest to the real images ... From the perspective of computer vision algorithm evaluation, resnet accuracy, and the yolov5 confidence of FGL-GAN, achieve 0.9386 and 0.7534, respectively, which are both better than other networks."*
- 비교 대상: 논문의 관련연구·실험에서 pix2pix, CycleGAN, QS-Attn, FIS-GAN 등과 비교. **개별 경쟁모델의 정확한 수치(각 네트워크의 FID/accuracy/confidence 값)는 Table 1이 이미지로 되어 있어 텍스트로 확보하지 못했다 — MDPI 원문 Table 1을 직접 확인 필요.** FGL-GAN 자체 수치(FID 29.75 / ResNet acc 0.9386 / YOLOv5 conf 0.7534 / global user 0.583 / local user 0.636)는 확정.

**(f) 【중요】 다운스트림 검출 성능(mAP/recall) 개선 측정 여부**
- **결론: FGL-GAN은 다운스트림 검출 성능(mAP/recall) 개선을 측정하지 않았다.** YOLOv5 confidence(0.7534)는 "합성 이미지 품질/사실성"의 대리 지표로 쓰였을 뿐, 합성 이미지로 검출기를 학습시켰을 때 검출 성능이 오르는지는 실험하지 않았다. 논문 목적은 초록에 명시: *"a large number of new flame images can be composited by FGL-GAN, which can provide extensive test data for fire detection equipment, based on deep learning algorithms."* — 즉 검출 장비의 **테스트 데이터** 제공이 목적이며 학습·성능개선 실험은 범위 밖이다.
- **설계적 함의:** 사용자의 설계(프리즈 검출기로 recall/FP/confidence를 재는 것)는 FGL-GAN보다 한 걸음 더 나아간 것으로, FGL-GAN이 "confidence를 품질 지표로 쓴" 선례는 되지만 "검출 성능 변화를 종속변수로 측정한" 선례는 아니다.

**(g) LGM/GCM 역할 원문**
- 원문 (Section 3.1.1): *"It contains two generation modules: local generation module (LGM) and global coordination module (GCM). LGM aims to make the network focus on local information (around the flame), thereby generating clearer, more realistic, and more detailed flame halo, and reflection. However, if the generator only contains LGM, the color of the generated local image and the background will be inconsistent ... Therefore, the generator blends the local image with the background color and blurs the boundaries through GCM, so that the final composite image is more natural and realistic."*
- 번역: "LGM은 네트워크가 국소 정보(불꽃 주변)에 집중하게 하여 더 선명·사실적·세밀한 halo와 반사광을 생성한다. LGM만 있으면 국소 이미지와 배경의 색이 불일치하고 경계가 지나치게 날카로워지므로, GCM이 국소 이미지를 배경 색과 blend하고 경계를 흐려 최종 이미지를 자연스럽게 만든다."

**(h) Ablation study 구성**
- 원문 (Abstract): *"Ablation study shows the effectiveness of the hierarchical Global-Local generator structure, fire mask, data augmentation, and MONCE loss of FGL-GAN."*
- 제거·측정한 요소: ① 계층적 Global-Local 생성기 구조, ② fire mask, ③ data augmentation(무화염 장면 투입), ④ MONCE loss. 번역: "ablation study는 계층적 Global-Local 생성기 구조, fire mask, data augmentation, MONCE loss의 유효성을 보인다."

---

### ■ 2. "검출기/분류기를 합성 품질 평가자로" 방법론 선례

**(i) Classification Accuracy Score (CAS)** — Ravuri & Vinyals, "Classification Accuracy Score for Conditional Generative Models," NeurIPS 2019, arXiv:1905.10887.
- 원문 (Abstract): *"To test this latter hypothesis, we use class-conditional generative models ... to infer the class labels of real data. We perform this inference by training an image classifier using only synthetic data and using the classifier to predict labels on real data. The performance on this task, which we call Classification Accuracy Score (CAS), reveals some surprising results not identified by traditional metrics."*
- 정량 결과(2차 문헌 Borji, "Pros and Cons of GAN Evaluation Measures," arXiv:2103.09396 요약): *"when using a state-of-the-art GAN (BigGAN-deep ...), Top-1 and Top-5 accuracies decrease by 27.9% and 41.6%, respectively, compared to the original data"* — 즉 최고 성능 GAN조차 downstream 분류에서 큰 정확도 저하를 보임.
- 번역: "우리는 합성 데이터만으로 분류기를 학습시키고 이를 실제 데이터 레이블 예측에 사용한다. 이 과제 성능을 CAS라 부르며, 전통 지표가 못 잡던 결과를 드러낸다."
- **설계 매핑:** "downstream task 성능으로 생성모델을 평가"하는 방법론의 정식 근거. 단 CAS는 **분류기를 합성으로 학습→실제로 평가**하는 방향인 반면, 사용자 설계는 **프리즈 검출기를 실제로 학습→합성으로 평가**하는 역방향이라는 차이가 있다.

**(j) Inception Score / FID의 원리** — 둘 다 pretrained classifier(Inception) 기반이며, FID는 FGL-GAN에서 29.75로 사용되었다. 두 지표 모두 "pretrained 분류기의 특징공간에서 합성 품질을 잰다"는 점에서 사용자 설계와 철학을 공유한다. (원리는 널리 확립되어 별도 verbatim은 생략.)

**(k) 【핵심 판정】** "프리즈된 검출기를 사실성 심판으로 사용하는 계단식 ablation" 전체에 대한 직접 선행 연구: **찾지 못했다.** 이는 (i) 검출기 confidence를 품질 지표로 쓰기(FGL-GAN), (ii) downstream 성능으로 생성모델 평가(CAS), (iii) 요소별 ablation(아래 §3)의 세 확립된 관행을 **조합**한 설계로 보아야 하며, 이 특정 조합을 화재·주방 도메인에서 수행한 선례는 확인되지 않는다. 정직하게 "방법론적으로는 표준 구성요소의 새로운 조합"이라고 기술하는 것이 정확하며, 이는 약점이 아니라 신규성 주장의 근거가 된다.

---

### ■ 3. Ablation / factorial design 방법론

**(l) Ablation study 정의** — Meyes, Lu, de Puiseau, Meisen, "Ablation Studies in Artificial Neural Networks," arXiv:1901.08644 (2019).
- 2차 인용 정의: *"Ablation analysis is used to determine whether a model component or input contributes unique predictive information by examining the change in performance after its removal (Meyes et al., 2019)."*
- 번역: "ablation 분석은 어떤 구성요소·입력을 제거한 뒤 성능 변화를 관찰하여, 그 요소가 고유한 예측 정보를 기여하는지 판정하는 데 쓰인다."
- **주의:** Meyes et al.의 원 논문은 신경망 **내부 뉴런/구조**를 제거하는 ablation이 주제다. 사용자의 "합성 파이프라인 요소(L1→L4)를 계단식으로 바꾸는" ablation은 이 개념을 데이터 파이프라인으로 확장한 것으로, 개념적 근거는 되나 원 논문의 직접 대상은 아니다 → **[설계 원칙]**에 가깝다.

**(m) 계단식(누적) ablation vs OFAT의 한계** — one-factor-at-a-time(OFAT)은 요인 간 **상호작용(교호작용) 효과**를 못 잡는다는 것이 실험설계(DOE)의 표준 지적이다. 사용자의 L1→L2→L3→L4 누적 계단은 OFAT/누적형이므로 blending×위치, scale×source 등의 교호작용을 완전히 분리하지 못한다. 사용자가 SOURCE×SCALE를 교차인자로 둔 것은 부분적으로 factorial을 도입한 것으로 바람직하다. **web_search 예산 소진으로 DOE 대표 논문의 verbatim 인용은 확보하지 못했으므로 이 항목은 [설계 원칙]으로 분류한다.**

**(n) CV에서 증강 요소별 기여 분리 측정 대표 사례** — YOLOv4 "Bag of Freebies" 관행(arXiv:1902.04103 등)에서 증강·트릭을 하나씩 켜고 mAP 변화를 측정하는 것이 표준이다. (예산 소진으로 특정 수치 verbatim은 미확보; 관행 자체는 확립.)

---

### ■ 4. Copy-paste에서 배치 위치/컨텍스트 (L3)

**(o) Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," CVPR 2021, arXiv:2012.07177 — 무작위 배치도 강력**
- 원문 (Abstract): *"Prior studies on Copy-Paste relied on modeling the surrounding visual context for pasting the objects. However, we find that the simple mechanism of pasting objects randomly is good enough and can provide solid gains on top of strong baselines. ... On COCO instance segmentation, we achieve 49.1 mask AP and 57.3 box AP, an improvement of +0.6 mask AP and +1.5 box AP over the previous state-of-the-art."*
- 번역: "기존 Copy-Paste 연구는 붙일 때 주변 시각 컨텍스트 모델링에 의존했다. 그러나 우리는 객체를 무작위로 붙이는 단순 방식만으로도 충분하며 강력한 baseline 위에 확실한 이득을 준다는 것을 발견했다. COCO에서 49.1 mask AP, 57.3 box AP(각 +0.6, +1.5 향상)를 달성."
- **설계 매핑:** 이 결과는 사용자의 L1(무작위 위치)이 이미 유효할 수 있음을 시사하며, **L3(컨텍스트 배치)의 우월성이 자명하지 않다**는 반대 증거다. 정직하게 "컨텍스트가 도움된다는 증거와 무작위도 충분하다는 증거가 공존"이라고 써야 한다.

**(p) Dvornik, Mairal, Schmid, "Modeling Visual Context is Key to Augmenting Object Detection Datasets," ECCV 2018, arXiv:1807.07428 — 컨텍스트 배치 우월**
- 원문 (Abstract): *"For this approach to be successful, we show that modeling appropriately the visual context surrounding objects is crucial to place them in the right environment. Otherwise, we show that the previous strategy actually hurts. With our context model, we achieve significant mean average precision improvements when few labeled examples are available on the VOC'12 benchmark."*
- 원문 (본문, 정확 수치): *"the visual context model always improve upon the random placement one, on average by 5%, and upon the baseline that uses only classical data augmentation, on average by 4%."* 또한 *"augmenting naively datasets with randomly placed objects slightly hurts the performance."*
- 번역: "컨텍스트 모델은 무작위 배치 대비 항상 향상되며 평균 5%, 고전적 데이터 증강만 쓴 baseline 대비 평균 4% 향상된다." / "무작위로 배치한 객체로 데이터셋을 순진하게 증강하면 오히려 성능이 약간 저하된다."
- **설계 매핑:** L3(컨텍스트 배치)의 정량적 직접 근거. Ghiasi(무작위 충분)와 상반되는 이유는 데이터 규모(Dvornik은 few-label 저자원 상황)에 따라 결론이 갈리기 때문으로 해석된다.

**(q) BIB-Copy-Paste 검증 【사용자 flagged 항목】** — Zhang, Xing, Wang et al., "Background Instance-Based Copy-Paste Data Augmentation for Object Detection," *Electronics* 2023, 12(18):3781, DOI: 10.3390/electronics12183781.
- **검증 결과: 논문은 실재하며 수치도 실재한다.** 원문 (Abstract): *"Several supervised object detectors were evaluated on the PASCAL VOC 2012 dataset, achieving a 1.1% average improvement in mean average precision. Ablation experiments with the BlitzNet object detector on the PASCAL VOC 2012 dataset showed an improvement of mAP by 1.19% using the proposed method, compared to a 0.18% improvement with random copy[-paste]."*
- **정정:** 사용자가 기억한 "+1.1% mAP(컨텍스트)"는 **여러 검출기 평균 +1.1%**이고, **BlitzNet 단독 ablation에서는 컨텍스트 +1.19% vs 무작위 +0.18%**가 정확한 대비 쌍이다. 즉 "+0.18% vs +1.1%"는 대략 맞으나 직접 대비 쌍은 +0.18% vs **+1.19%**.

**(r) Georgakis et al., "Synthesizing Training Data for Object Detection in Indoor Scenes," RSS 2017, arXiv:1702.07836 — 실내 물리적 타당 위치 배치**
- 원문 (Abstract): *"We superimpose 2D images of textured object models into images of real environments at variety of locations and scales. Our experiments evaluate different superimposition strategies ranging from purely image-based blending all the way to depth and semantics informed positioning of the object models into real scenes."*
- 원문 (RSS 논문 본문): *"augmenting some hand-labeled training data with synthetic examples carefully composed onto scenes yields object detectors with comparable performance to using much more hand-labeled data."*
- 번역: "질감 있는 2D 객체 모델을 실제 환경 이미지에 다양한 위치·스케일로 겹쳐 넣는다. 순수 이미지 기반 blending부터 깊이·의미 정보에 기반한 위치 지정까지 다양한 전략을 평가한다." / "합성 예시를 장면에 신중히 합성해 소량의 수동 라벨 데이터를 증강하면 훨씬 많은 수동 라벨을 쓴 것과 비슷한 성능을 낸다."
- **설계 매핑:** L3(레인지/팬 위 = 물리적으로 타당한 위치, 실내 표면 위 배치)의 직접 근거. 단 이들은 depth/semantics로 표면을 찾았고 사용자는 수동 배치다.

---

### ■ 5. Blending 방식이 검출 성능에 미치는 영향 (L1/L2)

**(s) Dwibedi, Misra, Hebert, "Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection," ICCV 2017, arXiv:1708.01642 — blending 혼합의 직접 실증 [최강 근거]**
- **naive pasting이 pixel artifact를 만들어 성능을 해침** (Abstract): *"We automatically 'cut' object instances and 'paste' them on random backgrounds. A naive way to do this results in pixel artifacts which result in poor performance for trained models."*
- **artifact가 shortcut처럼 학습됨** (Section 1 Introduction): *"However, naively placing object masks in scenes creates subtle pixel artifacts in the images. As these minor imperfections in the pixel space feed forward deeper into the layers of a ConvNet, they lead to noticeably different features and the training algorithm focuses on these discrepancies to detect objects, often ignoring to model their complex visual appearance."*
- **경계 artifact가 local feature 기반 검출을 저하** (Section 5.2.1 Blending): *"Directly pasting objects on background images creates boundary artifacts. ... Although these artifacts seem subtle, when such images are used to train detection algorithms, they give poor performance as seen in Table 1. As current detection methods strongly depend on local region-based features, boundary artifacts substantially degrade their performance."*
- **사용한 blending 모드** (Figure 6 캡션 + Section 5.2.1): 세 가지 모드 — *"No Blending, Gaussian Blurring, Poisson Blending"* — 에 Table 1의 조합 행 "All Blend", "All Blend + same image"가 더해진다. 원문: *"Each of these modes add different image variations, e.g., Poisson blending smooths edges and adds lighting variations. Although these blending methods do not yield visually 'perfect' results, they improve performance of the trained detectors."*
- **같은 장면을 여러 blending으로 렌더링하면 detector가 artifact에 불변이 됨** (Section 5.2.1): *"To make the training algorithm further ignore the effects of blending, we synthesize the exact same scene with the same object placement, and only vary the type of blending used. We denote this by 'All Blend + same image' in Table 1. Training on multiple such images where only the blending factor changes makes the training algorithm invariant to these blending factors and improves performance by 8 AP points over not using any form of blending."*
- **정확한 mAP 수치 (Table 1, GMU Kitchen 평가, mAP@IoU0.5):** No blending = 65.9, Gaussian Blurring = 68.9, Poisson = 58.4, All Blend = 72.4, All Blend + same image = 73.7. (즉 73.7 − 65.9 ≈ 8 AP. 주목: **Poisson 단독 58.4는 no-blending 65.9보다 낮다** — 이득은 모드 혼합에서 나온다.)
- 번역(핵심): "동일 장면·동일 배치로 blending 종류만 바꿔 여러 장 합성해 학습하면, 학습 알고리즘이 blending 요인에 불변이 되고 blending 미사용 대비 8 AP 향상된다."
- **설계 매핑:** 사용자의 페어링(같은 배경+불꽃+위치시드+스케일, blending만 변경)과 **정확히 동일한 프로토콜**이며, L1→L2 계단의 최강 직접 근거다. 단 사용자 도메인(발광 불꽃)에서의 재현은 미검증 → **[간접 근거]**. Poisson 단독이 오히려 성능을 낮췄다는 점은 발광체에 Poisson이 부적합할 것이라는 추론과도 정합적이다.

**(t) Poisson Image Editing** — Pérez, Gangnet, Blake, "Poisson image editing," SIGGRAPH 2003, ACM TOG 22(3):313.
- 원리 (원문 Abstract): *"Using generic interpolation machinery based on solving Poisson equations, a variety of novel tools are introduced for seamless editing of image regions."* 배경 조명에 맞추는 특성(구현 문헌 서술): *"This keeps the lighting conditions and colors of the destination image, but imports the object (shapes, texture and gradients) from the source image."*
- 번역: "Poisson 방정식 풀이 기반의 범용 보간 기법으로 이미지 영역의 seamless 편집 도구들을 도입한다." / "이는 목적(배경) 이미지의 조명·색을 유지하면서 소스(전경)로부터 형태·질감·gradient를 가져온다."
- **설계적 주의:** Poisson은 배경 조명에 전경을 맞추므로 **발광체(불꽃)에는 역효과 가능** — FGL-GAN이 "harmonization 부적합"이라 한 것 및 Dwibedi에서 Poisson 단독이 성능을 낮춘 것과 일치한다. L2에는 Poisson보다 **screen/additive**가 물리적으로 타당하다.

**(u) Porter & Duff, "Compositing Digital Images," SIGGRAPH 1984, Computer Graphics 18(3):253–259 — over operator 정의**
- 원문 (Abstract): *"This paper presents the case for four-channel pictures, demonstrating that a matte component can be computed similarly to the color channels, and guidelines for the generation of elements and the arithmetic for their arbitrary compositing are discussed."* over 연산 정의(premultiplied alpha): 전경 F over 배경 B = `F + B(1−αF)`.
- **설계 매핑:** L1의 "over 합성"(알파 컴포지팅)의 정식 정의 근거.

**(v) screen/additive가 발광체에 적합한 이유** — screen blend `1−(1−a)(1−b)` 및 additive는 광원을 배경 위에 **더하는** 연산으로, 발광·투명 요소 합성에 적합하다. Porter-Duff의 mixed/additive 계열 기술 문헌이 근거이나 "불꽃에 screen이 최적"이라는 도메인 직접 실증 논문은 **찾지 못함 → [간접 근거]**.

**(w) Shortcut Learning** — Geirhos et al., "Shortcut Learning in Deep Neural Networks," *Nature Machine Intelligence* 2(11):665–673 (2020), arXiv:2004.07780, DOI: 10.1038/s42256-020-00257-z.
- 원문 (Abstract): *"Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios."*
- 번역: "shortcut은 표준 벤치마크에서는 잘 작동하지만 실세계 같은 더 어려운 조건으로 전이되지 못하는 결정 규칙이다."
- **설계 매핑:** 검출기가 합성 경계 artifact를 "지름길"로 학습하는 위험의 근거. 사용자가 blending 계단으로 artifact 의존을 통제하려는 동기를 정당화한다.

---

### ■ 6. 객체 크기/스케일 (SCALE 인자)

**(x) COCO small/medium/large 정의** — Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014, arXiv:1405.0312.
- COCO API 정의(원 출처): small = area < 32² (1024px²), medium = 32²~96² (1024~9216px²), large = > 96² (9216px²). 2차 확인 (SNIP, arXiv:1711.08189): *"the area of small objects is less than 32x32, medium objects range from 32x32 to 96x96 and large objects are greater than 96x96."*
- **설계 매핑:** 사용자의 SCALE(64/128/256px 높이) 중 64px는 medium 경계 부근, 128/256px는 (면적 기준) large에 해당한다. 스케일 구간 설정의 **개념적** 근거는 COCO이나, **64/128/256이라는 특정 절대값 선택은 임의 [근거 없음]**.

**(y) small object 검출 난이도 정량화** — Kisantal et al., "Augmentation for small object detection," arXiv:1902.07296: *"the AP detection metric for small objects is 2-3 times lower than that for large objects."* (COCO 상위 제출작 기준.)
- 번역: "small object의 AP는 large object보다 2~3배 낮다."

**(z) SAHI 【flagged — 검증 완료】** — Akyon, Altinuc, Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," arXiv:2202.06934.
- 원문 (Abstract): *"the proposed inference method can increase object detection AP by 6.8%, 5.1% and 5.3% for FCOS, VFNet and TOOD detectors, respectively. Moreover, the detection accuracy can be further increased with a slicing aided fine-tuning, resulting in a cumulative increase of 12.7%, 13.4% and 14.5% AP in the same order."* (Visdrone·xView 항공 데이터셋 기준.)
- 번역: "제안된 추론 방법은 FCOS·VFNet·TOOD 검출기에서 각각 AP를 6.8%·5.1%·5.3% 높인다. 나아가 슬라이싱 파인튜닝을 병행하면 같은 순서로 누적 12.7%·13.4%·14.5% AP 증가한다."
- **주의:** 초록은 small object 향상 수치를 제시하나, 사용자가 물었던 "large object에서의 성능 저하 서술"의 정확한 verbatim은 초록에서 확인되지 않았다(본문 확인 필요). SAHI는 소형 객체용 기법이므로 대형 불꽃에는 이득이 제한적일 수 있다는 점만 정성적으로 유의.

---

### ■ 7. 평가 지표

**(aa) confidence를 연속 지표로, recall 포화 보완** — FGL-GAN이 YOLOv5 confidence(0.7534)를 품질 지표로 쓴 것이 도메인 내 선례(§1e)다. recall이 이진 임계 기반이라 포화하기 쉬운 반면 confidence는 연속이라 더 민감하다는 것은 통계적으로 타당하나, "confidence가 recall보다 민감하다"는 명제를 검출 실험에서 직접 실증한 특정 논문은 **찾지 못함 → [설계 원칙/간접]**.

**(bb) FID/KID의 한계 【flagged arXiv:2606.19817】** — **이 arXiv 번호(2606.xxxxx)는 2026년 6월 이후 식별자로, 현재(2026-08) 기준 미래 문헌이거나 오타로 판단된다. 해당 문헌을 확인하지 못했다.** FID/KID가 소표본에서 편향되고 지각적 품질과 어긋날 수 있다는 지적은 Borji(arXiv:2103.09396) 등에서 광범위하나, 지정된 arXiv 번호는 **검증 불가 → 사용 보류 권고**. Borji로 대체 인용을 권한다.

**(cc) paired comparison / McNemar** — 검출 실험에 McNemar test를 쓰는 것은 분류·검출 비교의 표준 통계 관행이나, 이 보고서 검색 범위에서 특정 대표 논문 verbatim은 확보하지 못했다 → **[설계 원칙]**.

---

### ■ 8. 실험 설계 일반 원칙

**(dd) pseudoreplication** — Hurlbert, "Pseudoreplication and the Design of Ecological Field Experiments," *Ecological Monographs* 54(2):187–211 (1984), DOI: 10.2307/1942661.
- 원문 (Abstract): *"Pseudoreplication is defined as the use of inferential statistics to test for treatment effects with data from experiments where either treatments are not replicated (though samples may be) or replicates are not statistically independent."*
- 번역: "pseudoreplication은 처리가 반복되지 않았거나(표본은 반복될 수 있음) 반복이 통계적으로 독립이 아닌 실험 데이터로 처리 효과를 추론통계로 검정하는 것으로 정의된다."
- **설계 매핑:** 같은 배경/불꽃에서 파생된 여러 합성 이미지는 **독립이 아니므로** 이미지 단위로 집계하면 pseudoreplication이다. 사용자가 "장면 단위 집계"를 규칙으로 둔 것이 정확히 이 문제를 회피한다 → **[직접 실증/원칙]으로 강한 근거**.

**(ee) 데이터 누출** — Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science," *Patterns* 4(9):100804 (2023), arXiv:2207.07048.
- 원문 (Abstract): *"We show that data leakage is indeed a widespread problem and has led to severe reproducibility failures. ... we find 17 fields where leakage has been found, collectively affecting 294 papers."*
- 번역: "데이터 누출이 실제로 광범위한 문제이며 심각한 재현 실패를 초래했음을 보인다. 누출이 발견된 17개 분야에서 총 294편 논문이 영향받았다."
- **설계 매핑:** 같은 배경/불꽃 소스가 train/test에 걸치면 누출이다. 사용자가 프리즈 검출기(재학습 없음)를 쓰므로 학습-누출 위험은 낮으나, 배경·불꽃 소스의 중복 제거·독립성은 여전히 관리해야 한다.

**(ff) paired design의 검정력** — 대응(paired) 설계가 비대응보다 개체 간 분산을 제거해 검정력이 높다는 것은 통계 표준이다. 사용자의 페어링(같은 배경+불꽃+위치+스케일, blending만 변경)은 이 원리의 정확한 적용이며 → **[설계 원칙]**, Dwibedi의 "same image, vary blending"(§5s)이 사실상 동일 프로토콜의 실증 선례다.

**(gg) 한 번에 하나의 요인** — controlled experiment의 기본 원칙. 사용자 L1→L4 계단이 이를 따르나, OFAT의 상호작용 미포착 한계(§3m)에 유의해야 한다.

---

### ■ 9. 기존 인용 수치 재확인 【flagged】

- **BoWFire (Chino et al., SIBGRAPI 2015, arXiv:1506.03495):** "false positive 최대 80% 감소"라는 **특정 수치 표현은 원문에서 확인되지 않았다.** 원문은 정성적으로만 서술: *"We explore the fact that color combined with texture can improve the detection of fire, reducing the number of false-positives as compared to related works from the literature."* — "80%"라는 정량 표현의 출처를 찾지 못했으므로 **"최대 80% 감소" 인용은 근거 미확인으로 철회 권고**.
- **DACBFIAM (IEEE 9979846, "Fire Image Augmentation based on Diverse Alpha Compositing for Fire Detection"):** web_search 예산 소진으로 초록 이상 정보를 확보하지 못함 → **미검증**.
- **Corsican Fire Database 【검증 완료】** — Toulouse et al., *Fire Safety Journal* 2017, 92:188–194. 원문: *"The dataset consists of 500 visible images of wildfire collected worldwide, 100 multi-modal (visible and near infrared) images, and 5 sequences of about 30 multi-modal images of outdoor experimental fires captured by the authors."* — **원본 규모는 "500 visible + 100 multimodal + 5 sequences"가 정확하다.** 후속 문헌(예: FIRe-GAN arXiv:2101.11745 등)이 인용하는 "640 pairs of visible and infrared fire images"는 **이후 확장판** 수치다. 두 수치 모두 실재하며 원본(500/100/5)과 확장판(640 pairs)의 구분이 확인된다.
- **ODGEN (arXiv:2405.15199) 【검증 완료】:** 원문 (Abstract): *"Adding training data generated by ODGEN improves up to 25.3% mAP@.50:.95 with object detectors like YOLOv5 and YOLOv7."* — "up to(최대)"이며 7개 도메인 벤치마크에서의 최댓값이다. 조건: 실제 데이터 + ODGEN 합성 데이터 추가 학습(예: COCO는 80k real + 20k synthetic).
- **SAHI (arXiv:2202.06934) 【검증 완료】:** §6z 참조 — 추론 단독 +6.8/5.1/5.3% AP(FCOS/VFNet/TOOD), 슬라이싱 파인튜닝 병행 시 누적 +12.7/13.4/14.5% AP.
- **Ultralytics background 0~10% 권장 【검증 완료】:** 원문 (YOLOv5 Tips for Best Training Results 및 issue #6281): *"Background images are images with no objects that are added to a dataset to reduce False Positives (FP). We recommend about 0-10% background images to help reduce FPs (COCO has 1000 background images for reference, 1% of the total). No labels are required for background images."* 유지관리자 부연: *"0-10% is based on empirical results with client datasets."* — **즉 이 권장치는 경험적 관행이지 논문 실증이 아니다.** 사용자의 조건 0-c(무화염 원본 배경, FP 기준선)의 근거.
- **D-Fire (de Venâncio et al., *Neural Computing and Applications* 2022, DOI: 10.1007/s00521-022-07467-z) 【검증 완료】:** 공식 GitHub 및 후속 논문 확인 — 총 **21,527 images** (Only Fire 1,164 / Only Smoke 5,867 / Fire and Smoke 4,658 / None(negative) 9,838), 총 **26,557 bounding boxes** (fire 14,692 / smoke 11,865), 해상도 416×416, YOLO 포맷.

---

## Recommendations

**1단계 — 보고서에 즉시 반영(근거 확정):** 다음은 원문 검증이 끝났으니 설계 근거표에 [직접 실증]/[간접 근거]로 인용하라. FGL-GAN(물리모델 a–d, LGM/GCM g, ablation h), Dwibedi blending 계단(s: no blend 65.9→all+same 73.7, +8 AP, 페어링 프로토콜), BIB-Copy-Paste(+1.19% vs +0.18%), Ghiasi(무작위도 강력, +0.6 mask/+1.5 box AP), Dvornik(컨텍스트가 무작위 대비 평균 +5%, 무작위 배치는 오히려 성능 저하), COCO 스케일 정의(x), SAHI(+6.8/5.1/5.3% 추론, 누적 +12.7/13.4/14.5%), Geirhos shortcut(w), Hurlbert pseudoreplication(dd → 장면 단위 집계 정당화), Kapoor&Narayanan 누출(ee), Ultralytics 0–10%(조건 0-c), D-Fire 구성, Corsican 원본 500/100/5.

**2단계 — 표현 수정·철회:** ① "BoWFire FP 80% 감소" → 정량 수치 근거 없음, 정성 표현으로 완화하거나 철회. ② FID/KID 한계 인용의 arXiv:2606.19817 → 미래/오타 식별자, 철회하고 Borji(arXiv:2103.09396)로 대체. ③ DACBFIAM 수치 → "미검증"으로 명시하거나 원문 직접 확인 후 사용.

**3단계 — 설계 정직성 문구 삽입:** 보고서에 "본 설계는 (i) 검출기 confidence를 합성 품질 지표로 쓰기(FGL-GAN), (ii) downstream 성능으로 생성모델 평가(CAS), (iii) 요소별 ablation의 조합이며, 이 특정 통합(프리즈 검출기 + 계단식 합성법 ablation)의 직접 선행 연구는 확인되지 않았다"를 명기하라. 이는 약점이 아니라 신규성 주장 근거가 된다.

**4단계 — 방법론 보강:** ① L3의 근거가 양방향(Ghiasi vs Dvornik)이므로 L3를 "가설"로 명시하고 무작위(L1) 대비 효과를 실제로 측정하는 것이 정당하다. 데이터 규모가 작을수록(사용자 케이스처럼 소량 합성) Dvornik 쪽(컨텍스트 유리) 결론이 재현될 가능성이 높다. ② OFAT 계단의 상호작용 미포착을 보완하려면 SCALE×SOURCE 완전 교차(이미 반영됨)를 유지하고 가능하면 blending×위치도 부분 교차. ③ 장면 단위 집계 + paired 분석 + (가능시) McNemar/부트스트랩으로 검정.

**벤치마크(판단 기준 변경점):** L2가 L1 대비 TP confidence를 유의미하게(예: paired 차이의 95% CI가 0을 배제) 올리지 못하면 screen blending 채택 근거가 약화 → over 유지. L3가 L1(무작위) 대비 recall/confidence 향상이 없으면 Ghiasi 쪽 결론(무작위 충분)을 채택. SCALE에서 64px(≈medium 경계)와 256px(large) 간 recall 격차가 Kisantal의 "2~3배" 수준으로 크게 나타나면, 소형 불꽃에 한해 SAHI식 슬라이싱 추론을 추가 검토.

---

## Caveats
- **web_search 예산(18회) 소진으로** DACBFIAM 원문, OFAT/factorial·McNemar 대표 논문의 verbatim, SAHI의 "large object 성능 저하" 본문 문장을 완전히 확보하지 못했다. 해당 항목은 본문에 "미검증/설계 원칙"으로 명시했다.
- **FGL-GAN Table 1의 경쟁모델별 정확 수치**(pix2pix/CycleGAN/QS-Attn/FIS-GAN 각각의 FID·accuracy·confidence)는 표가 이미지로 되어 있어 텍스트 추출에 실패했다. FGL-GAN 자체 수치만 확정. 정확한 비교표는 MDPI 원문 Table 1을 직접 확인할 것.
- **arXiv:2606.19817은 미래/오타 식별자로 판단**되어 검증하지 않았다.
- Dwibedi의 blending 실증은 **instance detection(비발광 주방 객체)** 도메인이며, 발광 불꽃에 그대로 전이된다는 보장은 없다(그래서 [간접 근거]). 사용자의 실험이 바로 그 전이를 검증하는 의의를 가진다.

---

## 부록 A: 설계요소별 근거 매핑표

| 설계 요소 | 근거 문헌(핵심) | 근거 유형 | 비고 |
|---|---|---|---|
| **L1: over 합성 + 무작위 위치** | Porter&Duff 1984(over 정의); Ghiasi 2021(무작위도 강력, +0.6 mask/+1.5 box AP) | [직접 실증] | 무작위 배치의 유효성 실증됨 |
| **L2: screen/blending 대체** | Dwibedi 2017(blending 혼합 +8 AP: 65.9→73.7); Pérez 2003(Poisson) | [간접 근거] | 발광체엔 Poisson 역효과 가능(Dwibedi에서 Poisson 단독 58.4<65.9); screen 적합성은 도메인 직접실증 없음 |
| **L3: 컨텍스트 배치(레인지/팬 위)** | Dvornik 2018(무작위 대비 평균 +5%, 무작위는 성능 저하); BIB-Copy-Paste 2023(+1.19% vs +0.18%); Georgakis 2017 ↔ 반대: Ghiasi 2021 | [직접 실증, 단 양방향] | 우월성 자명하지 않음 → 가설로 검증 권장 |
| **L4: 스필(반사광/글로우)** | FGL-GAN(불꽃=광원, 반사광이 배경광량·거리·색·형태에 따라 변함) | [직접 실증(원리)] | 물리모델의 직접 원문 근거 |
| **SCALE(64/128/256px)** | Lin et al. COCO(32²/96² 구간); Kisantal(small AP 2–3배 낮음); SAHI | [간접 근거] + [근거 없음] | 구간 개념은 근거 있음, 특정 절대값은 임의 |
| **SOURCE(VFX vs NIST 조리유)** | — | [근거 없음] | 도메인 일반화 인자로 합리적이나 특정 선행 실증 없음 |
| **조건0-a 하드 사각형(알파 없음)** | Dwibedi 2017(naive paste가 artifact→성능저하) | [직접 실증] | artifact 기준선의 근거 |
| **조건0-b 생성모델 세트** | FGL-GAN; ODGEN(+최대 25.3% mAP) | [직접 실증] | 생성 합성의 유효성 근거 |
| **조건0-c 무화염 배경(FP 기준선)** | Ultralytics(0–10% background로 FP 감소, 경험적) | [설계 원칙] | 논문 실증 아닌 경험적 권장 |
| **GT박스=배치박스 고정(스필 재추출 금지)** | FGL-GAN(GT는 flame mask, 스필은 렌더링 산물) | [설계 원칙] | 라벨 정의 일관성 원칙 |
| **페어링(blending만 변경)** | Dwibedi 2017("same image, vary blending"); paired design 통계원리 | [직접 실증] | 프로토콜이 문헌과 동일 |
| **장면 단위 집계** | Hurlbert 1984(pseudoreplication) | [직접 실증(원칙)] | 독립성 위반 회피의 정석 |
| **FP는 항상 0-c 대비** | Ultralytics(background=FP 측정 목적) | [설계 원칙] | |
| **평가: recall/FP/confidence** | FGL-GAN(confidence를 품질지표로); CAS(downstream 평가) | [간접 근거] | confidence>recall 민감성은 원칙적 |
| **평가: conf sweep PR/AP, IoU 분포** | COCO AP 관행 | [설계 원칙] | 표준 |
| **데이터 독립성/누출 관리** | Kapoor&Narayanan 2023 | [직접 실증(원칙)] | 소스 중복 제거 필요 |
| **프리즈 검출기를 심판으로(전체 설계)** | FGL-GAN + CAS + ablation의 조합 | **직접 선례 없음** | 표준요소의 신규 조합 |
| **불꽃 소스 클립 수(25~40), AI-Hub 4장 등** | — | [근거 없음] | 연구자 판단 |

## 부록 B: "논문 근거 없이 연구자 판단으로 정한 항목"
1. **SCALE의 특정 절대 픽셀값 64/128/256** (구간 개념만 COCO 근거)
2. **VFX 클립 수 25~40, NIST 프레임 4장, AI-Hub 크롭 4장(보류) 등 표본 수**
3. **SOURCE를 VFX(일반화염) vs NIST(조리유화염) 2수준으로 나눈 특정 분류**
4. **L1→L2→L3→L4의 특정 순서·누적 방식**(누적 ablation 자체는 관행이나 이 순서 배열은 판단)
5. **조건0의 a/b/c 세부 구성**(하드 사각형·생성모델 세트·무화염)의 특정 선택
6. **conf 스윕의 구체적 임계 구간, IoU 분석의 구체 방법**
7. **"프리즈 검출기를 심판으로 쓰는 계단식 ablation" 통합 설계 자체**(선행 직접 근거 없음 — 신규 조합)
