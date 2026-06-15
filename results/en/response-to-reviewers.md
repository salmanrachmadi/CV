# Response to Reviewers

**Manuscript:** A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcyclist Helmet-Use Detection

**Revision date:** 2026-06-13

---

Dear Editor and Reviewers,

We thank you for the thorough and constructive feedback on our manuscript. We have revised the manuscript comprehensively based on all comments from the five reviewers. Below is our point-by-point response to each issue raised.

Changes in the revised manuscript are indicated implicitly through the strengthening of arguments, the addition of references, and the expansion of the Limitations section.

---

## Response to the EIC (Editor-in-Chief)

### EIC-1: The claim that "attention does not help" is too general for a single dataset

**Type:** MAJOR · **Status:** RESOLVED

**Response:** We have narrowed the central claim throughout the manuscript. The Abstract (both Indonesian and English versions) now explicitly states that the finding "holds under the specific conditions tested: a single, relatively easy small-scale dataset, a training protocol configured for YOLO, and a resolution of 1280 pixels." The Conclusion section (§VII) adds a similar statement and characterizes this finding as "a practical caution against the assumption that adding attention always helps on small-scale helmet-detection tasks, not a general verdict on attention mechanisms." The Discussion section (§V) is also reinforced with the sentence: "This study does **not** conclude that attention is useless in general."

> See: Abstract (final paragraph), §V (seventh paragraph), §VII (second paragraph).

### EIC-2: Incremental contribution

**Type:** MAJOR · **Status:** RESOLVED

**Response:** We have strengthened the framing of the contribution (§I, contribution 3) as "a structured rebuttal of the assumption that adding attention always helps." The research question has been reinforced to read "can the improvements reported by attention-based studies on helmet detection be replicated in a controlled comparison?" — positioning the paper as a test of the generalizability of the positive claims present in the literature, rather than a mere routine comparison. A new subsection §II.D ("Attention in helmet detection") explicitly reviews studies that report benefits from attention [19], [20], [22] and identifies their methodological gaps, building the case for our study as a contribution that fills those gaps.

> See: §I (contribution 3, research question), §II.D (new subsection).

### EIC-3: Limited methodological novelty

**Type:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Response:** We acknowledge that the methodological novelty is indeed limited — the strength of the paper lies in the novelty of its findings (a controlled negative result) and the novelty of its framing (a structured rebuttal), not in the novelty of its method. We have reflected this in the discussion of the paper's position as a "practical caution" (§V, §VII). The selection of an appropriate venue (an applied workshop/conference) is consistent with this character.

---

## Response to Reviewer 1 — Methodology

### R1-1: Hyperparameter confounding (MAJOR)

**Type:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Response:** We acknowledge that a uniform hyperparameter protocol favors YOLO and may not yet maximize the potential of attention-based models. This limitation is now explicitly acknowledged as **Limitation (4)** in §VI: "Hyperparameters were uniform and not specifically *tuned* per architecture. This protocol favors YOLO... Transformers and attention-based models may demand different training recipes... the results of the attention-based models may represent a lower bound rather than their true potential." This counter-hypothesis is also noted in §V: "A plausible counter-hypothesis—that the attention-based models were not trained on their optimal recipe—cannot be ruled out and indeed becomes an agenda for further work." Per-architecture fine-tuning was added in §VII Future Work.

> See: §VI(4), §V (seventh paragraph), §VII (third suggestion).

### R1-2: Weak significance testing (MAJOR)

**Type:** MAJOR · **Status:** RESOLVED

**Response:** We have replaced the heuristic with a formal statistical test. §III.D now describes the explicit protocol: "a paired t-test on the mAP@0.5 differences of each *seed* pair, at a significance level of α = 0.05." A new **Table III** has been added in §IV.A summarizing all pairwise comparisons with t(df), p, Cohen's *d*, and 95% confidence interval values. Cohen's *d* for YOLO11s vs. baseline (0.61) is discussed as a medium-sized effect that does not reach significance owing to low power.

> See: §III.D (new paragraph), §IV.A (Table III and the effect-size discussion).

### R1-3: RT-DETR n=2 (MAJOR)

**Type:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Response:** The n=2 limitation for RT-DETR-l is now expanded as **Limitation (2)** in §VI with a discussion of concrete consequences: "its variance estimate is less precise (standard deviation 0.0151 vs. 0.0014–0.0046) and the 95% confidence interval is very wide ([−0.140; +0.142])." The claim of equivalence for RT-DETR-l is consistently stated as "absence of evidence of a difference" throughout the manuscript. The addition of a third seed was included in §VII Future Work.

> See: §VI(2), §IV.A (Table III), §VII (sixth suggestion).

### R1-4: Not size-matched (MAJOR)

**Type:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Response:** The capacity mismatch is now acknowledged in three places: (a) §III.B explicitly notes that "RT-DETR-l (~32 million parameters) is far larger" and explains the rationale for selecting standard models; (b) **Limitation (3)** in §VI discusses the implications and recommends "YOLOv8m (~25 million parameters) as a *size-matched* control"; (c) **Limitation (10)** in §VI emphasizes the need for size-matched architecture evaluation. The YOLOv8m comparison was added in §VII Future Work.

> See: §III.B (third paragraph), §VI(3), §VI(10), §VII (third suggestion).

### R1-5: Ceiling effect / low statistical power (MAJOR)

**Type:** MAJOR → CRITICAL · **Status:** RESOLVED

**Response:** The ceiling effect is now discussed in depth in **Limitation (5)** of §VI with concrete quantification: "with only 100 test images, a difference of one to two mislabeled annotations can already shift mAP@0.5 by approximately 0.01." All equivalence claims are consistently stated as "absence of evidence of a difference, not evidence of equivalence." The discussion in §V emphasizes: "concluding that 'attention does not help' in absolute terms would be a leap from *absence of evidence* to *evidence of absence*." Limitation (9) also adds that three seeds are sufficient for large effects (CBAM) but not for small effects.

> See: §VI(5), §VI(9), §V (seventh paragraph), Abstract.

### R1-6: The CBAM decline may be a design artifact (MAJOR)

**Type:** MAJOR · **Status:** RESOLVED

**Response:** We have added a sixth explanation in §V that explicitly discusses four counter-hypotheses for the CBAM decline: (a) suboptimal placement on the neck rather than the backbone; (b) fresh layers disrupting the COCO weights; (c) insufficient epochs; (d) the learning rate not tuned separately. **Limitation (7)** in §VI reiterates this point and states that "the decline in accuracy due to CBAM cannot be attributed outright to the ineffectiveness of the attention module itself." A systematic CBAM ablation (varying placement, from-scratch training, more epochs, separate LR) was added in §VII Future Work (fourth suggestion).

> See: §V (sixth explanation), §VI(7), §VII (fourth suggestion).

### R1-7: Inconsistent FPS protocol (MINOR)

**Type:** MINOR · **Status:** RESOLVED

**Response:** The FPS protocol is now described explicitly in two places: (a) §III.D states that "the reported FPS figures were obtained from a single *run* while the GPU was idle... using the built-in Ultralytics benchmark on the full test *split*"; (b) §IV.C adds a separate paragraph: "FPS was measured on a single *run* while the GPU was idle... Measurements were taken on the same hardware configuration (NVIDIA RTX 4090, batch size 1)." **Limitation (6)** in §VI acknowledges that the FPS figures are estimates at a single hardware point.

> See: §III.D, §IV.C (new paragraph), §VI(6).

---

## Response to Reviewer 2 — Domain

### R2-1: Thin literature coverage (MAJOR)

**Type:** MAJOR · **Status:** RESOLVED

**Response:** We have added 8 new references ([17]–[24]), increasing the total from 16 to 24 references. The additions include:
- Attention surveys: Guo et al. [17], Niu et al. [18]
- Small-object detection survey: Cheng et al. [23]
- Attention-based helmet detection: Zhang et al. [19], Li et al. [20], Jia et al. [22]
- Indonesian context: Hariyono et al. [21], Raj and Nair [24]
- The new subsection **§II.D "Attention in helmet detection"** reviews positive reports and identifies their methodological gaps.
- §II.A was expanded with the Indonesian context [21], [24].
- §II.C was expanded with attention surveys [17], [18] and the small-object survey [23].

> See: §II.A, §II.C, §II.D (new subsection), Reference List.

### R2-2: Framing of the contribution (MAJOR)

**Type:** MAJOR · **Status:** RESOLVED

**Response:** The contribution is now explicitly positioned as a "structured rebuttal" (§I, contribution 3). The research question has been reinforced to read "can the improvements reported by attention-based studies... be replicated in a controlled comparison?" Subsection §II.D builds the case that the existing positive studies lack adequate experimental control, positioning our paper as a filler of that methodological gap. §VII characterizes this finding as "a practical caution against the assumption that adding attention always helps on small-scale helmet-detection tasks."

> See: §I (research question, contribution 3), §II.D, §VII.

### R2-3: Position on the architecture map (MINOR)

**Type:** MINOR · **Status:** RESOLVED

**Response:** §II.C now explicitly refers to the Swin Transformer [15] and ViTDet [16] as points on the attention spectrum not represented in our experiments (a pure ViT backbone and hierarchical windowed attention). §II.D reviews studies that use attention in helmet detection. The limitation that four points do not cover the entire spectrum has been noted in §VI and §VII Future Work.

> See: §II.C (second paragraph), §II.D.

---

## Response to Reviewer 3 — Perspective / Impact

### R3-1: Gap to real-world deployment (MAJOR)

**Type:** MAJOR · **Status:** RESOLVED

**Response:** The fifth explanation in §V ("The gap toward real-world deployment") explicitly discusses that under real-world street CCTV conditions (occlusion, nighttime, rain, density), attention may provide benefits not observed on our clean dataset. We refer to Raj and Nair [24] on real-world CCTV challenges, Hariyono et al. [21] on the Indonesian context, and Cheng et al. [23] on small objects in dense frames. **Limitation (8)** in §VI also acknowledges the absence of cross-dataset validation. §VII Future Work (first suggestion) explicitly mentions testing on real-world CCTV.

> See: §V (fifth explanation), §VI(8), §VII (first suggestion).

### R3-2: The "so what" needs to be expanded (MINOR)

**Type:** MINOR · **Status:** RESOLVED

**Response:** The implications have been expanded in §V through the fourth explanation on the asymmetric cost of errors (~7.7% of helmets missed), the fifth explanation on real-world deployment, and the sixth explanation on the CBAM counter-hypotheses. §VII Future Work (fifth suggestion) now mentions the integration of license-plate detection, a two-stage approach, and a loss function that weights the *helmet* class more heavily.

> See: §V (fourth and fifth explanations), §VII (fifth suggestion).

### R3-3: Minority class & safety (MINOR)

**Type:** MINOR · **Status:** RESOLVED

**Response:** The fourth explanation in §V ("The asymmetric cost of errors in a safety context") explicitly discusses that *false negatives* on "no-helmet" detection are more harmful, and proposes a weighted loss function and metrics such as the *weighted F1-score* for deployments aimed at enforcing safety.

> See: §V (fourth explanation).

---

## Response to the Devil's Advocate

### DA-CRITICAL-1: Overgeneralization of the core claim

**Type:** CRITICAL · **Status:** RESOLVED

**Response:** All claims have been narrowed with consistent scope qualifiers. See the response to EIC-1 above for full details of the changes in the Abstract, §V, and §VII. The title was retained because it is descriptive ("A Comparison... of Attention-Based and Attention-Free Mechanisms"), not interpretive.

### DA-CRITICAL-2: The ceiling effect undermines the entire inference

**Type:** CRITICAL · **Status:** RESOLVED

**Response:** See the response to R1-5 above. The ceiling effect is now acknowledged quantitatively (§VI(5): 100 test images → 1–2 annotations shift mAP by ~0.01) and all claims are stated as "absence of evidence" rather than "evidence of absence."

### DA-MAJOR-1: Causal attribution of "attention" in a non-isolated comparison

**Type:** MAJOR · **Status:** DELIBERATE_LIMITATION

**Response:** We acknowledge that RT-DETR ≠ "YOLO + attention" and that the comparison mixes paradigm and capacity factors. §III.B (third paragraph) states this explicitly. §VI(3) and §VI(10) discuss the implications. §VII recommends YOLOv8m as a size-matched control. The phrase "controlled comparison" in contribution 1 is now followed by an explanation that "performance differences can be attributed to architectural factors, particularly the presence and type of attention" — with a note in §III.B that capacity also differs.

### DA-MAJOR-2: Inconsistency between the accuracy and FPS measurement protocols

**Type:** MAJOR · **Status:** RESOLVED

**Response:** See the response to R1-7 above. The FPS protocol is now described explicitly and consistently in §III.D, §IV.C, and §VI(6).

### DA-MINOR: Potential narrative cherry-picking of FPS

**Type:** MINOR · **Status:** RESOLVED

**Response:** §IV.C now states that "the FPS figures are estimates at a single hardware point and will differ under other configurations." §VI(6) acknowledges that FPS was measured on a single configuration and does not reflect end-to-end latency. We did not select the "most favorable" figures — the figures derive from the only single run available while the GPU was idle.

---

## Summary of Changes

| Metric | Value |
|--------|-------|
| Total comments addressed | 17 |
| Resolved | 12 |
| Deliberate Limitation | 5 |
| Reviewer Disagree | 0 |
| New references added | 8 (from 16 → 24) |
| New subsection | 1 (§II.D "Attention in helmet detection") |
| New table | 1 (Table III: Paired t-test results) |
| Limitations added | 4 (from 6 → 10) |
| Explanations added in §V | 3 (from 3 → 6) |
| Estimated word-count change | +25% |

We believe that this revision has substantially strengthened the manuscript, particularly through the narrowing of claims, the strengthening of the statistical methodology, the expansion of the literature, and the explicit acknowledgment of all the limitations raised by the reviewers.

Sincerely,

The Authors
