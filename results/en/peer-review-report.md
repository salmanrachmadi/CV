# Peer-Review Report (Simulated Panel of 5 Reviewers)

**Manuscript:** A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcyclist Helmet-Use Detection
**Field:** Computer Vision / Deep Learning · **Target:** IEEE Conference/Journal
**Review date:** 2026-06-13

> Note: this report is *read-only* with respect to the manuscript. Scores are on a 0–10 scale. Dimension weights: Originality 20%, Methodological Rigor 25%, Evidential Adequacy 25%, Argument Coherence 15%, Writing Quality 15%.

---

## Phase 0 — Field Analysis & Reviewer Configuration

- **Primary discipline:** Computer vision (object detection), transportation-safety application.
- **Paradigm:** Empirical-experimental (controlled architecture comparison).
- **Maturity:** Solid, well-structured draft; the central claim risks *overclaiming*.
- **Realistic target tier:** Applied workshop/conference or a mid-tier applied journal.

**Five reviewers:**
1. **EIC** — editor of an applied CV journal; assesses fit, originality, and significance.
2. **R1 Methodology** — expert in detection-model evaluation & experimental statistics.
3. **R2 Domain** — researcher in helmet detection / ITS and attention in detection.
4. **R3 Perspective** — ITS/edge deployment practitioner & real-world impact.
5. **Devil's Advocate** — challenger of the core claim & fallacies.

---

## Phase 1 — Five Independent Review Reports

### Reviewer EIC (Editor-in-Chief)

**Summary.** The manuscript addresses a relevant problem (helmet compliance) with a clean comparative design and—refreshingly—reports *negative results*. The writing is clear, reproducibility is emphasized, and the figures are adequate. However, significance is limited by external validity (a single "easy" dataset, ceiling effect) and a central claim that exceeds the evidence.

**Strengths:** (1) focused research question; (2) controlled protocol + multi-*seed*; (3) openness in reporting negative results & limitations.
**Main weaknesses:** (1) the claim "attention does not help" is too general for a single dataset; (2) incremental contribution; (3) limited methodological novelty.

**Scores:** Originality 6 · Rigor 5 · Evidence 5 · Coherence 6 · Writing 8.
**Recommendation:** **Major Revision.**

### Reviewer 1 — Methodology

**Critical findings (publication blockers):**
1. **Hyperparameter confounding (MAJOR).** All architectures use identical hyperparameters (`optimizer=auto`, the same lr, YOLO augmentation, 1280 px). Transformers such as RT-DETR generally require a different training schedule, *learning rate*, and augmentation. Thus the claim "attention does not help" is confounded with "attention was not *tuned* correctly". A recipe that is fair for one family is not necessarily fair for another.
2. **Weak significance testing (MAJOR).** The "real vs. noise" verdict uses a heuristic of |Δ| against the mean standard deviation, rather than a formal statistical test. With n=3, the std estimate itself is unstable. Use at least a paired t-test across *seeds* or a confidence interval; report the *effect size*.
3. **RT-DETR n=2 (MAJOR).** Two *seeds* are insufficient for a meaningful std; a std of 0.0151 indicates instability. The claim of RT-DETR equivalence is fragile. Complete it to n=3 or weaken the strength of the claim.
4. **Not *size-matched* (MAJOR).** RT-DETR-l (~32M) vs. YOLO-s (~9–11M) conflates capacity and architecture factors. Include a matched comparator (e.g., YOLOv8m ~25M) or a smaller RT-DETR variant.
5. **Ceiling effect / low statistical power (MAJOR).** A baseline of ~0.96 mAP leaves little room; the experiment may be unable to detect an attention benefit even if one exists. Create *headroom* (a smaller/harder subset, or a stricter metric).
6. **CBAM: the drop may be a design artifact (MAJOR).** Inserting a custom CBAM + fresh layers on top of pretrained weights could cause a drop due to placement/initialization, rather than because "attention is harmful". An ablation is needed: different placement, train-from-scratch, or longer training.
7. **Inconsistent FPS protocol (MINOR).** Accuracy is multi-*seed* but FPS comes from a single "idle" *run*—a different measurement protocol. Report the latency distribution under a uniform protocol.

**Scores:** Originality 5 · Rigor 4 · Evidence 4 · Coherence 6 · Writing 8.
**Recommendation:** **Major Revision.**

### Reviewer 2 — Domain

1. **Thin literature coverage (MAJOR).** Only 14 references. Missing: surveys of attention in detection, Swin Transformer & ViTDet (although discussed conceptually), recent attention-based helmet-detection work, and Indonesian-context studies (mentioned in the research materials but not cited). Add a comparison with literature findings on attention on small data.
2. **Contribution framing (MAJOR).** A negative result can be a strong contribution, but it must be positioned explicitly against the literature claims it challenges. At present it reads as a "routine test" rather than a structured rebuttal.
3. **Position on the architecture map (MINOR).** The "attention spectrum" is interesting but not yet formally formulated; clarify why these four points are representative and what is absent (e.g., a pure ViT backbone).

**Scores:** Originality 6 · Rigor 5 · Evidence 5 · Coherence 6 · Writing 8.
**Recommendation:** **Major Revision.**

### Reviewer 3 — Perspective / Impact

1. **Gap to real deployment (MAJOR).** The practical claim "use YOLOv8s" rests on a clean dataset. Precisely in real CCTV (occlusion, night, crowding) attention/context may well be useful. Without a cross-domain test, the practical recommendation is poorly grounded.
2. **The "so what" needs expanding (MINOR).** The implications are currently narrow. Connect them to the *edge* budget, energy cost, and enforcement scenarios (helmet + plate) to broaden relevance.
3. **Minority class & safety (MINOR).** Discuss the asymmetric cost of errors (failing to detect "no helmet" is more harmful)—relevant for a safety application.

**Scores:** Originality 6 · Rigor 5 · Evidence 5 · Coherence 7 · Writing 8.
**Recommendation:** **Minor–Major Revision.**

### Devil's Advocate

**Strongest counter-argument (≈250 words).**
The central claim—"attention mechanisms do not help helmet detection"—is an *overgeneralization* from a single confounded experiment. The exact same results are equally consistent with a wholly different rival hypothesis: "we did not optimize the attention-based models correctly". Three facts support this rival. First, the hyperparameters are locked to a recipe that favors YOLO; RT-DETR and CBAM are never given a chance with their own recipe. Second, the baseline is already ~0.96, so there is a ceiling effect—the experiment has almost no power to detect an improvement even if one exists. Third, the CBAM drop relies on a very small std and a heuristic, not a genuine test; with n=3 this is fragile. The manuscript thus leaps from "absence of evidence" to "evidence of absence"—a classic fallacy. The title, too, promises to isolate "attention-based vs. attention-free", whereas RT-DETR differs in many respects beyond attention, so attribution to attention is invalid for that point.

**List of issues:**
- **CRITICAL** — Overgeneralization of the core claim (Abstract, §V, §VII): change from "attention does not help" to a scoped claim ("under YOLO-tuned hyperparameters on a small, easy dataset, adding attention does not improve accuracy").
- **CRITICAL** — The ceiling effect weakens the entire inference (§IV–V): experimental *headroom* or an explicit acknowledgment of low statistical power is needed.
- **MAJOR** — Causal attribution to "attention" in an un-isolated comparison (RT-DETR ≠ "YOLO + attention").
- **MAJOR** — Inconsistency between the accuracy measurement protocol (multi-*seed*) vs. FPS (single *run*).
- **MINOR** — Potential narrative *cherry-picking*: selecting the most favorable FPS number for interpretation.

**Overlooked alternative explanations:** undertraining of the attention-based model; suboptimal CBAM placement; the dataset being too easy; excess capacity on small data.

**The "so what" test:** partially passes — the finding is useful as a practical caution, but only if the claim is narrowed.

---

## Phase 2 — Editorial Decision & Revision Map

### Panel consensus
- **Agreed 5/5:** the writing is clear; the controlled design + multi-*seed* is a strength; the central claim **exceeds the evidence**; external validity is weak (single dataset).
- **Agreed 4/5 (MAJOR):** hyperparameter confounding; not *size-matched*; weak statistical testing; thin literature.
- **Devil's Advocate CRITICAL:** overgeneralization + ceiling effect → **under the panel's rules, the decision may not be Accept.**

### Final scores (weighted average)

| Dimension | Weight | Score | Contribution |
|---|---|---|---|
| Originality | 20% | 5.8 | 1.16 |
| Methodological Rigor | 25% | 4.4 | 1.10 |
| Evidential Adequacy | 25% | 4.8 | 1.20 |
| Argument Coherence | 15% | 6.2 | 0.93 |
| Writing Quality | 15% | 8.0 | 1.20 |
| **Total** | | | **≈ 5.6 / 10** |

### ⚖️ Verdict: **MAJOR REVISION**

The manuscript has a strong foundation (a clean experiment, honest negative results, good writing), but its central claim is not yet adequately supported because of confounding and limited statistical power. This is not a rejection—the issues are fixable—but it requires major revision of the claims and/or experiments.

### Revision Map (prioritized)

**P1 — Mandatory (decides acceptance):**
1. **Narrow the core claim** in the Abstract, §V, §VII: from "attention does not help" → "under a YOLO-tuned protocol on a small, easy dataset, adding attention does not improve accuracy and adds cost". (Addresses DA-CRITICAL.)
2. **Address the ceiling effect / statistical power:** add a *headroom* condition (a smaller/harder subset or a second, more challenging dataset) **or** explicitly acknowledge the power limitation. (DA-CRITICAL.)
3. **Correct statistical testing:** replace the heuristic with a paired t-test/confidence interval + *effect size*.

**P2 — Strongly recommended:**
4. **A *size-matched* comparison** (e.g., YOLOv8m vs. RT-DETR-l) to separate capacity from architecture.
5. **Complete RT-DETR to n=3** or weaken its equivalence claim.
6. **CBAM ablation** (placement, train-from-scratch, duration) to test whether the drop is intrinsic or an artifact.

**P3 — Reinforcing:**
7. **Expand the literature** (surveys of attention-in-detection, Swin/ViTDet, attention-based helmet work, Indonesian studies).
8. **Cross-dataset validation** for external validity.
9. **A uniform FPS protocol** + latency report.
10. **Sharpen the contribution framing** as a structured rebuttal of the assumption "attention always helps".

> Note: if the experimental revisions (P2/P3) are beyond budget, the manuscript can still be upgraded by **decisively narrowing the claim (P1)** so that the conclusion matches the evidence—the fastest path to publishability.
