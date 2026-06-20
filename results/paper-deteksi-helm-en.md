---
title: "From Zero-shot to Fine-tuning: Measuring the Domain Gap and the Role of Attention in Motorcyclist Helmet Detection"
author: "Author (to be set)"
date: "2026"
lang: en
version: "v2 (post internal peer review)"
---

# From Zero-shot to Fine-tuning: Measuring the Domain Gap and the Role of Attention in Motorcyclist Helmet Detection

*Revised draft v2. IMRaD format, IEEE citations, English. Experimental numbers come from `experiments/*/metrics.json`; figures from `results/figures/`.*

---

## Abstract

Helmet use is a primary protective factor for motorcyclists, and automatic detection of compliance from traffic cameras can support road-safety enforcement. Two assumptions often appear in practice: that COCO pre-trained weights can be used directly on the helmet domain, and that adding an attention mechanism improves accuracy. This study tests both under controlled conditions on the public NCKH 2023 v19 dataset (1,803 images; three classes: *helmet*, *license_plate*, *motorcyclist*). As a sanity check and motivation, we first verify the size of the domain gap through *zero-shot* evaluation of COCO weights as-is (YOLOv8s and YOLO11s) with `pycocotools`. This stage is not intended as an architecture comparison; it confirms whether fine-tuning is necessary. The main contribution is a controlled multi-*seed* evaluation (n = 3) of three architectures after identical fine-tuning, namely YOLOv8s (pure CNN), YOLO11s (partial C2PSA attention), and YOLOv8s+CBAM (explicit channel-spatial attention), accompanied by paired t-tests and effect sizes. The *zero-shot* results confirm a large domain gap: mAP@0.5 of only 0.0986 to 0.1143, with the `helmet` and `license_plate` classes at zero *by design* (no COCO counterpart). After fine-tuning, accuracy jumps to about 0.96 mAP@0.5. Among the fine-tuned models, no attention variant surpasses the pure CNN on mAP@0.5: YOLOv8s reaches 0.9596 ± 0.0008, YOLO11s 0.9574 ± 0.0019, and YOLOv8s+CBAM 0.9481 ± 0.0006. On the stricter metric, mAP@[.5:.95], the order reverses: YOLO11s (0.6902 ± 0.0022) slightly exceeds YOLOv8s (0.6885 ± 0.0006), though the difference is not significant (t(2) = −1.16; p = 0.37). Adding CBAM lowers mAP@0.5 significantly (t(2) = 16.1; p = 0.004; Cohen's *d* ≈ 9.3). This last finding is limited to a single placement configuration and a single *learning rate*, so it cannot be generalized to all CBAM implementations. With n = 3 and a ceiling effect near 0.96, the statistical power to detect small differences is very limited. The results are better read as evidence that, under the conditions tested, there is no consistent benefit from added attention.

**Keywords:** helmet detection; object detection; YOLO; attention mechanism; CBAM; transfer learning; domain gap; mAP.

---

## I. Introduction

Motorcycles account for a large share of traffic-accident casualties, especially in developing countries where they dominate daily mobility. The helmet is the most decisive protective factor against fatal head injury, so monitoring helmet-use compliance is an important policy target for safety. Manual observation does not scale to real traffic volumes, so automatic object detection from camera footage offers a way to run continuously and consistently [1], [2].

Over the past decade, object detection has shifted from accurate but slow two-stage approaches to fast one-stage detectors, with the YOLO family as the de-facto standard for real-time applications [5], [6]. In parallel, attention mechanisms born from the Transformer [10] and adapted to images through the Vision Transformer [11] reshaped the architectural map. Transformer-based detectors such as DETR [8] and its real-time descendant RT-DETR [7] compete with CNNs, while lightweight attention modules such as Squeeze-and-Excitation [12] and CBAM [9] became a popular way to insert attention into an existing CNN *backbone*.

Adding attention to helmet-detection models has produced several positive reports. Zhang et al. [19] reported that inserting an attention module and *feature fusion* into YOLOv8 improved helmet detection; Li et al. [20] showed a similar improvement on YOLOv5s; and Jia et al. [22] demonstrated gains with *deformable attention*. It should be noted that those studies ran on datasets, base architectures, and training protocols that differ from this one. Their findings are not meant to be refuted directly; they form the context that motivates the need for a controlled evaluation. Those reports generally compare a single model variant without repeating across *seeds*, without a uniform training protocol, and without significance testing. As a result, the reported differences could come from hyperparameters, augmentation, or initialization luck rather than from attention itself.

Beyond the assumption about attention, a second assumption is rarely verified: that COCO pre-trained weights are already close enough to the helmet task. The helmet class schema (`helmet`, `license_plate`, `motorcyclist`) only partly overlaps with the 80 COCO classes [14], and two of the three classes have no counterpart at all. The size of this domain gap needs to be verified, not assumed, so that deployment implications can be assessed correctly.

This study closes both gaps. Its main methodological contribution lies in the controlled evaluation design, not in a conclusion that attention is bad. That design includes an identical protocol, multi-*seed* repetition, formal statistical testing, and a separation between the domain gap (the *zero-shot* stage) and the attention effect (the fine-tuning stage). Two focused research questions:

- **RQ1.** How large is the domain gap between COCO weights as-is and the helmet-detection task?
- **RQ2.** After fine-tuning with an identical protocol and multi-*seed* repetition, do attention mechanisms (C2PSA in YOLO11s, CBAM in YOLOv8s) surpass the pure CNN YOLOv8s in a statistically meaningful way?

This paper has three main contributions. First, verification of the domain gap through a *vanilla* baseline (zero-shot COCO) evaluated with `pycocotools` as a sanity check, including the acknowledgment that two of the three classes are zero *by design* so the aggregate mAP does not reflect direct helmet-detection ability. Second, a controlled multi-*seed* evaluation (n = 3) among pure CNN, partial attention, and explicit attention under identical hyperparameters, data, and augmentation, accompanied by paired t-tests, Cohen's *d* effect sizes, and confidence intervals, on both metrics mAP@0.5 and mAP@[.5:.95]. Third, a reproducible notebook-based pipeline with fixed *seeds*, a recorded environment, and single-source-of-truth metrics, so that every number can be traced and reproduced.

## II. Related Work

### A. Helmet-use detection

Helmet-detection research has grown from simple classification toward pipelines that track motorcycles across frames and distinguish drivers from passengers. Siebert and Lin [1] demonstrated large-scale helmet-use detection from traffic video using *deep learning* and released an annotated dataset that became a community reference. Lin et al. [2] extended it with CNN-based *multi-task learning* that tracks individual motorcycles while registering per-rider helmet use. Benchmark challenges such as the AI City Challenge 2023 [3] pushed multi-class helmet-violation detection that exposed extreme class-imbalance problems. In enforcement settings, several approaches combine helmet detection with license-plate localization [4]. In Indonesia, Hariyono et al. [21] integrated helmet detection into a Streamlit interface for enforcement scenarios, and Raj and Nair [24] highlighted occlusion and lighting challenges in real traffic CCTV.

### B. One-stage detectors and the YOLO family

One-stage detectors formulate detection as direct regression of *bounding boxes* and classes in a single network pass, trading some accuracy for high speed [5]; this accuracy gap was partly narrowed by loss functions such as *focal loss* [13]. YOLOv8 [6] represents a mature CNN generation without explicit attention modules, relying on efficient convolution blocks and a *feature pyramid* for multi-scale objects. YOLO11 [6] introduces positional attention blocks (C2PSA) into the feature path. Its position sits in the middle of the attention spectrum: a CNN that adds limited *self-attention* without fully moving to the transformer paradigm.

### C. Attention mechanisms in computer vision

Attention lets a network re-weight features by relevance. Since the Transformer [10] and ViT [11], attention has spread into detection through architectures such as the Swin Transformer [15] and *plain* ViT for detection [16]. In the CNN domain, lightweight modules such as SENet [12] (channel attention) and CBAM [9] (channel plus spatial attention) became popular because they can be inserted without overhauling the *backbone*. Comprehensive surveys [17], [18] summarize the diversity of these mechanisms, while small-object detection relevant to helmets and plates remains a challenge of its own [23].

### D. Attention for helmet detection and the research gap

Several studies report attention benefits specific to helmet detection [19], [20], [22]. Those positive findings come from different settings, namely datasets, base architectures, and training protocols that are not uniform, so they cannot be compared number for number with this study. The concern is the lack of experimental control: without a uniform protocol, without *seed* repetition, and without statistical testing, an improvement claim is hard to attribute to attention itself rather than to other confounding variables. In addition, most studies go straight to fine-tuning without first measuring the domain gap from the COCO starting point, so the benefit of fine-tuning and the benefit of attention become mixed. This study separates those two sources of improvement. The *zero-shot* stage isolates and verifies the domain gap, while the fine-tuning stage, with architecture as the only variable that differs, isolates the attention effect under tight control.

## III. Methodology

### A. Dataset

We use the public dataset **NCKH 2023 / helmet-detection-project v19** (Roboflow, MIT license) in YOLO annotation format. The dataset contains 1,803 images split fixedly into 1,563 train, 140 validation, and 100 test, with three classes: `helmet`, `license_plate`, and `motorcyclist`. The dataset tends to be daytime-oriented, showing the front or side of vehicles, with adequate lighting. These conditions are relatively favorable compared with real CCTV scenarios that often involve night-time, occlusion, and long distance. The split is fixed and not reshuffled across runs to avoid data leakage.

Across the 100 test images, the distribution of annotation instances reflects a dominance of `motorcyclist` and `helmet` as a natural pair, while `license_plate` appears less often at certain viewpoints. This natural imbalance is a domain characteristic that must be considered when interpreting aggregate mAP.

### B. Compared architectures

Three architectures were chosen to span the attention-use spectrum:

- **YOLOv8s**, a pure CNN without explicit attention modules (baseline).
- **YOLO11s**, a CNN with positional C2PSA attention blocks in the feature path (partial attention).
- **YOLOv8s+CBAM**, that is the identical YOLOv8s plus three CBAM blocks [9] on the P3/P4/P5 outputs before the detection *head* (explicit channel and spatial attention). The CBAM module is registered with the Ultralytics *parser* and compatible weights are transferred from the COCO pre-trained YOLOv8s.

Choosing the *small* (s) variant for YOLOv8 and YOLO11 keeps model capacity comparable, so performance differences can be attributed to the presence or type of attention rather than to model size. YOLOv8s+CBAM was tested in a single configuration only: P3/P4/P5 placement, a single *learning rate*, and no separate warmup for the attention module. These factors are acknowledged implementation limitations.

### C. Two evaluation protocols

*Protocol 1, vanilla baseline (zero-shot COCO).* The COCO pre-trained weights of YOLOv8s and YOLO11s are evaluated on the test *split* without any training, as a verification of the domain-gap size. Because COCO models recognize 80 classes that include neither `helmet` nor `license_plate`, predictions are mapped to our three-class schema: COCO `motorcycle` becomes `motorcyclist`, while the `helmet` and `license_plate` classes have no counterpart so their AP is zero *by design*. Metrics are computed with `pycocotools` (the COCO standard) at a low confidence threshold (0.001) so the precision-recall curve is correct. This protocol is not meant to compare architectures meaningfully, since both fail on two of three classes; it confirms that fine-tuning is mandatory.

*Protocol 2, fine-tuning (transfer learning).* The three architectures are trained from COCO pre-trained weights with an identical protocol: image resolution 1280 pixels (helmet and plate objects are small), 100 epochs, *auto-batch*, the `auto` *optimizer*, initial *learning rate* 0.01, *early stopping* with *patience* 25, and the same augmentation (mosaic, HSV, horizontal *flip*, *scale*). Each architecture is trained on three *seeds* (42, 0, 1) run sequentially and independently, with GPU memory cleared between runs to avoid interference.

### D. Evaluation metrics and statistical tests

Two main metrics are reported, both on the test *split*: mAP@0.5 (the helmet-detection community standard) and mAP@[.5:.95] (the stricter COCO-benchmark metric, sensitive to localization precision). Inference speed (FPS) is reported as a *trade-off* indicator that is indicative, not a controlled benchmark, because it was measured on a single run while the GPU was idle without a formal *warmup*.

For the fine-tuned models, we report the mean ± standard deviation across *seeds* for both metrics and run paired t-tests (paired per *seed*) between architectures, accompanied by Cohen's *d* effect sizes. The paired test is appropriate here because the three models share the same *seeds*. With n = 3, the t-test has limited power; we present p-values and effect sizes as supplementary information, aware that only large differences can be detected convincingly.

### E. Environment and reproducibility

Experiments were run with PyTorch 2.8.0, Ultralytics 8.4.65, and CUDA 12.8 on an NVIDIA RTX 4090 GPU. *Seeds* were fixed for `random`, `numpy`, and `torch` (deterministic mode). The entire pipeline is organized as *self-contained* notebooks; each run's metrics are stored as a single source of truth (`experiments/<run>/metrics.{json,csv}`), and figures are built directly from those metric files.

## IV. Results

### A. Domain-gap verification: vanilla baseline (zero-shot COCO)

Table I summarizes the *zero-shot* evaluation. The results confirm that COCO models as-is cannot be used for helmet detection: mAP@0.5 is only 0.0986 (YOLOv8s) and 0.1143 (YOLO11s). This aggregate is dominated by the `motorcyclist` class, the only class with a COCO counterpart (AP@0.5: 0.296 and 0.343), while `helmet` and `license_plate` are zero *by design* because they do not exist in the COCO schema. Figure 1 visualizes the mAP@0.5 of both, and Figure 2 shows the per-class breakdown.

The difference between YOLOv8s and YOLO11s under *zero-shot* (0.0157 on mAP@0.5) is not practically meaningful. Both fail to detect two of the three classes, and the small aggregate comes almost entirely from `motorcyclist`, whose mapping is itself imperfect because COCO `motorcycle` differs semantically from a `motorcyclist` already riding the vehicle. For that reason, the *zero-shot* condition is not used to compare architectures; it serves solely to confirm the need for fine-tuning.

**Table I. Domain-gap verification: vanilla zero-shot COCO baseline (test split; n = 1).**

| Model | mAP@0.5 | mAP@[.5:.95] | AP@0.5 helmet | AP@0.5 plate | AP@0.5 motorcyclist |
|---|---|---|---|---|---|
| YOLOv8s (vanilla) | 0.0986 | 0.0270 | 0.000 | 0.000 | 0.296 |
| YOLO11s (vanilla) | 0.1143 | 0.0320 | 0.000 | 0.000 | 0.343 |

*Note: AP = 0 for `helmet` and `license_plate` is a consequence of the class-mapping design (no COCO counterpart), not an indicator of architecture comparison.*

![mAP@0.5 comparison of the vanilla baseline YOLOv8s vs YOLO11s (zero-shot COCO)](figures/vanilla_mAP50.png)

**Figure 1.** Vanilla baseline (zero-shot COCO): mAP@0.5 of YOLOv8s and YOLO11s on the test *split*. Both perform very poorly, confirming a large domain gap and the need for fine-tuning.

![Per-class AP@0.5 under zero-shot COCO](figures/vanilla_per_class.png)

**Figure 2.** Per-class AP@0.5 under *zero-shot*. The `helmet` and `license_plate` classes are zero *by design* (no COCO counterpart); only `motorcyclist` gets a signal through the mapping from `motorcycle`.

This answers RQ1: the domain gap is very large and COCO models as-is cannot be used for helmet detection. This result confirms that fine-tuning is mandatory before deployment.

### B. After fine-tuning: architecture comparison

After fine-tuning, accuracy jumps sharply to the range 0.95 to 0.96 mAP@0.5 and closes the domain gap; this jump is visualized in Figure 3. Table II summarizes the multi-*seed* results for both metrics.

On mAP@0.5, the pure-CNN YOLOv8s reaches the highest value (0.9596 ± 0.0008), followed by YOLO11s (0.9574 ± 0.0019), then YOLOv8s+CBAM (0.9481 ± 0.0006). On the indicative speed measure, YOLOv8s is also the fastest. Figure 4 presents a direct comparison among the fine-tuned models, and Figure 5 shows qualitative detection examples.

On mAP@[.5:.95], the stricter metric for localization precision, the order changes. YOLO11s (0.6902 ± 0.0022) slightly exceeds YOLOv8s (0.6885 ± 0.0006), while YOLOv8s+CBAM remains the lowest (0.6832 ± 0.0034). The differences among models on this metric are also not statistically significant (see §IV.C), but the reversal matters: the claim that YOLOv8s is best holds only on mAP@0.5, not on the stricter localization metric.

**Table II. Multi-*seed* fine-tuning results (n = 3; mean ± standard deviation; test split).**

| Model | mAP@0.5 | mAP@[.5:.95] | FPS†|
|---|---|---|---|
| **YOLOv8s** | **0.9596 ± 0.0008** | 0.6885 ± 0.0006 | ~296 |
| YOLO11s | 0.9574 ± 0.0019 | **0.6902 ± 0.0022** | ~189 |
| YOLOv8s+CBAM | 0.9481 ± 0.0006 | 0.6832 ± 0.0034 | ~290 |

†FPS is indicative: measured on a single run while the GPU was idle (RTX 4090); not a controlled benchmark.

![Comparison of vanilla zero-shot vs fine-tuned](figures/compare_vanilla_vs_finetuned.png)

**Figure 3.** Accuracy jump from the vanilla baseline (zero-shot, ~0.10) to the fine-tuned models (~0.96) on mAP@0.5, closing the domain gap. The difference among architectures after fine-tuning (right-hand scale) is far smaller than the effect of fine-tuning itself.

![Comparison of fine-tuned models YOLOv8s vs YOLO11s vs CBAM](figures/compare_finetuned.png)

**Figure 4.** mAP@0.5 comparison (mean ± standard deviation, n = 3) among fine-tuned models. The pure-CNN YOLOv8s leads slightly on mAP@0.5; on mAP@[.5:.95] the order changes (YOLO11s highest, Table II).

![Qualitative detection example for YOLOv8s](figures/infer_yolov8s.png)

![Qualitative detection example for YOLO11s](figures/infer_yolo11s.png)

![Qualitative detection example for YOLOv8s+CBAM](figures/infer_cbam.png)

**Figure 5.** Qualitative detection examples on test *split* images for the three fine-tuned models (*seed* 42), in order: (a) YOLOv8s, (b) YOLO11s, (c) YOLOv8s+CBAM. All three produce visually similar detections, consistent with the closeness of their quantitative metrics.

### C. Significance testing

Table III presents paired t-tests between architectures on mAP@0.5 (Table IIIa) and mAP@[.5:.95] (Table IIIb). On mAP@0.5, adding CBAM lowers accuracy significantly and consistently (t(2) = 16.1; p = 0.004; *d* ≈ 9.3); CBAM is also lower than YOLO11s (p = 0.012). By contrast, the difference between YOLOv8s and YOLO11s is small (0.0022) and not significant (t(2) = 2.71; p = 0.11).

**Table IIIa. Paired t-tests across *seeds* on mAP@0.5.**

| Comparison | Mean Δ | t(2) | p | Cohen's *d* |
|---|---|---|---|---|
| YOLOv8s − CBAM | +0.0116 | 16.1 | 0.004 | 9.3 |
| YOLOv8s − YOLO11s | +0.0022 | 2.71 | 0.11 (n.s.) | 1.56 |
| YOLO11s − CBAM | +0.0093 | 8.92 | 0.012 | 5.15 |

**Table IIIb. Paired t-tests across *seeds* on mAP@[.5:.95].**

| Comparison | Mean Δ | t(2) | p | Cohen's *d* |
|---|---|---|---|---|
| YOLO11s − YOLOv8s | +0.0018 | 1.16 | 0.37 (n.s.) | 0.67 |
| YOLOv8s − CBAM | +0.0053 | 3.33 | 0.079 (n.s.) | 1.92 |
| YOLO11s − CBAM | +0.0071 | 2.52 | 0.13 (n.s.) | 1.45 |

On mAP@[.5:.95], there is no significant difference between any pair, including YOLOv8s versus CBAM (p = 0.079, marginal but not crossing α = 0.05 even without a *multiple-comparison* correction). The reversal between YOLOv8s and YOLO11s across metrics confirms that YOLOv8s's advantage is not consistent across all evaluation dimensions.

This answers RQ2: no attention mechanism statistically meaningfully surpasses the pure CNN under these conditions. The CBAM result on mAP@0.5 is significant and shows a decrease, but it is limited to a single implementation configuration (P3/P4/P5 placement, a single *learning rate*, no separate warmup). For YOLO11s, there is not enough evidence of a difference from the baseline, in either the better or the worse direction.

## V. Discussion

The jump from about 0.10 (zero-shot) to about 0.96 (fine-tuned) confirms that COCO weights as-is are far from ready to use for helmet detection. Two of the three classes (`helmet` and `license_plate`) do not even have a counterpart concept in COCO. The practical implication is clear: every deployment must budget for fine-tuning on domain data, and any performance claim must be reported after domain adaptation, not from a *zero-shot* test. Fine-tuning here is mandatory, not optional.

Among the fine-tuned models, no attention is consistently superior across all metrics. YOLOv8s leads on mAP@0.5, but YOLO11s is slightly ahead on mAP@[.5:.95], and the two do not differ significantly. This pattern aligns with the literature showing that the benefit of attention depends strongly on dataset scale and heterogeneity. On a small and relatively easy dataset (baseline around 0.96), the room for improvement is thin and insufficient to reveal an attention advantage.

Adding CBAM in the configuration tested lowers mAP@0.5 significantly. We offer several hypotheses that cannot be eliminated. First, randomly initialized CBAM blocks can disturb mature COCO features when trained with a single *learning rate* without a separate warmup. Second, placement at P3/P4/P5 may not be optimal for this dataset. Third, the ceiling effect near 0.96 makes even a small decrease appear consistent and statistically significant. This finding cannot be generalized to all CBAM implementations, since a different configuration (other placements, a separate *learning rate*, *from-scratch* training, or *warmup*) may produce different results. This is a result for one specific configuration, not a general statement about CBAM.

Comparing partial and explicit attention, YOLO11s (C2PSA) is statistically equivalent to the baseline on mAP@0.5, and slightly exceeds it on mAP@[.5:.95] without reaching significance. This pattern indicates that attention designed and trained as an integral part of the architecture (C2PSA) is more neutral than an attention module attached post-hoc to a pre-trained *backbone* (CBAM in the configuration tested). Even so, neither gives a convincing advantage on this dataset.

For ETLE or CCTV deployment scenarios that require real-time inference, this study suggests three things. Domain fine-tuning is an absolute prerequisite. Under good lighting and with objects that are not too small, YOLOv8s delivers competitive performance with lower latency. However, this conclusion is limited to the conditions tested, namely a daytime dataset, side views, and adequate quality, so it may not hold for more challenging scenarios. It should also be noted that an automatic helmet-detection system linked to license plates carries privacy and legal implications, so deployment within enforcement infrastructure requires an appropriate regulatory and ethical framework.

## VI. Threats to Validity and Limitations

- *Dataset generalizability.* The findings are tested on a single small and relatively easy dataset (baseline around 0.96; daytime conditions, limited angles). The conclusions may differ on large, diverse, or occlusion-heavy datasets, which are precisely the conditions where attention tends to be more useful.
- *Statistical power.* With n = 3 *seeds* and a ceiling effect near 0.96, the power to detect small improvements is very limited. A retrospective power analysis indicates that to approach 80% power on an effect of Δ = 0.002 (the YOLOv8s vs YOLO11s difference), n > 10 *seeds* would be needed. The claim that YOLO11s equals YOLOv8s is an absence of evidence of a difference, not evidence of equivalence; p = 0.11 with n = 3 is not enough for an equivalence claim.
- *CBAM ablation scope.* CBAM was tested in a single placement configuration (P3/P4/P5) with a single *learning rate* and no separate warmup. Other placements, *from scratch* training, or a separate *learning rate* for the attention module have not been explored and may produce different results.
- *Vanilla baseline interpretation.* Because `helmet` and `license_plate` are zero *by design*, the *zero-shot* baseline measures the domain gap, not intrinsic helmet-detection ability. The `motorcycle` to `motorcyclist` mapping also carries semantic mismatch, and architecture differences under this condition are not meaningful.
- *A single model size.* Only the *small* variant was compared, so behavior on *medium* or *large* variants may differ.
- *FPS as a rough indicator.* FPS was measured on a single run without a formal *warmup*, without a *multiple-batch benchmark*, and without isolating GPU load. Its value is indicative, not a benchmark reproducible across systems.
- *Privacy and deployment ethics.* A system that combines helmet detection with license plates could be used to identify individuals. Deployment within law-enforcement infrastructure requires an appropriate regulatory framework and ethical oversight, which is outside the scope of this technical study.

## VII. Conclusion and Recommendations

This study separates two sources of improvement that are often mixed in the helmet-detection literature, using a methodology that prioritizes experimental control and reporting care. Its main contribution is methodological: a multi-*seed* evaluation with an identical protocol, a separation between the fine-tuning effect and the attention effect, and reporting on two metrics with formal statistical testing.

The key findings are as follows. The domain gap from the COCO starting point is very large (mAP@0.5 around 0.10 under *zero-shot*), so fine-tuning is mandatory. After fine-tuning with an identical protocol, no attention mechanism consistently surpasses the pure CNN. YOLOv8s leads on mAP@0.5 (0.9596), while YOLO11s is slightly ahead on mAP@[.5:.95] (0.6902), and the two do not differ significantly. CBAM in the configuration tested lowers mAP@0.5 significantly (p = 0.004), but this finding is limited to one specific implementation.

The main limitations, namely n = 3, a single easy dataset, and a single CBAM configuration, constrain generalization. Priority next steps cover four areas: (i) a study with n ≥ 5 *seeds* and an *a priori* power analysis; (ii) systematic CBAM ablation (placement, *from scratch*, separate *learning rate*) to isolate the source of the decrease; (iii) testing on larger and more challenging datasets (occlusion, night-time, long distance) that likely give attention mechanisms more room; and (iv) integrating a two-stage pipeline (motorcycle then helmet detection) and plate recognition for enforcement scenarios.

## Statements

**Data and code availability.** The dataset is public (Roboflow NCKH 2023 v19, MIT license). The experimental pipeline, per-run metrics, and figure-generation scripts are available in the project repository (notebooks `01` through `03`).

**Use of AI.** Drafting and analysis were assisted by AI tools; all numbers, claims, and interpretations were verified by the authors against the experimental outputs. The authors take full responsibility for the content.

**Conflict of interest.** The authors declare no conflict of interest.

**Ethics note.** A helmet-detection system integrated with plate reading carries privacy and individual-identification implications. The technology described in this study is intended for technical research; use in a law-enforcement context must observe the applicable regulatory framework and consent.

## References

[1] F. W. Siebert and H. Lin, "Detecting motorcycle helmet use with deep learning," *Accident Analysis & Prevention*, vol. 134, p. 105319, 2020.

[2] H. Lin, J. D. Deng, D. Albers, and F. W. Siebert, "Helmet use detection of tracked motorcycles using CNN-based multi-task learning," *IEEE Access*, vol. 8, pp. 162073–162084, 2020.

[3] M. Naphade et al., "The 7th AI City Challenge," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023.

[4] W. Jia et al., "Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector," *IET Image Processing*, vol. 15, no. 14, pp. 3623–3637, 2021.

[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics. Accessed: Jun. 2026.

[7] Y. Zhao et al., "DETRs beat YOLOs on real-time object detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024.

[8] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 213–229.

[9] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. European Conf. Computer Vision (ECCV)*, 2018, pp. 3–19.

[10] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.

[11] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[12] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[13] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980–2988.

[14] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. European Conf. Computer Vision (ECCV)*, 2014, pp. 740–755.

[15] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 10012–10022.

[16] Y. Li, H. Mao, R. Girshick, and K. He, "Exploring plain vision transformer backbones for object detection," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 280–296.

[17] M.-H. Guo et al., "Attention mechanisms in computer vision: A survey," *Computational Visual Media*, vol. 8, no. 3, pp. 331–369, 2022.

[18] Z. Niu, G. Zhong, and H. Yu, "A review on the attention mechanism of deep learning," *Neurocomputing*, vol. 452, pp. 48–62, 2021.

[19] H. Zhang, Q. Luo, and C. Yin, "YOLOv8 safety helmet detection algorithm based on attention mechanism and feature fusion," *J. Phys.: Conf. Ser.*, vol. 3135, no. 1, p. 012025, 2025.

[20] J. Li, X. Zhang, and Z. Liu, "Safety helmet detection based on improved YOLOv5s with attention mechanism," *Sensors*, vol. 23, no. 14, p. 6500, 2023.

[21] M. H. D. Hariyono, A. N. Ihsan, and D. S. Kusumo, "Streamlit application for helmet detection based on YOLOS: Case study Indonesia," in *Proc. Data Science and Its Applications Conf. (DASA)*, 2024.

[22] W. Jia et al., "DAAM-YOLOv5: Helmet detection combined with attention mechanism," *Electronics*, vol. 12, no. 9, p. 2094, 2023.

[23] G. Cheng, X. Yuan, X. Yao, K. Yan, Q. Zeng, and J. Han, "Towards large-scale small object detection: Survey and benchmarks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 1, pp. 521–539, 2024.

[24] R. D. Raj and M. S. Nair, "A YOLO-based approach to detecting helmetless riders through CCTV," *SITECH: J. Inf. Technol.*, 2024.
