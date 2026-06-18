---
title: "A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcycle-Helmet Use Detection"
author: "Author (to be adjusted)"
date: "2026"
lang: en
---

# A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcycle-Helmet Use Detection

*Paper draft — IMRaD format, IEEE citation style. Prepared with the assistance of ARS academic-paper (full mode).*

---

## Abstract (Indonesian)

Helmet use is the primary protective factor for motorcyclists, and the automatic detection of compliance through traffic cameras can support the enforcement of road-safety regulations. Attention mechanisms have become a dominant trend in computer vision, yet their benefit for helmet detection on small-scale datasets has not been widely examined under controlled settings. This study compares four object-detection architectures that occupy the attention-usage spectrum: YOLOv8s (pure CNN, no attention), YOLO11s (partial attention through C2PSA blocks), RT-DETR-l (a transformer with full attention), and YOLOv8s+CBAM (a CNN with a channel-spatial attention module). All models were trained on a public helmet-detection dataset comprising 1,803 images with three classes (*helmet*, *license_plate*, *motorcyclist*), using an identical protocol (transfer learning from COCO, 1280 resolution, the same augmentation) and evaluated on a held-out test split with several seeds and paired t-tests for statistical validity. As a result, under the unified training protocol, no attention variant surpassed the plain YOLOv8s: YOLOv8s achieved the highest mAP@0.5 (0.9592 ± 0.0014) together with the best inference speed (~296 FPS). A paired t-test across seeds showed that adding the CBAM module significantly reduced mAP@0.5 by 0.0113 (t(2) = 10.9; p = 0.008), whereas YOLO11s and RT-DETR-l did not differ significantly from the baseline. Because the baseline already approaches 0.96 (a ceiling effect), the statistical power to detect small improvements is limited, so equivalence claims—particularly for RT-DETR-l, which was tested on only two seeds—must be interpreted with caution. These findings indicate that on a small, relatively easy dataset and under hyperparameters tuned for YOLO, the added complexity of attention does not pay off; a simple CNN architecture remains the most efficient choice for real-time helmet detection under these conditions.

**Keywords:** helmet detection; object detection; attention mechanism; YOLO; RT-DETR; CBAM; motorcycle.

## Abstract (English)

Helmet use is the primary protective factor for motorcyclists, and automatic compliance detection through traffic cameras can support road-safety enforcement. Attention mechanisms have become a dominant trend in computer vision, yet their benefit for helmet detection on small-scale datasets remains under-examined in controlled settings. This study compares four object-detection architectures spanning the attention spectrum: YOLOv8s (pure CNN, no attention), YOLO11s (partial attention via C2PSA blocks), RT-DETR-l (a transformer with full attention), and YOLOv8s+CBAM (a CNN augmented with a channel-spatial attention module). All models were trained on a public helmet-detection dataset of 1,803 images with three classes (helmet, license_plate, motorcyclist) under an identical protocol (COCO transfer learning, 1280 resolution, matched augmentation) and evaluated on a held-out test split with several seeds and paired t-tests for statistical validity. Under the unified training protocol, no attention variant surpassed the plain YOLOv8s: it achieved the highest mAP@0.5 (0.9592 ± 0.0014) and the best inference speed (~296 FPS). A paired t-test across seeds showed that adding the CBAM module significantly reduced mAP@0.5 by 0.0113 (t(2) = 10.9; p = 0.008), whereas YOLO11s and RT-DETR-l did not differ significantly from the baseline. Because the baseline already approaches 0.96 (a ceiling effect), statistical power to detect small improvements is limited, so the equivalence claims—particularly for RT-DETR-l, tested on only two seeds—must be read with caution. The findings indicate that on a small, relatively easy dataset and under YOLO-tuned hyperparameters, the added complexity of attention does not pay off; a simple CNN architecture remains the most efficient choice for real-time helmet detection under these conditions.

**Keywords:** helmet detection; object detection; attention mechanism; YOLO; RT-DETR; CBAM; motorcycle.

---

## I. Introduction

Motorcycles account for a large proportion of traffic-accident casualties, particularly in developing countries where this mode dominates daily mobility. The helmet is the most decisive protective factor against fatal head injuries, so monitoring helmet-use compliance is an important target of safety policy. Manual observation does not scale to real traffic volumes: officers cannot possibly monitor every intersection at all times, and momentary surveys yield a biased picture. This is where automatic object detection from camera footage offers a way out that can operate continuously and consistently [1], [2].

Over the past decade, object detection has shifted from accurate but slow two-stage approaches to fast single-stage detectors, with the YOLO family as the de-facto standard for real-time applications [5]. In parallel, attention mechanisms—born from the Transformer [10] and adapted to images through the Vision Transformer [11]—have transformed the architectural landscape. Transformer-based detectors such as DETR [8] and its real-time derivative RT-DETR [7] now compete with CNNs, while lightweight attention modules such as Squeeze-and-Excitation [12] and CBAM [9] have become a popular way to insert attention into existing CNN backbones.

This trend gives rise to an implicit assumption that adding attention tends to improve performance. That assumption is reasonable on large-scale benchmarks such as COCO, where the diversity and volume of data give attention room to learn useful long-range dependencies. In the practical context of helmet detection, however—which often uses small-scale, curated datasets with a limited number of classes—it is not clear whether the complexity of attention truly pays off. Many reports compare a single model variant without strict experimental control, without repetition across seeds, and with differing training protocols. As a result, the reported performance differences could stem from hyperparameters, augmentation, or luck in initialization—rather than from the attention mechanism itself.

This study closes that gap with a single focused research question: **does an attention mechanism—whether partial, a full transformer, or the CBAM module—improve the accuracy of motorcyclist helmet-use detection compared with a pure CNN on a small-scale dataset?** To answer it, we devised a controlled comparison of four architectures deliberately chosen to occupy different points on the attention-usage spectrum, with a uniform training and evaluation protocol and repetition across seeds to assess practical significance.

The main contributions of this paper:

1. **A controlled comparison** of four detection architectures (YOLOv8s, YOLO11s, RT-DETR-l, YOLOv8s+CBAM) on the helmet-detection task with identical hyperparameters, data, and augmentation, so that performance differences can be attributed to architectural factors, particularly the presence and type of attention.
2. **A multi-seed evaluation** with reporting of mean ± standard deviation, so that the conclusions are not vulnerable to the luck of a single initialization.
3. **Counterintuitive empirical evidence**: on this dataset, no attention mechanism surpassed the pure CNN, and the CBAM module even reduced accuracy consistently. We trace the cause and discuss its implications for the practice of model selection.

## II. Literature Review

### A. Helmet-use detection

Research on helmet detection has evolved from simple classification toward pipelines capable of tracking motorcycles across frames and distinguishing drivers from passengers. Siebert and Lin [1] demonstrated large-scale helmet-use detection from Myanmar traffic video using a deep-learning approach with a single-stage detector, and released an annotated dataset that has become a reference for the community. Lin et al. [2] extended this with CNN-based multi-task learning that tracks individual motorcycles while registering per-rider helmet use, thereby handling pillion-passenger cases and distinguishing drivers from passengers.

Benchmark challenges such as the AI City Challenge 2023 Track 5 [3] have driven attention toward multi-class helmet-violation detection, including distinguishing the helmet status of the driver and of the first and second passengers. This fine-grained class scheme exposes the problem of extreme class imbalance, since categories such as a second passenger without a helmet appear very rarely. In the enforcement context, a number of approaches combine helmet detection with license-plate localization and character recognition to identify violators [4], often supplemented with detection of pillion riding by more than two people.

### B. Single-stage detectors and the YOLO family

Single-stage detectors formulate detection as direct regression of bounding boxes and classes in a single network pass, trading some accuracy for high speed [5], and this accuracy gap is partly narrowed by loss functions such as the focal loss, which handles class imbalance in dense detection [13]. The YOLO family has become the backbone of real-time applications, with iterations that continually improve the backbone, neck, label-assignment strategy, and augmentation. YOLOv8 represents a mature CNN generation without explicit attention modules; its architecture relies on efficient convolution blocks and a feature pyramid to handle multi-scale objects. YOLO11 introduces a positional attention block (C2PSA) into the feature path, making it an interesting midpoint on the attention spectrum—a CNN that adds limited self-attention without fully moving to the transformer paradigm.

### C. Attention mechanisms in computer vision

Attention allows a model to selectively reweight information, highlighting relevant parts and suppressing irrelevant ones. The Transformer [10] introduced self-attention that models global dependencies among elements, and the Vision Transformer [11] proved this paradigm competitive for images when the training data is sufficiently large. In detection, DETR [8] formulates detection as set prediction with a Transformer encoder-decoder and removes hand-crafted components such as anchors and non-maximum suppression. DETR's weaknesses—slow convergence and high computational cost—prompted the emergence of RT-DETR [7], which adapts those ideas to real time through an efficient hybrid encoder. Another direction applies attention at the backbone level: the Swin Transformer [15] introduced an efficient hierarchical windowed attention as a detection backbone, while ViTDet [16] showed that a plain ViT backbone can be used for detection. These approaches generally excel precisely when training data is abundant.

On a different front, lightweight attention modules insert attention into a CNN without replacing the base architecture. Squeeze-and-Excitation [12] provides inter-channel attention by learning the importance weight of each feature channel. CBAM [9] extends this by combining channel attention and spatial attention sequentially, so that the model can emphasize both "what" is important and "where". These modules are attractive because they are cheap, do not add many parameters, and are easy to attach to existing networks. However, their effectiveness depends on the task and the scale of the data: on small datasets, a randomly initialized module must learn from limited signal.

### D. Research gap

The helmet-detection literature tends to report a single model configuration with differing protocols, often without repetition across seeds, so that causal attribution to the attention mechanism becomes weak. When a study reports that an attention-based model outperforms the baseline, it is hard to be sure whether the advantage comes from attention or from differences in resolution, augmentation, number of epochs, or initialization. This study fills that gap through a controlled and repeated comparison that explicitly varies the presence and type of attention as the principal axis, while holding all other factors constant.

## III. Methodology

### A. Dataset

We used a public helmet-detection dataset from Roboflow Universe ("NCKH 2023 / Helmet Detection Project", version 19, MIT license). The dataset contains 1,803 images annotated in YOLO format with three classes: *helmet*, *license_plate*, and *motorcyclist*. A fixed data split was used throughout the experiments so that there was no re-randomization across runs: 1,563 training images, 140 validation, and 100 test. All final metrics were computed on the test split that was never seen during training to avoid data leakage. The annotations cover three classes directly relevant to the enforcement scenario: the rider as context, the helmet as the compliance object, and the license plate as the identification anchor.

### B. Compared architectures

Four architectures were chosen to occupy different points on the attention-usage spectrum (Table I). This selection is deliberate: from a CNN with no attention at all, to a CNN with partial attention, to a transformer with full attention, and finally a controlled attention intervention that only adds a module to the baseline.

**TABLE I. The four architectures and their positions on the attention spectrum.**

| Model | Paradigm | Attention mechanism | Parameters |
|---|---|---|---|
| YOLOv8s | Single-stage CNN | None (baseline) | ~11.17 M |
| YOLO11s | Single-stage CNN | Partial (C2PSA block) | ~9.4 M |
| RT-DETR-l | Transformer | Full (*self-attention*) | ~32 M |
| YOLOv8s+CBAM | CNN + attention module | Channel + spatial (CBAM) | ~11.51 M |

The YOLOv8s+CBAM variant is a custom architecture that forms the core of the controlled attention experiment. Three CBAM modules were inserted at the outputs of the three detection scales (P3, P4, and P5) just before the detection head, so that the model is identical to the plain YOLOv8s except for that added attention. Placement at all three scales ensures attention operates on small features (P3, for objects such as helmets and plates), medium features (P4), and large features (P5). This addition only raises the parameter count from 11.17 million to 11.51 million—about 3%—making it a lightweight, isolated intervention so that any change in performance can be linked directly to the attention module.

### C. Training protocol

To ensure that performance differences can be attributed to architecture, all models were trained with an identical protocol using the Ultralytics framework [6] on top of PyTorch and a single NVIDIA RTX 4090 GPU. Each model was initialized with transfer learning from COCO pretrained weights [14]; for the CBAM variant, the attention layers were randomly initialized while the rest inherited the COCO weights. The uniform configuration included an input resolution of 1280 pixels, automatic batch-size determination according to GPU memory, automatic optimizer selection, the same augmentation (mosaic, HSV shifting, and horizontal flipping), and early stopping with a patience of 25 epochs out of a maximum of 100 epochs.

The high resolution of 1280 pixels was chosen because preliminary analysis showed that the *helmet* and *license_plate* classes are small within the frame, so a low resolution would harm their detection. Reproducibility was enforced by fixing the seed for the *random*, NumPy, and PyTorch libraries in deterministic mode, and by saving the configuration and environment information (library versions, seed, and device) for each run. For statistical validity, each model was retrained on several seeds (42, 0, and 1) and reported as mean ± standard deviation on the test split. RT-DETR-l was run on only two seeds because its training cost is much higher, around 190 minutes per run compared with about 26 minutes for the YOLO variants; this limitation is noted explicitly and accounted for when interpreting its variance.

### D. Evaluation metrics

The primary metrics are mAP@0.5 and mAP@[.5:.95] following object-detection convention, complemented by precision, recall, and inference speed in frames per second (FPS). Because FPS measurements in the multi-seed sweep are affected by the GPU load running back-to-back, the reported FPS figures were taken from a single run while the GPU was idle so as to reflect the true speed. This distinction is important so that speed comparisons are not misleading due to measurement artifacts.

To assess the significance of accuracy differences, we paired the results across models based on the same seed and applied a **paired t-test** on the mAP@0.5 difference, with a significance level of α = 0.05, and reported the 95% confidence interval for the mean difference. Pairing by seed controls for variation in initialization so that the test is more sensitive to architectural effects. Given the small number of seeds (n = 3 for the YOLO variants, n = 2 for RT-DETR-l), this test is indicative: it can confirm a consistent difference, but is underpowered to reject equivalence—a limitation we discuss explicitly in Section VI.

## IV. Results

Table II summarizes the performance of the four architectures on the test split. YOLOv8s achieved the highest mAP@0.5 together with the best inference speed. YOLO11s and RT-DETR-l were comparable in accuracy but slower, whereas YOLOv8s+CBAM was in fact the lowest on mAP@0.5.

**TABLE II. Performance comparison on the test split (mean ± standard deviation).**

| Model | n *seeds* | mAP@0.5 | mAP@[.5:.95] | FPS |
|---|---|---|---|---|
| **YOLOv8s** | 3 | **0.9592 ± 0.0014** | 0.6862 ± 0.0046 | **~296** |
| YOLO11s | 3 | 0.9569 ± 0.0046 | **0.6898 ± 0.0030** | ~189 |
| RT-DETR-l | 2 | 0.9588 ± 0.0151 | 0.6764 ± 0.0205 | ~55 |
| YOLOv8s+CBAM | 3 | 0.9479 ± 0.0004 | 0.6810 ± 0.0045 | ~290 |

### A. Effect of the attention mechanism

The principal axis of the study—the presence and type of attention—shows no advantage. Figure 1 visualizes the mAP@0.5 of the four models together with the across-seed standard deviation. YOLOv8s without attention leads. The paired t-test confirms that neither YOLO11s (Δ = +0.0023; t(2) = 0.73; p = 0.54; 95% CI [−0.011; +0.016]) nor RT-DETR-l (Δ = +0.0012; t(1) = 0.11; p = 0.93; 95% CI [−0.140; +0.142]) **differs significantly** from the baseline. RT-DETR-l's very wide confidence interval—due to only two seeds with widely separated results (0.9482 and 0.9695)—indicates an unstable estimate; its equivalence is therefore more an absence of evidence of difference than evidence of equivalence.

![Figure 1. Effect of the attention mechanism on mAP@0.5 on the test split. Error bars show the across-seed standard deviation. No attention variant surpasses YOLOv8s; CBAM is in fact the lowest.](../figures/fig1_attention_mAP50.png)

In contrast to the two models above, adding the CBAM module reduced mAP@0.5 by 0.0113, and this reduction is **statistically significant** (paired t-test: t(2) = 10.9; p = 0.008; 95% CI of the reduction [0.0068; 0.0157]). The consistently small standard deviation in both models (0.0014 for the baseline, 0.0004 for the CBAM variant) confirms that this reduction is stable across all initializations tested. It should be emphasized that this result holds under a training protocol tuned for YOLO; it shows that inserting CBAM is detrimental *under those conditions*, not that attention is universally detrimental.

### B. Per-class performance

For the YOLOv8s baseline, per-class accuracy shows a pattern consistent with the size and distinctiveness of objects (Figure 2). The *motorcyclist* class is the easiest to detect (mAP@0.5 = 0.990) because it is large and prominent in the frame. The *license_plate* class follows (0.969) with its distinctive rectangular shape. The *helmet* class is the most challenging (0.923) because of its small size, its similarity to other objects on the rider's head such as caps or hair, and its variation in color and viewpoint. This pattern confirms that the *helmet* class—which is in fact the most relevant for the application's purpose—is the chief determinant of room for improvement, and any improvement effort should be focused there.

![Figure 2. Per-class performance of YOLOv8s on the test split. The helmet class is the most challenging due to its small size and visual ambiguity.](../figures/fig2_per_class.png)

### C. Speed trade-off

The differences in speed are far larger than the differences in accuracy (Figure 3). RT-DETR-l runs at about 55 FPS, roughly five times slower than YOLOv8s, which reaches about 296 FPS, and demands a training time of about 190 minutes per run compared with about 26 minutes for YOLO. YOLO11s sits in the middle at about 189 FPS, while the CBAM variant is almost as fast as the baseline because the added module is lightweight. When the accuracy of the four models is practically equal, the computational cost of the transformer does not pay off on this task. Figure 3 places YOLOv8s in the most advantageous corner: the highest accuracy together with the highest speed.

![Figure 3. Trade-off between accuracy (mAP@0.5) and speed (FPS). YOLOv8s occupies the most advantageous position, leading on both axes simultaneously.](../figures/fig3_accuracy_speed.png)

### D. Qualitative analysis

Figure 4 shows an example of YOLOv8s detection output on a single test image. The model detects the rider, the helmet, and the license plate simultaneously with tight bounding boxes, illustrating a typical case where all three classes are present in a single scene. Visual inspection of the test samples shows that the remaining errors tend to appear on very small or partially occluded helmets, consistent with the quantitative finding that the *helmet* class is the most difficult.

![Figure 4. Example of YOLOv8s detection output on a test image: rider, helmet, and license plate detected simultaneously.](../figures/fig4_detection_example.png)

## V. Discussion

The central finding of this study runs counter to the common intuition that attention tends to help: on a small-scale helmet-detection dataset, no attention mechanism surpassed the pure CNN, and the CBAM module was even detrimental. Three complementary explanations can account for this pattern.

First, **the dataset is relatively small and easy**. With 1,563 training images and a baseline that already reaches about 0.96 mAP@0.5, the room for improvement is very narrow. When a simple model already approaches the upper bound achievable on these data, there is almost no gap left for a more complex mechanism to demonstrate an advantage. Attention mechanisms—especially data-hungry transformers—require large volume and diversity to learn useful dependency patterns. In the small-data regime, their theoretical advantage does not materialize and the model instead bears the burden of unused capacity.

Second, **randomly initialized attention layers can disrupt pretrained features**. In the CBAM variant, most weights are inherited from COCO while the attention module starts from scratch. This untrained module inserts a transformation that, at the start of training, disrupts the mature feature flow from transfer learning. With limited data, there is not enough gradient signal to recover from that disruption, let alone surpass the baseline. The result is a small but consistent reduction that is observed to be stable across all seeds. This explanation is consistent with the common observation that added modules are most beneficial when trained jointly from the start on large data, rather than attached to a pretrained network with limited data.

Third, **complexity is not free**. Attention adds parameters and computation. When there is no accuracy gain, this addition becomes purely a cost. The consequence is most evident in RT-DETR-l, which is five times slower without any mAP return, but also applies conceptually to CBAM, which adds a computation path at every detection scale. In a real-time setting where the computational budget is limited, this cost means a reduction in the number of frames that can be processed per second.

These findings can be placed in the context of the broader literature. The advantages of transformer architectures and attention modules are most consistently reported on large-scale benchmarks with high diversity. On specialized tasks with limited data, a number of studies find that a well-tuned CNN remains competitive or superior. Our results add evidence in that direction specifically for the helmet-detection domain, and underscore that architecture selection should be guided by the characteristics of the data and the computational budget, not by architectural trends alone.

It is important to delimit this claim so as not to overgeneralize. This study does **not** conclude that attention is generally useless. What we show is narrower: under one training protocol tuned for YOLO, on one small and relatively easy dataset, and with a baseline already approaching the ceiling, adding attention does not improve accuracy and—in the case of CBAM—actually reduces it. The equivalence of YOLO11s and RT-DETR-l is an absence of evidence of difference, not evidence of equivalence; the ceiling effect suppresses statistical power, so a small benefit from attention, if any, is hard to detect. A plausible counter-hypothesis—that the attention-based models have not been trained with their optimal recipe—cannot be ruled out and indeed becomes an agenda for further work. In other words, these findings are best read as a practical warning against the assumption that "adding attention must help," not as a verdict on attention itself.

The practical implication, within that scope, is clear: for real-time helmet detection on similar data, a simple CNN architecture such as YOLOv8s is the most rational choice—the highest accuracy, the lightest, and the fastest. These findings also serve as a methodological reminder that claims of architectural superiority must be tested under controlled and repeated conditions before being adopted. Without repetition across seeds, small differences such as those observed here are easily misread as a real advantage when in fact they lie within the range of random variation.

## VI. Threats to Validity and Limitations

This study has several limitations that constrain the generalization of its findings. First, the evaluation is limited to **a single, relatively clean dataset**; generalization to real-world conditions—street CCTV footage with occlusion, fast motion, poor lighting, and high density—has not been tested. It may well be that on more challenging data, attention's ability to model context becomes useful after all. Second, **RT-DETR-l was run on only two seeds** because of computational cost, so its variance estimate is less precise and its equivalence conclusion must be interpreted with caution. Third, the comparison is **not size-matched**: RT-DETR-l (~32 million parameters) is far larger than the YOLO-s variants (~9–11 million), so the results reflect a comparison of standard architectures rather than equal capacity; a comparison equalizing the number of parameters could give a different picture. Fourth, the hyperparameters were unified and not specifically tuned per architecture, even though transformers and CNNs may demand different training recipes to reach their peak potential; as a result, the attention-based models' results may be a lower bound rather than their true potential. Fifth, **the ceiling effect suppresses statistical power**: with the baseline already at ~0.96, the room for improvement is narrow and the number of seeds is small (n = 2–3), so the t-test is underpowered to detect small improvements. Consequently, equivalence claims (YOLO11s, RT-DETR-l) must be read as an absence of evidence of difference, not evidence of equivalence; concluding "attention does not help" in absolute terms would be a leap from *absence of evidence* to *evidence of absence*. Sixth, FPS measurement in the multi-seed sweep is affected by GPU load; the speed figures were taken from a single run to mitigate this bias, but they remain estimates on one hardware configuration.

## VII. Conclusion and Recommendations

We compared four object-detection architectures occupying the attention-usage spectrum for motorcyclist helmet-use detection, under an identical and repeated training and evaluation protocol. Under a protocol tuned for YOLO and on a single small, relatively easy dataset, no attention mechanism—partial, full transformer, or the CBAM module—surpassed the pure-CNN YOLOv8s. YOLOv8s delivered the highest accuracy together with the best speed, while adding CBAM significantly reduced mAP@0.5 (paired t-test, p = 0.008); the equivalence of YOLO11s and RT-DETR-l is interpreted as an absence of evidence of difference, not evidence of equivalence, given the ceiling effect. Within this scope, the complexity of attention does not pay off—a practical warning against the assumption that adding attention must help, not a general verdict on attention mechanisms.

The recommendations for further work follow directly from the limitations above. First, to test generalization on a larger, more diverse, and more challenging dataset, including real CCTV footage with difficult conditions. Second, to evaluate attention in a large-data regime where its advantages have a chance to emerge, as well as with training from scratch rather than merely attaching a module to a pretrained network. Third, to explore a two-stage approach—motorcycle detection followed by helmet classification—and the integration of license-plate detection for enforcement scenarios. Fourth, to conduct a size-matched comparison and per-architecture hyperparameter fine-tuning so that each model is tested under its best conditions.

---

## Statements

**Data Availability.** The dataset is public via Roboflow Universe ("NCKH 2023 / Helmet Detection Project" v19, MIT license). The training scripts, configurations, and experiment notebooks are available in the authors' research repository.

**Ethics Statement.** The study uses a public, openly licensed dataset without personal data directly identifiable outside the public-traffic context; it does not involve human subjects or experiments requiring ethical approval.

**Author Contributions (CRediT).** Conceptualization, methodology, software, analysis, and manuscript writing: the author. (To be adjusted to the actual list of authors.)

**Conflict of Interest.** The author declares no conflict of interest.

**Funding.** No specific funding was reported for this study.

**AI Disclosure.** The design of the experiments, the analysis, and the writing of this manuscript were assisted by AI-based tools (Claude Code together with the Academic Research Skills skill). All methodological decisions, verification of numerical results, and ultimate responsibility for the content rest with the author. The cited references were independently verified for their existence.

---

## References

[1] F. W. Siebert and H. Lin, "Detecting motorcycle helmet use with deep learning," *Accident Analysis & Prevention*, vol. 134, p. 105319, 2020.

[2] H. Lin, J. D. Deng, D. Albers, and F. W. Siebert, "Helmet use detection of tracked motorcycles using CNN-based multi-task learning," *IEEE Access*, vol. 8, pp. 162073–162084, 2020.

[3] M. Naphade et al., "The 7th AI City Challenge," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023.

[4] W. Jia et al., "Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector," *IET Image Processing*, vol. 15, no. 14, pp. 3623–3637, 2021.

[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

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
