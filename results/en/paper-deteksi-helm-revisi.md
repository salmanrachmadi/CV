---
title: "A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcyclist Helmet-Use Detection"
author: "Author (to be filled in)"
date: "2026"
lang: en
---

# A Comparison of Attention-Based and Attention-Free Object Detection Architectures for Motorcyclist Helmet-Use Detection

*Revised paper — IMRaD format, IEEE citations. Prepared with the assistance of ARS academic-paper (full mode). Revision assisted by Claude Code.*

---

## Abstract (Indonesian)

Helmet use is the primary protective factor for motorcyclists, and the automatic detection of compliance through traffic cameras can support the enforcement of road-safety regulations. Attention mechanisms have become a dominant trend in computer vision, yet their benefit for helmet detection on small-scale datasets has not been extensively examined under controlled conditions. This study compares four object-detection architectures that occupy a spectrum of attention usage: YOLOv8s (pure CNN, no attention), YOLO11s (partial attention via C2PSA blocks), RT-DETR-l (a transformer with full attention), and YOLOv8s+CBAM (a CNN with a channel-spatial attention module). All models were trained on a public helmet-detection dataset comprising 1,803 images with three classes (*helmet*, *license_plate*, *motorcyclist*), using an identical protocol (transfer learning from COCO, 1280 resolution, identical augmentation) and evaluated on a test split with several seeds as well as paired t-tests for statistical validity. As a result, under a unified training protocol arranged for YOLO, no attention-based variant surpassed the plain YOLOv8s: YOLOv8s attained the highest mAP@0.5 (0.9592 ± 0.0014) together with the best inference speed (~296 FPS). A paired t-test across seeds showed that adding the CBAM module significantly reduced mAP@0.5 by 0.0113 (t(2) = 10.9; p = 0.008; Cohen's *d* ≈ 9.1), whereas YOLO11s and RT-DETR-l did not differ significantly from the baseline (p > 0.05). Because the baseline already approaches 0.96 (a ceiling effect) and the number of seeds is limited (n = 2–3), the statistical power to detect small improvements is severely limited, so the equivalence claims—particularly for RT-DETR-l, which was tested on only two seeds—represent an absence of evidence of a difference rather than evidence of equivalence. These findings hold under the specific conditions tested: a single, small, relatively easy dataset, a training protocol arranged for YOLO, and a resolution of 1280 pixels. Under these conditions, the added complexity of attention does not pay off; a simple CNN architecture remains the most efficient choice for real-time helmet detection.

**Keywords:** helmet detection; object detection; attention mechanism; YOLO; RT-DETR; CBAM; motorcycle.

## Abstract (English)

Helmet use is the primary protective factor for motorcyclists, and automatic compliance detection through traffic cameras can support road-safety enforcement. Attention mechanisms have become a dominant trend in computer vision, yet their benefit for helmet detection on small-scale datasets remains under-examined in controlled settings. This study compares four object-detection architectures spanning the attention spectrum: YOLOv8s (pure CNN, no attention), YOLO11s (partial attention via C2PSA blocks), RT-DETR-l (a transformer with full attention), and YOLOv8s+CBAM (a CNN augmented with a channel-spatial attention module). All models were trained on a public helmet-detection dataset of 1,803 images with three classes (helmet, license_plate, motorcyclist) under an identical protocol (COCO transfer learning, 1280 resolution, matched augmentation) and evaluated on a held-out test split with several seeds and paired t-tests for statistical validity. Under a unified training protocol tuned for YOLO, no attention variant surpassed the plain YOLOv8s: it achieved the highest mAP@0.5 (0.9592 ± 0.0014) and the best inference speed (~296 FPS). A paired t-test across seeds showed that adding the CBAM module significantly reduced mAP@0.5 by 0.0113 (t(2) = 10.9; p = 0.008; Cohen's *d* ≈ 9.1), whereas YOLO11s and RT-DETR-l did not differ significantly from the baseline (p > 0.05). Because the baseline already approaches 0.96 (a ceiling effect) and the number of seeds is limited (n = 2–3), statistical power to detect small improvements is severely limited, so the equivalence claims—particularly for RT-DETR-l, tested on only two seeds—represent an absence of evidence rather than evidence of absence. These findings hold under the specific conditions tested: a single small, relatively easy dataset, a training protocol tuned for YOLO, and 1280-pixel resolution. Under these conditions, the added complexity of attention does not pay off; a simple CNN architecture remains the most efficient choice for real-time helmet detection.

**Keywords:** helmet detection; object detection; attention mechanism; YOLO; RT-DETR; CBAM; motorcycle.

---

## I. Introduction

Motorcycles account for a large proportion of traffic-accident casualties, especially in developing countries where this mode dominates daily mobility. The helmet is the most decisive protective factor against fatal head injuries, so monitoring helmet-use compliance has become an important target of safety policy. Manual observation does not scale to real traffic volumes: officers cannot possibly observe every intersection at all times, and point-in-time surveys provide a biased picture. This is where automatic object detection from camera footage offers a way out that can operate continuously and consistently [1], [2].

Over the past decade, object detection has shifted from accurate but slow two-stage approaches to fast single-stage detectors, with the YOLO family as the de facto standard for real-time applications [5]. In parallel, attention mechanisms—born from the Transformer [10] and adapted to images via the Vision Transformer [11]—have transformed the architectural landscape. Transformer-based detectors such as DETR [8] and its real-time descendant RT-DETR [7] now compete with CNNs, while lightweight attention modules such as Squeeze-and-Excitation [12] and CBAM [9] have become a popular way to inject attention into existing CNN backbones.

The trend of adding attention to helmet-detection models has produced a number of positive reports. Zhang et al. [19] reported that inserting attention modules and feature fusion into YOLOv8 improved helmet detection. Li et al. [20] showed a similar improvement with an attention module on YOLOv5s, and Jia et al. [22] demonstrated that combining deformable attention with YOLOv5 improved helmet-detection performance. These reports consistently claim a benefit from attention, yet they generally compare a single model variant without rigorous experimental controls—without repetition across seeds, without a unified training protocol, and without statistical significance testing. As a result, the reported performance differences could stem from hyperparameters, augmentation, or initialization luck—rather than from the attention mechanism itself.

This trend gives rise to the implicit assumption that adding attention tends to improve performance. That assumption is reasonable on large-scale benchmarks such as COCO, where the diversity and volume of data give attention room to learn useful long-range dependencies. However, in the practical context of helmet detection—which often uses small-scale, curated datasets with a limited number of classes—it is not yet clear whether the complexity of attention truly pays off.

This study closes that gap with a single focused research question: **can the improvements reported by attention-based studies on helmet detection be replicated in a controlled comparison with a unified protocol, repetition across seeds, and significance testing?** This question positions the study as a structured test of the generalizability of the claimed benefits of attention in a controlled, small-scale setting.

The main contributions of this paper are:

1. A **controlled comparison** of four detection architectures (YOLOv8s, YOLO11s, RT-DETR-l, YOLOv8s+CBAM) on the helmet-detection task with identical hyperparameters, data, and augmentation, so that performance differences can be attributed to architectural factors, particularly the presence and type of attention.
2. A **multi-seed evaluation** with reporting of mean ± standard deviation, paired t-tests, Cohen's *d* effect size, and 95% confidence intervals, so that the conclusions are not vulnerable to the luck of a single initialization.
3. A **structured rebuttal** of the assumption that adding attention always helps: on this dataset and under a training protocol arranged for YOLO, no attention mechanism surpassed the pure CNN, and the CBAM module even reduced accuracy significantly and consistently. We trace the cause, discuss rival hypotheses, and summarize the implications for model-selection practice.

## II. Literature Review

### A. Helmet-use detection

Research on helmet detection has evolved from simple classification toward pipelines capable of tracking motorcycles across frames and distinguishing drivers from passengers. Siebert and Lin [1] demonstrated large-scale helmet-use detection from Myanmar traffic video using a deep-learning approach with a single-stage detector, and released an annotated dataset that has become a community reference. Lin et al. [2] extended this with CNN-based multi-task learning that tracks individual motorcycles while registering helmet use per rider, thereby being able to handle pillion-rider cases and distinguish drivers from passengers.

Benchmark challenges such as the AI City Challenge 2023 Track 5 [3] have driven attention toward multi-class helmet-violation detection, including distinguishing the helmet status of the driver and the first and second passengers. This detailed class scheme exposes the problem of extreme class imbalance, since categories such as a helmetless second passenger appear very rarely. In the enforcement context, several approaches combine helmet detection with license-plate localization and character recognition to identify violators [4], often supplemented by detection of more than two riders sharing a motorcycle.

In Indonesia, where motorcycles dominate the transportation mode, automatic detection of helmet compliance has direct relevance. Hariyono et al. [21] demonstrated a YOLOS-based helmet-detection application integrated into a Streamlit interface for enforcement scenarios in Indonesia, showing the potential for practical deployment. Raj and Nair [24] proposed a YOLO-based approach to detect helmetless riders through CCTV footage, emphasizing the occlusion and lighting-variation challenges that are typical in real traffic settings.

### B. Single-stage detectors and the YOLO family

Single-stage detectors formulate detection as direct regression of bounding boxes and classes in a single network pass, trading some accuracy for high speed [5], and this accuracy gap is partly narrowed by loss functions such as focal loss that address class imbalance in dense detection [13]. The YOLO family has become the backbone of real-time applications, with iterations that continually improve the backbone, neck, label-assignment strategy, and augmentation. YOLOv8 represents a mature CNN generation without explicit attention modules; its architecture relies on efficient convolutional blocks and a feature pyramid to handle multi-scale objects. YOLO11 introduces positional attention blocks (C2PSA) into the feature path, making it an interesting midpoint on the attention spectrum—a CNN that adds limited self-attention without fully switching to the transformer paradigm.

### C. Attention mechanisms in computer vision

Attention enables a model to selectively reweight information, highlighting relevant parts and suppressing irrelevant ones. Guo et al. [17] comprehensively surveyed attention mechanisms in computer vision and classified them into several categories: channel, spatial, temporal, and self-attention, each with its own strengths and limitations. The survey by Niu et al. [18] reviewed attention from a broader deep-learning perspective and emphasized that the effectiveness of attention depends heavily on the volume of training data and the complexity of the task.

The Transformer [10] introduced self-attention that models global dependencies among elements, and the Vision Transformer [11] proved this paradigm competitive for images when the training data is sufficiently large. In detection, DETR [8] formulates detection as set prediction with a Transformer encoder-decoder and removes hand-crafted components such as anchors and non-maximum suppression. DETR's weaknesses—slow convergence and high computational cost—prompted the emergence of RT-DETR [7], which adapts the idea into a real-time form through an efficient hybrid encoder. Another direction applies attention at the backbone level: the Swin Transformer [15] introduced an efficient hierarchical windowed attention as a detection backbone, while ViTDet [16] showed that a plain ViT backbone can be used for detection. These approaches generally excel precisely when training data is abundant.

On a different side, lightweight attention modules inject attention into a CNN without replacing the base architecture. Squeeze-and-Excitation [12] provides inter-channel attention by learning the importance weight of each feature channel. CBAM [9] extends this by combining channel attention and spatial attention sequentially, so that the model can emphasize "what" is important as well as "where." These modules are attractive because they are cheap, do not add many parameters, and are easy to attach to existing networks. However, their effectiveness depends on the task and the scale of the data: on small datasets, a randomly initialized module must learn from limited signal.

For the task of small-object detection, Cheng et al. [23] surveyed large-scale approaches and emphasized that small objects such as helmets and license plates in wide frames pose a particular challenge requiring multi-scale feature strategies, high resolution, and specially designed augmentation. The findings of this survey are relevant to our study, given that the *helmet* and *license_plate* classes in the dataset used are relatively small within the frame.

### D. Attention in helmet detection

Several recent studies have specifically integrated attention mechanisms into helmet-detection models. Zhang et al. [19] proposed modifications to YOLOv8 with the addition of attention modules and an extended feature-fusion strategy, and reported an mAP improvement on their helmet dataset. Li et al. [20] inserted a channel-spatial attention module into YOLOv5s and reported an improvement in helmet-detection accuracy. Jia et al. [22] introduced deformable attention (DAAM) into the YOLOv5 architecture and demonstrated an improvement on the helmet-detection task.

Although these reports consistently show a benefit from attention, there is a methodological pattern worth noting. First, most studies report results from a single configuration without repetition across seeds, so it cannot be concluded whether the reported improvement is consistent across initializations. Second, the training protocol—including augmentation, resolution, number of epochs, and learning rate—is generally tuned separately per model, so the performance differences reflect the combined effect of architecture and hyperparameters rather than attention alone. Third, no study reports statistical significance testing of the performance differences. This gap is the direct motivation for our study, which is designed to isolate the contribution of the attention mechanism through a unified protocol, multi-seed repetition, and formal statistical testing.

### E. Research gap

The helmet-detection literature tends to report a single model configuration with varying protocols, often without repetition across seeds, so the causal attribution to the attention mechanism is weak. When a study reports that an attention-based model outperforms a baseline, it is difficult to be certain whether that advantage comes from attention or from differences in resolution, augmentation, number of epochs, or initialization. The studies reviewed in Section II.D are no exception: they report a benefit from attention but without the experimental controls needed to attribute the cause firmly. This study fills that gap through a controlled, repeated comparison that explicitly varies the presence and type of attention as the main axis, while holding all other factors constant, including hyperparameters, augmentation, data, and input resolution.

## III. Methodology

### A. Dataset

We use a public helmet-detection dataset from Roboflow Universe ("NCKH 2023 / Helmet Detection Project," version 19, MIT license). The dataset contains 1,803 annotated images in YOLO format with three classes: *helmet*, *license_plate*, and *motorcyclist*. A fixed data split was used throughout the experiments to ensure no re-randomization across runs: 1,563 training images, 140 validation images, and 100 test images. All final metrics were computed on the test split, which was never seen during training, to avoid data leakage. The annotations cover three classes directly relevant to the enforcement scenario: the rider as context, the helmet as the compliance object, and the license plate as an identification anchor.

### B. Architectures compared

Four architectures were chosen so as to occupy different points on the spectrum of attention usage (Table I). This selection is deliberate: from a CNN with no attention at all, to a CNN with partial attention, to a transformer with full attention, and finally a controlled attention intervention that merely adds a module to the baseline.

**TABLE I. The four architectures and their position on the attention spectrum.**

| Model | Paradigm | Attention mechanism | Parameters |
|---|---|---|---|
| YOLOv8s | Single-stage CNN | None (baseline) | ~11.17 M |
| YOLO11s | Single-stage CNN | Partial (C2PSA blocks) | ~9.4 M |
| RT-DETR-l | Transformer | Full (*self-attention*) | ~32 M |
| YOLOv8s+CBAM | CNN + attention module | Channel + spatial (CBAM) | ~11.51 M |

The YOLOv8s+CBAM variant is a custom architecture that is the core of the controlled attention experiment. Three CBAM modules were inserted at the outputs of the three detection scales (P3, P4, and P5) just before the detection head, so that the model is identical to the plain YOLOv8s except for the addition of that attention. Placement at all three scales ensures that attention operates on small features (P3, for objects such as helmets and plates), medium features (P4), and large features (P5). This addition only raises the parameter count from 11.17 million to 11.51 million—about 3%—making it a lightweight and isolated intervention so that any change in performance can be linked directly to the attention module.

It should be noted that RT-DETR-l (~32 million parameters) is far larger than the YOLO-s variants (~9–11 million), so this comparison mixes the factors of architectural paradigm and model capacity. This choice of standard models is deliberate, to reflect the practical choices faced by practitioners, but this limitation is acknowledged and discussed explicitly in Section VI.

### C. Training protocol

To ensure that performance differences can be attributed to the architecture, all models were trained with an identical protocol using the Ultralytics framework [6] on top of PyTorch and a single NVIDIA RTX 4090 GPU. Each model was initialized with transfer learning from COCO-pretrained weights [14]; for the CBAM variant, the attention layers were randomly initialized while the rest inherited COCO weights. The uniform configuration included an input resolution of 1280 pixels, automatic batch-size determination according to GPU memory, automatic optimizer selection, identical augmentation (mosaic, HSV shifting, and horizontal flipping), and early stopping with a patience of 25 epochs out of a maximum of 100 epochs.

The high resolution of 1280 pixels was chosen because preliminary analysis showed that the *helmet* and *license_plate* classes are small within the frame, so a low resolution harms their detection. Reproducibility was enforced by fixing the seed for the *random*, NumPy, and PyTorch libraries in deterministic mode, as well as saving the configuration and environment information (library versions, seed, and device) for each run. For statistical validity, each model was retrained on several seeds (42, 0, and 1) and the mean ± standard deviation on the test split was reported. RT-DETR-l was run on only two seeds because its training cost is far higher, about 190 minutes per run compared with about 26 minutes for the YOLO variants; this limitation is noted explicitly and taken into account when interpreting its variance.

### D. Evaluation metrics and statistical testing protocol

The primary metrics are mAP@0.5 and mAP@[.5:.95] following object-detection convention, supplemented by precision, recall, and inference speed in frames per second (FPS). Because FPS measurement in a multi-seed sweep is affected by the GPU load running back to back, the reported FPS figures were taken from a single run when the GPU was idle (no other training or inference process running), using the built-in Ultralytics benchmark on the full test split, so that they reflect the true speed on the hardware configuration used. This distinction is important so that the speed comparison is not misleading due to measurement artifacts.

To assess the significance of accuracy differences, we paired the results between models on the same seed and applied a **paired t-test** to the mAP@0.5 difference of each seed pair, with a significance level of α = 0.05. In addition to the *p*-value, we report the Cohen's *d* effect size as well as the 95% confidence interval (95% CI) for the mean difference. Pairing by seed controls for initialization variation so that the test is more sensitive to the architectural effect. Given the small number of seeds (n = 3 for the YOLO variants, n = 2 for RT-DETR-l), this test is indicative: it can confirm consistent differences, but it has low power to reject equivalence—a limitation we discuss explicitly in Section VI.

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

The main axis of the study—the presence and type of attention—shows no advantage under a training protocol arranged for YOLO. Figure 1 visualizes the mAP@0.5 of the four models along with the standard deviation across seeds. YOLOv8s without attention leads.

**TABLE III. Paired t-test results for the mAP@0.5 difference.**

| Comparison | Δ mAP@0.5 | t(df) | p | Cohen's *d* | 95% CI |
|---|---|---|---|---|---|
| YOLOv8s vs YOLO11s | +0.0023 | t(2) = 0.73 | 0.54 | 0.61 | [−0.011; +0.016] |
| YOLOv8s vs RT-DETR-l | +0.0004 | t(1) = 0.11 | 0.93 | — | [−0.140; +0.142] |
| YOLOv8s vs CBAM | +0.0113 | t(2) = 10.9 | 0.008 | 9.1 | [0.0068; 0.0157] |

Table III presents the paired t-test results for the mAP@0.5 difference between the baseline and each variant. The paired t-test confirms that neither YOLO11s (Δ = +0.0023; t(2) = 0.73; p = 0.54; Cohen's *d* = 0.61; 95% CI [−0.011; +0.016]) nor RT-DETR-l (Δ = +0.0004; t(1) = 0.11; p = 0.93; 95% CI [−0.140; +0.142]) **differs significantly** from the baseline. The very wide confidence interval for RT-DETR-l—due to only two seeds with far-apart results (0.9482 and 0.9695)—indicates an unstable estimate; its equivalence is therefore more an absence of evidence of a difference than evidence of equivalence. It should be noted that Cohen's *d* for YOLO11s vs baseline (0.61) indicates a medium-sized effect by convention, but it does not reach significance because of the very low statistical power with n = 3.

In contrast to the two models above, adding the CBAM module reduced mAP@0.5 by 0.0113, and this reduction is **statistically significant** (paired t-test: t(2) = 10.9; p = 0.008; Cohen's *d* ≈ 9.1; 95% CI of the reduction [0.0068; 0.0157]). The very large Cohen's *d* indicates a substantial and consistent effect. The small standard deviation in both models (0.0014 for the baseline, 0.0004 for the CBAM variant) confirms that this reduction is stable across the initializations tested. It must be emphasized that this result holds under a training protocol arranged for YOLO; it shows that inserting CBAM is detrimental *under those conditions*, not that attention is universally detrimental.

![Figure 1. Effect of the attention mechanism on mAP@0.5 on the test split. Error bars show the standard deviation across seeds. No attention-based variant surpasses YOLOv8s; CBAM is in fact the lowest.](../figures/fig1_attention_mAP50.png)

### B. Per-class performance

On the YOLOv8s baseline, the per-class accuracy shows a pattern consistent with the size and distinctiveness of the objects (Figure 2). The *motorcyclist* class is the easiest to detect (mAP@0.5 = 0.990) because it is large and prominent in the frame. The *license_plate* class follows (0.969) with its distinctive rectangular shape. The *helmet* class is the most challenging (0.923) because of its small size, its resemblance to other objects on the rider's head such as caps or hair, and its variation in color and viewing angle. This pattern confirms that the *helmet* class—which is in fact the most relevant for the application's purpose—is the main determinant of the room for improvement, and any improvement effort should be focused there.

![Figure 2. Per-class performance of YOLOv8s on the test split. The helmet class is the most challenging because of its small size and visual ambiguity.](../figures/fig2_per_class.png)

### C. Speed trade-off

The speed difference is far larger than the accuracy difference (Figure 3). RT-DETR-l runs at about 55 FPS, roughly five times slower than YOLOv8s, which reaches about 296 FPS, and it demands a training time of about 190 minutes per run compared with about 26 minutes for YOLO. YOLO11s sits in the middle at about 189 FPS, while the CBAM variant is nearly as fast as the baseline because the added module is lightweight. When the accuracy of the four models is practically equal, the computational cost of the transformer does not pay off on this task. Figure 3 places YOLOv8s in the most advantageous corner: the highest accuracy together with the highest speed.

FPS was measured on a single run when the GPU was idle (no other training or inference process running), using the built-in Ultralytics benchmark on the full test split. The measurement was performed on the same hardware configuration (NVIDIA RTX 4090, batch size 1) for all models. The FPS figures are estimates at a single hardware point and will differ on other configurations.

![Figure 3. Accuracy (mAP@0.5) versus speed (FPS) trade-off. YOLOv8s occupies the most advantageous position, leading on both axes simultaneously.](../figures/fig3_accuracy_speed.png)

### D. Qualitative analysis

Figure 4 shows an example of YOLOv8s detection output on one test image. The model detects the rider, the helmet, and the license plate simultaneously with tight bounding boxes, illustrating the typical case where all three classes are present in a single scene. Visual inspection of the test sample shows that the remaining errors tend to appear on very small or partially occluded helmets, consistent with the quantitative finding that the *helmet* class is the most difficult.

![Figure 4. Example of YOLOv8s detection output on a test image: rider, helmet, and license plate detected simultaneously.](../figures/fig4_detection_example.png)

## V. Discussion

The central finding of this study runs counter to the common intuition shaped by the attention-based helmet-detection literature (Section II.D): on a small-scale helmet-detection dataset and under a training protocol arranged for YOLO, no attention mechanism surpassed the pure CNN, and the CBAM module was even significantly detrimental. Six complementary explanations can account for this pattern.

First, **the dataset is relatively small and easy**. With 1,563 training images and a baseline that already reaches about 0.96 mAP@0.5, the room for improvement is very narrow. When a simple model already approaches the upper bound achievable on this data, there is almost no gap for a more complex mechanism to show an advantage. Attention mechanisms—especially data-hungry transformers—require large volume and diversity to learn useful dependency patterns [17], [18]. In the small-data regime, their theoretical advantage does not materialize and the model instead bears the burden of unused capacity.

Second, **randomly initialized attention layers can disrupt pretrained features**. In the CBAM variant, most weights are inherited from COCO while the attention module starts from scratch. This untrained module inserts a transformation that, at the start of training, disrupts the mature feature flow from transfer learning. With limited data, there is not enough gradient signal to recover from that disruption, let alone surpass the baseline. The result is a small but consistent reduction observed to be stable across seeds. This explanation is consistent with the general observation that additional modules are most beneficial when trained jointly from the start on large data, rather than attached to a pretrained network with limited data.

Third, **complexity is not free**. Attention adds parameters and computation. When there is no accuracy gain, this addition becomes pure cost. The consequence is most evident in RT-DETR-l, which is five times slower with no mAP reward, but it also applies conceptually to CBAM, which adds a computational path at every detection scale. In real-time settings where the computational budget is limited, this cost means a reduction in the number of frames that can be processed per second.

Fourth, **the cost of errors is asymmetric in a safety context**. In safety-enforcement applications, the consequence of a false negative (failing to detect a helmetless rider) is far more serious than that of a false positive (flagging a helmeted rider as a violator). On the YOLOv8s baseline, the mAP@0.5 for the *helmet* class is 0.923, which means that about 7.7% of helmets are not correctly detected. In the context of automatic enforcement, this implies that nearly one in thirteen violators escapes undetected. Increasing recall for the *helmet* class—even at the expense of precision—may be more valued in real applications than overall mAP. The design of a loss function that weights the *helmet* class more heavily, or the use of metrics that explicitly account for the asymmetric cost (for example, a weighted F1-score), is worth exploring for deployments aimed at enforcing safety.

Fifth, **the gap toward real deployment**. The dataset used is relatively clean: good lighting, minimal occlusion, and low object density. Under real road-CCTV conditions—with the challenges of heavy occlusion by other riders, nighttime lighting, rain, and high traffic density—the capacity of attention to model global context may become useful. Raj and Nair [24] note that lighting variation and occlusion are the main challenges in real CCTV, and Hariyono et al. [21] emphasize the need for robustness to such conditions in the Indonesian context. Cheng et al. [23] emphasize that small objects in dense frames require more sophisticated feature strategies. Under more challenging conditions, attention mechanisms may provide benefits not observed on the relatively easy dataset in this study.

Sixth, **a rival hypothesis for the CBAM reduction**. The reduction in mAP@0.5 due to CBAM observed in this study does not automatically mean that CBAM is useless for helmet detection. Several alternative design factors can explain the reduction: (a) possibly suboptimal module placement—we only inserted CBAM at the detection head (P3, P4, P5), not in the backbone, whereas placement in the backbone as done in several studies [19], [20] might yield different results; (b) the fresh layers disrupting the COCO-pretrained weights without enough data to recover them; (c) a number of training epochs that may not be sufficient for the new module to converge; (d) a learning rate not tuned separately for the attention-module parameters. A systematic ablation—with variations in placement, training from scratch, more epochs, and a separate learning rate—is needed to isolate the exact cause and constitutes an agenda for future work.

These findings can be placed in the context of the broader literature. The advantages of transformer architectures and attention modules are most consistently reported on large-scale benchmarks with high diversity [17], [18]. On specialized tasks with limited data, several studies find that a well-arranged CNN remains competitive or superior. Our results add evidence in that direction specifically for the helmet-detection domain, and underscore that architecture selection should be guided by the characteristics of the data and the computational budget, not by architectural trends alone.

It is important to delimit this claim to avoid overgeneralization. This study does **not** conclude that attention is useless in general. What we show is narrower: under one training protocol arranged for YOLO, on one small and relatively easy dataset, and with a baseline that already approaches the ceiling, adding attention did not improve accuracy and—in the case of CBAM—even reduced it significantly. The equivalence of YOLO11s and RT-DETR-l is an absence of evidence of a difference, not evidence of equivalence; the ceiling effect suppresses statistical power so that a small benefit from attention, if any, is hard to detect. The plausible rival hypothesis—that the attention-based models have not been trained on their optimal recipe—cannot be ruled out and in fact becomes an agenda for future work. In other words, this finding is most aptly read as a practical caution against the assumption that "adding attention is sure to help," not as a verdict against attention mechanisms in general.

The practical implication, within that scope, is clear: for real-time helmet detection on similar data, a simple CNN architecture such as YOLOv8s is the most rational choice—the highest accuracy, the lightest, and the fastest. This finding also serves as a methodological reminder that claims of architectural superiority must be tested under controlled and repeated conditions before being adopted. Without repetition across seeds, small differences such as those observed here are easily misinterpreted as real advantages when they are within the range of random variation.

## VI. Threats to Validity and Limitations

This study has a number of limitations that constrain the generalization of the findings. Each limitation is described explicitly below.

**(1) Evaluation limited to a single, relatively clean dataset.** Generalization to real conditions—road-CCTV footage with occlusion, fast motion, poor lighting, and high density—has not been tested. It may be that on more challenging data, the ability of attention to model context actually becomes useful.

**(2) RT-DETR-l was run on only two seeds.** Because of the far higher computational cost (~190 minutes per run), RT-DETR-l was evaluated on only two seeds. As a consequence, its variance estimate is less precise (standard deviation 0.0151 vs 0.0014–0.0046 for the YOLO variants) and the 95% confidence interval is very wide ([−0.140; +0.142]). Its equivalence conclusion cannot be considered with the same confidence as the comparisons among YOLO variants that use three seeds. The RT-DETR-l equivalence claim needs to have its level of certainty lowered.

**(3) The comparison is not size-matched.** RT-DETR-l (~32 million parameters) is far larger than the YOLO-s variants (~9–11 million). This comparison reflects the standard model choices faced by practitioners, but it mixes the factors of architectural paradigm and model capacity. A comparison that equalizes the parameter count—for example, by evaluating YOLOv8m (~25 million parameters) as a size-matched control for RT-DETR-l—could provide a purer picture of the paradigm effect (transformer vs CNN) independent of capacity.

**(4) Hyperparameter confounding.** Hyperparameters were unified and not tuned specifically per architecture. This protocol favors YOLO, which is the more mature architecture and more sensitive to the default Ultralytics settings. Transformers and attention-based models may require a different training recipe—for example, a longer warm-up, a different learning rate, or a special decay schedule—to reach their peak potential. As a result, the results of the attention-based models may be a lower bound rather than their true potential.

**(5) The ceiling effect suppresses statistical power.** With the baseline already at ~0.96, the room for improvement is very narrow, and the number of seeds is small (n = 2–3), so the statistical power of the t-test is very low. For a concrete picture: with only 100 test images, a difference of one or two mislabeled annotations can already shift mAP@0.5 by about 0.01. This means that the 0.002–0.004 difference observed between YOLOv8s and YOLO11s is within a range of variation that could be caused by a difference on one or two images. As a consequence, the equivalence claims (YOLO11s, RT-DETR-l) must be read as an absence of evidence of a difference, not evidence of equivalence; concluding "attention does not help" in absolute terms would be a leap from *absence of evidence* to *evidence of absence*.

**(6) FPS measured on a single configuration.** The speed figures were taken from a single run to mitigate inter-run interference, but they remain estimates on a single hardware configuration (NVIDIA RTX 4090, batch size 1). The FPS results will differ on other devices and do not reflect end-to-end latency including pre-processing and post-processing.

**(7) The CBAM reduction may be an artifact of the experimental design.** The accuracy reduction due to CBAM cannot be automatically attributed to the ineffectiveness of the attention module itself. Several alternative design factors can explain the reduction: (a) suboptimal placement—CBAM was inserted only at the detection head, not in the backbone; (b) the fresh layers disrupting the COCO-pretrained weights; (c) an insufficient number of epochs for the new module to converge; (d) a learning rate not tuned separately for the attention parameters. A systematic ablation with variations in placement, training from scratch, more epochs, and a separate learning rate is needed to isolate the exact cause.

**(8) A single dataset without difficulty variation.** The use of a single dataset with relatively easy conditions (good lighting, minimal occlusion) limits the ability to test whether attention provides benefits under more challenging conditions. Generalization to real CCTV with heavy occlusion, nighttime lighting, rain, and high traffic density has not been verified.

**(9) The overall number of seeds is limited.** Although three seeds are sufficient to detect large effects such as the CBAM reduction (Cohen's *d* ≈ 9.1), the statistical power to detect small effects is very limited. To confirm equivalence with an adequate level of confidence, a larger number of seeds or more sophisticated inference methods (for example, Bayesian equivalence testing) are required.

**(10) No evaluation on a size-matched attention-based architecture.** A comparison of YOLOv8m (~25 million parameters) or other medium-sized YOLO variants as a size-matched control for RT-DETR-l has not been carried out. Without this comparison, it cannot be ascertained whether the equivalent RT-DETR-l result reflects the ineffectiveness of the transformer paradigm or merely a model-scale mismatch.

## VII. Conclusion and Recommendations

We compared four object-detection architectures occupying a spectrum of attention usage for motorcyclist helmet-use detection, under an identical and repeated training and evaluation protocol. Under a protocol arranged for YOLO and on a single, small, relatively easy dataset, no attention mechanism—partial, full transformer, or the CBAM module—surpassed the pure-CNN YOLOv8s. YOLOv8s delivered the highest accuracy (mAP@0.5 = 0.9592 ± 0.0014) together with the best speed (~296 FPS), while adding CBAM reduced mAP@0.5 by 0.0113 significantly (paired t-test, t(2) = 10.9, p = 0.008, Cohen's *d* ≈ 9.1); the equivalence of YOLO11s (p = 0.54) and RT-DETR-l (p = 0.93) is interpreted as an absence of evidence of a difference, not evidence of equivalence, given the ceiling effect and the limited number of seeds.

**These findings hold under the specific conditions tested:** a single, small, relatively easy dataset, a training protocol arranged for YOLO, and a resolution of 1280 pixels. Generalization to other conditions—larger and more challenging datasets, per-architecture-tuned hyperparameters, or architectures with comparable capacity—requires further validation. Within this scope, this finding is a **practical caution against the assumption that adding attention always helps on small-scale helmet-detection tasks, not a general verdict on attention mechanisms.**

The recommendations for future work follow directly from the limitations above. First, to test generalization on larger, more diverse, and more challenging datasets, including real CCTV footage with difficult conditions (occlusion, nighttime lighting, rain, high density). Second, to evaluate attention in a large-data regime where its advantage has a chance to emerge, as well as with training from scratch instead of merely attaching modules to a pretrained network. Third, to add size-matched comparisons—particularly YOLOv8m (~25 million parameters) as a capacity control for RT-DETR-l—as well as per-architecture hyperparameter fine-tuning so that each model is tested under its best conditions. Fourth, to conduct a systematic CBAM ablation: variations in placement (backbone vs neck vs detection head), training from scratch, more epochs, and a separate learning rate for the attention-module parameters. Fifth, to explore a two-stage approach—motorcycle detection followed by helmet classification—and the integration of license-plate detection for enforcement scenarios, with a loss function that weights the *helmet* class more heavily to handle the asymmetric error cost. Sixth, to increase the number of seeds (at least n = 5) or use Bayesian inference methods to strengthen the equivalence or difference claims.

---

## Statements

**Data Availability.** The dataset is public via Roboflow Universe ("NCKH 2023 / Helmet Detection Project" v19, MIT license). The training scripts, configurations, and experiment notebooks are available in the authors' research repository.

**Ethics Statement.** The study uses a public, openly licensed dataset without personally identifiable data outside the public-traffic context; it does not involve human subjects or experiments requiring ethical approval.

**Author Contributions (CRediT).** Conceptualization, methodology, software, analysis, and manuscript writing: the authors. (Adjust to the actual list of authors.)

**Conflict of Interest.** The authors declare no conflict of interest.

**Funding.** No specific funding is reported for this study.

**AI Disclosure.** The design of the experiments, the analysis, and the writing of this manuscript were assisted by AI-based tools (Claude Code together with the Academic Research Skills skill). Revision of the manuscript in response to peer-review feedback was also assisted by Claude Code. All methodological decisions, verification of numerical results, and final responsibility for the content rest with the authors. The cited sources were independently verified for their existence.

---

## References

[1] F. W. Siebert and H. Lin, "Detecting motorcycle helmet use with deep learning," *Accident Analysis & Prevention*, vol. 134, p. 105319, 2020.

[2] H. Lin, J. D. Deng, D. Albers, and F. W. Siebert, "Helmet use detection of tracked motorcycles using CNN-based multi-task learning," *IEEE Access*, vol. 8, pp. 162073–162084, 2020.

[3] M. Naphade et al., "The 7th AI City Challenge," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023.

[4] W. Jia et al., "Real-time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector," *IET Image Processing*, vol. 15, no. 14, pp. 3623–3637, 2021.

[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics. Accessed: Jun. 13, 2026.

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

[17] M.-H. Guo, T.-X. Xu, J.-J. Jiang, N. Liu, Z.-N. Liu, P.-T. Jiang, T.-J. Mu, and S.-M. Hu, "Attention mechanisms in computer vision: A survey," *Computational Visual Media*, vol. 8, no. 3, pp. 331–369, 2022.

[18] Z. Niu, G. Zhong, and H. Yu, "A review on the attention mechanism of deep learning," *Neurocomputing*, vol. 452, pp. 48–62, 2021.

[19] H. Zhang, Q. Luo, and C. Yin, "YOLOv8 safety helmet detection algorithm based on attention mechanism and feature fusion," *J. Phys.: Conf. Ser.*, vol. 3135, no. 1, p. 012025, 2025.

[20] J. Li, X. Zhang, and Z. Liu, "Safety helmet detection based on improved YOLOv5s with attention mechanism," *Sensors*, vol. 23, no. 14, p. 6500, 2023.

[21] M. H. D. Hariyono, A. N. Ihsan, and D. S. Kusumo, "Streamlit application for helmet detection based on YOLOS: Case study Indonesia," in *Proc. Data Science and Its Applications Conf. (DASA)*, 2024.

[22] W. Jia et al., "DAAM-YOLOv5: Helmet detection combined with attention mechanism," *Electronics*, vol. 12, no. 9, p. 2094, 2023.

[23] G. Cheng, X. Yuan, X. Yao, K. Yan, Q. Zeng, and J. Han, "Towards large-scale small object detection: Survey and benchmarks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 1, pp. 521–539, 2024.

[24] R. D. Raj and M. S. Nair, "A YOLO-based approach to detecting helmetless riders through CCTV," *SITECH: J. Inf. Technol.*, 2024.
