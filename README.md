
# Brain Tumor MRI Classification

Deep-learning classification of brain MRI images across **10 pathological categories**, with a strong focus on **data leakage, evaluation methodology, model generalization, MRI modality robustness, and explainability**.

> **Key finding:** measured performance changed substantially depending on how sample independence was defined. EfficientNetV2B0 achieved approximately **30% accuracy under a conservative group-disjoint evaluation**, compared with **62.35% test accuracy under a duplicate-safe image-level diagnostic protocol**.

This gap became one of the central findings of the project and highlights the importance of provenance and split methodology when evaluating medical imaging models.

---

## Project Overview

The initial objective was to train a deep-learning model capable of classifying brain MRI images into 10 pathological categories.

However, exploratory data analysis revealed several characteristics that made a conventional random image-level train/test split potentially misleading:

- repeated and exact duplicate images;
- groups of apparently related acquisitions;
- heterogeneous image framing and intensity characteristics;
- no verified patient identifiers.

The project therefore evolved from a straightforward image-classification task into a broader investigation of:

1. **data quality and leakage risk**;
2. **generalization under different independence assumptions**;
3. **class- and modality-specific failure modes**;
4. **whether model attention is spatially associated with annotated lesions**.

Although the original dataset is organized into 30 pathology–modality combinations, these combinations were not treated as distinct diagnostic classes. The three MRI modalities (T1, T1C+, and T2) represent different imaging sequences rather than different pathologies. The prediction target was therefore defined as 10 pathology classes, while modality was retained as metadata for subgroup evaluation.

---

## Dataset

The project uses the **[Brain Tumor 11300 MRI Images 30 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes)** dataset from Kaggle.

### Main characteristics

- **11,300 MRI images**
- Resolution: **512 × 512**
- Three MRI modalities:
  - T1
  - T1C+
  - T2
- 30 original folders corresponding to combinations of pathology and modality
- 10 pathological categories:

| Pathology |
|---|
| Astrocytoma |
| Ependymoma |
| Glioma |
| Hemangiopericytoma |
| Meningioma |
| Neurocytoma |
| Normal |
| Oligodendroglioma |
| Other |
| Schwannoma |

For this project, **pathology was used as the classification target**, while MRI modality was retained as metadata for subsequent robustness analysis.

The dataset also provides metadata including lesion-center coordinates for tumor images and textual descriptions.

Textual descriptions were **not used as model inputs**, since they contain strong pathology-specific information and could introduce target leakage.

---

## Data Audit

A substantial data audit was performed before model training.

### Reconstructed case groups

Because verified patient identifiers were unavailable, image naming patterns, metadata, modality relationships, descriptions, and duplicate analysis were used to reconstruct groups of apparently related acquisitions.

An initial **361 groups** were identified.

Exact duplicate analysis subsequently revealed four pairs of groups that contained identical images across group boundaries. These groups were merged, resulting in:

**357 reconstructed `split_group` units.**

These groups were used as conservative proxies for related cases.

Importantly, `split_group` must **not** be interpreted as a verified patient identifier.

---

### Exact duplicates

Pixel-level hashing revealed substantial duplication:

- **11,300 total images**
- **9,435 unique pixel hashes**
- **3,317 images** belonging to exact-duplicate groups
- **1,452 exact-duplicate groups**

Some duplicates initially crossed reconstructed case-group boundaries, which helped refine the grouping procedure.

Exact duplicates were explicitly controlled during subsequent evaluation.

---

### Near-duplicate investigation

Perceptual hashing was also used to search for visually similar images that might represent additional leakage.

Candidate pairs with very small perceptual-hash distances were visually inspected.

The candidates were generally distinct MRI images rather than convincing duplicates. Consequently, reconstructed groups were **not merged based solely on perceptual similarity**.

This prevented aggressive grouping based on normal anatomical similarity between brain MRI images.

---

### Effective class imbalance

Image counts alone did not accurately represent the amount of independent information available for each pathology.

For example, some classes contained many images but relatively few reconstructed groups.

This distinction became important for group-disjoint evaluation, where the effective sample size is closer to the number of independent groups than to the raw number of images.

---

### Image framing and intensity

The dataset exhibited substantial heterogeneity in:

- black-background proportions;
- image framing;
- intensity distributions;
- acquisition appearance.

Some of these characteristics were correlated with pathology.

This creates a potential source of shortcut learning, where a model may partially exploit acquisition or preprocessing characteristics rather than pathology-specific tumor features.

---

## Modeling

### Architecture

The classification model uses **EfficientNetV2B0** pretrained on ImageNet.

The training strategy consisted of:

1. transfer learning with a frozen convolutional backbone;
2. a classification head with global average pooling and dropout;
3. partial fine-tuning of the final EfficientNetV2 blocks;
4. conservative data augmentation.

Input images were resized to:

`224 × 224`

The final classification layer predicts the 10 pathological categories.

### Data augmentation

Training augmentation included:

- horizontal flipping;
- small rotations;
- translations;
- zoom.

Vertical flipping was deliberately excluded because arbitrary superior-inferior inversion is not necessarily a medically plausible augmentation for brain MRI.

---

# Evaluation Methodology

Evaluation methodology became a central component of this project.

Two complementary protocols were studied.

## 1. Group-disjoint evaluation

Reconstructed `split_group` units were kept entirely within a single partition.

Therefore, images belonging to the same reconstructed group could not appear across training, validation, and test sets.

Several mitigation strategies were investigated, including group-size weighting and group-aware sampling.

Despite these modifications, performance remained approximately **30%**, with the strongest tested configuration reaching roughly:

**32.0% test accuracy**

and substantial variation between reconstructed groups.

This suggests poor generalization when the model is required to classify images from entirely unseen reconstructed groups.

However, because these groups are reconstructed rather than verified patient identities, this result should be interpreted as a **conservative generalization stress test**, not as a definitive patient-level estimate.

---

## 2. Duplicate-safe image-level diagnostic evaluation

A second experiment was designed to determine whether the low group-disjoint performance reflected an inability to learn the classification task or sensitivity to the reconstructed grouping assumption.

Images were partitioned at the level of their **exact pixel hash**.

Therefore:

- identical images could never cross partitions;
- exact-duplicate leakage was eliminated;
- different images belonging to the same reconstructed group were allowed across partitions.

Under this protocol:

- **0 exact duplicate hashes crossed partitions**
- **93.56% of reconstructed groups crossed more than one partition**

EfficientNetV2B0 achieved:

| Metric | Result |
|---|---:|
| Test accuracy | **62.35%** |
| Macro F1 | **0.630** |
| Weighted F1 | **0.614** |

The large increase from approximately 30% to 62.35% cannot be explained by identical-pixel leakage, since exact duplicates were explicitly kept within a single partition.

Instead, it demonstrates that performance is highly sensitive to the assumed unit of statistical independence.

### Interpretation

| Protocol | Exact duplicates across splits | Reconstructed groups across splits | Test accuracy | Interpretation |
|---|---:|---:|---:|---|
| Group-disjoint | 0 | 0 | ~32% | Conservative unseen-group robustness |
| Duplicate-safe diagnostic | 0 | 93.56% | **62.35%** | Image-level learnability under group overlap |

---

# Per-Class Performance

Performance varied substantially across pathologies.

Under the duplicate-safe diagnostic protocol, some categories such as **Neurocytoma, Normal, Schwannoma, and Oligodendroglioma** showed relatively strong recall, while **Astrocytoma and Hemangiopericytoma** remained considerably more difficult.

Astrocytoma in particular showed substantial confusion with several other categories, including Meningioma, Glioma, and Other.

These results demonstrate why overall accuracy alone is insufficient to characterize performance in a multi-class medical imaging problem.

---

# MRI Modality Robustness

Performance was also evaluated separately for each MRI modality.

| Modality | Accuracy | Macro F1 |
|---|---:|---:|
| **T1C+** | **67.4%** | **0.696** |
| T1 | 63.0% | 0.627 |
| T2 | 53.5% | 0.486 |

T1C+ produced the strongest aggregate results, while T2 was substantially more challenging.

However, modality effects were strongly pathology-dependent.

For example, T2 recall was particularly poor for some tumor categories while remaining very high for others such as Normal and Neurocytoma.

This indicates an important **pathology × modality interaction** that would be hidden by aggregate test metrics alone.

---

# Explainability with Grad-CAM

Grad-CAM was used as a **post-hoc explanation method** to investigate which image regions contributed to the classifier's predictions.

The dataset provides an approximate lesion-center coordinate for tumor images, allowing model attention to be evaluated relative to known lesion locations.

Grad-CAM was evaluated on **1,516 tumor images** from the duplicate-safe test set.

### Lesion attention and classification success

Correctly classified images showed stronger Grad-CAM activation around the annotated lesion center.

Median lesion-center activation percentile:

- **Correct predictions:** 78.29
- **Incorrect predictions:** 62.18

The difference was statistically significant:

`p = 2.52 × 10⁻¹¹`

This indicates that stronger attention around the lesion was associated with successful classification.

However, association does not demonstrate that the model systematically bases its predictions on tumor tissue.

---

## Lesion vs anatomical control

To test whether lesion-centered activation represented true lesion preference rather than broad activation of brain anatomy, each lesion location was compared with a mirrored control location at the same distance from the image center.

Results:

- median lesion local activation: **0.271**
- median mirrored-control activation: **0.242**
- lesion activation greater than control in only **48.88%** of images
- one-sided paired Wilcoxon test: **p = 0.065**

Therefore, there was **no statistically significant evidence that the classifier systematically prioritized the lesion region over a comparable anatomical control location**.

Grad-CAM patterns also varied substantially between pathologies.

The explainability analysis therefore suggests that lesion-related information contributes to successful predictions in some cases, while the classifier also relies on broader and pathology-dependent spatial patterns.

---

# Key Findings

The most important result of this project is not a single classification score.

Instead, the experiments demonstrate that:

- medical-image classification performance can change substantially depending on how sample independence is defined;
- removing exact duplicate leakage alone does not guarantee patient-independent evaluation;
- missing patient provenance creates fundamental uncertainty when interpreting generalization;
- image-level metrics can conceal large class- and modality-specific failure modes;
- stronger lesion-centered attention is associated with correct predictions, but the classifier does not systematically prioritize the lesion over comparable anatomical regions;
- explainability visualizations should be quantitatively challenged rather than interpreted solely from visually convincing heatmaps.

---

# Limitations

The main limitation is the absence of verified patient identifiers.

Although reconstructed groups are supported by multiple internal signals, they cannot be considered confirmed patient identities.

---

# Reproducibility

The project was developed primarily with:

- Python
- TensorFlow / Keras
- EfficientNetV2
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Pillow
- ImageHash

Model training was performed in **Google Colab** to access accelerated compute resources.

Random seeds were fixed where applicable, although complete deterministic reproducibility across GPU environments is not guaranteed.

---

# Conclusion

This project demonstrates why medical imaging machine learning should not be reduced to training a classifier and reporting its highest accuracy.

EfficientNetV2B0 achieved **62.35% test accuracy** under a duplicate-safe image-level protocol, while performance fell to approximately **30% when reconstructed case groups were kept entirely disjoint**.

Rather than selecting whichever protocol produced the strongest score, both results were retained because they answer different questions and expose the uncertainty introduced by missing patient provenance.

Additional modality-specific and Grad-CAM analyses showed that aggregate classification metrics alone also conceal substantial heterogeneity in both performance and model behavior.

---

## Disclaimer

This project is intended for **machine-learning research and educational purposes only**.
