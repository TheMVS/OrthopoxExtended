# Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models

**Authors:**  
Alejandro Puente-Castro, Enrique Fernandez-Blanco, Daniel Rivero, Álvaro Rodríguez, Andres Molares-Ulloa

## Author Contributions

* **Alejandro Puente-Castro**: Conceptualization, Methodology, Software, Validation, Formal Analysis, Resources, Data Curation, Writing – Original Draft Preparation, Visualization, Investigation
* **Enrique Fernandez-Blanco**: Writing – Reviewing and Editing, Project Administration, Funding Acquisition
* **Daniel Rivero**: Writing – Reviewing and Editing, Supervision, Methodological Guidance
* **Álvaro Rodríguez**: Writing – Reviewing and Editing, Methodological Validation, Experimental Design Support, Statistical Review
* **Andres Molares-Ulloa**: Writing – Reviewing and Editing, Methodological Validation

---

# 1. Project Overview

This project implements a rigorous and reproducible experimental pipeline for Orthopox-related skin image classification.

The pipeline compares:

1. **Fine-tuned deep learning models**
   - EfficientNetV2B0
   - ConvNeXtTiny
   - MonkeyNet-like DenseNet201 baseline

2. **Hybrid deep learning + machine learning models**
   - Frozen CNN embeddings
   - Logistic Regression
   - SVM
   - Random Forest
   - Optional SMOTE and SMOTEENN in embedding space

3. **Evaluation protocols**
   - Original images only
   - Train-only augmentation
   - Paper-like augmented split

4. **Explainability methods**
   - Grad-CAM
   - Grad-CAM++
   - Occlusion sensitivity

The main objective is to evaluate models under a leakage-free protocol and to quantify whether more optimistic protocols, such as augmenting before splitting, can inflate performance.

---

# 2. Dataset Structure

Expected dataset structure:

```text
data/
 └── original/
     ├── Chickenpox/
     ├── Measles/
     ├── Monkeypox/
     └── Normal/
```

The code also supports:

```text
data/
 ├── Chickenpox/
 ├── Measles/
 ├── Monkeypox/
 └── Normal/
```

Images may be `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`.

---

# 3. Experimental Workflow

## 5.1 Search stage

```text
Dataset
 → candidate configurations
 → SEARCH_CV stratified K-fold
 → metric computation
 → ranking
 → top-N selection
```

The number of configurations is controlled by:

```python
MAX_SEARCH_CONFIGS
```

The number of top models retained is controlled by:

```python
TOP_N_TO_EVALUATE
```

---

## 5.2 Final DL evaluation

For each selected configuration:

```text
model
 × condition
 × seed
 × fold
```

Conditions:

1. `original`
   - no training augmentation

2. `augmented`
   - augmentation applied only to the training fold

The test fold is never augmented.

---

## 5.3 ML baseline evaluation

For each embedding backbone:

```text
frozen CNN → embeddings
```

Then the code trains:

- Logistic Regression
- SVM
- Random Forest

With:

- no balancing
- SMOTE
- SMOTEENN

All sampling is applied only to the training embeddings of each fold.

---

## 5.4 Protocol comparison

For the best model, three protocols are compared:

### `original_only_clean_split`

```text
original 770 images
 → split
 → train without augmentation
 → test without augmentation
```

### `rigorous_clean_split_train_only_aug`

```text
original 770 images
 → split
 → augment train only
 → test untouched
```

### `paper_like_leaky_augmented_split`

```text
original images
 → offline augmentation
 → split augmented dataset
```

This approximates a paper-like protocol where augmentation is performed before splitting.

---

## 5.5 Augmentation ablation

For the best model, the code compares:

- no augmentation
- light augmentation
- moderate augmentation

This helps determine whether augmentation truly improves performance under a clean protocol.

---

# 11. Detailed Experimental Design

The project is organized around two main execution stages controlled by:

```python
Config.RUN_RANDOM_SEARCH
```

## 11.1 Random Search Stage

When:

```python
RUN_RANDOM_SEARCH = True
```

the pipeline executes `search.py`.

The goal of this stage is not to produce the final paper results, but to identify promising model configurations. The search stage:

1. Loads the dataset.
2. Samples candidate hyperparameter configurations.
3. Trains each candidate using stratified cross-validation.
4. Computes search metrics.
5. Saves partial results incrementally.
6. Selects the best configurations for final evaluation.

The updated version guarantees that the random search explores at least one configuration from each model family. This avoids the risk that random sampling accidentally selects only one family of architectures.

Relevant concept: random search for hyperparameter optimization  
https://www.jmlr.org/papers/v13/bergstra12a.html

---

## 11.2 Final Evaluation Stage

When:

```python
RUN_RANDOM_SEARCH = False
```

the pipeline executes `evaluate.py`.

The final evaluation is the paper-quality stage. It evaluates the selected configurations using:

- repeated stratified K-fold cross-validation
- multiple random seeds
- original and augmented training conditions
- calibration-aware metrics
- class-wise results
- confusion matrices
- statistical tests
- explainability methods
- protocol comparison

This stage is designed to answer whether differences between models are robust or merely due to split variability.

Stratified K-fold documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

---

# 4. Model Families and Architectures

This section describes the architectural families evaluated in the repository.

The project is intentionally organized around model families instead of isolated architectures. This improves scientific fairness during model selection and avoids overrepresenting one family during random search.

The experimental pipeline distinguishes between:

1. Fine-Tuned Deep Learning Models
2. Hybrid Deep Learning + Machine Learning Models
3. Lightweight Deployment-Oriented Models
4. Family-Based Experimental Design

---

## 4.1 Fine-Tuned Deep Learning Models

These models are trained end-to-end using transfer learning and partial fine-tuning.

General workflow:

```text
image
 → pretrained backbone
 → classifier head
 → partial fine-tuning
 → prediction
```

The goal is to evaluate whether modern CNN backbones can achieve strong performance under a rigorous leakage-free protocol.

### EfficientNetV2 Family

EfficientNetV2 is a modern convolutional architecture designed for:

- strong accuracy
- parameter efficiency
- fast training

In this repository, EfficientNetV2 models act as efficient fine-tuned CNN baselines.

Useful links:

- EfficientNetV2 paper: https://arxiv.org/abs/2104.00298
- TensorFlow EfficientNetV2B0: https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B0
- Keras Applications overview: https://keras.io/api/applications/

---

### ConvNeXt Family

ConvNeXt modernizes CNNs using ideas inspired by Vision Transformers while remaining fully convolutional.

Advantages:

- strong transfer-learning capability
- stable optimization
- modern CNN design

Useful links:

- ConvNeXt paper: https://arxiv.org/abs/2201.03545
- Keras ConvNeXt documentation: https://keras.io/api/applications/convnext/

---

### MonkeyNet-Style DenseNet201 Family

The repository includes a MonkeyNet-inspired DenseNet201 baseline.

Conceptually:

```text
DenseNet201
 → adaptation layers
 → GAP
 → classifier head
```

This architecture approximates the design philosophy used in the MonkeyNet paper.

Useful links:

- DenseNet paper: https://arxiv.org/abs/1608.06993
- Keras DenseNet documentation: https://keras.io/api/applications/densenet/
- MonkeyNet paper DOI page: https://doi.org/10.1016/j.neunet.2023.02.022

---

### Classical CNN Baselines

The repository can also include standard medical-imaging baselines such as:

- ResNet50V2
- Xception

These are important because many reviewers expect comparisons against widely used CNN architectures.

Useful links:

- ResNet paper: https://arxiv.org/abs/1512.03385
- Keras ResNet documentation: https://keras.io/api/applications/resnet/
- Xception paper: https://arxiv.org/abs/1610.02357
- Keras Xception documentation: https://keras.io/api/applications/xception/

---

## 4.2 Hybrid Deep Learning + Machine Learning Models

The repository also evaluates hybrid pipelines combining deep feature extraction with classical machine learning.

General workflow:

```text
image
 → frozen CNN backbone
 → embedding vector
 → optional balancing
 → ML classifier
 → prediction
```

This design tests whether pretrained CNN embeddings alone are sufficient for robust classification.

### Logistic Regression

Linear probabilistic classifier.

Advantages:

- interpretable
- stable
- strong baseline

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

---

### Support Vector Machine (SVM)

Nonlinear classifier often strong on small datasets.

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

---

### Random Forest

Tree ensemble classifier.

Advantages:

- nonlinear decision boundaries
- robustness
- interpretability

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

---

## 4.3 Lightweight Deployment-Oriented Models

The repository can also evaluate lightweight architectures designed for low-resource deployment.

Example:

- MobileNetV3

These models are useful for:

- mobile inference
- edge AI
- embedded medical systems
- low-resource clinical deployment

Useful links:

- MobileNetV3 paper: https://arxiv.org/abs/1905.02244
- Keras MobileNet documentation: https://keras.io/api/applications/mobilenet/

---

## 4.4 Family-Based Experimental Design

The updated pipeline uses family-balanced model selection.

Instead of selecting only the global top-N configurations, the pipeline can retain:

```text
best configuration per family
+ optional global top extras
```

This improves:

- experimental diversity
- fairness
- reproducibility
- scientific interpretability

The repository therefore supports comparisons between:

- modern CNNs
- classical CNNs
- MonkeyNet-style architectures
- lightweight models
- hybrid DL+ML systems

under the same rigorous evaluation protocol.


# 5. Hybrid Deep Learning + Machine Learning Baselines

The hybrid branch evaluates whether frozen CNN embeddings are sufficient for classification.

The workflow is:

```text
image
 → frozen CNN backbone
 → embedding vector
 → optional embedding-space balancing
 → classical ML classifier
 → prediction
```

This is useful because, in small medical datasets, frozen features plus a classical classifier can sometimes compete with fine-tuning while being cheaper and more stable.

---

## 13.1 Logistic Regression

Logistic Regression is a linear probabilistic classifier. It is simple, strong, and interpretable.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

---

## 13.2 Support Vector Machine

SVMs can be strong on small datasets, especially with RBF kernels.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

---

## 13.3 Random Forest

Random Forests are nonlinear tree ensembles. They are robust and often useful as classical baselines.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

---

# 6. Balancing Methods

## 14.1 Class Weights

Class weights compensate for imbalanced class frequencies during neural network training. They increase the penalty associated with errors in minority classes.

Scikit-learn utility:  
https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

---

## 14.2 SMOTE

SMOTE creates synthetic samples by interpolating between minority-class examples.

In this project, SMOTE is applied only to embedding vectors and only inside the training fold. It is not applied to raw images.

Documentation:  
https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

Original paper:  
https://arxiv.org/abs/1106.1813

---

## 14.3 SMOTEENN

SMOTEENN combines SMOTE oversampling with Edited Nearest Neighbours cleaning. It can both rebalance the dataset and remove ambiguous samples.

Documentation:  
https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html

---

# 7. Data Augmentation

The pipeline distinguishes between rigorous train-only augmentation and paper-like offline augmentation.

---

## 15.1 Train-Only Online Augmentation

This is the recommended protocol.

```text
split first
 → augment training fold only
 → validation/test remain untouched
```

Possible transformations include:

- horizontal flip
- rotation
- zoom
- contrast adjustment
- brightness adjustment

Keras preprocessing layers:  
https://keras.io/api/layers/preprocessing_layers/image_augmentation/

---

## 15.2 Paper-Like Offline Augmentation Before Split

This protocol is included only for methodological comparison.

```text
augment full dataset
 → split augmented dataset
```

This can inflate results because near-duplicate augmented versions of the same original image can appear in both train and test partitions.

This experiment helps quantify protocol-induced performance inflation.

---

# 8. Test-Time Techniques and Calibration

Test-Time Augmentation (TTA) is an inference-time technique. It does not retrain the model.

The idea is:

```text
test image
 → deterministic transformations
 → multiple predictions
 → average probabilities
```

For example:

```text
original image
horizontal flip
vertical flip
small rotations
```

The final prediction is obtained by averaging the predicted probabilities.

TTA can improve robustness because the prediction becomes less dependent on one exact image orientation or crop.

Important distinction:

- Training augmentation modifies training batches during learning.
- TTA modifies test images only at inference time and averages predictions.
- TTA does not use test labels and does not update model weights.

Introductory explanation:  
https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/

---

# 17. Weighted TTA

Weighted TTA is a variant of TTA where not all transformed predictions receive the same weight.

For example, the original image can receive higher weight than transformed versions:

```text
final_probability =
0.50 * original_prediction
+ 0.25 * horizontal_flip_prediction
+ 0.25 * vertical_flip_prediction
```

This can be useful when transformed images are helpful but should not dominate the original view.

---

# 18. Monte Carlo Dropout

Monte Carlo Dropout keeps dropout active during inference and performs several stochastic forward passes.

The result is:

- average prediction
- uncertainty estimate from prediction variability

This is useful in medical imaging because the model can indicate when it is uncertain.

Original paper:  
https://arxiv.org/abs/1506.02142

---

# 19. Temperature Scaling

Temperature Scaling is a post-hoc calibration method.

It modifies softmax confidence using:

```text
softmax(logits / T)
```

where `T` is the temperature.

- `T = 1`: unchanged probabilities
- `T > 1`: softer probabilities
- `T < 1`: sharper probabilities

The class prediction usually remains the same, but the confidence becomes better calibrated.

Paper:  
https://arxiv.org/abs/1706.04599

---

# 20. Entropy-Based Rejection / Abstention

Entropy measures prediction uncertainty.

High entropy means the model is uncertain across classes. The system can abstain from making a prediction when entropy is too high.

This is useful for medical decision support because uncertain cases can be flagged for human review.

Entropy explanation:  
https://en.wikipedia.org/wiki/Entropy_(information_theory)

---

# 9. Explainable AI Methods

## 22.1 Grad-CAM

Grad-CAM highlights image regions that contribute to the predicted class.

Paper:  
https://arxiv.org/abs/1610.02391

---

## 22.2 Grad-CAM++

Grad-CAM++ is an extension of Grad-CAM that can better localize small or multiple discriminative regions.

Paper:  
https://arxiv.org/abs/1710.11063

---

## 22.3 Occlusion Sensitivity

Occlusion sensitivity hides image patches and measures how much the predicted probability drops.

It is slower than Grad-CAM but useful as a perturbation-based sanity check.

Explanation:  
https://christophm.github.io/interpretable-ml-book/pixel-attribution.html

---

# 10. Evaluation Metrics

## 21.1 Accuracy

Accuracy measures the percentage of correct predictions.

Limitation: it can be misleading in imbalanced datasets.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

---

## 21.2 Macro F1

Macro F1 computes F1 per class and averages all classes equally.

It is the recommended primary metric for this project because all classes should matter equally.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

---

## 21.3 Balanced Accuracy

Balanced accuracy is the average recall across classes.

It is useful when the class distribution is imbalanced.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

---

## 21.4 Cohen's Kappa

Cohen's kappa measures agreement corrected for chance.

It helps detect when accuracy is inflated by class prevalence.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

---

## 21.5 Expected Calibration Error

Expected Calibration Error (ECE) measures the gap between predicted confidence and empirical correctness.

A well-calibrated model should satisfy:

```text
among predictions made with 80% confidence,
approximately 80% should be correct
```

Explanation:  
https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d

---

## 21.6 Brier Score

The Brier score measures the squared error between predicted probabilities and the true one-hot label.

Lower values are better.

Documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

---

# 11. Statistical Analysis

The statistical unit is the seed-level result, not individual image predictions. This avoids treating many predictions from the same trained model as independent observations.

---

## 23.1 Friedman Test

The Friedman test is used as an omnibus test when comparing more than two configurations across matched repeated units.

Explanation:  
https://en.wikipedia.org/wiki/Friedman_test

---

## 23.2 Shapiro-Wilk Test

Shapiro-Wilk checks whether paired differences are approximately normally distributed.

Documentation:  
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

---

## 23.3 Paired t-test

Used when paired differences are compatible with normality.

Documentation:  
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html

---

## 23.4 Wilcoxon Signed-Rank Test

Used when paired differences are not normally distributed.

Documentation:  
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

---

## 23.5 Holm Correction

Holm correction adjusts p-values for multiple comparisons.

Explanation:  
https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method

---

## 23.6 Cohen's dz

Cohen's dz is an effect size for paired designs.

Explanation:  
https://real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/

---

# 12. Project Files

## `Program.py`

Main entry point.

It decides which stage to execute based on:

```python
Config.RUN_RANDOM_SEARCH
```

If:

```python
RUN_RANDOM_SEARCH = True
```

then it runs:

```text
search.py
```

If:

```python
RUN_RANDOM_SEARCH = False
```

then it runs:

```text
evaluate.py
```

You normally execute only:

```bash
python Program.py
```

---

## `Config.py`

Central configuration file.

This file controls:

- dataset paths
- model families
- search space
- number of folds
- seeds
- augmentation policies
- ML baselines
- SMOTE/SMOTEENN settings
- Grad-CAM and XAI settings
- output file names

All important experimental decisions are centralized here so that experiments are reproducible.

A detailed explanation of `Config.py` is included in Section 7.

---

## `search.py`

Implements the configuration search stage.

It:

1. Loads the image dataset.
2. Samples candidate configurations from `Config.SEARCH_SPACE`.
3. Evaluates each configuration with `SEARCH_CV` stratified folds.
4. Trains each candidate in two stages:
   - head training
   - partial fine-tuning
5. Computes search metrics.
6. Saves partial results after every completed configuration.
7. Saves the top-N configurations at the end.

Important output:

```text
results/search_results_with_monkeynet_like.csv
results/top_configs.json
results/search_summary.json
```

The results from `search.py` are not final paper results. They are used only for model selection.

---

## `evaluate.py`

Implements the final experimental evaluation.

It:

1. Loads `top_configs.json`.
2. Evaluates the selected models using repeated stratified K-fold.
3. Compares original vs train-only augmentation.
4. Computes all final metrics.
5. Generates class-wise metrics.
6. Generates confusion matrices.
7. Performs error analysis.
8. Runs statistical tests.
9. Runs ML baselines on frozen embeddings.
10. Runs SMOTE/SMOTEENN comparisons.
11. Runs protocol comparison against paper-like augmentation.
12. Saves Grad-CAM, Grad-CAM++, and occlusion explanations.

This is the file that generates the results to use in the paper.

---

## `model_factory.py`

Creates neural network models and feature extractors.

It defines:

### Fine-tuned DL models

- `EfficientNetV2B0`
- `ConvNeXtTiny`
- `MonkeyNetDenseNet201`

These models are trained end-to-end with partial fine-tuning.

### Frozen embedding extractors

For ML baselines, CNNs are used only as feature extractors. The CNN weights are frozen and each image is converted into a feature vector.

Those vectors are then used by classical ML models in `ml_baselines.py`.

---

## `data_utils.py`

Handles image loading and data preparation.

It:

- loads images from class folders
- resizes images to `Config.IMAGE_SIZE`
- converts images to NumPy arrays
- creates `tf.data.Dataset` objects
- builds Keras augmentation layers
- computes class weights
- creates paper-like offline augmented datasets

Important methodological detail:

- online augmentation is applied only when `training=True`
- test data is never augmented in the rigorous protocol

---

## `metrics.py`

Defines all evaluation metrics.

It computes:

- accuracy
- macro F1
- balanced accuracy
- Cohen’s kappa
- Expected Calibration Error
- Brier score
- per-class precision/recall/F1
- confusion matrices
- confused class pairs

This file ensures all model families use the same metric definitions.

---

## `ml_baselines.py`

Implements the hybrid DL + ML baselines.

Pipeline:

```text
image
 → frozen CNN backbone
 → embedding vector
 → optional SMOTE / SMOTEENN
 → classical ML classifier
 → prediction
```

Classifiers:

- Logistic Regression
- SVM
- Random Forest

Sampling strategies:

- none
- SMOTE
- SMOTEENN

Important methodological detail:

SMOTE and SMOTEENN are applied only to embedding vectors and only inside the training fold. They are never applied to raw images and never before splitting.

---

## `stats_utils.py`

Contains the statistical analysis tools.

It implements:

- Friedman test
- Shapiro-Wilk test
- paired t-test
- Wilcoxon signed-rank test
- Holm correction
- Cohen’s dz effect size
- 95% confidence intervals

These tests are used to determine whether differences between configurations are statistically supported.

---

## `gradcam_utils.py`

Contains explainability utilities.

It implements:

- Grad-CAM
- Grad-CAM++
- Occlusion sensitivity

Each explanation is saved as a PNG containing:

- original image
- heatmap overlay
- true class
- predicted class
- confidence

Outputs are stored under:

```text
results/gradcam_test_png/
```

---

## `requirements.txt`

Lists the required Python dependencies.

Install them with:

```bash
pip install -r requirements.txt
```

On CESGA or another cluster, prefer loading an appropriate TensorFlow GPU module instead of installing TensorFlow manually.

---

## `run_orthopox.slurm`

Example SLURM script for running the project on a GPU node.

It loads the required modules, moves to the project directory, checks TensorFlow/GPU availability, and runs:

```bash
python Program.py
```

---

# 13. Detailed Explanation of `Config.py`

## Main execution switch

```python
RUN_RANDOM_SEARCH = True
```

Controls whether the project runs search or final evaluation.

- `True`: run `search.py`
- `False`: run `evaluate.py`

---

## Paths

```python
BASE_PATH = "./data"
VARIANT_NAME = "original"
RESULTS_DIR = "./results"
```

These define where the dataset is read from and where results are saved.

---

## Image setup

```python
IMAGE_SIZE = (224, 224)
CHANNELS = 3
```

All images are resized to 224×224 RGB.

---

## Seeds and cross-validation

```python
SEARCH_SEED = 42
EVAL_SEEDS = [42, 52, 62, 72, 82]
SEARCH_CV = 3
OUTER_CV = 5
```

- `SEARCH_SEED`: controls random search sampling and search folds
- `EVAL_SEEDS`: repeated evaluation seeds
- `SEARCH_CV`: folds used during search
- `OUTER_CV`: folds used during final evaluation

Final evaluation total repetitions per configuration:

```text
len(EVAL_SEEDS) × OUTER_CV
```

Example:

```text
5 seeds × 5 folds = 25 evaluations
```

---

## Search size

```python
MAX_SEARCH_CONFIGS = 20
TOP_N_TO_EVALUATE = 5
```

- `MAX_SEARCH_CONFIGS`: number of candidate configurations tested during search
- `TOP_N_TO_EVALUATE`: number of best configurations evaluated in the final stage

---

## Model search space

```python
BACKBONES = [
    "EfficientNetV2B0",
    "ConvNeXtTiny",
    "MonkeyNetDenseNet201",
]
```

Backbones used in the DL search.

```python
SEARCH_SPACE = {...}
```

Defines all candidate hyperparameter values.

Each sampled configuration is one combination of values from this dictionary.

---

## Optimization

```python
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
WEIGHT_DECAY = 1e-5
USE_CLASS_WEIGHTS = True
```

Controls training regularization.

Class weights are used to reduce the effect of class imbalance in DL models.

---

## Augmentation policies

```python
AUGMENTATION = {
    "none": ...,
    "light": ...,
    "moderate": ...,
}
```

Defines training augmentation strength.

Important:

- augmentation is applied only to training data in the rigorous protocol
- test data is not augmented

---

## Test-time augmentation

```python
TTA_TRANSFORMS = ["identity", "flip_left_right"]
```

If a configuration enables TTA, predictions are averaged across these deterministic transforms.

TTA does not train on test data.

---

## Grad-CAM and XAI

```python
SAVE_GRADCAM = True
SAVE_GRADCAM_PLUS_PLUS = True
SAVE_OCCLUSION_SENSITIVITY = True
XAI_MAX_IMAGES_PER_FOLD = None
```

Controls visual explanation output.

Use:

```python
XAI_MAX_IMAGES_PER_FOLD = 5
```

for debugging or to reduce runtime/storage.

---

## ML baselines

```python
RUN_ML_BASELINES = True
ML_EMBEDDING_BACKBONES = [...]
ML_CLASSIFIERS = [...]
ML_SAMPLING_STRATEGIES = [...]
```

Controls frozen embedding + ML experiments.

Sampling strategies:

- `none`
- `smote`
- `smoteenn`

These are applied only inside training folds.

---

## Protocol comparison

```python
RUN_PROTOCOL_COMPARISON_FOR_BEST = True
PROTOCOL_COMPARISON_SEEDS = [42, 52, 62, 72, 82]
PROTOCOL_COMPARISON_TEST_SIZE = 0.20
PAPER_LIKE_AUG_REPEATS = 11
```

Controls the comparison between:

- original clean split
- train-only augmentation
- paper-like augmented split

---

## Reference paper values

```python
REFERENCE_PAPER_REPORTED_ACCURACY_AUGMENTED = 0.9891
REFERENCE_PAPER_REPORTED_ACCURACY_ORIGINAL = 0.9319
```

These values are saved for convenience when preparing comparison tables.

They should be checked manually before submission.

---

## Output file paths

The final section of `Config.py` defines all CSV and JSON output paths.

This makes output locations easy to modify without editing the pipeline logic.

---

# 14. Output Files

All outputs are saved in:

```text
results/
```

## Search outputs

### `search_results_with_monkeynet_like.csv`

One row per searched configuration.

Contains:

- configuration name
- backbone
- dropout
- dense units
- learning rates
- batch size
- augmentation policy
- TTA flag
- search macro F1 mean/std
- search balanced accuracy
- search kappa
- search ECE
- search Brier

This file is saved incrementally after each completed configuration.

---

### `top_configs.json`

Contains the best `TOP_N_TO_EVALUATE` configurations selected after search.

Used by `evaluate.py`.

---

### `search_summary.json`

Summary of the search stage.

Contains:

- number of evaluated candidates
- number of saved top configurations
- paths to search outputs
- total search time

---

## Main DL evaluation outputs

### `evaluation_dl_plus_monkeynet_stratified_kfold_fold_results.csv`

One row per:

```text
config × condition × seed × fold
```

Contains fold-level metrics.

Use this file for detailed inspection.

---

### `evaluation_dl_plus_monkeynet_stratified_kfold_seed_results.csv`

One row per:

```text
config × condition × seed
```

Each row averages over folds for that seed.

This is the preferred input for statistical testing because seeds are the repeated units.

---

### `evaluation_dl_plus_monkeynet_stratified_kfold_summary.csv`

Final summary table.

Contains:

- mean
- standard deviation
- 95% confidence interval

for each metric and configuration.

This is one of the main tables for the paper.

---

### `evaluation_classwise_precision_recall_f1.csv`

Per-class metrics.

Contains precision, recall, F1, and support for each class.

Useful for clinical interpretation.

---

### `evaluation_confusion_matrices_long_format.csv`

Confusion matrices in long format.

Each row contains:

```text
true class
predicted class
count
```

Useful to build confusion matrix figures.

---

### `evaluation_error_analysis_confused_class_pairs.csv`

Aggregated off-diagonal confusion pairs.

Shows which classes are most frequently confused.

Useful for discussion, especially for clinically relevant errors.

---

## Statistical outputs

### `evaluation_friedman_omnibus_stats.csv`

Friedman omnibus test over DL configurations.

Answers:

> Is there evidence that at least one configuration differs from the others?

---

### `evaluation_pairwise_shapiro_ttest_wilcoxon_holm_stats.csv`

Pairwise comparisons between configurations.

Includes:

- Shapiro-Wilk normality test
- paired t-test or Wilcoxon
- raw p-value
- Holm-adjusted p-value
- Cohen’s dz effect size
- significance flag

---

## ML baseline outputs

### `ml_smote_smoteenn_stratified_kfold_fold_results.csv`

Fold-level results for frozen embeddings + ML models.

---

### `ml_smote_smoteenn_stratified_kfold_seed_results.csv`

Seed-level results for frozen embeddings + ML models.

---

### `ml_smote_smoteenn_summary.csv`

Summary of ML baselines.

Use this to compare:

- Logistic Regression
- SVM
- Random Forest
- none vs SMOTE vs SMOTEENN

---

## Combined DL vs ML outputs

### `combined_dl_vs_ml_smote_smoteenn_summary.csv`

Combined table containing:

- fine-tuned DL models
- MonkeyNet-like baseline
- frozen embeddings + ML baselines

This is useful for the main comparative results table.

---

### `combined_dl_vs_ml_friedman_omnibus_stats.csv`

Friedman test over combined DL and ML configurations.

---

### `combined_dl_vs_ml_pairwise_shapiro_ttest_wilcoxon_holm_stats.csv`

Pairwise tests over combined DL and ML configurations.

---

## Augmentation ablation outputs

### `augmentation_ablation_fold_results.csv`

Fold-level results for augmentation ablation.

---

### `augmentation_ablation_seed_results.csv`

Seed-level results for augmentation ablation.

---

### `augmentation_ablation_summary.csv`

Summary of:

- no augmentation
- light augmentation
- moderate augmentation

---

## Protocol comparison outputs

### `best_model_protocol_comparison_runs.csv`

Seed-level results for the three protocols:

- original-only clean split
- rigorous train-only augmentation
- paper-like augmented split

---

### `best_model_protocol_comparison_summary.csv`

Summary of protocol comparison.

This is important for demonstrating whether paper-like augmentation inflates performance.

---

### `best_model_protocol_comparison_shapiro_ttest_wilcoxon_stats.csv`

Pairwise statistical comparison between protocols.

---

### `best_model_protocol_comparison_reference_paper_metrics.csv`

Stores reference values reported in the dataset paper.

This file is included to simplify table creation, but the reported values should be manually verified before submission.

---

## XAI outputs

### `results/gradcam_test_png/`

Contains visual explanations.

Subfolders:

```text
gradcam/
gradcam_plus_plus/
occlusion_sensitivity/
```

Each PNG includes:

- original image
- explanation overlay
- true class
- predicted class
- confidence

---

# 15. How to Use the Project

## Step 1: Prepare the dataset

Place the dataset in:

```text
data/original/<class_name>/
```

Example:

```text
data/original/Monkeypox/img001.jpg
data/original/Normal/img002.jpg
```

---

## Step 2: Install requirements

Local machine:

```bash
pip install -r requirements.txt
```

CESGA example:

```bash
module purge
module load cesga/2025 gcc/system openmpi/4.1.8 tensorflow/2.15.1-CUDA-system
```

TensorFlow 2.5 should not be used because it does not include ConvNeXt.

---

## Step 3: Run a quick debug search

In `Config.py`, set:

```python
RUN_RANDOM_SEARCH = True
MAX_SEARCH_CONFIGS = 2
SEARCH_CV = 2
```

Then run:

```bash
python Program.py
```

Check that this file appears:

```text
results/search_results_with_monkeynet_like.csv
```

---

## Step 4: Run the full search

In `Config.py`, set for example:

```python
RUN_RANDOM_SEARCH = True
MAX_SEARCH_CONFIGS = 20
SEARCH_CV = 3
```

Then run:

```bash
python Program.py
```

At the end, the code saves:

```text
results/top_configs.json
```

This file contains the best configurations selected during search.

---

## Step 5: Run final evaluation

In `Config.py`, set:

```python
RUN_RANDOM_SEARCH = False
```

Then run:

```bash
python Program.py
```

This will run the full experimental pipeline and generate the final result files.

---

# 16. Recommended SLURM Usage

Example:

```bash
#!/bin/bash
#SBATCH -J orthopox_%j
#SBATCH -o orthopox_%j.o
#SBATCH -e orthopox_%j.e
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-cpu=4G

module purge
module load cesga/2025 gcc/system openmpi/4.1.8 tensorflow/2.15.1-CUDA-system

cd /path/to/Orthopox || exit 1

python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

srun python Program.py
```

No progress bars are used, because progress bars are inconvenient in SLURM logs. The code uses structured `print(..., flush=True)` messages instead.

---

# 17. Recommended Debug Configuration

Before a full run, use:

```python
MAX_SEARCH_CONFIGS = 2
SEARCH_CV = 2
EVAL_SEEDS = [42]
OUTER_CV = 2
XAI_MAX_IMAGES_PER_FOLD = 5
```

After verifying that everything works, restore the full experimental values.

---

# 18. Notes on Computational Cost

The selected advanced techniques are mostly inference-time methods and do not require additional training:

- TTA: extra inference passes
- weighted TTA: almost no extra memory
- temperature scaling: post-processing
- entropy rejection: post-processing
- MC Dropout: multiple stochastic inference passes

The main cost is inference time, not training memory.

For memory control:

- avoid loading multiple models simultaneously
- accumulate TTA probabilities incrementally
- save outputs incrementally
- limit XAI images per fold
- use `float32`
- clear TensorFlow sessions between folds

TensorFlow memory growth guide:  
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

---

# 19. Recommended Reporting in the Paper

The updated search procedure can retain top configurations by model family instead of only selecting the global top-N.

This is recommended because a purely global top-N can select several very similar models and exclude other families.

A stronger design is:

```text
best model per family
+ optional global top extras
```

This supports fairer final comparisons between model families.

---

# 25. Recommended Reporting in the Paper

Recommended tables:

1. Search summary table
2. Final DL summary table
3. Best model per family table
4. Original vs augmented comparison
5. TTA / MC Dropout / calibration ablation
6. ML baseline summary
7. Protocol comparison table
8. Class-wise metrics
9. Statistical test table
10. Calibration table with ECE and Brier score

Recommended figures:

1. Experimental workflow diagram
2. Confusion matrix for best model
3. Grad-CAM / Grad-CAM++ examples
4. Calibration plots
5. Protocol comparison bar chart
6. Family comparison plot

---

# 20. Additional Useful Documentation

- TensorFlow Keras Applications: https://keras.io/api/applications/
- TensorFlow data pipelines: https://www.tensorflow.org/guide/data
- Scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Imbalanced-learn user guide: https://imbalanced-learn.org/stable/user_guide.html
- Keras image augmentation: https://keras.io/api/layers/preprocessing_layers/image_augmentation/
- TensorFlow model training: https://www.tensorflow.org/guide/keras/training_with_built_in_methods
