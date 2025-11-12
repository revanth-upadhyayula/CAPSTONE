# 🧠 Ensemble Learning for Cervical Cell Classification

## 📋 Project Overview

This repository is part of a **Final Year B.Tech Project** on automated cervical cell classification using ensemble deep learning and fuzzy logic.
The project builds upon state-of-the-art medical image analysis research and is divided into two complementary repositories:

1. **Current Repository (CAPSTONE)** – Implements the **ensemble modeling and combination logic** for classification.
2. **[Fuzzy Ensemble Repository](https://github.com/revanth-upadhyayula/fuzzy_ensemble)** – Contains the **Flask-based web application** for real-time fuzzy ensemble prediction and image retrieval.

---

## 🔬 Research Context

Cervical cancer remains one of the most common malignancies among women worldwide. Manual screening of pap-smear images is labor-intensive, subjective, and prone to diagnostic variation. The need for **automated, AI-assisted screening systems** has driven extensive research into computer-aided diagnosis (CAD) and image-based learning models.

Our work draws motivation from the review by **Wasswa et al. (2018)**, which comprehensively evaluated image analysis and machine learning techniques for automated cervical cancer screening, emphasizing the potential of ensemble and hybrid classifiers for improving diagnostic accuracy.

The proposed ensemble approach in this project is **adapted from the fuzzy-rank-based ensemble network by Deb & Jha (2023)**, originally developed for breast ultrasound classification. Their method effectively integrates multiple CNN architectures using three nonlinear fuzzy membership functions—**exponential**, **tanh**, and **sigmoid**—to reduce uncertainty in predictions.

This project extends that fuzzy-ensemble principle to **five cervical cell types** from the **SIPaKMeD dataset**, creating a robust and explainable cervical cell classification pipeline.

---

## 🎯 Objectives

* Train multiple CNN models for cervical cell classification
* Apply fuzzy membership-based ensemble fusion
* Achieve stable accuracy across five distinct cell classes
* Perform robust **3-fold cross-validation**
* Export models for integration into the **Flask-based web interface**

---

## 🧫 Cell Classification Categories

The SIPaKMeD dataset includes five cervical cell categories:

1. **im_Dyskeratotic** – Abnormal keratinized cells
2. **im_Koilocytotic** – HPV-related morphological changes
3. **im_Metaplastic** – Transformation zone cells
4. **im_Parabasal** – Basal epithelial cells
5. **im_Superficial-Intermediate** – Superficial epithelial cells

---

## 🏗️ Architecture

### Base Models Used

Seven pre-trained CNN architectures were evaluated:

* DenseNet169
* InceptionV3
* Xception
* InceptionResNetV2
* VGG19
* MobileNetV2
* ResNet50V2

---

### Fuzzy Ensemble Logic

The ensemble applies **fuzzy membership transformation** to model probabilities using three nonlinear functions inspired by Deb & Jha (2023):

```python
expo_equ(p) = 1 - exp(-((p - 1)**2 / 2))
tanh_equ(p) = 1 - tanh(((p - 1)**2 / 2))
norm_equ(p) = 1 / (1 + exp(-p))

final_score(p) = expo_equ(p) × tanh_equ(p) × norm_equ(p)
```

Prediction is based on selecting the class with the **minimum aggregated score** across all selected models.

---

## 📊 Results Summary

### Cross-Validation Performance

| Fold | Best Model Combination                               | Accuracy |
| ---- | ---------------------------------------------------- | -------- |
| 1    | DenseNet169 + InceptionResNetV2 + VGG19 + ResNet50V2 | 98.07%   |
| 2    | InceptionResNetV2 + VGG19 + ResNet50V2               | 98.05%   |
| 3    | InceptionResNetV2 + VGG19 + MobileNetV2 + ResNet50V2 | 98.31%   |

**Average Accuracy:** 98.14%
**Most Consistent Ensemble:** InceptionResNetV2 + VGG19 + ResNet50V2

---

## 📁 Repository Structure

```
CAPSTONE/
├── ensemble1.ipynb                 # Main experiment notebook
├── ensemble_predictions.csv        # Final ensemble predictions
├── top_5_accuracies_fold_1.csv
├── top_5_accuracies_fold_2.csv
├── top_5_accuracies_fold_3.csv
├── top_10_model_combinations_fold_*.docx
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn python-docx jupyter
```

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/revanth-upadhyayula/CAPSTONE
   cd CAPSTONE
   ```
2. Open the notebook:

   ```bash
   jupyter notebook ensemble1.ipynb
   ```
3. Run cells sequentially to:

   * Load model probabilities
   * Evaluate model combinations
   * Generate reports and save results

---

## 🔍 Key Features

* **Comprehensive Ensemble Analysis** – Evaluates all model combinations (2–7 models)
* **Fuzzy Logic Fusion** – Three-component nonlinear membership integration
* **Automated Reporting** – Exports classification summaries in CSV & DOCX
* **Cross-Validation Support** – Ensures robust generalization

---

## 📈 Methodology

1. Extract model prediction probabilities
2. Apply fuzzy membership functions
3. Aggregate fuzzy scores across models
4. Select class with minimum ensemble score
5. Compute performance metrics and rank results

---

## 🔗 Related Repository

➡️ **[fuzzy_ensemble](https://github.com/revanth-upadhyayula/fuzzy_ensemble)**
Flask-based web app for real-time prediction and image retrieval.

---

## 📄 License

This repository is part of an academic research project.
For reuse or modification, please contact the authors.

---

## 🙏 Acknowledgments

* Department of Computer Science and Engineering, SRM University, Amaravati
* SIPaKMeD Dataset (Kaggle)
* TensorFlow and Keras open-source libraries

---

## 📚 References

1. **Wasswa, W., Ware, A., Basaza-Ejiri, A. H., & Obungoloch, J. (2018).**
   *A review of image analysis and machine learning techniques for automated cervical cancer screening from pap-smear images.*
   *Computer Methods and Programs in Biomedicine, 164*, 15–22.
   [https://doi.org/10.1016/j.cmpb.2018.05.034](https://doi.org/10.1016/j.cmpb.2018.05.034)

   *Relevance*: Provides the foundational background on automating cervical cancer screening through machine learning and image analysis techniques.

2. **Deb, S. D., & Jha, R. K. (2023).**
   *Breast Ultrasound Image Classification Using Fuzzy-Rank-Based Ensemble Network.*
   *Biomedical Signal Processing and Control, 85*, 104871.
   [https://doi.org/10.1016/j.bspc.2023.104871](https://doi.org/10.1016/j.bspc.2023.104871)

   *Relevance*: Forms the theoretical foundation for the fuzzy ensemble framework used in this project. The ensemble logic was adapted from this work to suit cervical cytology image classification.

---

**Note:**
This is a research-oriented repository focusing on model evaluation and ensemble logic.
For the complete web implementation, please refer to the [Fuzzy Ensemble Repository](https://github.com/revanth-upadhyayula/fuzzy_ensemble).

---
