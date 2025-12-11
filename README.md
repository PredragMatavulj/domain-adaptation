# Weakly supervised domain adaptation for improving automatic airborne pollen classification

This repository presents a novel approach for improving the identification of pollen particles using weakly supervised domain adaptation. The method is designed to work well even with only a small amount of labeled data. By combining transfer learning, specifically domain adaptation, nested cross-validation, and uncertainty analysis, the approach provides more accurate and reliable pollen classification across different imaging devices.

The results and method are published here:  
**https://link.springer.com/article/10.1007/s10489-024-06021-9**
---

## Method Summary

Our approach builds on a CNN trained to classify pollen types from airborne measurement data by integrating weakly supervised domain adaptation. We first train a baseline multi-modal CNN with a labelled dataset for pollen classification. Then, to bridge the gap between controlled labelled datasets and real-world operational data, we retrain the model using expert-verified measurements from manual pollen counts. This domain adaptation step leverages large amounts of unlabeled operational data while minimizing the discrepancy between automatic predictions and manual standards. By fine-tuning the classifier, the adapted model improves the correlation with manual measurements by 23% and reduces prediction uncertainty by 38% compared to the baseline model.

---

## Repository Structure

### `__crossvalidate_CNN.py`
Runs nested cross-validation on a CNN model.  
This script helps evaluate the model in a robust way and selects the best hyperparameters.

### `__train_model.py`
Trains the final model using all available data and the hyperparameters selected during cross-validation.

### `__retrain_model.py`
Retrains the previously trained model by adding more data (often treated as ground truth).  
This step helps fine-tune and improve the model’s performance. More information is available in the paper **ola2023** (found in the repository).

### `__calculate_correlation_distributions.py`
Calculates correlation distributions to measure uncertainty in both the model predictions and the ground truth.  
This provides insights into how reliable the classifications are. Details can be found in the **stoten2022** paper (included in the repository).

All scripts use helper functions and modules stored in the `Libraries/` folder.

---

## Why Use This Approach?

- Works well with small labeled datasets  
- Can be used across different devices  
- Improves accuracy through transfer learning and retraining  
- Provides uncertainty estimates for better interpretation of results  
- Fully reproducible workflow for scientific use

---

## Citation

If you use this work in your research, please cite:
Matavulj, P., Jelic, S., Severdija, D. et al. Domain adaptation for improving automatic airborne pollen classification with expert-verified measurements. Appl Intell 55, 430 (2025). https://doi.org/10.1007/s10489-024-06021-9

---

## Contributions

Contributions and suggestions are welcome.  
Feel free to open an issue or submit a pull request.
