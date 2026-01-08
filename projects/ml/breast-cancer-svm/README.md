# Results — Breast Cancer Classification (SVM)

## Dataset
- Breast Cancer Wisconsin (scikit-learn)
- 569 samples
- Target: malignant vs benign
- Moderate class imbalance (benign majority)

For clinical relevance, the positive class corresponds to malignant tumors.


## Model Selection and Tuning

Several models were compared:
- Logistic Regression
- SVM (linear kernel)
- SVM (RBF kernel)


Hyperparameters were optimized using **Optuna** with cross-validation, targeting **recall on the malignant class**.


## Final Model

The best-performing model was an **SVM with RBF kernel**, tuned with optimized values of `C` and `gamma`.

At threshold = 0.50 on the test set:
- High ROC-AUC (strong global discrimination)
- Very high recall for malignant tumors
- Low number of false negatives

The confusion matrix shows that most malignant cases are correctly identified, with only a few misclassifications.


## Interpretation

Exploratory analysis (2D scatter plots and PCA) showed that:
- Several features are highly discriminative
- Classes are almost linearly separable, with local non-linear overlaps

This explains why SVM performs particularly well, and why the RBF kernel slightly improves performance over the linear version.


## Conclusion

- SVM is well suited for high-dimensional medical data with correlated features
- Feature scaling is critical for good performance
- Non-linear kernels help capture subtle overlaps between classes
- Optimizing recall is essential in a medical context to minimize missed cancer cases

### Limitations
- Single dataset
- No external validation
- Threshold fixed at 0.5 (could be optimized further for clinical use)
