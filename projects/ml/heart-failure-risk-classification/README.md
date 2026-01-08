# Results — Heart Failure Risk Classification

## Dataset
- Total samples: 299
- Death events: ~32%
- Non-death events: ~68%

The dataset shows a moderate class imbalance, which motivates the use of PR-AUC, class weighting, and recall-oriented threshold analysis to minimize false negatives.
 
## Training and Evaluation

Random Forest achieves the best global discrimination (ROC-AUC, PR-AUC), while Decision Tree and HistGradientBoosting show higher recall at the default threshold.

In a clinical context, missing high-risk patients is costly.

A decision threshold was selected independently for each model to satisfy the constraint of Recall (DEATH_EVENT = 1) ≥ 0.80


## Conclusion

- Random Forest provides the best balance between discrimination and clinical sensitivity. (figures)
- Threshold tuning is essential: default 0.5 is suboptimal for medical risk prediction.
- PR-AUC and recall-oriented evaluation are more informative than accuracy alone in this context.

### Limitations
- Small dataset from a single cohort
- No external validation
- Potential instability of thresholds across populations