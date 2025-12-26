# Results

## Baseline
- Baseline model: predict train mean
- Baseline MAE: 64.0

## Final Model
- Model: Linear Regression 
- MAE (test): 42.7
- Improvement over baseline: 33%

## Error Analysis
- Residuals are roughly centered around 0, but with some large outliers.
- No strong non-linear pattern observed
- Slight increase in variance for high predictions

## Feature Importance
Top contributing features for :
Larger effects:
- s5
- bmi 

Smaller effects:
- sex
- s1    

## Conclusion
- The model outperforms the baseline
- Linear assumptions seem reasonable
- Further improvement could come from non-linear features or regularization
