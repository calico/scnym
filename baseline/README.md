# Baseline Cell Identity Classification Methods

Here, we provide code for training and predicting cell identity annotations using a varienty of baseline models.
Relevant code for models implemented in `python` is in `baseline.py` and code for models implemented in `R` is in `baseline.R`

## Python Baselines

1. scmap-cell-exact: scmap-cell with exact kNN search
2. Support Vector Machine (SVM)
3. Harmony-SVM: integration with `harmony`, followed by SVM classification.
4. scANVI

## R Baselines

1. LIGER-SVM: Integration with `LIGER`, followed by SVM classification.
2. scmap-cell
3. scmap-cell-cluster
4. singleCellNet
5. CHETAH
6. scPred