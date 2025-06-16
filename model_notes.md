# This file documents model version warnings and compatibility notes

## Version Warnings

The following warnings were observed when loading models:

1. scikit-learn model version warning:
   - Models were created with scikit-learn 1.4.2
   - Current environment uses scikit-learn 1.4.1.post1
   - While these warnings don't prevent the app from functioning, they indicate potential compatibility issues

2. XGBoost serialization warning:
   - XGBoost recommends using model.save_model() instead of serializing with pickle/joblib
   - See: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

## Compatibility Notes

These warnings are informational only and don't affect the current functionality of the application. However, for production deployment, consider:

1. Updating scikit-learn to match the version used when training the models
2. Re-saving XGBoost models using the recommended method
