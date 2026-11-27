# Breast Cancer Risk Predictor (multi-model ML hackathon project)

## Setup
    pip install -r requirements.txt
    python train.py          # trains 5 models, saves the best
    streamlit run app.py     # launches the web app

## Pipeline
Data (sklearn built-in) -> train/test split -> 5 models
(Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP)
-> 5-fold CV + test metrics -> best model by CV-AUC -> Streamlit app.

## Ideas to extend for the hackathon
- Add XGBoost / LightGBM, or hyperparameter tuning (GridSearchCV / Optuna)
- Add SHAP explanations to the app
- Swap in your own CSV: replace the data-loading block in train.py
- Deploy free on Streamlit Community Cloud or Hugging Face Spaces
