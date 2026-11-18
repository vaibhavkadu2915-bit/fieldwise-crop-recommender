# Soil-based Crop Recommender

A multinomial logistic regression model learns crop labels from four soil measurements (`N`, `P`, `K`, and `ph`). A small Streamlit app collects these measurements and displays the model prediction and its three highest class probabilities.

## Project layout

```text
data/soil_measures.csv    # training data (not committed)
src/train_model.py        # validation, training, evaluation, model export
models/crop_model.joblib  # trained model used by the deployed app
app.py                    # Streamlit user interface
```

## Dataset format

Place `soil_measures.csv` in `data/` to retrain locally. The dataset is excluded from the public repository; the committed model artifact lets the deployed app make predictions without exposing training data. The CSV must contain numeric columns `N`, `P`, `K`, `ph` and a target column named `crop`. Each row should represent an observed soil sample with its associated crop label. Use data you have permission to use, and record its source for your project report.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train_model.py
streamlit run app.py
```

The training script creates a stratified 80/20 holdout split, fits a standardized multinomial logistic regression model, prints accuracy and weighted/macro F1, and saves the fitted pipeline. Run training again whenever the dataset or training configuration changes.

## Notes

The app assumes `N`, `P`, and `K` use the same units and scale as the training dataset. Its probability values are model scores, not calibrated agronomic certainty. The benchmark dataset in the accompanying report has only four soil measurements; if you choose a dataset with additional variables, update the feature list and form together before training.

## Deploy on Streamlit Community Cloud

Create a new app from this public repository, select the `master` branch, and set the main file path to `app.py`. The app loads `models/crop_model.joblib`, which is included in the repository. The raw CSV is intentionally not published.
