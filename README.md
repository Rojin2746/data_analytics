# Diabetes Disease Prediction – Data Analytics & Machine Learning

A data analytics and machine learning mini project that explores the **Pima Indians Diabetes Dataset**, visualizes patient features, compares classification models, and provides a reusable prediction function.

> **Disclaimer:** This project is for educational purposes only. Its predictions are not medical diagnoses and must not be used as a substitute for professional medical advice.

## Project objectives

- Integrate the Pima Indians Diabetes Dataset into the project.
- Explore the dataset and visualize outcome distribution, feature relationships, and patient measurements.
- Handle zero values that represent missing measurements in selected clinical features.
- Split the data into training and test sets, impute missing values, and standardize features.
- Train and compare five machine learning classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion-matrix counts.
- Save the best-performing model and use it to make predictions on new input data.

## Dataset

The project downloads the Pima Indians Diabetes Dataset from:

`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

The dataset contains these columns:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin-fold thickness |
| `Insulin` | 2-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | Target label: 0 = non-diabetes, 1 = diabetes |

For `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, zero values are treated as missing measurements during analysis and preprocessing.

## Technologies used

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- scikit-learn
- Joblib

## Project structure

The notebook creates and uses the following directories and files:

```text
.
├── Rojin_20241CSE0678_data_analytics.ipynb
├── README.md
├── data/
│   ├── diabetes.csv
│   ├── train_processed.csv
│   └── test_processed.csv
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── models/
│   ├── scaler.pkl
│   ├── imputation_values.pkl
│   ├── best_model.pkl
│   └── best_model_info.txt
└── plots/
    ├── outcome_distribution.png
    ├── correlation_heatmap.png
    ├── glucose_vs_bmi.png
    ├── model_comparison.csv
    └── confusion-matrix and dashboard images
```

Generated files appear after the relevant notebook cells have been run. The exact set of plot files depends on which notebook sections are executed.

## Installation

Use Python 3.9 or later (a compatible Python environment with the listed packages is recommended).

1. Clone or download this project.
2. Open a terminal in the project directory.
3. Install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly joblib
```

4. Launch Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

5. Open `Rojin_20241CSE0678_data_analytics.ipynb`.

## How to run

The notebook is the main entry point. Run its cells **from top to bottom** so that the data, generated Python modules, preprocessing artifacts, and trained model are available to later sections.

The workflow is:

1. **Project setup and dataset integration** – creates the folders and downloads the CSV file to `data/diabetes.csv`.
2. **Exploratory data analysis** – inspects the data and saves charts in `plots/`.
3. **Visualization dashboard** – displays distributions and feature comparisons.
4. **Preprocessing** – creates a stratified 80/20 train/test split, replaces selected zero values with missing values, imputes using training-set medians grouped by outcome, and standardizes the features. The scaler and imputation values are saved under `models/`.
5. **Model training and evaluation** – trains the five classifiers and saves comparison metrics and the best model.
6. **Prediction** – runs a sample prediction using the saved preprocessing artifacts and model.

The notebook also writes the reusable modules into `src/` using its file-writing cells.

## Machine learning models

The project compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

The model with the highest **F1-score** on the test set is selected as the best model and saved as `models/best_model.pkl`.

## Evaluation metrics

The notebook reports:

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of positive predictions that are correct.
- **Recall:** proportion of actual positive cases identified.
- **F1-score:** harmonic mean of precision and recall.
- **TN:** true negatives.
- **FP:** false positives.
- **FN:** false negatives.
- **TP:** true positives.

The results are saved to `plots/model_comparison.csv`, with confusion-matrix visualizations saved under `plots/`.

## Making a prediction

After running the notebook through model training, use the prediction function from the project root:

```python
from src.prediction import predict_diabetes

result = predict_diabetes(
    pregnancies=1,
    glucose=140,
    bp=74,
    skin=25,
    insulin=110,
    bmi=28.4,
    dpf=0.55,
    age=32
)

print(result)
```

The function returns a prediction label, an estimated probability, or an input-validation error. The saved scaler, imputation values, and best model must exist before calling it.

## Notes and limitations

- The train/test split uses `test_size=0.2`, `random_state=42`, and stratification by the target.
- Imputation and scaling parameters are learned from the training data and applied to the test data.
- The prediction function validates that inputs are non-negative and uses saved model artifacts.
- Results depend on the dataset and the environment in which the notebook is run.
- This is a learning project, not a clinically validated diagnostic system.

## Author

**Rojin**  
**Project:** Data Analytics – Diabetes Disease Prediction
