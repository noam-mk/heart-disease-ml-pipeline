# Heart Disease Prediction – ML Pipeline

## Project Overview

This project implements a structured machine learning pipeline to predict heart disease using a RandomForest classifier.

The objective was to build a modular and reproducible ML project with a clean architecture suitable for production-oriented workflows.

---

## Project Structure

heart_ml_project/

- data/ → raw dataset  
- models/ → trained model and generated artifacts  
- src/ → modular pipeline  
- notebooks/ → experimentation (optional)  
- requirements.txt → dependencies  
- README.md → project documentation  

---

## Pipeline Steps

1. Data loading  
2. Feature / target separation  
3. Train / test split  
4. Model training  
5. Cross-validation  
6. Evaluation (confusion matrix, classification report)  
7. Feature importance visualization  
8. Model persistence (joblib)

---

## Model

- RandomForestClassifier (scikit-learn)

---

## Metrics

- Accuracy  
- Cross-validation accuracy  
- Confusion matrix  
- Precision / Recall / F1-score  

---

## How to Run

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
## Run the full pipeline
python -m src.run