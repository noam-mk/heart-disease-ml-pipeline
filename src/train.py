from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.config import TEST_SIZE, RANDOM_STATE, CV_FOLDS


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS)
    print("\nCross-validation accuracy:", cv_scores.mean())

    return model, X_test, y_test

def save_model(model, path="models/heart_model.pkl"):
    joblib.dump(model, path)