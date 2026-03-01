from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_

    plt.figure(figsize=(10,6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance - RandomForest")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.show()