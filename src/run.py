from src.data_loader import load_data
from src.preprocessing import split_features_target
from src.train import train_model, save_model
from src.evaluate import evaluate

def main():
    df = load_data("data/heart - heart.csv")
    X, y = split_features_target(df)

    model, X_test, y_test = train_model(X, y)

    evaluate(model, X_test, y_test, X.columns)

    save_model(model)

if __name__ == "__main__":
    main()