def split_features_target(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y