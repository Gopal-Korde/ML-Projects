import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path):
    """
    Loads dataset, encodes categorical features,
    scales numerical features, and separates target.
    """

    # Load dataset
    df = pd.read_csv(path)

    # Drop completely empty columns (safety)
    df.dropna(axis=1, how="all", inplace=True)

    # Automatically detect target column (assumes last column is target)
    target_column = df.columns[-1]

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = encoder.fit_transform(df[col])

    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
