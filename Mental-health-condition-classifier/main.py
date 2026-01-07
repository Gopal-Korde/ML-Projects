from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_models

X, y, scaler = preprocess_data(r"C:\Users\gopal\Downloads\archive (2)\mental_health_dataset.csv")
models, X_test, y_test = train_models(X, y)
evaluate_models(models, X_test, y_test)
