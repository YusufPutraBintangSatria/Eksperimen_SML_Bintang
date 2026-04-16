import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    X_train = pd.read_csv('titanic_preprocessing/X_train.csv')
    X_test = pd.read_csv('titanic_preprocessing/X_test.csv')
    y_train = pd.read_csv('titanic_preprocessing/y_train.csv')
    y_test = pd.read_csv('titanic_preprocessing/y_test.csv')

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 🔥 TAMBAHAN PENTING (WAJIB)
        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    train()