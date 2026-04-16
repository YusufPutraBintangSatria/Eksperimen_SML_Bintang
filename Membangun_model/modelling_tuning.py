import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def load_data():
    X_train = pd.read_csv('titanic_preprocessing/X_train.csv')
    X_test = pd.read_csv('titanic_preprocessing/X_test.csv')
    y_train = pd.read_csv('titanic_preprocessing/y_train.csv')
    y_test = pd.read_csv('titanic_preprocessing/y_test.csv')

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train():
    X_train, X_test, y_train, y_test = load_data()

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    with mlflow.start_run():
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_score", grid_search.best_score_)

        mlflow.sklearn.log_model(best_model, "model")

        print("Best Params:", best_params)
        print("Accuracy:", acc)

if __name__ == "__main__":
    train()