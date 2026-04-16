# automate_Bintang.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df


def encode_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    return df

def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_data(X_train, X_test, y_train, y_test, output_path):
    os.makedirs(output_path, exist_ok=True)

    X_train.to_csv(f'{output_path}/X_train.csv', index=False)
    X_test.to_csv(f'{output_path}/X_test.csv', index=False)
    y_train.to_csv(f'{output_path}/y_train.csv', index=False)
    y_test.to_csv(f'{output_path}/y_test.csv', index=False)

def main():
    input_path = '../titanic_raw/Titanic-Dataset.csv'
    output_path = 'titanic_preprocessing'

    print("📥 Loading data...")
    df = load_data(input_path)

    print("🧹 Cleaning data...")
    df = clean_data(df)

    print("🔄 Encoding data...")
    df = encode_data(df)

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("💾 Saving processed data...")
    save_data(X_train, X_test, y_train, y_test, output_path)

    print("✅ Preprocessing selesai!")

if __name__ == "__main__":
    main()