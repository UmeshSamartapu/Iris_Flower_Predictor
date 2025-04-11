# app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim

import joblib
from sklearn.datasets import load_iris


def visualize_data(df):
    sns.pairplot(df, hue='target')
    plt.savefig("pairplot.png")  # Save plot instead of showing in non-interactive environments


def train_models(X_train, y_train, X_test, y_test):
    print("Training models...\n")

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    print("RandomForest Accuracy:", rf_model.score(X_test, y_test))

    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    print("XGBoost Accuracy:", xgb_model.score(X_test, y_test))

    # TensorFlow / Keras
    keras_model = Sequential([
        Dense(8, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    keras_model.fit(X_train, y_train, epochs=10, verbose=0)
    keras_acc = keras_model.evaluate(X_test, y_test)[1]
    print("Keras Accuracy:", keras_acc)

    # PyTorch model
    class PyTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(X_train.shape[1], 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.fc(x)

    pytorch_model = PyTorchNet()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = pytorch_model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate PyTorch model
    with torch.no_grad():
        y_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_pytorch = pytorch_model(y_test_tensor).numpy()
        y_pred_labels = (y_pred_pytorch > 0.5).astype(int)
        pytorch_acc = np.mean(y_pred_labels.flatten() == y_test)
        print("PyTorch Accuracy:", pytorch_acc)

    # Evaluation reports
    print("RandomForest Confusion Matrix:\n", confusion_matrix(y_test, rf_model.predict(X_test)))
    print("XGBoost Report:\n", classification_report(y_test, xgb_model.predict(X_test)))

    # Save scikit-learn model
    joblib.dump(rf_model, 'rf_model.pkl')
    print("RandomForest model saved as rf_model.pkl")

    return rf_model, xgb_model, keras_model, pytorch_model


def main():
    # Load and preprocess data
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target'] = (df['target'] == 0).astype(int)  # Binary classification

    visualize_data(df)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    train_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
