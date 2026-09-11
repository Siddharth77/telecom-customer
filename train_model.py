from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'TelcoCustomerChurn.csv'
MODEL_PATH = ROOT / 'model' / 'churn_model.pkl'
MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f'Dataset not found at {DATA_PATH}. Please ensure the file is copied into the data folder.')


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, np.inf],
        labels=['0-12', '13-24', '25-48', '48+'],
        right=False,
    )

    service_cols = [
        'PhoneService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]
    service_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
    service_values = df[service_cols].replace(service_map).astype(int)
    df['has_multiple_services'] = (service_values.sum(axis=1) > 1).astype(int)
    df['charges_per_tenure'] = df['MonthlyCharges'] / df['tenure'].replace(0, 1)
    return df


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = add_engineered_features(df)
    return df


def prepare_customer_record(raw_record: dict) -> dict:
    record = dict(raw_record)
    record['TotalCharges'] = pd.to_numeric(record.get('TotalCharges', 0), errors='coerce')
    record['MonthlyCharges'] = pd.to_numeric(record.get('MonthlyCharges', 0), errors='coerce')
    record['tenure_bucket'] = pd.cut(
        [record.get('tenure', 0)],
        bins=[0, 12, 24, 48, np.inf],
        labels=['0-12', '13-24', '25-48', '48+'],
        right=False,
    )[0]

    service_cols = [
        'PhoneService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]
    record['has_multiple_services'] = int(
        sum(
            1
            for col in service_cols
            if str(record.get(col, 'No')).lower() in {'yes', '1', 'true', 'no internet service'}
        )
        > 1
    )
    record['charges_per_tenure'] = float(record.get('MonthlyCharges', 0) / max(record.get('tenure', 1), 1))
    return record


def build_pipeline(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop',
    )
    return preprocessor


def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str):
    y_pred = pipe.predict(X_test)
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    print(f'\n=== {name} ===')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1 Score: {metrics["f1"]:.4f}')
    print('Confusion Matrix:\n', cm)
    return metrics


def main():
    df = load_and_prepare_data()
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    models = [
        ('tree_depth_3', DecisionTreeClassifier(max_depth=3, random_state=42)),
        ('tree_depth_4', DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('tree_depth_5', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('tree_depth_5_leaf', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)),
        ('tree_depth_6', DecisionTreeClassifier(max_depth=6, random_state=42)),
        ('tree_depth_6_balanced', DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)),
        ('tree_depth_8_balanced', DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)),
    ]

    results = []
    for name, model in models:
        preprocessor = build_pipeline(X_train)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        results.append((name, pipeline, metrics))

    best_name, best_pipeline, best_metrics = max(
        results,
        key=lambda item: (item[2]['recall'], item[2]['f1'], item[2]['accuracy']),
    )

    print(f'\nBest model selected: {best_name}')
    print(json.dumps(best_metrics, indent=2))

    bundle = {
        'pipeline': best_pipeline,
        'feature_columns': X.columns.tolist(),
        'target': 'Churn',
        'model_name': best_name,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f'\nSaved final pipeline to {MODEL_PATH}')

    # EDA visual summary
    notebook_dir = ROOT / 'notebook'
    notebook_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(12, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'churn_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
    plt.title('Monthly Charges by Churn Status')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'monthly_charges_by_churn.png', dpi=200)
    plt.close()

    print('Created summary EDA plots in notebook/')

    # Save a small summary as JSON for later reporting
    summary_path = notebook_dir / 'eda_summary.json'
    summary = {
        'dataset_shape': list(df.shape),
        'churn_counts': df['Churn'].value_counts().to_dict(),
        'best_model': best_name,
        'metrics': best_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
