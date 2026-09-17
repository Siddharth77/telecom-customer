from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


SERVICE_COLS = [
    'PhoneService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
]


def service_indicator(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {'yes', '1', 'true'}:
        return 1
    if normalized in {'no', '0', 'false', 'no phone service', 'no internet service', 'nan', 'none'}:
        return 0
    return 0


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    service_values = df[SERVICE_COLS].apply(lambda column: column.map(service_indicator)).astype(int)
    df['has_multiple_services'] = (service_values.sum(axis=1) > 1).astype(int)
    df['charges_per_tenure'] = df['MonthlyCharges'] / df['tenure'].replace(0, 1)
    return df


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
    return add_business_features(df)


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = add_engineered_features(df)
    return df


def prepare_customer_record(raw_record: dict) -> dict:
    record = dict(raw_record)
    record['TotalCharges'] = pd.to_numeric(record.get('TotalCharges', 0), errors='coerce')
    record['MonthlyCharges'] = pd.to_numeric(record.get('MonthlyCharges', 0), errors='coerce')
    record['tenure'] = int(record.get('tenure', 0) or 0)
    record['tenure_bucket'] = pd.cut(
        [record.get('tenure', 0)],
        bins=[0, 12, 24, 48, np.inf],
        labels=['0-12', '13-24', '25-48', '48+'],
        right=False,
    )[0]

    record['has_multiple_services'] = int(
        sum(service_indicator(record.get(col, 'No')) for col in SERVICE_COLS) > 1
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


def build_business_insights(df: pd.DataFrame) -> list[str]:
    churn_share = df['Churn'].value_counts(normalize=True).mul(100)
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda values: (values == 'Yes').mean()).mul(100)
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda values: (values == 'Yes').mean()).mul(100)
    tenure_churn = df.groupby('tenure_bucket', observed=False)['Churn'].apply(lambda values: (values == 'Yes').mean()).mul(100)
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda values: (values == 'Yes').mean()).mul(100)

    return [
        f"Churn is imbalanced: {churn_share['Yes']:.1f}% of customers leave, so recall matters more than raw accuracy.",
        f"Month-to-month customers churn the most at {contract_churn.get('Month-to-month', 0):.1f}%, making contract type a strong retention signal.",
        f"Fiber optic customers show the highest churn among internet segments at {internet_churn.max():.1f}%, suggesting service experience or price sensitivity.",
        f"Customers in the first tenure bucket (0-12 months) churn at {tenure_churn.get('0-12', 0):.1f}%, so early-life retention offers are likely to pay off.",
        f"The highest-risk payment method is {payment_churn.idxmax()} at {payment_churn.max():.1f}% churn, which can help target billing-related interventions.",
    ]


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
        (
            'logistic_regression_balanced',
            LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42),
        ),
        ('decision_tree_balanced', DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)),
        (
            'random_forest_balanced',
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            ),
        ),
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
    sns.countplot(data=df, x='Churn', hue='Churn', palette='Set2', legend=False)
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'churn_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', hue='Churn', palette='Set2', legend=False)
    plt.title('Monthly Charges by Churn Status')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'monthly_charges_by_churn.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Churn by Contract Type')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(notebook_dir / 'contract_type_by_churn.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set2')
    plt.title('Churn by Internet Service')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'internet_service_by_churn.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x='tenure_bucket', hue='Churn', palette='Set2')
    plt.title('Churn by Tenure Bucket')
    plt.tight_layout()
    plt.savefig(notebook_dir / 'tenure_bucket_by_churn.png', dpi=200)
    plt.close()

    print('Created summary EDA plots in notebook/')

    # Save a small summary as JSON for later reporting
    summary_path = notebook_dir / 'eda_summary.json'
    summary = {
        'dataset_shape': list(df.shape),
        'churn_counts': df['Churn'].value_counts().to_dict(),
        'best_model': best_name,
        'metrics': best_metrics,
        'visualizations': [
            'churn_distribution.png',
            'monthly_charges_by_churn.png',
            'contract_type_by_churn.png',
            'internet_service_by_churn.png',
            'tenure_bucket_by_churn.png',
        ],
        'business_insights': build_business_insights(df),
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
