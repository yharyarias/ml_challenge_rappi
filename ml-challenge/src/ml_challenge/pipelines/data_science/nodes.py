# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.pipeline
from typing import Any, Tuple

from ...data_processing.processing import DataPreprocessor
from ...models.classifier_enum import ClassifierType, get_model
from ...models.model_provider import create_pipeline

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split


# Global variables
numerical_columns = ["Age", "Fare", "SibSp", "Parch"]
categorical_columns = ["Pclass", "Sex", "Embarked"]


# Functions
def future_selection(train: pd.DataFrame, test: pd.DataFrame) -> Tuple:
    target_column = "Survived"
    all_features = numerical_columns + categorical_columns
    all_ = numerical_columns + categorical_columns + [target_column]

    # Training data
    train_data = train[all_]
    train_data = train_data[train_data['Embarked'].notna()]
    train_data = train_data.fillna(0)
    train_features = train_data[all_features]
    train_target = train_data[target_column]

    # Test data
    test_features = test[all_features + ["PassengerId"]]
    test_features = test_features[test_features["Embarked"].notna()]
    test_features = test_features.fillna(0)
    return train_features, train_target, test_features


def split_data(train_features: pd.DataFrame, train_target: pd.Series, parameters) -> Tuple:
    data_train, data_validation, target_train, target_validation = train_test_split(
        train_features, train_target, random_state=parameters["random_state"],
        shuffle=True, test_size=parameters["test_size"])
    return data_train, data_validation, target_train, target_validation


def train_model(train_features: pd.DataFrame, train_target: pd.Series) -> sklearn.pipeline.Pipeline:
    preprocessor = DataPreprocessor(categorical_columns, numerical_columns).get_processor()
    model = create_pipeline(preprocessor, get_model(ClassifierType.RANDOM_FOREST))
    model.fit(train_features, train_target)
    return model


def make_predictions(model: sklearn.pipeline.Pipeline, data_validation: pd.DataFrame) -> Any:
    target_predicted = model.predict(data_validation)
    return target_predicted


def evaluate_model(target_validation: pd.Series, target_predicted: np.ndarray) -> Any:
    accuracy = accuracy_score(target_validation, target_predicted)
    balanced_accuracy = balanced_accuracy_score(target_validation, target_predicted)
    precision = precision_score(target_validation, target_predicted, pos_label=1)
    recall = recall_score(target_validation, target_predicted, pos_label=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, balanced_accuracy, precision, recall, f1_score


def plot_result_and_save(model: sklearn.pipeline.Pipeline, data_validation: pd.DataFrame, target_validation: pd.Series):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    disp = PrecisionRecallDisplay.from_estimator(
        model, data_validation, target_validation, pos_label=1, ax=ax1
    )
    plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
    disp.ax_.set_title("Precision-recall curve")
    disp = RocCurveDisplay.from_estimator(
        model, data_validation, target_validation, pos_label=1, ax=ax2
    )
    plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
    disp.ax_.set_title("ROC AUC curve")
    plt.savefig("data/01_raw/plot_result.png")
    plt.close()


def save_predictions(model: sklearn.pipeline.Pipeline, test_features: pd.DataFrame):
    predictions = model.predict(test_features)
    output = pd.DataFrame({'PassengerId': test_features.PassengerId, 'Survived': predictions})
    output.to_csv('data/01_raw/submission.csv', index=False)
