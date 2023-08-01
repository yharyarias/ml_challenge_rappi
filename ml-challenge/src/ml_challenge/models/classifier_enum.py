from enum import Enum

import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class ClassifierType(Enum):
    RANDOM_FOREST = "RANDOM FOREST"
    TREE_MODEL = "TREE MODEL"
    LOGISTIC_REGRESSION_MODEL = "LOGISTIC REGRESSION MODEL"


def get_model(classifier_type):
    if classifier_type == ClassifierType.LOGISTIC_REGRESSION_MODEL:
        return RandomForestClassifier()
    elif classifier_type == ClassifierType.TREE_MODEL:
        return DecisionTreeClassifier()
    else:
        return LogisticRegression()

