import sklearn.pipeline
from sklearn.pipeline import Pipeline


def create_pipeline(preprocessor, classifier) -> sklearn.pipeline.Pipeline:
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline
