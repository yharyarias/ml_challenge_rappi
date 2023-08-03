from kedro.pipeline import Pipeline, node, pipeline

from .nodes import future_selection, split_data, train_model, make_predictions, evaluate_model, plot_result_and_save, save_predictions


def create_pipeline(**kwargs) -> Pipeline:

    # Add corresponding inputs and outputs to each node

    return pipeline(
        [
            node(
                func=future_selection,
                inputs=["train", "test"],
                outputs=["train_features", "train_target", "test_features"],
                name="future_selection",
            ),
            node(
                func=split_data,
                inputs=["train_features", "train_target", "parameters"],
                outputs=["data_train", "data_validation", "target_train", "target_validation"],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=["train_features", "train_target"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=make_predictions,
                inputs=["model", "data_validation"],
                outputs="target_predicted",
                name="make_predictions",
            ),
            node(
                func=evaluate_model,
                inputs=["target_validation", "target_predicted"],
                outputs=["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"],
                name="evaluate_model",
            ), node(
                func=plot_result_and_save,
                inputs=["model", "data_validation", "target_validation"],
                outputs=None,
                name="plot_result_and_save"
            ), node(
                func=save_predictions,
                inputs=["model", "test_features"],
                outputs=None,
                name="save_predictions"
            )
        ]
    )
