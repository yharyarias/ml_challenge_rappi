# hooks/mlflow_hook.py
from typing import Any, Dict
import kedro
from kedro.framework.hooks import hook_impl
import mlflow
from kedro.pipeline.node import Node
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


class ModelTrackingHooks:
    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        node_name = Node.name

        # For the nodes of interest, we log the corresponding information
        if node_name == "train_model":
            model = outputs["model"]
            with mlflow.start_run() as run:
                mlflow.sklearn.log_model(model, "model")
                mlflow.set_tag("node_name", node_name)
                mlflow.set_tag("task", "train_model")

        elif node_name == "make_predictions":
            model = mlflow.sklearn.load_model("model")
            # data_validation = outputs["data_validation"]
            # target_predicted = model.predict(data_validation)
            with mlflow.start_run() as run:
                mlflow.set_tag("node_name", node_name)
                mlflow.set_tag("task", "make_predictions")
                mlflow.log_params(outputs)

        elif node_name == "evaluate_model":
            mlflow.log_metrics(outputs)
            with mlflow.start_run() as run:
                mlflow.set_tag("node_name", node_name)
                mlflow.set_tag("task", "evaluate_model")

