import os
import sys
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import json
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, read_yaml, setup_mlflow


class ModelEvaluation:
    def __init__(self):
        try:
            self.params = read_yaml("params.yaml")
        except Exception as e:
            raise CustomException(e, sys)

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="weighted", zero_division=0)
        recall = recall_score(actual, pred, average="weighted", zero_division=0)
        f1 = f1_score(actual, pred, average="weighted", zero_division=0)
        return accuracy, precision, recall, f1

    def initiate_model_evaluation(self):
        try:
            processed_data_path = os.path.join("data", "processed")
            dataset_names = self.params["data"]["dataset_names"]

            setup_mlflow()

            all_metrics = {}

            # ✅ Create metrics directory
            metrics_dir = os.path.join("artifacts", "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            for dataset_full_name in dataset_names:
                dataset_folder = dataset_full_name.split("/")[-1]
                dataset_path = os.path.join(processed_data_path, dataset_folder)

                logging.info(f"Evaluating model for dataset: {dataset_folder}")

                test_path = os.path.join(dataset_path, "test.csv")

                # ✅ FIXED MODEL PATH
                model_file_path = os.path.join(
                    "artifacts",
                    "models",
                    f"{dataset_folder}_model.pkl",
                )

                if not os.path.exists(model_file_path):
                    logging.warning(
                        f"Model not found for {dataset_folder} at {model_file_path}, skipping."
                    )
                    continue

                model = load_object(model_file_path)
                test_df = pd.read_csv(test_path)

                model_config = self.params["models"].get(dataset_folder)
                target_column = (
                    model_config["target_column"]
                    if model_config
                    else test_df.columns[-1]
                )
                drop_cols = model_config.get("drop_columns", []) if model_config else []

                drop_list = [target_column] + drop_cols
                existing_drop_list = [
                    col for col in drop_list if col in test_df.columns
                ]

                X_test = test_df.drop(columns=existing_drop_list, axis=1)
                y_test = test_df[target_column]

                prediction = model.predict(X_test)

                accuracy, precision, recall, f1 = self.eval_metrics(
                    y_test, prediction
                )

                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                }

                all_metrics[dataset_folder] = metrics

                logging.info(f"{dataset_folder} metrics: {metrics}")

                with mlflow.start_run(run_name=f"Eval_{dataset_folder}"):
                    mlflow.log_metrics(metrics)
                    mlflow.log_param("dataset", dataset_folder)

            # ✅ Always create metrics.json (even if empty)
            metrics_path = os.path.join(metrics_dir, "metrics.json")

            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=4)

            logging.info(f"Metrics saved at {metrics_path}")

            return all_metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelEvaluation()
    obj.initiate_model_evaluation()
