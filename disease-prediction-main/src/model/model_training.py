import os
import sys
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml, save_object, setup_mlflow


class ModelTrainer:
    def __init__(self):
        self.params = read_yaml("params.yaml")

    def initiate_model_trainer(self):
        try:
            processed_data_path = os.path.join("data", "processed")
            dataset_names = self.params["data"]["dataset_names"]

            # Setup MLflow
            setup_mlflow()

            trained_models = {}

            for dataset_full_name in dataset_names:
                dataset_folder = dataset_full_name.split("/")[-1]
                dataset_path = os.path.join(processed_data_path, dataset_folder)

                logging.info(f"Training model for dataset: {dataset_folder}")

                train_path = os.path.join(dataset_path, "train.csv")
                train_df = pd.read_csv(train_path)

                model_config = self.params["models"].get(dataset_folder)
                if not model_config:
                    logging.warning(f"No config found for {dataset_folder}, skipping.")
                    continue

                target_column = model_config["target_column"]
                model_type = model_config["model_type"]
                model_params = model_config["params"]
                drop_cols = model_config.get("drop_columns", [])

                if target_column not in train_df.columns:
                    target_column = train_df.columns[-1]

                drop_list = [target_column] + drop_cols
                existing_drop_list = [col for col in drop_list if col in train_df.columns]

                X_train = train_df.drop(columns=existing_drop_list, axis=1)
                y_train = train_df[target_column]

                # Model selection
                if model_type == "logistic_regression":
                    model = LogisticRegression(**model_params)
                elif model_type == "svm":
                    model = svm.SVC(**model_params)
                elif model_type == "random_forest":
                    model = RandomForestClassifier(**model_params)
                else:
                    raise Exception(f"Unsupported model type: {model_type}")

                with mlflow.start_run(run_name=f"Train_{dataset_folder}"):

                    model.fit(X_train, y_train)

                    mlflow.log_params(model_params)
                    mlflow.log_param("dataset", dataset_folder)
                    mlflow.log_param("model_type", model_type)

                    # Log model in MLflow (MLflow 3.x uses `name=` not `artifact_path=`)
                    mlflow.sklearn.log_model(model, name="model")

                    # Save model locally (DVC tracked folder)
                    model_dir = os.path.join("artifacts", "models")
                    os.makedirs(model_dir, exist_ok=True)

                    model_file_path = os.path.join(
                        model_dir, f"{dataset_folder}_model.pkl"
                    )
                    save_object(model_file_path, model)

                    logging.info(f"Model saved at {model_file_path}")

                    trained_models[dataset_folder] = model_file_path

            return trained_models

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()
