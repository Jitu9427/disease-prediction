import sys
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml, setup_mlflow


class ModelRegister:
    def __init__(self):
        try:
            self.params = read_yaml("params.yaml")
        except Exception as e:
            raise CustomException(e, sys)

    def register_models(self):
        try:
            dataset_names = self.params["data"]["dataset_names"]

            setup_mlflow()
            client = MlflowClient()

            for dataset_full_name in dataset_names:
                dataset_folder = dataset_full_name.split("/")[-1]
                model_name = f"Model_{dataset_folder}"

                logging.info(f"Registering model for: {dataset_folder}")

                model_path = os.path.join(
                    "artifacts",
                    "models",
                    f"{dataset_folder}_model.pkl"
                )

                if not os.path.exists(model_path):
                    logging.warning(
                        f"Model file not found for {dataset_folder}"
                    )
                    continue

                try:
                    # Load model
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                    # 🔥 Start run and log model correctly
                    with mlflow.start_run(run_name=f"Register_{dataset_folder}") as run:

                        logged_model = mlflow.sklearn.log_model(
                            sk_model=model,
                            name="model"   # ✅ IMPORTANT (not artifact_path)
                        )

                        run_id = run.info.run_id

                    # 🔥 Register using model URI from logged model
                    model_uri = f"runs:/{run_id}/model"

                    mv = mlflow.register_model(
                        model_uri=model_uri,
                        name=model_name,
                    )

                    logging.info(
                        f"Registered {model_name} version {mv.version}"
                    )

                    # Move to staging
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Staging",
                        archive_existing_versions=True,
                    )

                    logging.info(
                        f"{model_name} moved to Staging"
                    )

                except Exception as e:
                    logging.error(
                        f"Registration failed for {model_name}: {e}"
                    )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelRegister()
    obj.register_models()
