# src/data_pre/data_ingestion.py

import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml
from dotenv import load_dotenv
load_dotenv()

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw")
    processed_data_path: str = os.path.join("data", "processed")
    dataset_names: list = None


class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()

            params = read_yaml("params.yaml")

            self.ingestion_config.dataset_names = params["data"]["dataset_names"]

            self.test_size = params["data"].get("test_size", 0.2)
            self.random_state = params["data"].get("random_state", 42)

        except Exception as e:
            raise CustomException(e, sys)

    def download_and_split_dataset(self, dataset_name):
        try:
            import kaggle
            kaggle.api.authenticate()

            dataset_folder_name = dataset_name.split("/")[-1]
            dataset_raw_path = os.path.join(
                self.ingestion_config.raw_data_path, dataset_folder_name
            )

            os.makedirs(dataset_raw_path, exist_ok=True)

            logging.info(f"Downloading dataset: {dataset_name}")

            kaggle.api.dataset_download_files(
                dataset_name,
                path=dataset_raw_path,
                unzip=True,
            )

            logging.info("Download completed")

            # Find CSV file
            csv_files = [
                f for f in os.listdir(dataset_raw_path) if f.endswith(".csv")
            ]

            if not csv_files:
                raise Exception(f"No CSV file found in {dataset_name}")

            data_file_path = os.path.join(dataset_raw_path, csv_files[0])

            df = pd.read_csv(data_file_path)

            train_set, test_set = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            dataset_processed_path = os.path.join(
                self.ingestion_config.processed_data_path, dataset_folder_name
            )

            os.makedirs(dataset_processed_path, exist_ok=True)

            train_path = os.path.join(dataset_processed_path, "train.csv")
            test_path = os.path.join(dataset_processed_path, "test.csv")

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)

            logging.info(f"Saved train/test for {dataset_folder_name}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.processed_data_path, exist_ok=True)

            for dataset in self.ingestion_config.dataset_names:
                self.download_and_split_dataset(dataset)

            logging.info("All datasets downloaded and split successfully")

            return self.ingestion_config.processed_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
