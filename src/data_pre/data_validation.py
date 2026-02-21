import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml

class DataValidation:
    def __init__(self):
        try:
            params = read_yaml("params.yaml")
            self.dataset_names = params["data"]["dataset_names"]
            self.processed_data_path = os.path.join("data", "processed")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_file(self, file_path):
        try:
            logging.info(f"Validating file: {file_path}")
            df = pd.read_csv(file_path)

            # Check for empty dataset
            if df.empty:
                raise Exception(f"Dataset at {file_path} is empty")

            # Check for null values
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                logging.warning(f"Dataset at {file_path} contains {null_count} null values")
                # User requirement: "Raise exception if validation fails"
                # We can choose to be strict or lenient. Given "clean code" requested:
                # Let's fail if it's significant, or just warn. 
                # User asked for "Raise exception if validation fails".
                # But typically real datasets have missing values. 
                # I will Log Error and raise exception to satisfy the requirement strictly.
                raise Exception(f"Validation failed: Dataset at {file_path} contains {null_count} null values")

            logging.info(f"Validation successful for {file_path}")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        logging.info("Initiating data validation for all datasets")
        try:
            validation_status = True
            
            for dataset_name in self.dataset_names:
                dataset_folder_name = dataset_name.split("/")[-1]
                dataset_path = os.path.join(self.processed_data_path, dataset_folder_name)
                
                train_path = os.path.join(dataset_path, "train.csv")
                test_path = os.path.join(dataset_path, "test.csv")

                if not os.path.exists(train_path) or not os.path.exists(test_path):
                     raise Exception(f"Train/Test files not found for {dataset_name}")

                # Validate train and test
                self.validate_file(train_path)
                self.validate_file(test_path)
            
            logging.info("All datasets validated successfully")
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataValidation()
    obj.initiate_data_validation()
