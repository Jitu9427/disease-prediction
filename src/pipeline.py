import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.data_pre.data_ingestion import DataIngestion
from src.data_pre.data_validation import DataValidation
from src.model.model_training import ModelTrainer
from src.model.model_evaluation import ModelEvaluation
from src.model.model_register import ModelRegister

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Pipeline started")
            
            # Data Ingestion
            logging.info(">>>>> Stage: Data Ingestion <<<<<")
            ingestion = DataIngestion()
            processed_data_path = ingestion.initiate_data_ingestion()
            
            # Data Validation
            logging.info(">>>>> Stage: Data Validation <<<<<")
            validation = DataValidation()
            validation.initiate_data_validation()
            
            # Model Training
            logging.info(">>>>> Stage: Model Training <<<<<")
            trainer = ModelTrainer()
            trainer.initiate_model_trainer()
            
            # Model Evaluation
            logging.info(">>>>> Stage: Model Evaluation <<<<<")
            evaluation = ModelEvaluation()
            evaluation.initiate_model_evaluation()
            
            # Model Registration
            logging.info(">>>>> Stage: Model Registration <<<<<")
            register = ModelRegister()
            register.register_models()
            
            logging.info("Pipeline completed successfully")

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
