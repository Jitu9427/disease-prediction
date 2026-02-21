import sys
import yaml
import pickle
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

from src.exception import CustomException

def setup_mlflow():
    """
    Sets up MLflow tracking URI and credentials.
    Uses dagshub.init if available for DAGsHub URIs to ensure artifact upload works.
    """
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
             return

        # Check if it's a DAGsHub URI
        if "dagshub.com" in tracking_uri:
            try:
                import dagshub
                # Extract repo owner and name from URI
                # URI format: https://dagshub.com/<owner>/<repo>.mlflow
                parts = tracking_uri.split("dagshub.com/")[-1].split(".mlflow")[0].split("/")
                if len(parts) == 2:
                    owner, repo = parts
                    
                    # Ensure token is set for dagshub client
                    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
                    if token:
                        os.environ["DAGSHUB_USER_TOKEN"] = token
                    
                    print(f"Initializing DAGsHub for {owner}/{repo}")
                    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
                    return
            except ImportError:
                print("dagshub package not installed, falling back to standard mlflow setup.")
            except Exception as e:
                print(f"Failed to init dagshub: {e}")

        # Fallback
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Multiple-Disease-Prediction")
        
    except Exception as e:
        raise CustomException(e, sys)

def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)
