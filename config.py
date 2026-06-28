from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_FILE = DATA_DIR / "customer_churn_1M.csv"

TARGET_COL = "churn"
ID_COL = "customer_id"
DATE_COL = "signup_date"

CATEGORICAL_COLS = [
    "gender", "education", "marital_status", "contract",
    "payment_method", "paperless_billing",
]
BINARY_COLS = [
    "senior_citizen", "has_phone_service", "has_internet_service",
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
]
NUMERICAL_COLS = [
    "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
    "num_services", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb",
    "days_since_last_interaction", "credit_score", "dependents",
]

MODELS_TO_TRAIN = ["lightgbm", "xgboost", "logistic_regression"]
BEST_MODEL_NAME = "best_model"
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_FEATURES = 25

TARGET_ENCODE_COLS = ["contract", "payment_method", "education"]
OPTUNA_TRIALS = 50

API_HOST = "0.0.0.0"
API_PORT = 8000
