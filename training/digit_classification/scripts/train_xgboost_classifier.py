import h5py
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Removed seaborn and matplotlib to avoid GUI blocking

# --- Load HDF5 embeddings ---
h5_path = "/home/viny/CV-sudoku-solver/models/weights/digit_embeddings.h5"
with h5py.File(h5_path, "r") as f:
    filenames = f["filenames"][:].astype(str)  # convert bytes to str if necessary
    X = f["features"][:]
    Y = f["labels"][:]

print(f"Loaded {len(filenames)} samples with features shape {X.shape}")

# --- Split Data (Train: 70%, Eval: 15%, Test: 15%) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train size: {len(X_train)}, Eval size: {len(X_eval)}, Test size: {len(X_test)}")

# --- Convert to DMatrix for XGBoost (GPU-enabled) ---
dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
deval = xgb.DMatrix(X_eval, label=y_eval, nthread=-1)
dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)

# --- Define model parameters ---
params = {
    "objective": "multi:softmax",
    "num_class": 10,
    "eval_metric": "mlogloss",
    "tree_method": "hist",  # GPU-accelerated
    "device": "cuda",
    "verbosity": 1
}

# --- Train XGBoost Model ---
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(deval, "eval")],
    early_stopping_rounds=10
)
model.save_model("/home/viny/CV-sudoku-solver/models/weights/xgboost_digit_classifier.model")
print("XGBoost model saved as /home/viny/CV-sudoku-solver/models/weights/xgboost_digit_classifier.model")

# --- Load model with XGBClassifier for prediction ---
model_clf = xgb.XGBClassifier()
model_clf.load_model("/home/viny/CV-sudoku-solver/models/weights/xgboost_digit_classifier.model")

# Predictions
y_pred = model_clf.predict(X_test)

# --- Evaluation Metrics ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix (text only to avoid GUI blocking)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
