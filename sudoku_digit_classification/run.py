import subprocess
from loguru import logger

logger.info("Creating")
subprocess.run(["python", "create_dataset.py"])
logger.info("Creating Features")
subprocess.run(["python", "create_resnet_feature_dataset.py"])
logger.info("Training Classifier")
subprocess.run(["python", "train_xgboost_classifier.py"])