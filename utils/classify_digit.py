import cv2
import numpy as np
import tensorflow as tf

from loguru import logger
from typing import Tuple


def classify(image: np.ndarray) -> Tuple[int, float]:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.shape != (28,28):
        image = cv2.resize(image,(28,28))
    model = tf.keras.models.load_model('./model/best_model.keras')
    logger.info("Successfully loaded model.")
    #normalize image and convert to float
    image = (image / 255.0).astype(np.float32)
    #add addtional dimensions
    image = image.reshape((1,28,28,1))

    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_digit, confidence