import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import confusion_matrix, classification_report


# Load and normalize MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

logger.info("Dataset loaded and normalized.")

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

logger.info("Model compiled successfully.")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("./model/best_model.keras", save_best_only=True)

logger.info("Starting training...")

# Train the model
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    validation_split=0.2, 
                    batch_size=64,  
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)  # Suppress console output

logger.success(f"Training completed. Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
logger.success(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")

# Generate predictions
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

# Compute confusion matrix
# Classification report
class_report = classification_report(y_test, y_pred)
logger.info("\n" + class_report)

# Save the model
model.save('./model/digit_classifier.keras')
logger.success("Model saved as 'digit_classifier.keras'.")
