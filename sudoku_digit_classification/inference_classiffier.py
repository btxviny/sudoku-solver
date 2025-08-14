import torch
import torch.nn as nn
from torchvision import models, transforms
import xgboost as xgb
import numpy as np
import cv2
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the pretrained ResNet18 model (as feature extractor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
resnet.to(device)
resnet.eval()

# Load the trained XGBoost classifier
model = xgb.XGBClassifier()
model.load_model("xgboost_digit_classifier.model")

def classify_digit(image):
    # Step 2: Apply transformations
    if len(image.shape) != 3:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU if available

    # Step 3: Extract features using ResNet18
    with torch.no_grad():
        features = resnet(transformed_image)  # Output shape: [batch, 512, 1, 1]
        features = features.view(features.size(0), -1).cpu().numpy()  # Flatten and move to CPU

    # Step 4: Classify the extracted features using the XGBoost model
    probs = model.predict_proba(features)  # Get class probabilities
    predicted_class = np.argmax(probs, axis=1)  # Get the class with highest probability
    confidence = np.max(probs, axis=1) 
    return predicted_class, confidence, probs


if __name__ == "__main__":
    # Test the function
    image_path  = '/home/viny/Desktop/Sudoku-Solver/sudoku_digit_classification/digits/images/mnist_4.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred, confidence, probs = classify_digit(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    print(f"Prediction: {pred}, Confidence: {confidence}")