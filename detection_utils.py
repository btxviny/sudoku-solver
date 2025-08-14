#detect_border.py

import cv2
import os 
import numpy as np


def adjust_brightness_contrast(image, alpha=1.2, beta=50):
    """
    Adjust the brightness and contrast of the image.
    alpha: contrast control, beta: brightness control
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized

def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce noise and smooth the image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred

def adaptive_threshold(image):
    """
    Apply adaptive thresholding to handle varying lighting conditions.
    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    return adaptive_thresh

def erode(image, kernel_size=(3, 3)):
    """
    Apply morphological erosion to remove noise.
    """
    kernel = np.ones(kernel_size, np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded

def preprocess_image(image):
    """
    Preprocess the image: adjust brightness/contrast, apply CLAHE, Gaussian blur, and adaptive thresholding.
    """
    adjusted_image = adjust_brightness_contrast(image, alpha=1.2, beta=50)  # Adjust brightness/contrast
    equalized_image = clahe(adjusted_image)  # Apply CLAHE for local contrast adjustment
    blurred_image = gaussian_blur(equalized_image)  # Gaussian blur to reduce noise
    thresholded_image = adaptive_threshold(blurred_image)  # Adaptive thresholding
    eroded_image = erode(thresholded_image)  # Erosion to remove noise
    return eroded_image

def detect_border(image):
    # Preprocess the image to enhance the grid visibility
    image_area = image.shape[0] * image.shape[1]
    preprocessed_image = preprocess_image(image)
    
    # Apply Canny edge detection on the preprocessed image
    edges = cv2.Canny(preprocessed_image , 50, 150)  # Adjust Canny thresholds for better edge detection
    
    # Find contours (only external contours for the whole grid)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_with_contours = image.copy()
    pts = None
    for contour in contours:
        if cv2.contourArea(contour) >= 0.1 * image_area: 
            pts = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)  # Tighter epsilon
            if len(pts) == 4:  # Looking for a 4-point quadrilateral (the grid)
                # Draw a rectangle around the largest contour with 4 points
                cv2.polylines(image_with_contours, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                pts = pts.squeeze(1)
                break
            else: 
                pts = None
    return preprocessed_image, image_with_contours, pts
    
def order_points(pts):
    """
    Orders 4 points in [top_left, top_right, bottom_right, bottom_left] order.
    """
    # Sort points by y-coordinate (top two first, bottom two later)
    pts = pts[np.argsort(pts[:, 1])]
    # Separate top and bottom points
    top_points = pts[:2]
    bottom_points = pts[2:]
    # Sort left to right
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def warp_image(image, pts):
    height, width = 128,128
    src_points = order_points(pts)
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    mat, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    warped = cv2.warpPerspective(image, mat, (width, height))
    return warped

def detect_and_warp(image):
    _, _, pts = detect_border(image)
    if pts is not None:
        return warp_image(image,pts)
    else:
        return image