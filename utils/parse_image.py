import cv2
import numpy as np

def preprocess_image(image):
    # Load the image and convert it to graysca
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Apply morphological operations to close gaps in edges
    '''kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)'''
    return edges

def find_largest_contour(edges):
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and get the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return approx
    return None

def warp_perspective(image, contour):
    # Get the 4 points of the largest contour
    pts = contour.reshape(4, 2)
    
    # Order points (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Compute width and height
    width = height = 450  # Standard Sudoku size
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def extract_cells(warped):
    grid = np.zeros((9,9,28,28), dtype=np.uint8)
    grid_size = 9
    cell_size = warped.shape[0] // grid_size
    cells = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            x_start, y_start = j * cell_size, i * cell_size
            x_end, y_end = (j + 1) * cell_size, (i + 1) * cell_size
            cell = warped[y_start:y_end, x_start:x_end]
            cell = cv2.resize(cell, (28,28))
            grid[i,j,...] = cell
    return grid

def main(image_path):
    image, edges = preprocess_image(image_path)
    contour = find_largest_contour(edges)
    if contour is None:
        print("No Sudoku grid detected.")
        return
    
    warped = warp_perspective(image, contour)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cells = extract_cells(warped_gray)
    
    # Display extracted cells for testing
    for i in range(9):
        for j in range(9):
            cv2.imshow(f'Cell {i},{j}', cells[i][j])
            cv2.waitKey(100)
    cv2.destroyAllWindows()


