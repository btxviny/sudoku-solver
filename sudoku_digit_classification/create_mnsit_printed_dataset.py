import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from torchvision.datasets import MNIST

# ========== Synthetic digit generation (your code) ==========
def apply_perspective_transform(img):
    h, w = img.shape[:2]
    max_x_offset = int(w * 0.05)
    max_y_offset = int(h * 0.05)
    
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_pts = np.float32([
        [random.randint(0, max_x_offset), random.randint(0, max_y_offset)], 
        [w-1-random.randint(0, max_x_offset), random.randint(0, max_y_offset)], 
        [random.randint(0, max_x_offset), h-1-random.randint(0, max_y_offset)], 
        [w-1-random.randint(0, max_x_offset), h-1-random.randint(0, max_y_offset)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h))

def apply_random_rotation(img):
    angle = random.randint(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def add_noise(image):
    h, w = image.shape
    noise = np.random.normal(0, 5, (h, w))
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def generate_digit_image(digit, size=(28,28)):
    img = np.zeros((100,100), dtype=np.uint8)
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    thickness = random.randint(1,3)
    font_scale = random.uniform(2,3.5)
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, str(digit), (text_x,text_y), font, font_scale, 255, thickness)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = apply_random_rotation(img)
    img = apply_perspective_transform(img)
    img = add_noise(img)
    return img

def generate_empty_image(size=(28,28)):
    img = np.zeros((100,100), dtype=np.uint8)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = apply_random_rotation(img)
    img = apply_perspective_transform(img)
    img = add_noise(img)
    return img

def save_synthetic_digit(idx, images_path):
    # 10% empty cells
    if random.randint(1,10) == 1:
        img = generate_empty_image()
        label = 0
    else:
        label = random.randint(1,9)
        img = generate_digit_image(label)
    path = f"{images_path}/{idx}.jpg"
    cv2.imwrite(path, img)
    return f"{path},{label}\n"

# ========== MNIST integration ==========
def save_mnist_digit(idx, mnist_img, mnist_label, images_path):
    img = np.array(mnist_img, dtype=np.uint8)
    img = cv2.resize(img, (28,28))
    # Optional: add small synthetic transforms
    img = apply_random_rotation(img)
    img = apply_perspective_transform(img)
    path = f"{images_path}/mnist_{idx}.jpg"
    cv2.imwrite(path, img)
    return f"{path},{mnist_label}\n"

# ========== Main Dataset Creation ==========
if __name__ == "__main__":
    logger.info("Creating Hybrid Dataset (Synthetic + MNIST)")
    
    images_path = "digits/images"
    labels_path = "digits/labels.txt"
    os.makedirs(images_path, exist_ok=True)
    
    # Load MNIST first to know how many images to match
    mnist_data = MNIST(root="mnist_data", train=True, download=True)
    num_mnist = len(mnist_data)  # 60000
    num_synthetic = num_mnist    # Match MNIST size
    num_threads = 10
    
    labels = []
    
    # Generate synthetic digits
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        labels += list(tqdm(executor.map(lambda idx: save_synthetic_digit(idx, images_path),
                                         range(num_synthetic)), total=num_synthetic, desc="Generating Synthetic Digits"))
    
    # Save MNIST digits
    for idx in tqdm(range(num_mnist), total=num_mnist, desc="Saving MNIST Digits"):
        mnist_img, mnist_label = mnist_data[idx]
        labels.append(save_mnist_digit(idx, mnist_img, mnist_label, images_path))
    
    # Write labels
    with open(labels_path, "w") as f:
        f.writelines(labels)
    
    logger.info(f"Hybrid dataset creation completed! Total images: {len(labels)}")

